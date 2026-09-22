# HeadAtlas — 多模态注意力头功能图谱：详细实验计划

目标会议：CVPR 2027（摘要 2026-11-10 / 全文 2026-11-16 AOE）。起点 2026-09-22，共 58 天。

## 0. 论文骨架

一句话：首次对开源 MLLM（LLaVA-1.5-7B / Qwen2.5-VL-7B / InternVL2.5-8B）逐层逐头绘制
功能图谱（grounding / 空间 / OCR / sink / 指令敏感 / 幻觉关联 六类角色），用置零消融 +
activation patching 证明因果性，并给出 head-selective 免训练注意力重偏置，在 POPE/CHAIR
上超过全局偏置基线 PAI 与对比解码 VCD。

- 贡献 1（分析，保底）：Head Taxonomy Atlas —— ~2600 个头 × 6 类功能指纹 × 3 模型。
- 贡献 2（协议）：头级因果验证协议（消融 ΔAcc + 头输出 patching 翻转率）。
- 贡献 3（方法）：head-selective 重偏置（按图谱分工定 per-head λ），training-free。

## 1. 代码基础设施（本仓库）

| 组件 | 选型 | 位置 |
|---|---|---|
| 评测骨架 | lmms-eval 思路自建轻量 harness（POPE/CHAIR 自实现，避免重依赖） | `headatlas/eval/` |
| 模型加载 | HF：`llava-hf/llava-1.5-7b-hf`、`Qwen/Qwen2.5-VL-7B-Instruct`、`OpenGVLab/InternVL2_5-8B`(实验) | `models/wrapper.py` |
| 注意力提取 | `attn_implementation="eager"` + forward hook 逐层切片聚合（只保留需要的 query 行/列，不落盘全矩阵） | `models/attention_grabber.py` |
| 注意力干预 | 自注册 attention function（`headatlas_eager`），KV 列加 per-head bias | `models/biased_attention.py` |
| 头消融/patching | o_proj 输入按头切片置零 / 均值替换 / 干扰 run 替换 | `interventions/` |
| 聚类与可视化 | scikit-learn（k-means + silhouette + PCA），umap-learn 可选 | `metrics/taxonomy.py` |

关键工程点（务必遵守）：
1. 必须用 eager attention 才能导出每头注意力权重；上下文保持短（图 576 token + 问题 ~40），单层
   临时权重 < 1 GB，24 GB 卡安全。
2. GQA 模型（Qwen/InternVL）注意力矩阵按 **query head** 展开（28/32 头），LLaVA 是 32/32 MHA。
3. 只提取「答题位（prompt 末 token）→ 图像 token 列」的行切片和「图像 token 行 → 前 K 列」的块切片，
   在线聚合，显存与磁盘都可控。
4. 所有实验 bf16；评测 greedy decode，max_new_tokens ≤ 32。

## 2. S1 探针电池（每模型每探针 400 样本，3 seeds 子采样）

| 探针 | 功能假设 | 数据 | GT | 每头指纹特征 |
|---|---|---|---|---|
| P1 grounding | 指称对象定位头 | RefCOCO/+/g val 子采样 | GT box | mass-in-box、top-10% patch precision、pointing 命中率、box IoU |
| P2 spatial | 空间关系头 | GQA spatial 子集（可选，缺数据用 RefCOCO 空间短语） | GT box | 同 P1，按谓词（left/right/between）分组 |
| P3 conditioning | 指令敏感/属性跟随头 | 同图双问（颜色修饰 vs 无修饰），RefCOCO 构造 | — | 两问的图像注意力 KLD（条件敏感度）、注意力指向 GT 的切换正确率 |
| P4 ocr | 文本锚定头 | PIL 合成文字图（零下载，GT 精确） | 文字 box | mass-in-textbox、top-k precision |
| P5 sink | sink/惰性头 | 均匀噪声图 + 短指令（无内容） | — | 前 16 token 注意力份额、跨噪声图稳定性(1-CV)、头熵 |
| P6 occlusion | 失觉重路由头 | RefCOCO 图 + GT box 黑块遮挡 | GT box | 遮挡后仍在遮挡区的质量（错误绑定）、注意力重路由熵 |
| P8 halluc-link | 幻觉责任头 | POPE random/popular/adversarial | yes/no + yes 类 GT box | 答案正确性 × 头质量-in-GT 点二列相关；yes 偏置头的 mass 偏差 |

产出：`outputs/raw/{model}/{probe}.npz` → `outputs/fingerprints/{model}.npz/csv`。

## 3. S2 因果验证

1. **头消融**：按 S1 聚类把头分为 6 类；每类取 top-3 特征头，o_proj 输入切片置零，
   在 P1 grounding / POPE / P3 上测 ΔAcc。预期：grounding 头消融 → P1 崩、POPE 缓；
   sink 头消融 → 几乎无影响（对照）。
2. **Activation patching**：clean run（正确指称）↔ corrupted run（指向同图另一对象的指称），
   在「prompt 末 token + 首 token 生成位」互换目标头的 o_proj 输入切片 → 记录答案翻转率。
   grounding 头翻转率应显著高于随机头（这把相关性升格为因果证据）。
3. 产出：任务×头类因果热图、翻转率表、3 组消融曲线图。

## 4. S3 方法组件：head-selective 重偏置

- λ 向量：grounding 类头 +α、sink/惰性头 −α（α∈{0.25,0.5,1,2} 网格），只加在图像 token 的 KV 列。
- 对照：① 无干预；② uniform-λ（全头 +α，等价 PAI 的消融形式）；③ VCD（对比解码，α=1）。
- 评测：POPE 三 split F1/acc、CHAIR@COCO-val500（CHAIRs/CHAIRi）、AMBER（可选）。
- 附加消融：随机头选 λ、反角色选 λ、单模型 vs 跨模型迁移 λ（Qwen 图谱迁移到 LLaVA）。

## 5. 数据集与下载

| 数据集 | 用途 | 获取方式（`scripts/prepare_data.py`） | 大小 |
|---|---|---|---|
| COCO val2014 图像（懒下载） | P1/P6/P8 | images.cocodataset.org 单文件 URL + 本地缓存 | ~150 MB（1200 张） |
| POPE 三 split json | P8/S3 | RUCAIBox/POPE GitHub raw | <10 MB |
| RefCOCO/+/g | P1/P3/P6/P2 备选 | 官网 refs(unc).pkl + instances_val2014.json（README 给镜像），转 JSONL | ~1 GB |
| GQA（可选） | P2/P7 | HF `lmms-lab/GQA` 或本地 | 可选 |
| CLEVR（可选） | P3 加强 | 本地目录 | 可选 |
| 合成文字图 | P4 | 生成式，无需下载 | 0 |

---

## 6. 指标与统计规范

- 空间对齐：mass-in-box（注意力质量比）、top-k precision（k=10% patch，报告 chance=box 面积占比作对照）、
  pointing 命中率、box-IoU。
- Sink：前 16 token 份额、跨样本稳定性 = 1 − std/mean、头内注意力熵。
- 条件敏感：双问图像注意力 KLD（对称化）。
- 因果：消融 ΔAcc（百分点）、patching 答案翻转率（含 95% CI）。
- 幻觉：POPE acc/precision/recall/F1（yes 为正类）、CHAIRs/CHAIRi。
- 统计：所有逐头分数报 bootstrap 95% CI（400 样本 × 3 子采样）；跨层比较先做层内 z-score
  （控制 softmax 层间尺度差）；聚类质量报 silhouette，头角色指认需同时满足特征阈值 + 消融因果验证。

## 7. 算力预算（实验机：2×RTX 4090 24GB，已就绪）

| 阶段 | 推理次数(估) | GPU·h(估) | 说明 |
|---|---|---|---|
| S1 指纹 | ~30k 次 eager 前向（3 模型 × 6 探针 × 400 × 双问对） | 90–110 | 双 4090 按模型切分并行 |
| S2 消融+patching | 6 类 × top3 × 3 任务 × 3 模型 + patching 2k 次 | 90–120 | patching 双倍前向 |
| S3 重偏置+基线 | 3 模型 × 3 方法 × 3 α × POPE(3×900) + CHAIR 500×caption | 70–90 | VCD 单步双前向 |
| 调试/返工余量 | ×1.3 | — | — |
| **合计** | — | **≈ 320–420 GPU·h** | 双 4090 ≈ 1.5–2 周墙钟 |

单卡 24GB 上限：7–8B bf16 (~16GB 权重) + 短上下文临时权重 + KV —— 安全。InternVL2.5-8B 走
trust_remote_code（实验性），若工程受阻则以「LLaVA + Qwen2.5-VL + Qwen2.5-VL-3B(规模行)」替代三模型故事。

## 8. 58 天执行周历（双 4090）

| 周 | 日期 | 任务 | 硬检查点（不过则触发风控线） |
|---|---|---|---|
| W1 | 9/22–9/28 | 仓库代码跑通（hook 单测 + 真模型 smoke）；POPE/RefCOCO 数据准备；精读 6 篇前作 | 合成探针全管线出数 |
| W2 | 9/29–10/5 | LLaVA-1.5 全探针 S1；rollout/value-zeroing 对照复现 | LLaVA 指纹矩阵 + 首版聚类 |
| W3 | 10/6–10/12 | Qwen2.5-VL(+3B 规模行) S1；跨模型聚类对齐；Atlas 主图 | **Atlas 保底贡献完成** |
| W4 | 10/13–10/19 | S2 消融 + patching；InternVL2.5 并行补第三模型（或降级） | grounding 头消融→P1 崩的因果信号成立 |
| W5 | 10/20–10/26 | S3 rebias v1 + PAI/VCD 基线；α 网格；跨模型迁移 λ | POPE F1 > 2 基线则锁方法组件 |
| W6 | 10/27–11/2 | 全量消融/失败案例/可视化案例图；图表冻结 | 所有主表定稿 |
| W7 | 11/3–11/9 | LaTeX 写作（图先行）：Atlas 图、因果热图、方法主表 | 完整 PDF |
| W8 | 11/10–11/16 | 润色 + supplementary + 提交(11/16 AOE)；预写 rebuttal | 提交 |

## 9. 风险线

1. **W3 末**：若聚类弥散（silhouette 低、无清晰角色）→ 主叙事改为「头分工是分布式的，与 FastV 类
   逐头剪枝的隐含假设矛盾」，转方法学批判 + 对剪枝方法的头感知改进（连接效率赛道）。
2. **W5 末**：若 rebias 打不过 PAI → 砍 S3，投纯分析+因果论文（CVPR 仍可投，或 EMNLP/NeurIPS D&B）。
3. **数据风险**：RefCOCO 镜像失效 → P1 换 GQA spatial/合成指称数据（prepare_data 已内置合成路径）。
4. **版本风险**：transformers API 漂移（attention 接口）→ 仓库内 `tests/test_hooks.py` 以
   output_attentions 参考值为金标对照，任何版本先过单测再实验。

## 10. 论文图表清单

- Fig.1 Atlas：UMAP/PCA 头分布（3 模型并排，颜色=功能类）
- Fig.2 层×头功能热图（grounding 分数 / sink 分数两张）
- Fig.3 消融因果热图 + 翻转率柱状图
- Fig.4 遮挡重路由与 sink 案例（注意力叠加原图可视化）
- Fig.5 方法主表：POPE/CHAIR × {vanilla, uniform(PAI), VCD, head-selective(ours)}
- Tab.1 聚类画像（每类头数量/层分布/跨模型一致性）
- Tab.2 头消融 ΔAcc 表
- Tab.3 α 与消融敏感性