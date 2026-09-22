# HeadAtlas — 多模态注意力头功能图谱（CVPR 2027）

对开源 MLLM 逐层逐头绘制功能图谱（grounding / 空间 / OCR / sink / 指令敏感 / 幻觉关联），
用置零消融 + activation patching 证明因果性，并给出 head-selective 免训练注意力重偏置。
详细实验计划见 `PLAN.md`。

## 环境

```bash
pip install -r requirements.txt
python tests/test_hooks.py          # hook 机制金标单测（无网络、CPU 可跑）
```

模型（HF 权重，单卡 24GB 够用，bf16）：
`llava-hf/llava-1.5-7b-hf`、`Qwen/Qwen2.5-VL-7B-Instruct`、`OpenGVLab/InternVL2_5-8B`(需 transformers≥4.52)

## 数据

```bash
python scripts/prepare_data.py --data data --pope --ensure-images --coco-instances data/instances_val2014.json
python scripts/prepare_data.py --data data --refcoco refs(unc).pkl data/instances_val2014.json
```

- POPE：自动下载 3 个 split json；COCO val2014 图像按需单文件懒下载（~150MB/1200 张）。
- RefCOCO：从官网或镜像取 `refs(unc).pkl` 与 `instances_val2014.json`，脚本转成 canonical JSONL。
- P4/P5（OCR 文字图、噪声 sink 探针）纯合成，无需下载。

## S1：逐头指纹

```bash
python scripts/run_probes.py --model llava-hf/llava-1.5-7b-hf \
    --probes synth_grounding,ocr_synth,sink --max-samples 400
python scripts/run_probes.py --model Qwen/Qwen2.5-VL-7B-Instruct \
    --probes grounding,cond,occlusion,pope --data data
python scripts/run_taxonomy.py --models llava-hf/llava-1.5-7b-hf Qwen/Qwen2.5-VL-7B-Instruct
```

产出：`outputs/raw/<model>/*.npz`（逐头注意力切片）→
`outputs/taxonomy/<model>/heads.csv`（每头角色标签）+ Atlas 图（fig_atlas.png、fig_heat_*.png）。

## S2：因果验证

```bash
python scripts/run_ablation.py  --model <M> --heads-csv outputs/taxonomy/<M>/heads.csv --probe grounding
python scripts/run_patching.py  --model <M> --heads-csv outputs/taxonomy/<M>/heads.csv --role grounding
```

## S3：head-selective 重偏置 vs PAI/VCD

```bash
python scripts/run_rebias.py --model <M> --heads-csv outputs/taxonomy/<M>/heads.csv \
    --splits random,popular,adversarial --alphas 0.5,1,2 --vcd
```

## 端到端冒烟（真模型）

```bash
python scripts/smoke_test.py --model Qwen/Qwen2-VL-2B-Instruct --n 4
```

## 代码结构

```
src/headatlas/
  models/wrapper.py           模型加载/输入构造/图像 token 位置/网格映射 (llava+qwen 全支持, internvl 实验性)
  models/attention_grabber.py per-head 注意力 hook 抓取（行切片/查询块聚合，权重置 None 防爆显存）
  models/biased_attention.py  headatlas_eager 注意力函数注册 + per-head KV 列偏置（PAI 为其均匀特例）
  probes/                     探针电池（RefCOCO grounding / 合成 OCR / 噪声 sink / 遮挡 / 条件对 / POPE）
  metrics/                    空间对齐指标、指纹汇总、聚类 taxonomy、跨模型对齐
  interventions/head_ops.py   o_proj 头切片编辑器：置零/均值消融/patching/捕获
  interventions/{ablation,patching,rebias}.py   S2/S3 runner
  eval/pope_runner.py         POPE 评测（可选注意力捕获 + 偏置模式）
  eval/contrastive.py         VCD 对比解码基线
tests/test_hooks.py           以 output_attentions 参考值为金标的 hook 单测（版本漂移防护）
```

## 工程要点

- 注意力提取必须 `attn_implementation="eager"`（FA2/SDPA 不返回权重）；探针上下文短，24GB 卡安全。
- GQA 模型按 query head 展开（Qwen 28 头 / InternVL 32 头）；LLaVA-1.5 为 32 头 MHA。
- 所有干预/捕获均为 bsz=1 推理，greedy 解码，max_new_tokens≤32。
- 先跑 `tests/test_hooks.py` 再上真模型；transformers 版本变化时以单测为金标排查。
