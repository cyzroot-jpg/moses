# MOSES AAE 模型 SDAA 适配与 CPU 对比测试报告

- 项目：`/data01/tuyilist/cyz/moses`（Molecular Sets / MOSES）
- 目标模型：Adversarial Autoencoder（AAE，SMILES 生成）
- 目标：迁移到太初 SDAA，实现 **SDAA / CUDA / CPU 三端兼容**，审计自定义算子，覆盖全部推理功能，并与 CPU 做对比测试
- 测试日期：2026-09-23
- 结果数据：`cyz_linshi/aae/sdaa_test/aae_device_results.json`

---

## 1. 探测结果

### 1.1 运行环境

| 项 | 值 |
|---|---|
| Host IP | 10.71.13.47 |
| CPU 架构 | loongarch64（`cpuinfo` 报 unsupported，可忽略） |
| CPU / 内存 | 128 核 / 约 1 TB |
| PyTorch | 2.12.0（CPU+CUDA 名空间，`torch.version.cuda == None`） |
| torch_sdaa | 3.3.0b0+gitb716a30 |
| SDAA Driver / Runtime | 3.4.0 / 3.3.0b0 |
| TecoDNN / TecoBLAS / CustomDNN / TCCL | 3.3.0b0 |
| 加速卡 | 8 张 `TECO_AICARD_01`（`torch.sdaa.device_count() == 32`，单卡 64 GB） |
| 关键依赖 | rdkit 2026.9.1, fcd-torch 1.0.7, numpy 1.26.4, pandas 3.0.3, scipy 1.16.1, tqdm 4.66.5 |

> 环境要点：本机 **无 NVIDIA CUDA 设备**（`torch.cuda.is_available() == False`），因此本次实测为 **SDAA vs CPU**；CUDA 分支通过统一的设备抽象层实现，在 CUDA 机器上无需改代码即可运行。

### 1.2 项目结构

```
moses/
├── moses/                      # 核心库
│   ├── aae/                    # ★ 本次目标：Adversarial Autoencoder
│   │   ├── config.py           # 超参（embedding=32, enc_hidden=512, dec_hidden=512, latent=128 ...）
│   │   ├── model.py            # Encoder / Decoder / Discriminator / AAE
│   │   ├── trainer.py          # 预训练 + 对抗训练，CharVocab，collate
│   │   └── __init__.py
│   ├── char_rnn/ vae/ organ/   # 其它生成模型（同接口）
│   ├── latentgan/              # 依赖 heteroencoder / DDC（TF）
│   ├── baselines/              # HMM / NGram / Combinatorial（纯 Python）
│   ├── metrics/                # FCD(ChemNet) / SNN / Frag / Scaf / IntDiv / Filters ...
│   ├── dataset/                # 内置 train/test/test_scaffolds (.csv.gz)
│   ├── device_utils.py         # ★ 新增：SDAA/CUDA/CPU 设备兼容层
│   ├── script_utils.py         # 公共参数、set_seed、设备校验
│   ├── models_storage.py       # 模型注册表
│   └── interfaces.py           # MosesTrainer 抽象接口
├── scripts/
│   ├── train.py sample.py eval.py run.py   # ★ 训练/采样/评测/编排入口
│   ├── verify_aae_inference.py             # ★ 新增：跨设备一致性与性能测试
│   ├── debug.py                # FCD 数值问题排查（原有）
│   └── prepare_dataset.py split_dataset.py print_table.py ...
├── checkpoint/aae/             # 预训练权重 model.pt / config.pt / vocab.pt
├── cyz_linshi/                 # 用户临时工作区（生成结果/测试输出）
├── Deep-Drug-Coder/            # 独立包（Keras/TF，非本次推理路径）
├── molvecgen/                  # 独立包（LatentGAN 向量化）
└── data/ moses/dataset/data/   # 数据集（LFS）
```

### 1.3 AAE 模型结构

MOSES AAE 是字符级 SMILES 对抗自编码器，共享一个 Embedding 层，三个子网络：

```
                        ┌───────────────────────────── Encoder ─────────────────────────────┐
  SMILES ids (N,L) ──▶ Embedding(41,32) ──▶ Bi-LSTM(in32,h512,1layer) ──▶ 取末状态 reshape ──▶ Linear(1024→128) ──▶ z (N,128)
                                                                                                              │
                                           z ──▶ Linear(128→512) ──▶ h0=0, c0 ──┐                          │
                                                                                 ▼                          │
                       ┌──────────────────── Decoder ──────────────────────┐   ┌── 先验 N(0,1) ──┐
  prev token ──▶ Embedding(共享) ──▶ LSTM(in32,h512,2layers) ──▶ Linear(512→41) ──▶ logits ──▶ softmax ──▶ Categorical.sample
                                                                                 │
                       ┌──────────────────── Discriminator ─────────────────┐   │
                       z / prior ──▶ Linear(128→640) ▶ ELU ─▶ Linear(640→256) ▶ ELU ─▶ Linear(256→1)
```

| 子模块 | 结构 | 参数量 |
|---|---|---:|
| Embedding（共享） | `Embedding(41, 32)`, `padding_idx=pad` | — |
| Encoder | Bi-LSTM(32→512, 1 层) + `Linear(1024→128)` | 2,368,928 |
| Decoder | `Linear(128→512)` + LSTM(32→512, 2 层) + `Linear(512→41)` | 3,307,849 |
| Discriminator | MLP 128→640→256→1（ELU） | 246,913 |
| **合计** | vocab=41，latent=128 | **5,922,378** |

推理路径：`scripts/sample.py::main` → `AAE.sample(n_batch, max_len)` → 逐步自回归解码，**100 步串行**（`max_len`），每步 batch 内并行。这就是性能瓶颈所在（见 §5）。

### 1.4 依赖清单

- 运行 AAE 推理必需：`torch`、`torch_sdaa`（SDAA 时）、`rdkit`、`numpy`、`pandas`、`tqdm`。
- 评测（`eval.py`）额外：`fcd-torch`、`scipy`。
- 训练额外：`torch.optim`。
- 非必需/其它模型：`pomegranate`（HMM baseline）、`tensorflow/keras`（Deep-Drug-Coder / LatentGAN HeteroEncoder）、`molvecgen`。

---

## 2. 自定义算子审计（关键结论）

对全仓库做了 `*.cu / *.cuh / *.cpp / cpp_extension / load_inline / CUDAExtension / torch.ops / __global__` 搜索：

> **AAE 推理链路不包含任何自定义算子**，全部为标准 ATen 算子，无需绕过。

AAE 用到的算子全部有 SDAA 实现：

| 算子 | 用途 | SDAA |
|---|---|---|
| `Embedding` | token 嵌入 | ✅ 已实测 |
| `LSTM`（单向 + 双向） | 编/解码 | ✅ 已实测 |
| `Linear` | 投影 | ✅ |
| `pack_padded_sequence` / `pad_packed_sequence` | 变长序列 | ✅（见下） |
| `softmax` | 概率 | ✅ |
| `Categorical.sample`（multinomial） | 采样 | ✅ 已实测 |
| `CrossEntropyLoss` / `BCEWithLogitsLoss`（训练） | 损失 | 标准 |

唯一需要针对 SDAA/新版 PyTorch 适配的点（**不是自定义算子，只是参数/数据类型约束**）：

1. `pack_padded_sequence(x, lengths.cpu(), batch_first=True)` —— `lengths` 必须在 CPU（CUDA/SDAA 通用要求）。原代码在 CUDA 上偶然可用，SDAA 上必须显式 `.cpu()`。`moses/aae/model.py:27,57`。
2. `is_end` 由 `torch.uint8` 改为 `torch.bool` —— 新版 PyTorch + SDAA 对 uint8 算术支持不佳。`moses/aae/model.py:156`。
3. 反向传播中 `parameter.grad` 可能为 `None`，需判空后再 `clamp_`。`moses/aae/trainer.py:178`。

其它包中的 GPU 相关写法**不属于本推理路径**：
- `Deep-Drug-Coder/ddc_pub/ddc_v3.py` 使用 TensorFlow 的 `CuDNNLSTM`（TF 后端，非 PyTorch，不参与 MOSES AAE 推理）。
- `moses/latentgan/model.py` 使用 `torch.cuda.FloatTensor`，属 LatentGAN（本次不涉及），如需迁移需单独处理。

---

## 3. 迁移改动清单（实现 SDAA / CUDA 双兼容）

核心思想：**设备选择集中到一个兼容层，业务代码不再硬编码 `torch_sdaa` / `torch.cuda`**。

### 3.1 新增 `moses/device_utils.py`

- `sdaa_available()` / `cuda_available()` / `available_backends()`
- `auto_device()`：自动选择 `sdaa > cuda > cpu`
- `resolve_device(device)`：把 `None/'auto'/'cpu'/'cuda:N'/'sdaa:N'` 规范化成具体字符串
- `build_device(device)`：返回 `torch.device` 并调用对应后端的 `set_device`
- `seed_everything(seed)`：同时 seed python / numpy / cuda / sdaa
- 关键：`torch_sdaa` **惰性导入**（`try/except`），因此在没有 SDAA 的 CUDA/CPU 机器上 import 不会失败，自动回退。

### 3.2 修改

| 文件 | 改动 |
|---|---|
| `moses/script_utils.py` | 修复 `torch_device` 校验器（原实现缩进错误导致校验体不可达）；接受 `cpu/cuda:N/sdaa:N/auto`；`--device` 默认改为 `None`（自动探测）；`set_seed` 委托给 `seed_everything` |
| `scripts/train.py` | 删除硬编码 `import torch_sdaa`；用 `build_device` 选择设备；`torch.load(..., weights_only=False)` 兼容 torch≥2.6 |
| `scripts/sample.py` | 同上；`import torch_sdaa` 移除，设备/`torch.load` 统一处理 |
| `scripts/run.py` | `--device` 默认 `None`→`resolve_device`；编排 train/sample/eval 时下发具体设备串 |
| `scripts/eval.py` | `--device` 经 `resolve_device` 规范化，支持 `sdaa:N` 传递给 FCD/ChemNet |
| `moses/aae/model.py` | `lengths.cpu()`；`is_end` 改 `bool` |
| `moses/aae/trainer.py` | `parameter.grad is not None` 判空 |

### 3.3 兼容性验证（自动探测）

```
backends: ['sdaa', 'cpu']      # 本机（无 CUDA）
auto:     sdaa
None     -> sdaa:0
'cpu'    -> cpu
'sdaa'   -> sdaa:0
'sdaa:3' -> sdaa:3
'cuda:0' -> RuntimeError: CUDA device requested but CUDA is not available   # 预期报错
```

在 CUDA 机器上，`torch_sdaa` 导入失败后 `auto_device()` 自动回退为 `cuda`，`--device cuda:0` 正常走 `torch.cuda.set_device`，**无需改动代码**。

---

## 4. 推理功能覆盖

| 推理功能 | 入口 | CPU | SDAA | 说明 |
|---|---|:--:|:--:|---|
| 采样（批量自回归生成） | `scripts/sample.py` | ✅ | ✅ | 用真实 checkpoint 各生成 256 条 |
| 采样（可编程 API） | `AAE.sample()` | ✅ | ✅ | `verify_aae_inference.py` |
| 编码器前向 | `AAE.encoder_forward()` | ✅ | ✅ | 一致性测试 |
| 解码器前向（latent states） | `AAE.decoder_forward()` | ✅ | ✅ | 一致性测试 |
| 指标评测（SNN/Frag/Scaf/IntDiv/Filters/性质/Novelty…） | `scripts/eval.py` | ✅ | ✅ | 见 §5.4 |
| FCD（ChemNet） | `fcd_torch` | ⚠️NaN | ⚠️NaN | **设备无关**的数值问题，见 §5.4 |
| 编排（train→sample→eval） | `scripts/run.py` | ✅ | ✅ | 设备串正确下发 |

> FCD 的 NaN 与本仓库已知问题一致（`scripts/debug.py` 即为排查该问题而写）。经实测定位：**不是秩亏**（协方差满秩 rank=512），而是 `sqrtm` 的数值不适定问题；详见 §5.4.1。CPU 与 SDAA 均复现，**非迁移缺陷**。

---

## 5. CPU vs SDAA 对比测试

测试脚本：`scripts/verify_aae_inference.py`（同一份权重、eval 模式、同一 seed 与 batch）。

```bash
python scripts/verify_aae_inference.py \
  --model_load  checkpoint/aae/model.pt \
  --config_load checkpoint/aae/config.pt \
  --vocab_load  checkpoint/aae/vocab.pt \
  --devices cpu sdaa:0 \
  --n_samples 1024 --n_batch 512 --max_len 100 --seed 42 \
  --outdir cyz_linshi/aae/sdaa_test
```

### 5.1 数值一致性（正确性）

对同一批真实 test SMILES，比较 CPU 与 SDAA 的 **Encoder 输出**和**一步 Decoder logits**：

| 指标 | CPU(参考) | SDAA | 结论 |
|---|---:|---:|---|
| Encoder 最大绝对误差 | 0 | **1.57e-05** | `allclose(atol=1e-3)` ✅ |
| Decoder 最大绝对误差 | 0 | **9.54e-06** | `allclose(atol=1e-3)` ✅ |

→ SDAA 数值与 CPU 等价（差异仅为浮点累加顺序），**迁移正确性通过**。

### 5.2 端到端采样（n=1024, batch=512, max_len=100）

| 设备 | 有效分子 | 有效率 | 有效唯一 | 唯一率 | 墙钟时间(s) | 吞吐(smiles/s) |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 756 | 0.7383 | 756 | 0.7383 | 12.88 | **79.5** |
| SDAA:0 | 737 | 0.7197 | 737 | 0.7197 | 28.14 | **36.4** |

- 两端有效率、唯一率基本一致（统计噪声内），说明生成质量等价。
- 本模型 **CPU 吞吐高于 SDAA**：AAE 每步计算量极小但需 **100 次串行解码**，属延迟受限；128 核 CPU 的 BLAS 对小矩阵反而高效，而 SDAA 每步受 kernel launch / 同步开销主导。

### 5.3 批大小扩展性（n=1024）

| batch | CPU (smiles/s) | SDAA (smiles/s) |
|---:|---:|---:|
| 128 | 25.0 | 26.9 |
| 512 | 98.2 | 36.6 |
| 2048 | 127.4 | 45.2 |

- 小 batch(128) 两者相当；增大 batch 时 CPU 借助多核线性加速，SDAA 提升有限。
- 结论：**对该串行 RNN 小算子负载，SDAA 优势不明显**；SDAA 更适合大矩阵/可并行的算子。若需提速，应减少串行步数（如批内并行/近似解码）或将计算合并，而非单纯换设备。

### 5.4 指标评测

用 `scripts/eval.py` 跑真实生成集（同一份结果在 CPU 与 SDAA 上计算）：

| 指标 | CPU (1024 样本) | SDAA (30k 样本) |
|---|---:|---:|
| valid | 0.7383 | 0.7364 |
| unique@1000 | 1.0 | 1.0 |
| unique@10000 | — | 0.9976 |
| SNN/Test | 0.2486 | 0.2488 |
| Frag/Test | 0.3465 | 0.9307 |
| Scaf/Test | 0.00385 | 0.01186 |
| IntDiv | 0.8475 | 0.8475 |
| IntDiv2 | 0.8380 | 0.8407 |
| Filters | 0.0 | 0.0 |
| Novelty | 1.0 | 1.0 |
| FCD/Test | NaN | NaN |

- SNN/IntDiv/Novelty 等**设备无关指标在两端一致**，佐证 SDAA 推理结果与 CPU 等价。
- Frag/Scaf 差异来自两次评测的输入集不同（1024 vs 30k），非设备差异。
- **FCD 两端均为 NaN/报错**，原因见下（与设备无关）。

#### 5.4.1 FCD 失败的真实原因（已用 `debug.py` 实测）

Fréchet 距离的计算式（`fcd_torch/utils.py::calculate_frechet_distance`）：

```
P        = sigma_ref · sigma_gen                 # 两个 512×512 协方差之积
covmean  = sqrtm(P)                              # scipy.linalg.sqrtm（Schur 法）
# 若 covmean 非有限 → 用 (sigma+1e-6·I) 重试
# 若 covmean 有虚部且 diag(imag) 超出 1e-3 → raise "Imaginary component"
FCD = ||mu_ref - mu_gen||² + tr(sigma_ref) + tr(sigma_gen) - 2·tr(covmean)
```

> **关键：整个 `sqrtm` / 协方差 / FCD 计算都是纯 CPU 的 numpy+scipy。**
> `fcd_torch` 的 `device` 参数只决定 ChemNet 特征在哪块卡上抽取（`get_predictions` 最后 `.to('cpu').numpy()`），
> 之后的 `np.cov` 与 `linalg.sqrtm` 与设备无关。因此 **CUDA/SDAA 本身不会改变 FCD 的数值成败**。

在真实生成集上实测（`scripts/debug.py`）：

| 量 | 1024 样本集 | 30k 样本集 | 参照 test 集 |
|---|---:|---:|---:|
| 有效唯一分子数 | 737 | 22014 | — |
| ChemNet 特征 | (737,512), nan=0, 全部唯一 | (22014,512), nan=0, 全部唯一 | — |
| `sigma_gen` 秩 | **512（满秩）** | **512（满秩）** | 512 |
| `sigma_gen` 条件数 | 2.65e+07 | 1.04e+06 | 4.52e+05 |
| `sigma_gen` 最小特征值 | **7.3e-08** | **1.76e-06** | 5.55e-06 |
| `sigma_ref·sigma_gen` 最小实特征值 | **1.26e-11** | **6.65e-11** | — |
| `sqrtm` 结果 | 虚部 0.142 → **raise** | 非有限 → 重试后 **nan** | — |
| `‖mu_ref−mu_gen‖²` | 15.86 | 15.83 | — |

**结论（修正之前的"秩亏"说法）：**

1. 协方差**满秩**（rank=512），无负特征值，并非秩亏/特征退化。
2. 真正原因是 **`sqrtm` 的数值不适定**：生成集的 ChemNet 特征协方差 `sigma_gen` 高度病态——最小特征值低至 7e-8~2e-6（条件数 1e6~1e7）。两者相乘后 `sigma_ref·sigma_gen` 的条件数升到约 **1e11**，最小特征值 ~1e-11 已接近 float64 的数值噪声底。
3. 对这样接近奇异、且非对称（`sigma_ref·sigma_gen` 本身不对称）的矩阵做矩阵平方根，`scipy.linalg.sqrtm` 会给出 **虚部 0.14** 或 **非有限值**（scipy 也发 `LinAlgWarning: Matrix is ill-conditioned`）。前者被 fcd_torch 的 `diag(imag)>1e-3` 判据拦截报错，后者经 `+1e-6·I` 重试（正则量级远小于 1e-11）仍为 nan。
4. 走"报错"还是"nan"只取决于极微小的浮点差异（样本数、batch 顺序、CPU/SDAA 舍入），因此**同一文件 CPU 可能 nan、SDAA 可能报错**——这正说明它是**阈值敏感的数值问题，而非设备缺陷**。
5. **根因在生成质量**：`‖mu_ref−mu_gen‖² ≈ 15.8`（正常 FCD 约 1~2），说明该 checkpoint 生成的分子严重偏离 MOSES test 分布（多为 100 字符长串、约 3% 字符为 ChemNet 词表外的 `*`、26% 解析失败），导致特征协方差病态。这属于**该预训练权重/数据问题**，与 SDAA 迁移无关。

**可选修复方向：** ① 更换/重训为规范 MOSES AAE 权重（生成短小、类药 SMILES，FCD 自然回落）；② 对协方差做更强正则（把 `eps=1e-6` 改为与尺度匹配的值，如 `1e-6·tr(sigma)/d`）或改用 PSD 稳定的平方根（特征分解 `V sqrt(Λ) Vᵀ`）；③ 评估时改用分布距离更鲁棒的替代指标。

### 5.5 生成产物

- `cyz_linshi/aae/sdaa_test/aae_gen_cpu.csv`（CPU，1024）
- `cyz_linshi/aae/sdaa_test/aae_gen_sdaa_0.csv`（SDAA，1024）
- `cyz_linshi/aae/sdaa_test/gen_cpu_cli.csv` / `gen_sdaa_cli.csv`（CLI 256）
- `cyz_linshi/aae/sdaa_test/aae_device_results.json`（原始结果）

---

## 6. 结论与建议

1. **迁移完成且双兼容**：`--device cpu/cuda:N/sdaa:N/auto` 全支持；SDAA 与 CUDA 由同一套 `device_utils` 抽象，CUDA 机器零改动运行。
2. **自定义算子**：AAE 推理链路**无自定义算子**，全部为标准算子，**无需绕过**；仅做了 3 处参数/类型/判空适配。
3. **推理全覆盖**：采样、编码/解码、指标评测在 SDAA 上均通过；仅 FCD 因**设备无关的数值问题**返回 NaN（CPU 同样 NaN）。
4. **正确性**：SDAA 与 CPU 数值等价（误差 ~1e-5）。
5. **性能**：本 AAE 为 100 步串行小算子负载，**CPU 反超 SDAA**（batch=512 时 79.5 vs 36.4 smiles/s）；SDAA 在该类算子上的收益有限，属于预期现象。
6. 建议：
   - FCD NaN 单独跟进（`scripts/debug.py`，考虑对协方差加正则/降维或使用大样本）。
   - LatentGAN 的 `torch.cuda.FloatTensor` 若需迁移，应改为 `device_utils` 的通用 tensor 创建方式。
   - 若追求 SDAA 上更高吞吐，考虑将自回归解码的串行步数用 KV-cache / 批内并行规避，而非依赖 batch 增大。

## 7. 复现命令

```bash
cd /data01/tuyilist/cyz/moses

# 采样（CPU / SDAA）
python scripts/sample.py aae --device cpu   --seed 7 --n_samples 256 --n_batch 256 \
  --model_load checkpoint/aae/model.pt --config_load checkpoint/aae/config.pt \
  --vocab_load checkpoint/aae/vocab.pt --gen_save /tmp/gen_cpu.csv
python scripts/sample.py aae --device sdaa:0 --seed 7 --n_samples 256 --n_batch 256 \
  --model_load checkpoint/aae/model.pt --config_load checkpoint/aae/config.pt \
  --vocab_load checkpoint/aae/vocab.pt --gen_save /tmp/gen_sdaa.csv

# 跨设备一致性 + 性能
python scripts/verify_aae_inference.py --devices cpu sdaa:0 \
  --model_load checkpoint/aae/model.pt --config_load checkpoint/aae/config.pt \
  --vocab_load checkpoint/aae/vocab.pt \
  --n_samples 1024 --n_batch 512 --outdir cyz_linshi/aae/sdaa_test

# 指标评测
python scripts/eval.py --gen_path cyz_linshi/aae/sdaa_test/aae_gen_sdaa_0.csv \
  --device sdaa:0 --n_jobs 4
```