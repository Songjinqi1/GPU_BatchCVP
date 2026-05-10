# GPU_BatchCVP
本项目基于 [G6K](https://github.com/fplll/g6k)（General Sieve Kernel）hybrid attack artifact 进行扩展，在原有攻击流程上增加了同一格基条件下的批量目标生成、目标数据转储与重放机制，以及 CPU/GPU 双后端评分调用。该实现用于分析 Batch-CVP 在线评分阶段的性能瓶颈，并评估基于 GPU 的矩阵化投影计算对整体攻击流程的加速效果。

## 核心特性

- **批量候选生成** — 对同一格基条件批量生成 wrong/correct 两类 target candidates，支持多种 LWE 错误分布（binomial、ternary、ternary_sparse）
- **Dump / Replay 机制** — 将批量候选序列化为 pickle dump，支持离线重放评分，确保 CPU/GPU 对比实验的输入完全一致
- **CPU/GPU 双后端评分** — CPU 路径基于 NumPy，GPU 路径基于 CuPy，支持自动回退（GPU 不可用时回退到 CPU）
- **矩阵化投影计算** — 将逐目标投影合并为矩阵运算（GEMM + blocked Babai nearest-plane），充分利用 GPU 并行性
- **多层 GPU 优化** — pinned memory staging、可复用设备缓冲区、RawKernel fused panel Babai、row-major / column-major 双布局支持

## 目录结构

```
GPU_BatchCVP/
├── gpu_repo/                  # 核心仓库（基于 G6K 扩展）
│   ├── hybrid_attack.py       # 混合攻击主算法（alg_2_batched, alg_3_debug_v2 等）
│   ├── gpu_projected_batch_backend.py  # GPU/CPU 双后端投影评分
│   ├── run_prog_hyb.py        # 统一命令行入口（attack / dump / replay）
│   ├── batch_scheduler.py     # 跨实例批量调度器
│   ├── candidate_enumerator.py # 确定性候选枚举器
│   ├── projection_cache.py    # 投影矩阵缓存
│   ├── global_consts.py       # 全局配置常量
│   ├── preprocessing.py       # 格基预处理（BKZ 缩减）
│   ├── hyb_batch/             # Batch dump/replay/config 子模块（第三轮重构）
│   │   ├── config.py          # 数据类型定义与参数配置
│   │   ├── dump.py            # 批量候选生成与序列化
│   │   ├── replay.py          # 离线重放与评分管线
│   │   └── common.py          # 公共工具
│   ├── bkz.py / hkz.py        # 格基缩减算法
│   ├── slicer_python.py       # Slicer 接口
│   ├── svp_exact.py           # 精确 SVP 求解
│   └── ...                    # 其他 sieving / 实验工具
├── configs/                   # 运行参数配置
│   └── default.env            # 默认实验参数
├── scripts/                   # 一键运行脚本
│   ├── run_once.sh            # 单次 CPU/GPU 完整对比流程
│   └── run_repeat_bench.sh    # 重复 replay benchmark
├── tools/                     # 结果收集与汇总工具
│   └── collect_replay_summary.py
├── runs/                      # 输出目录
│   ├── dumps/                 # 批量候选 pickle dump
│   ├── logs/                  # 运行日志
│   ├── summaries/             # JSON 结构化摘要
│   └── reports/               # CPU/GPU 对比报告
└── README.md
```

## 依赖

- Python 3.10+
- [fpylll](https://github.com/fplll/fpylll)
- [G6K](https://github.com/fplll/g6k)（含 Cython 编译扩展：siever、slicer、slicerww）
- NumPy, SciPy
- Cython, cysignals
- [CuPy](https://cupy.dev/)（可选，GPU 后端需要）
- multiprocess（可选，替代 multiprocessing）

## 快速开始

### 环境配置

**前置要求：** Python 3.10+、GNU Autotools（autoreconf）、libgmp、CUDA Toolkit（可选，GPU 后端需要）。

<details>
<summary><b>Ubuntu/Debian 系统依赖一键安装</b></summary>

```bash
sudo apt install -y autoconf automake libtool libgmp-dev g++ make git
```
</details>

```bash
# 1. 进入 gpu_repo 目录
cd gpu_repo

# 2. 安装 Python 依赖
pip install fpylll numpy scipy cython cysignals
pip install -r requirements.txt

# 3. 编译 G6K Cython 扩展及 C++ 内核
python setup.py build_ext --inplace

# 4.（可选）安装 GPU 后端依赖
pip install cupy-cuda12x
# 并确保 libnvrtc.so 在库搜索路径中（可通过 conda install -c conda-forge cuda-nvrtc 安装）
```

**验证安装：**

```bash
# 在 gpu_repo 目录下运行
python -c "
from g6k.siever import Siever; from g6k.slicer import RandomizedSlicer
from gpu_projected_batch_backend import batched_nearest_plane_cpu
print('G6K + projected backend: OK')
"
```

> **说明：** 如使用 conda 环境，步骤 3 可能需要先安装 `cuda-nvrtc` 以支持 CuPy 的 JIT 编译。`install-dependencies.sh` 为 G6K 上游脚本（需 `$VIRTUAL_ENV` 环境变量），在 conda 中建议直接通过 pip 安装 fpylll。

### 单次 CPU/GPU 对比实验

```bash
bash scripts/run_once.sh configs/default.env
```

流程：
1. 格基预处理（BKZ 缩减）
2. 批量候选生成与 dump
3. GPU 投影评分重放
4. CPU 投影评分重放
5. CPU/GPU 评分结果对比
6. 汇总摘要输出

### 重复 Benchmark

```bash
bash scripts/run_repeat_bench.sh
```

若 dump 不存在，脚本会自动先生成 dump 再执行重复 benchmark。

### 命令行入口

```bash
# 生成批量候选 dump
python gpu_repo/run_prog_hyb.py --dump_batch_candidates \
    --n 140 --q 3329 --dist binomial --dist_param 3 \
    --n_guess_coord 6 --beta_pre 46 --n_slicer_coord 47

# GPU 重放评分
python gpu_repo/run_prog_hyb.py --replay_batch_candidates <dump.pkl> --force_gpu_backend

# CPU 重放评分
python gpu_repo/run_prog_hyb.py --replay_batch_candidates <dump.pkl> --force_cpu_backend

# JSON 格式结构化输出
python gpu_repo/run_prog_hyb.py --replay_batch_candidates <dump.pkl> \
    --force_gpu_backend --json-out gpu_summary.json
```

## 配置参数

编辑 `configs/default.env` 可调整以下参数：

| 参数 | 说明 |
|------|------|
| `N` / `Q` | LWE 实例维度与模数 |
| `DIST` / `DIST_PARAM` | 错误分布类型及参数 |
| `N_GUESS_COORD` | 猜测坐标数 |
| `BETA_PRE` | BKZ 预处理块大小 |
| `N_SLICER_COORD` / `DELTA_SLICER_COORD` | Slicer 维度范围 |
| `GPU_RUNS` / `CPU_RUNS` | GPU/CPU 重复运行次数 |
| `GPU_WARMUP` | GPU 预热运行次数 |

GPU 后端优化参数可通过 `gpu_repo/global_consts.py` 或环境变量调整：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `HYB_GPU_NP_IMPL` | Nearest-plane 实现：`loop` / `blocked_exact` / `blocked_exact_fused` | `blocked_exact_fused` |
| `HYB_GPU_NP_BLOCK_ROWS` | Blocked Babai 面板行数 | 32 |
| `HYB_BATCH_MAX_TARGETS` | 单次调度最大目标数 | 16384 |
| `HYB_GPU_USE_ROWMAJOR_T` | 启用 row-major target 布局 | True |
| `HYB_GPU_DIRECT_STAGE_T` | 直接 staging 非连续 target 视图 | True |

## 重要约定

- 父目录优先调用 `hyb_batch/` 子模块 CLI；`run_prog_hyb.py` 为统一入口
- CPU/GPU 对比依赖同一份 dump，保证输入一致性
- 所有输出统一放在 `runs/` 目录下，避免污染仓库根目录
- dump / replay / repeat benchmark 均支持稳定 JSON summary，便于接入 CI 或批量实验
- GPU 路径在 CuPy 不可用或发生异常时自动回退到 CPU（可通过 `HYB_GPU_ALLOW_FALLBACK` 控制）

## 许可

本项目基于 G6K 扩展开发，遵循 [GNU General Public License v2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)。

