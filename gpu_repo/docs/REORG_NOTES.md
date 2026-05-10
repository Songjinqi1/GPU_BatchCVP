# GPU 仓库整理说明

当前仓库在保留原有核心算法代码布局的前提下，对新增的 GPU / benchmark / pipeline 脚本做了轻量归类：

## 新增归类目录

### 第二轮重构新增

- `hyb_batch/`
  - `common.py`：dump / replay 共用工具
  - `config.py`：batch 配置 dataclass / JSON 输出工具
  - `dump.py`：候选构建与 dump 生成
  - `replay.py`：replay、验证与 summary 聚合
- `pipelines/`
  - `dump_batch_candidates.py`
  - `replay_batch_candidates.py`
  - `run_guess6_full_pipeline.py`
- `benchmarks/`
  - `bench_replay_repeat.py`
- `tools/`
  - `compare_gpu_cpu_scores.py`
  - `compare_loop_blocked_exact_correctness.py`
  - `analyze_projected_scores.py`
- `docs/`
  - 文档说明

## 为什么没有把所有 Python 文件都大规模挪走

因为当前项目的核心算法脚本仍大量依赖“仓库根目录同级导入”方式，例如：

- `run_prog_hyb.py`
- `preprocessing.py`
- `hybrid_attack.py`
- `gpu_projected_batch_backend.py`
- `projection_cache.py`
- `batch_scheduler.py`
- `candidate_enumerator.py`

如果一次性把这些也全部搬走，需要系统性重构 import 路径，风险会明显增大。

所以当前整理策略是：

1. **保留核心算法层在仓库根目录**，确保原有主流程不被破坏
2. **把新增的 pipeline / benchmark / compare 脚本归类**
3. **把 `run_prog_hyb.py` 中的 dump / replay 逻辑下沉到 `hyb_batch/` 子模块**
4. **引入 dataclass 配置对象，减少 scattered globals 的直接扩散**
5. **给 dump / replay 增加统一 JSON summary 输出**
6. **保留 `run_prog_hyb.py` 兼容入口，避免旧命令行失效**
7. **为被迁移脚本补上基于 `REPO_ROOT` 的路径解析**
8. **在父目录中新增统一调度层**

这是一种更稳的“先整理入口，再逐步模块化”的做法。
