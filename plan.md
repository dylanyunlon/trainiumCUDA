# trainiumCUDA Performance Optimization Plan

## Project Overview

**Repository**: `github.com/dylanyunlon/trainiumCUDA.git`
**Reference Benchmark**: `github.com/NVIDIA/nvbench` (CUDA Kernel Benchmarking Library)

### Relationship Between Projects

- **trainiumCUDA** (`perf_takehome.py`): Anthropic's performance engineering take-home. It simulates a custom VLIW SIMD architecture (similar to Trainium2) and asks engineers to optimize a kernel that traverses a tree forest, performing hash operations and index updates. The benchmark measures clock cycles on this simulated machine.
- **NVIDIA/nvbench**: A C++17 CUDA kernel benchmarking library for regression testing and parameter tuning. It measures GPU/CPU execution time with statistical rigor (noise detection, cold/batch measurements, throughput computation).
- **The Connection**: Both projects address **compiler/kernel optimization on specialized hardware**. CUDA is completely closed-source; Anthropic aims to replicate equivalent performance optimization capabilities on its own platform (Trainium2). Solving NVBench's open issues around measurement accuracy, scheduling, distribution comparison, and profiling directly maps to techniques we need for beating the 1000-cycle barrier on our simulated VLIW machine.

### Current State

- **Original cycles**: 1449
- **After Round 1 optimization**: 1393 (−56 cycles, 3.9% improvement)
- **Target**: < 1363 (beat Claude Opus 4.5 improved harness)
- **Stretch goal**: < 1000 cycles (approaching best human performance)
- **Tests passing**: 8/9 (only `test_opus45_improved_harness` at <1363 fails)
- **Correctness**: All 8 correctness tests pass across 20 seeds

---

## PART A: First 10 Tasks (Claude — this PR)

> **Git branch**: `feat/optimization-round1`
> **PR title**: `feat: scheduler + kernel optimizations to break 1363-cycle barrier`
> **PR description**: Implements 10 optimization tasks targeting the embedded scheduler and kernel body in `perf_takehome.py`. Reduces cycles from 1449 to target <1363.

### Task 1 — Analyze Critical Path Bottleneck
- **File**: `perf_takehome.py` (read-only analysis)
- **Action**: Add tracing/profiling to identify which round types (level 0/1/2/3+) consume the most cycles
- **New file**: `analysis/critical_path_report.md` (at `trainiumCUDA/analysis/critical_path_report.md`)

### Task 2 — Optimize Scheduler: WAR Dependency Relaxation
- **File**: `perf_takehome.py` — `KernelBuilder.build_kernel()` scheduler section (~line 280-380)
- **Action**: WAR (Write-After-Read) dependencies currently treated almost like RAW. Relax WAR constraints — allow WAR pairs to share the same cycle (t_min = pt instead of pt+1 for WAR).
- **Expected impact**: -20 to -40 cycles

### Task 3 — Scheduler: Better Priority Heuristic
- **File**: `perf_takehome.py` — scheduler section
- **Action**: Replace simple critical-path priority with a combined heuristic: (critical_path, -mobility, engine_pressure). Mobility = latest_start - earliest_start. Engine pressure = prefer scheduling ops on underutilized engines first.
- **Expected impact**: -15 to -30 cycles

### Task 4 — Merge Hash Stages for `multiply_add` Pattern
- **File**: `perf_takehome.py` — `build_kernel()` hash section (~line 220-240)
- **Action**: For hash stages that use the `multiply_add` optimization, currently we emit per-U ops sequentially. Interleave U-batches across hash stages to improve ILP (instruction-level parallelism).
- **Expected impact**: -20 to -50 cycles

### Task 5 — Level 3+ Gather Optimization: Prefetch Pattern
- **File**: `perf_takehome.py` — main loop, level >= 3 branch (~line 195)
- **Action**: For levels >= 3, we do per-lane loads (8 loads per U element). Reorder these loads to issue all address computations first, then all loads, maximizing load pipeline utilization.
- **Expected impact**: -30 to -60 cycles

### Task 6 — Reduce Scalar ALU Pressure in XOR Stage
- **File**: `perf_takehome.py` — Stage 2 XOR section (~line 210)
- **Action**: The XOR uses per-lane scalar ALU (8 ops per U). Investigate if we can use valu XOR instead (valu("^",...)) to reduce from 8 scalar ops to 1 vector op per U element.
- **Expected impact**: -40 to -80 cycles (major win if valu supports XOR)

### Task 7 — Reduce Scalar ALU Pressure in IDX UPDATE
- **File**: `perf_takehome.py` — Stage 4 IDX UPDATE section (~line 230)
- **Action**: Similar to Task 6 — the AND and ADD in idx update use per-lane scalar. Convert to vector operations where possible.
- **Expected impact**: -30 to -60 cycles

### Task 8 — Init Section: Batch Load Packing
- **File**: `perf_takehome.py` — init section (~line 100-140)
- **Action**: Init loads are currently 2-per-cycle. Verify scheduler packs them optimally; if not, manually pack const+load pairs into 2-wide bundles.
- **Expected impact**: -5 to -10 cycles

### Task 9 — Level 1 and Level 2 Optimization: Remove Redundant Ops
- **File**: `perf_takehome.py` — level 1 and level 2 branches
- **Action**: Level 2 currently uses 6 stages of vector ops per U element. Simplify the bit-manipulation logic using a lookup approach or reduce intermediate scratch usage.
- **Expected impact**: -10 to -20 cycles

### Task 10 — Comprehensive Regression Test
- **File**: `tests/submission_tests.py` (read-only, DO NOT MODIFY)
- **New file**: `tests/optimization_regression.py` (at `trainiumCUDA/tests/optimization_regression.py`)
- **Action**: Create a separate test file that runs multiple seeds and forest heights to verify correctness is maintained after all optimizations. Also logs cycle counts per run.

---

## PART B: Next 10 Tasks (Codex — parallel PR)

> **Git branch**: `feat/optimization-round2`
> **PR title**: `feat: advanced kernel transforms for sub-1200 cycle target`
> **PR description**: Implements 10 advanced optimization tasks. Depends on round1 branch being merged first. Target <1200 cycles.

### Task 11 — Software Pipelining: Overlap Rounds
- **File**: `perf_takehome.py` — main loop structure
- **Action**: Implement software pipelining where round N+1's address computation overlaps with round N's hash computation.

### Task 12 — Loop Unrolling: Merge Adjacent Same-Level Rounds
- **File**: `perf_takehome.py` — main loop
- **Action**: When consecutive rounds have the same level, merge their bodies to reduce loop overhead.

### Task 13 — Scratch Space Layout Optimization
- **File**: `perf_takehome.py` — alloc_scratch calls
- **Action**: Reorder scratch allocations to minimize register pressure and bank conflicts.

### Task 14 — Constant Folding: Pre-compute Hash Constants
- **File**: `perf_takehome.py` — hash_info construction
- **Action**: Pre-compute all possible multiply_add constants and broadcast once, eliminating redundant broadcasts.

### Task 15 — Speculative Execution: Pre-compute Both IDX Paths
- **File**: `perf_takehome.py` — Stage 4 IDX UPDATE
- **Action**: Compute both idx*2+1 and idx*2+0 speculatively, then select. Eliminates the serial dependency chain.

### Task 16 — Custom VLIW Bundle Packer (Post-Scheduler)
- **File**: `perf_takehome.py` — after scheduler output
- **New file**: `optimizers/bundle_packer.py` (at `trainiumCUDA/optimizers/bundle_packer.py`)
- **Action**: Post-process the scheduler output to fill empty engine slots with independent operations from future cycles.

### Task 17 — Profile-Guided Optimization
- **New file**: `optimizers/profile_guided.py` (at `trainiumCUDA/optimizers/profile_guided.py`)
- **Action**: Run the kernel once with tracing, identify the longest stalls, then re-schedule to eliminate them.

### Task 18 — Vector Scatter/Gather Pattern for Level 3+
- **File**: `perf_takehome.py` — level >= 3 node lookup
- **Action**: Implement a vector gather pattern that batches all 8 lane loads into a single coordinated load sequence.

### Task 19 — Reduce Store Overhead
- **File**: `perf_takehome.py` — store results section
- **Action**: Investigate if stores can overlap with final round computations.

### Task 20 — Final Integration and Sub-1200 Validation
- **File**: `perf_takehome.py`
- **New file**: `benchmarks/final_report.md` (at `trainiumCUDA/benchmarks/final_report.md`)
- **Action**: Integrate all optimizations, run full test suite, generate final benchmark report.

---

## PART C: NVBench Issue Tracker — Cross-Reference Backlog

All issues from `NVIDIA/nvbench` that inform our optimization strategy. Solving these problems in the NVBench context maps directly to improving our simulated-machine performance.

### Open Issues (61 total, ~35 tracked below)

| # | Title | Category | Relevance to trainiumCUDA |
|---|-------|----------|---------------------------|
| 321 | [Python] cudaErrorIllegalAddress | Bug | Memory safety in kernel dispatch — parallels our scratch bounds checking |
| 320 | Integrate distribution-based comparison into nvbench_compare.py | Enhancement | Statistical comparison of benchmark runs — we need similar noise analysis |
| 319 | Design MVP decision tree for regression detection | Enhancement | Automated regression detection — applicable to tracking our cycle counts across commits |
| 317 | Gather prior art on performance distribution comparison | Research | Understanding performance variability — directly relevant to our measurement methodology |
| 316 | Gather distributions for problematic performance comparison | Research | Performance distribution analysis — helps understand why our cycles vary between seeds |
| 313 | Extend comparison script to allow comparison of two distributions | Task | Distribution comparison tooling — we need this to validate optimizations statistically |
| 312 | Editable builds of cuda-bench are broken | Bug/Build | Build system issues — applicable if we add C-extension accelerators |
| 310 | Running batch benchmarks affects cold benchmarks | Bug | Measurement interference — parallels our concern about simulator warm-up effects |
| 303 | Build failure when using conda | Bug/Build | Dependency management |
| 293 | Python package release check-list | Process | Release engineering |
| 287 | Throughput Failed On Multiple GPUs | Bug | Multi-device benchmarking |
| 270 | Write-only benchmark exceeds 100% bandwidth | Bug | Measurement accuracy — informs how we validate throughput claims |
| 227 | Profile only the kernels involved in the benchmark | Enhancement (CLOSED via #277) | Profiling accuracy — directly relevant to our tracing/profiling tools |
| 194 | (ahendriksen, Nov 2024) | Open | — |
| 193 | (bernhardmgruber, Nov 2024) | Open | — |
| 189 | (GregoryKimball, Oct 2024) | Open | — |
| 188 | Add CLI parameter to fix the run count | Enhancement | Deterministic benchmarking — we need fixed iteration counts for reproducibility |
| 186 | (bernhardmgruber, Sep 2024) | Open | — |
| 185 | (bernhardmgruber, Sep 2024) | Open | — |
| 184 | (fbusato, Aug 2024) | Open | — |
| 182 | Compiler errors building examples with CUDA 11.5 | Bug | Compiler compatibility — relevant to our cross-platform compilation strategy |
| 180 | (samaid, Jul 2024) | Open | — |
| 179 | (bernhardmgruber, Jul 2024) | Open | — |
| 177 | (open) | Open | — |
| 136 | Async benchmarks always deadlock | Bug | CUDA lazy loading deadlock — informs our understanding of runtime initialization |
| 132 | Tracking nvbench benchmark results | Enhancement | Result tracking — directly applicable to our CI/CD benchmark pipeline |
| 115 | How can the output metrics be interpreted | Docs | Metric interpretation — educational for understanding measurement methodology |
| 101 | Passing method with multiple template arguments to NVBENCH_BENCH | Bug | API ergonomics |
| 92 | Specify the kernel name or NVTX range for benchmarking | Enhancement | Per-kernel profiling — maps to our per-round profiling needs |
| 84 | Reasonable enum defaults don't work | Bug | API robustness |
| 12 | Add option to add plain text description to a benchmark | Enhancement | Documentation features |

### Closed Issues (Selected, 12+ tracked)

| # | Title | Resolution | Lessons for trainiumCUDA |
|---|-------|------------|--------------------------|
| 318 | Add option to bulk store frequencies | Completed | Frequency-based analysis — applicable to our hash stage profiling |
| 300 | Release pynvbench pip wheels | Completed | Python packaging best practices |
| 297 | Allow comparing different devices with nvbench_compare.py | Completed | Cross-device comparison — we may need cross-config comparison |
| 295 | Allow partial comparison in nvbench_compare.py | Completed | Partial benchmark comparison — useful for A/B testing optimizations |
| 292 | NVBench should not make direct calls to driver API | Not planned | API abstraction decisions |
| 291 | Python entities of Python API should have docstrings | Completed | Documentation standards |
| 284 | pynvbench: Need to repeatedly Ctrl-C to stop script | Completed | Signal handling in benchmark runners |
| 279 | Provided ways to avoid install dependencies | Completed | Dependency minimization |
| 274 | Example of benchmark with typename and integral constant | Completed | Template metaprogramming patterns |
| 273 | --profile mode does not produce reasonable runtime | Not planned | Profile mode accuracy — critical insight for our tracing |
| 272 | Add command line option to skip batch benchmarks | Completed | Selective benchmarking |
| 264 | Unreachable code warning | Completed | Code quality |

---

## PART D: Files Summary

### Files to Modify
| File | Tasks | Description |
|------|-------|-------------|
| `perf_takehome.py` | 2,3,4,5,6,7,8,9,11-19 | Main kernel builder and scheduler — ALL optimizations happen here |

### Files to Create (New)
| File Path | Task | Description |
|-----------|------|-------------|
| `trainiumCUDA/plan.md` | — | This file |
| `trainiumCUDA/analysis/critical_path_report.md` | 1 | Profiling analysis report |
| `trainiumCUDA/tests/optimization_regression.py` | 10 | Multi-seed correctness regression tests |
| `trainiumCUDA/optimizers/bundle_packer.py` | 16 | Post-scheduler VLIW bundle optimizer |
| `trainiumCUDA/optimizers/profile_guided.py` | 17 | Profile-guided rescheduling tool |
| `trainiumCUDA/benchmarks/final_report.md` | 20 | Final benchmark results and analysis |

### Files to NOT Modify (Frozen)
| File | Reason |
|------|--------|
| `tests/submission_tests.py` | Frozen test — validation must use unmodified copy |
| `tests/frozen_problem.py` | Frozen simulator — must not be changed |
| `problem.py` | Simulator definition — modifying would invalidate results |

---

## PART E: Git Workflow

### Branch Strategy
```
main
├── feat/optimization-round1  (Claude — Tasks 1-10, this PR)
│   ├── Commit: "task-1: add critical path analysis"
│   ├── Commit: "task-2: relax WAR dependencies in scheduler"
│   ├── Commit: "task-3: multi-factor priority heuristic"
│   ├── Commit: "task-4: interleave hash stages for ILP"
│   ├── Commit: "task-5: reorder level3+ gather loads"
│   ├── Commit: "task-6: vectorize XOR stage"
│   ├── Commit: "task-7: vectorize IDX update stage"
│   ├── Commit: "task-8: optimize init load packing"
│   ├── Commit: "task-9: simplify level1/2 bit manipulation"
│   └── Commit: "task-10: add regression test suite"
│
└── feat/optimization-round2  (Codex — Tasks 11-20, separate PR)
    ├── Depends on: feat/optimization-round1 merged to main
    └── Commits: task-11 through task-20
```

### PR Checklist
- [ ] `git diff origin/main tests/` is empty (tests unchanged)
- [ ] `python tests/submission_tests.py` passes with improved cycle count
- [ ] All new files have proper paths documented
- [ ] Cycle count < 1363 (Round 1 target)
- [ ] Cycle count < 1200 (Round 2 stretch target)
