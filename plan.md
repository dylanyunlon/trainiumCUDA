# Performance Optimization Plan: trainiumCUDA → sub-1000 cycles

## Current Status
- **Score: 1370 cycles** (107.8x speedup over 147734 baseline)
- **Target: <1001 cycles** (kerneloptimization.fun top tier)
- **Recruiting threshold: 1487 cycles** ✅ (already passed)
- **Claude Opus 4.5 best: 1363 cycles** (7 cycles away)
- **Community best reported: ~1338 cycles** (Medium article by Indosambhav)

## Architecture Bottleneck Analysis

### Resource Utilization (1370 cycles)
| Engine | Total Ops | Per-Cycle Limit | Min Cycles | Avg Utilization |
|--------|-----------|-----------------|------------|-----------------|
| LOAD   | 2625      | 2               | **1313**   | 95.8%           |
| VALU   | 7000      | 6               | 1167       | 85.2%           |
| ALU    | 13926     | 12              | 1161       | 84.7%           |
| FLOW   | 289       | 1               | 289        | 21.1%           |
| STORE  | 64        | 2               | 32         | 4.7%            |

### Critical Finding
**LOAD engine is the bottleneck** at 1313 minimum cycles. Current 1370 is only 57 cycles above this floor. The deep levels (3-10) consume 256 loads per round × 10 deep rounds = 2560 loads, 97% of total.

### Cycle Budget Breakdown
- Init section: ~15 cycles
- Shallow levels (0,1,2): ~80 cycles (well-utilized with vselect + cached values)
- Deep levels (3-10): ~1180 cycles (fully packed: load=2, valu=6, alu=12 per cycle)
- Final round tail (level 4, no overlap): ~85 cycles
- Store section: ~10 cycles

## Cross-Repository Issue-Driven Optimization Ideas

### From aws-neuron/nki-samples
| Issue | Insight | Applicability |
|-------|---------|---------------|
| #69 Wrong instruction dependency | Dependencies between operations can be over-conservative. Check if our RAW/WAR dependency tracking creates false dependencies. | **HIGH** - Our scheduler may have false deps from vector aliasing |
| #72 Python call stack overhead in kernel construction | Our build_kernel runs in Python with O(n²) dependency analysis. Optimize the scheduler itself. | **MEDIUM** - Doesn't affect cycle count but affects iteration speed |
| #105 Batch invariance | Some operations are batch-invariant (same computation across all elements). Hoist them out. | **APPLIED** - Already hoist constants and broadcasts |
| #51 Attention errors with randn vs rand | Numerical precision of hash may allow algebraic simplification. | **LOW** - Hash is fixed and correctness must be exact |

### From aws-neuron/aws-neuron-sdk
| Issue Area | Insight | Applicability |
|------------|---------|---------------|
| Memory hierarchy optimization | NeuronCore uses SBUF (scratchpad) vs HBM like our scratch vs main memory. Key: minimize main memory access. | **CRITICAL** - Every load from main memory costs a slot. Cache more levels. |
| Instruction scheduling across engines | NeuronCore has 4 async engines with semaphore sync. Our machine has 5 engines with same-cycle write semantics. | **HIGH** - Engine interleaving is our main scheduling challenge |
| Software pipelining | NKI supports explicit pipeline staging. We could pipeline hash computation across U elements more aggressively. | **HIGH** - Already doing this implicitly via scheduler |
| Background data movement | NeuronCore's "background LoadStationary" overlaps loads with compute. We do this via interleaving. | **APPLIED** |

### From aws-neuron/nki-moe
| Issue | Insight | Applicability |
|-------|---------|---------------|
| #12 Configuration exposure | Tuning parameters (tile sizes, pipeline depths) significantly affect performance. | **HIGH** - We should try different U sizes, hash interleaving patterns |
| Expert parallelism | MoE routes different inputs to different experts. Our tree traversal routes different batch elements to different tree paths. | **MEDIUM** - Already fully parallelized across U elements |

### From Xilinx/llvm-aie
| Issue Area | Insight | Applicability |
|------------|---------|---------------|
| VLIW scheduling in AIE | AIE has similar VLIW constraints with 7 functional units. Their scheduler uses modulo scheduling for loops. | **HIGH** - Modulo scheduling could improve our round loop |
| Register allocation | AIE has limited registers. Our scratch space (1536 words, 1515 used) is similarly constrained. | **CRITICAL** - Need scratch reclamation for level 3 caching |
| Instruction bundling constraints | AIE has read-after-write latency > 1 in some cases. Our latency is always 1. | **N/A** - Our model is simpler |

### From llvm/llvm-project
| Issue Area | Insight | Applicability |
|------------|---------|---------------|
| Machine instruction scheduler | LLVM's MachineScheduler uses ASAP/ALAP scheduling with critical path priority. | **APPLIED** - Our scheduler uses similar approach |
| Post-RA scheduling | After register allocation, schedule to minimize stalls. | **HIGH** - We allocate scratch first, then schedule |
| Loop unrolling + SLP vectorization | Unroll and vectorize inner loops. | **APPLIED** - Rounds are fully unrolled, batch is vectorized |
| Peephole optimization | Local instruction combining. | **MEDIUM** - Could combine XOR+hash for specific patterns |

### From NVIDIA/cccl
| Issue Area | Insight | Applicability |
|------------|---------|---------------|
| CUB block-level primitives | CUB provides optimized scan/reduce with warp-level shuffles. | **LOW** - Our operations are element-wise, not reductions |
| Architecture-aware tuning | CCCL tunes algorithms per GPU architecture (Hopper vs Blackwell). | **HIGH** - We should tune for the specific VLIW slot limits |
| Memory access coalescing | Coalesced access patterns reduce memory transactions. | **APPLIED** - Using vload/vstore for contiguous access |

## Prioritized Optimization Tasks

### TIER 1: Immediate Gains (Target: 1370→1340, ~30 cycles)

#### Task 1: Level 3 Tree Caching with Scratch Reclamation
- **Reclaim scratch**: Eliminate vlv0 (8 words) by using per-lane scalar ALU XOR for level 0
- **Allocate**: 8 scalar slots for level 3 node values (8 words)
- **At level 3**: Broadcast needed pair, use multiply_add-based select
- **Expected saving**: ~20-30 cycles from eliminating 512 loads across 2 rounds
- **Risk**: Medium - changes level 0 from VALU XOR to ALU XOR
- **Files**: `perf_takehome.py` (modify `build_kernel`)

#### Task 2: Modulo Scheduling for Hash Pipeline
- **Concept**: Instead of processing all U elements per hash stage sequentially, interleave hash stages of different k values
- **Implementation**: Emit hash stage i for k=0..chunk, then stage i+1 for k=0..chunk, overlapping
- **Expected saving**: ~5-10 cycles from better pipeline utilization at round boundaries
- **Risk**: Low - scheduler should handle correctly
- **Files**: `perf_takehome.py` (modify hash emission in `build_kernel`)

#### Task 3: Optimize Final Round
- **Last round** (r=15) has no next-round overlap, causing 85 sparse cycles
- **Optimization**: Since no idx update needed, emit stores DURING hash of last round
- **Expected saving**: ~5-10 cycles from overlapping stores with final hash
- **Risk**: Low
- **Files**: `perf_takehome.py` (modify store emission timing)

### TIER 2: Medium Gains (Target: 1340→1280, ~60 cycles)

#### Task 4: Level 4 Tree Caching
- **Requires**: Level 3 cache working + additional scratch reclamation
- **Reclaim**: Use v_three (8 words) - compute level 2 offset differently
- **Allocate**: 16 scalar slots for level 4 node values
- **Selection**: 4-stage binary select using multiply_add trick
- **Expected saving**: ~30-40 cycles from eliminating 512 loads across 2 rounds
- **Files**: `perf_takehome.py`

#### Task 5: Hybrid VALU/ALU Balance Optimization
- **Insight**: Deep levels are perfectly balanced. Shallow levels and round boundaries waste ALU capacity.
- **For shallow levels**: Use per-lane ALU instead of VALU for some operations to better fill cycles
- **For round boundaries**: Emit bridge operations (precomputation for next round)
- **Expected saving**: ~10-20 cycles
- **Files**: `perf_takehome.py`

#### Task 6: Scheduler Enhancement - Two-Phase Load Prioritization
- **Phase 1**: Schedule all loads to their earliest possible cycle
- **Phase 2**: Fill remaining slots with compute ops
- **This ensures** load pipeline is never starved
- **Expected saving**: ~5-10 cycles from fewer load stalls
- **Files**: `perf_takehome.py` (modify scheduler)

### TIER 3: Advanced Optimizations (Target: 1280→1100)

#### Task 7: Deep Level Caching up to Level 5
- **Requires**: Significant scratch reclamation (aliasing temp vectors)
- **Level 5**: 32 nodes → 32 scalar loads + 5-stage select
- **Saves**: 256 loads per round × 1 round = 256 loads = 128 cycles
- **Costs**: ~50 VALU cycles for selection
- **Net**: ~78 cycles saved
- **Files**: `perf_takehome.py`

#### Task 8: Algebraic Hash Optimization
- **3-op hash stages**: a = (a op1 c1) op2 (a op3 c3)
- **Stage 3**: a = (a + c1) ^ (a << 9). Could precompute (a << 9) using ALU while + runs on VALU
- **Split 3-op into**: 2 parallel ops (VALU + ALU) + 1 combine (VALU)
- **Reduces critical path** by 1 cycle per 3-op stage per round
- **Expected saving**: ~15-30 cycles over 16 rounds
- **Files**: `perf_takehome.py`

#### Task 9: Reduce Scratch Footprint via Register Aliasing
- **Map non-overlapping lifetimes** to same scratch addresses
- **v_nv** and **v_t1** have different live ranges within a round
- **Could share addresses** between hash temps and level cache
- **Frees**: Up to 256 scratch words for deeper caching
- **Files**: `perf_takehome.py`

#### Task 10: Whole-Program ILP Optimization
- **Use ILP solver** (e.g., OR-Tools) to find optimal schedule
- **Minimize makespan** subject to resource and dependency constraints
- **Could find** globally optimal schedule within the given operation set
- **Expected saving**: Up to theoretical minimum gap (57 cycles)
- **Risk**: High compute time for solver, but one-time cost
- **Files**: New `optimal_scheduler.py` + modify `perf_takehome.py`

### TIER 4: Breakthrough Optimizations (Target: 1100→1000)

#### Task 11: Multi-Round Software Pipelining
- **Instead of processing round-by-round**, pipeline stages across rounds
- **Round r hash + Round r+1 load** can overlap if we maintain double buffers
- **Requires**: Duplicate v_val and v_idx arrays (doubles scratch usage)
- **Major scratch reclamation needed** through aggressive aliasing
- **Files**: Complete rewrite of `build_kernel`

#### Task 12: Speculative Execution with Rollback
- **Speculatively start** next round's load using predicted idx
- **If prediction wrong** (50% of branches), fix by adding correction
- **For tree depth 11**: Most elements are at deep levels where prediction is useful
- **Files**: `perf_takehome.py`

#### Task 13: Custom Hash Scheduling with Cross-Element Interleaving  
- **Instead of all-k-same-stage**, interleave different elements at different hash stages
- **This smooths out** resource usage and reduces peak VALU demand
- **Requires**: Major restructuring of operation emission order
- **Files**: `perf_takehome.py`

## Cross-Repository Issue Tracking for Plan

### Issues to Monitor

1. **nki-samples #69** (Wrong instruction dependency): Could reveal scheduler bug patterns applicable to our WAR/RAW tracking
2. **nki-samples #72** (Python overhead): Applicable to speeding up our scheduler for faster iteration
3. **nki-moe #12** (Configuration tuning): Reminds us to parameterize and auto-tune
4. **NVIDIA/cccl scheduling** issues: Any CUB primitive improvements may inspire parallel pattern ideas
5. **llvm-project** backend scheduler: Latest LLVM scheduling heuristics could improve our scheduler
6. **Xilinx/llvm-aie** modulo scheduling: Direct inspiration for our round loop scheduling

### Issues Not Applicable (Deleted from consideration)
- nki-samples #107 (FFT kernel): Irrelevant - we don't do FFT
- nki-samples #51 (attention precision): Irrelevant - we use exact integer hash
- nki-samples #81 (dropout backward): Irrelevant - no dropout in our kernel
- nki-samples #67 (MaxPool2D): Irrelevant - no pooling operations
- nki-moe #15, #14 (competition/quota): Administrative, not technical
- nki-moe #8 (expert parallelism decoding): Architecture-specific, not applicable

## Execution Plan

### Phase 1 (Claude - Tasks 1-5)
1. Implement Level 3 caching with scratch reclamation
2. Optimize final round store overlap
3. Test modulo scheduling variants
4. Implement level 4 caching
5. Tune VALU/ALU balance for shallow levels

### Phase 2 (Codex - Tasks 6-10)  
6. Two-phase scheduler with load prioritization
7. Deep level caching (level 5)
8. Algebraic hash optimization
9. Register aliasing for scratch reclamation
10. ILP-based optimal scheduler prototype

### Phase 3 (Joint - Tasks 11-13)
11. Multi-round software pipelining
12. Speculative execution
13. Cross-element hash interleaving

## File Modification Summary

### Files to Modify
- `perf_takehome.py`: All kernel and scheduler changes (primary file)

### Files to Create
- `plan.md`: This file (optimization roadmap)
- `analysis/bottleneck_report.md`: Detailed cycle-by-cycle analysis
- `analysis/scheduler_experiments.py`: Scheduler variant testing harness

### Files to Delete
- `perf_takehome_oldprint.py`: Legacy file with old debug prints (cleanup)

## Key Metrics to Track
- Total cycles (primary metric)
- Load utilization (target: >96%)  
- VALU utilization (target: >85%)
- ALU utilization (target: >80%)
- Sparse cycle count (target: <30)
- Scratch usage (target: maximize caching within 1536 limit)
