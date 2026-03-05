# Performance Optimization Plan: trainiumCUDA → sub-1000 cycles



The kernel is **95.8% load-bound**:
- Load engine: 2625 ops, active 1301/1370 cycles at max capacity (2/2)
- VALU engine: 7000 ops, 85.2% utilization  
- ALU engine: 13926 ops, 84.7% utilization
- Flow engine: 289 ops, 21.1% utilization
- Store engine: 64 ops, 2.3% utilization

**To reach 1001 cycles, we must eliminate ~738 load operations (369 load-cycles).**

---

## Architecture Summary

| Engine | Slots/Cycle | Current Utilization | Status |
|--------|------------|-------------------|--------|
| ALU    | 12         | 84.7%             | Supporting |
| VALU   | 6          | 85.2%             | Near-saturated |
| Load   | 2          | **95.8%**         | **BOTTLENECK** |
| Store  | 2          | 2.3%              | Idle |
| Flow   | 1          | 21.1%             | Available |

### Per-Round Analysis (16 rounds, tree_depth=11)

| Round | Level | Load Ops | Load Cycles | VALU Cycles | Bottleneck |
|-------|-------|----------|-------------|-------------|------------|
| 0,11  | 0     | 0        | 0           | ~70         | VALU       |
| 1,12  | 1     | 0        | 0           | ~75         | VALU+Flow  |
| 2,13  | 2     | 0        | 0           | ~96         | VALU+Flow  |
| 3,14  | 3     | 256      | 128         | ~70         | **LOAD**   |
| 4,15  | 4     | 256      | 128         | ~70         | **LOAD**   |
| 5     | 5     | 256      | 128         | ~70         | **LOAD**   |
| 6     | 6     | 256      | 128         | ~70         | **LOAD**   |
| 7     | 7     | 256      | 128         | ~70         | **LOAD**   |
| 8     | 8     | 256      | 128         | ~70         | **LOAD**   |
| 9     | 9     | 256      | 128         | ~70         | **LOAD**   |
| 10    | 10    | 256      | 128         | ~64         | **LOAD**   |

---

## Optimization Approaches (Ordered by Impact)

### TIER 1: Critical (Required for sub-1001)

#### 1. [P0] Cache Tree Level 3 — Save ~94 cycles
**Status: BLOCKED by scratch space (need 64 words, have 21 free)**
- Level 3 has only 8 nodes, appears in rounds 3,14
- Cache 8 scalar values at init, broadcast to 8 vectors
- Eliminates 512 load ops (256 per round × 2 rounds)
- Each round becomes VALU-bound (~70 cy) instead of load-bound (128 cy)
- Net savings: (128-70) × 2 = ~116 cycles minus vselect overhead
- **Prerequisite**: Free scratch via approach #5

#### 2. [P0] Cache Tree Level 4 — Save ~94 cycles  
**Status: BLOCKED by scratch space (need 128 words)**
- Level 4 has 16 nodes, appears in rounds 4,15
- Same approach as level 3 but with 4-level binary vselect
- **Prerequisite**: Free scratch via approach #5

#### 3. [P0] Cache Tree Level 5 — Save ~47 cycles
**Status: BLOCKED by scratch space (need 256 words)**
- Level 5 has 32 nodes, appears in round 5 only
- **Prerequisite**: Free scratch via approach #5

#### 4. [P0] Eliminate vselect Overhead for Cached Levels
- For level 3 (8 nodes): binary search needs 7 vselects × 32 = 224 flow cycles
- This is WORSE than 128 load cycles!
- **Solution**: Use multiply_add branchless mux chains (algebraic selection)
- For 8 nodes: 3 stages × ~5 VALU = 15 VALU × 32 = 480 ops = 80 cycles
- For 16 nodes: 4 stages ≈ 100 cycles  
- Must balance VALU pressure vs flow availability

#### 5. [P0] Free Scratch Space — Enable levels 3-5 caching
**Options (pick combination):**
- a) Alias v_nv with v_t1 for levels 3+ (save 256 words) — Complex, level 2 conflicts
- b) Process in 2 halves (U=16 per half, loop) — Save 640 words but 2x overhead
- c) Reduce U to 24 (3 passes of 8 vectors) — Save 320 words  
- d) Store v_nv values directly in v_ad after addr compute (save 256 words)
- e) Remove unused scratch allocations — Save 4 words (already identified)
- f) Reuse lv_addr after init — Save 7 words
- g) Compact scalar constants — Share with unused var space

### TIER 2: Important (Required for competitive submission)

#### 6. [P1] Improve Scheduler Load Packing
- 69 cycles currently have <2 loads (wasted bandwidth)
- Improve priority heuristic to always maximize load fill
- Target: save 20-40 cycles

#### 7. [P1] Software Pipelining Across Rounds
- Overlap hash computation of round N with loads of round N+1
- Challenge: round N+1's addresses depend on round N's hash result
- Speculative approach: precompute v_idx*2+1 before hash completes
- Then only need 1 correction bit after hash → overlap most of addr computation

#### 8. [P1] Convert Level 0 XOR to VALU
- Currently 256 per-lane ALU ops for XOR with cached root value
- Could be 32 VALU ops — saves 16 cycles per level 0 round
- Risk: adds VALU pressure on non-load-bound rounds
- **Only beneficial if VALU has spare capacity** (check per-round)

### TIER 3: Nice to Have

#### 9. [P2] Store Optimization — vstore Batching
- Currently 64 store ops at end
- Can overlap stores with final hash computations

#### 10. [P2] Init Phase Optimization  
- Better packing of constant loads and broadcasts during init

---

## Issue Tracker: Related Compiler/SDK Problems

### From aws-neuron/nki-samples
| # | Issue | Our Relevance | Status |
|---|-------|--------------|--------|
| 72 | Python call stack overhead in kernel construction | Build-time optimization, not runtime | Low |
| 69 | Wrong instruction dependency | **CRITICAL**: Same class of dep tracking bugs in our scheduler | High |
| 51 | Fused attention errors with randn | Numeric precision in fused ops | Low |
| 36 | MatMul result/store semantics | Memory layout for tile operations | Medium |

### From aws-neuron/aws-neuron-sdk
| # | Issue | Our Relevance | Status |
|---|-------|--------------|--------|
| 1015 | Error benchmarking auto-generated NKI | Benchmark methodology | Low |
| - | VLIW instruction scheduling passes | Direct analog to our scheduler | High |
| - | Memory allocation optimization | Scratch space management ≈ SBUF allocation | High |

### From Xilinx/llvm-aie (AIEngine LLVM)
| Area | Relevance | Transferable Insight |
|------|-----------|---------------------|
| VLIW scheduling | Direct analog: AIEngine uses VLIW | Priority functions, resource balancing |
| Software pipelining | Modulo scheduling for loops | Cross-round overlap techniques |
| Register allocation | Scratch = register file | Aliasing, spilling strategies |

### From llvm/llvm-project
| Area | Relevance | Transferable Insight |
|------|-----------|---------------------|
| MachineScheduler | List scheduling with priority | Better heuristics for our scheduler |
| ModuloScheduler | Software pipelining | Cross-iteration overlap |
| RegisterScavenger | Register reuse | Scratch space aliasing/reuse |
| PostRAScheduler | Post-allocation scheduling | Schedule after scratch allocation |

### From NVIDIA CCCL / GPU MODE
| Insight | Transferable to Our Problem |
|---------|----------------------------|
| CUB sort uses architecture-tuned policies | Our scheduler should tune to slot limits |
| Vectorized memory access patterns | Maximize vload where possible |
| Warp-level primitives for reduction | Vector-level reduction in hash |

---

## Task List for Implementation

### Phase 1: Scratch Space Recovery (Claude)
- [ ] Task 1: Remove unused var allocations (rounds, n_nodes, etc.) — Save 4 words
- [ ] Task 2: Reuse lv_addr after init loads — Save 7 words
- [ ] Task 3: Alias v_nv with v_ad for levels 3+ — Save 256 words
- [ ] Task 4: Compact scalar constants into freed space
- [ ] Task 5: Verify scratch usage fits with level 3 cache (64 words needed)

### Phase 2: Level 3-4 Caching (Claude)
- [ ] Task 6: Implement level 3 cache (8 nodes, scalar load + broadcast)
- [ ] Task 7: Implement branchless mux for level 3 (multiply_add chain)
- [ ] Task 8: Test and verify correctness with level 3 cache
- [ ] Task 9: Implement level 4 cache if scratch allows (16 nodes)
- [ ] Task 10: Benchmark after level caching

### Phase 3: Scheduler Improvements (Codex)
- [ ] Task 11: Profile per-cycle load utilization gaps
- [ ] Task 12: Improve load-feeder priority to reduce gaps
- [ ] Task 13: Implement speculative addr precomputation for next round
- [ ] Task 14: Add inter-round dependency analysis to scheduler
- [ ] Task 15: Test cross-round software pipelining

### Phase 4: Micro-optimizations (Codex)
- [ ] Task 16: Convert level 0 XOR to VALU (if VALU has capacity)
- [ ] Task 17: Optimize store phase overlap
- [ ] Task 18: Reduce init phase cycles
- [ ] Task 19: Fine-tune scheduler priority weights
- [ ] Task 20: Final benchmark and validation

---

## Theoretical Limits

With perfect scheduling and level 3-5 caching:
- Non-load rounds (0-5, 11-15): 11 rounds × ~70 VALU cycles = 770 cycles
- Load rounds (6-10): 5 rounds × 128 load cycles = 640 cycles  
- Overlap between rounds: ~200 cycles (hash under load shadow)
- Store + init overhead: ~30 cycles
- **Theoretical minimum: ~970 cycles** (with perfect overlap)
- **Practical target: ~1050-1100 cycles**

To actually reach sub-1001 requires either:
1. Caching level 5+ (eliminates more load rounds), OR
2. Processing in 2 halves with smaller U, OR
3. Fundamentally reducing loads via memory layout changes

---

## Git Commit Strategy

Each optimization phase should be a separate commit:
```
feat: phase-1 scratch-recovery - free 267 words via aliasing
feat: phase-2 level3-cache - cache 8 tree nodes, save ~94 cycles  
feat: phase-2 level4-cache - cache 16 tree nodes, save ~94 cycles
feat: phase-3 scheduler-improve - better load packing heuristics
feat: phase-4 micro-opts - XOR to VALU, store overlap
```

PR description should include:
- Before/after cycle count
- `git diff origin/main tests/` output (must be empty)
- `python tests/submission_tests.py` output

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


# Performance Optimization Plan: trainiumCUDA → sub-1001 cycles

## Current Status: 1370 cycles (107.8x speedup over baseline 147734)

## Architecture Profile (at 1370 cycles)

| Engine | Slots/Cycle | Total Ops | Min Cycles | Utilization |
|--------|------------|-----------|------------|-------------|
| Load   | 2          | 2625      | 1313       | **95.8%**   |
| VALU   | 6          | 7000      | 1167       | 85.2%       |
| ALU    | 12         | 13926     | 1161       | 84.7%       |
| Flow   | 1          | 289       | 289        | 21.1%       |
| Store  | 2          | 64        | 32         | 2.3%        |

**Bottleneck: LOAD engine at 95.8% utilization**

Load slot utilization:
- 2 loads (full): 1301 cycles (95.0%)
- 1 load (half): 23 cycles (1.7%)
- 0 loads: 46 cycles (3.4%)

## Theoretical Minimum Analysis

### Per-Round Structure (tree_depth=11, 16 rounds)

**Load-bound rounds (levels 3-10, rounds 3-10,14-15): 10 rounds**
- Load bottleneck: 256 load_offset / 2 = 128 cycles per round
- Total: 10 × 128 = 1280 cycles

**Cached rounds (levels 0-2, rounds 0-2,11-13): 6 rounds**
- VALU bottleneck: ~416 ops / 6 = ~70 cycles per round
- Total: 6 × 70 = 420 cycles

**Round transitions**: 15 × ~4 = 60 cycles
**Init + stores**: ~16 cycles

**Theoretical minimum**: max(1296, 1167, 1161) + 76 = **~1372 cycles**
**Current: 1370 — within 2 cycles of theoretical minimum!**

## Path to Sub-1001

Since we're at the theoretical minimum for the current op set, we must **reduce load ops** via tree level caching.

| Level | Nodes | Rounds | Loads Saved | Cycles Saved |
|-------|-------|--------|-------------|--------------|
| 3     | 8     | 3, 14  | 512         | ~60-196      |
| 4     | 16    | 4, 15  | 512         | ~60-196      |
| 5     | 32    | 5      | 256         | ~30-98       |
| **Total** | | | **1280** | **~150-490** |

**Caching levels 3+4 → target ~978-1210 cycles**

### Key Challenge: Scratch Space

- Current usage: 1515 / 1536 (21 free)
- Level 3 cache needs: ~64 words (8 vectors)
- Level 4 cache needs: ~128 words (16 vectors)
- **Solution**: Alias v_nv with cache vectors (v_nv only used levels 1,2; cache for 3,4)
- v_nv[0..7] → level 3 cache, v_nv[8..23] → level 4 cache
- Already moved load dest from v_nv to v_ad for levels 3+ (done)

### Key Challenge: Selection Mechanism

With 1 flow slot/cycle, vselect is expensive:
- 8 values (level 3): 7 vselects × 32 elements = 224 flow cycles > 128 load cycles
- Need alternative: staged pair selection, multiply_add mux, or hybrid approach
- Research: 3-stage cascade may work (96 flow cycles, saving 32/round)

---

## Task List

### Phase 1: Claude — Tasks 1-5 (Target: 1370→1250)

- [x] Task 1: Move load_offset to v_ad for levels 3+ (done, 1370 maintained)
- [ ] Task 2: Scratch aliasing — v_nv[0..7] ← level 3 cache vectors
- [ ] Task 3: Level 3 cache with best available mux strategy
- [ ] Task 4: Level 4 cache implementation
- [ ] Task 5: Scheduler load packing (fill 23 half-empty load cycles)

### Phase 2: Codex — Tasks 6-10

- [ ] Task 6: Software pipelining across rounds
- [ ] Task 7: Hash 3-op stage optimization (parallel ALU+VALU split)
- [ ] Task 8: Store overlap with final round
- [ ] Task 9: Level 5 cache (32 nodes, 1 round)
- [ ] Task 10: ILP-based optimal scheduler prototype

### Phase 3: Joint — Tasks 11-13

- [ ] Task 11: Multi-round software pipelining with double buffering
- [ ] Task 12: Speculative execution with double-load
- [ ] Task 13: Cross-element hash interleaving

---

## Cross-Repository Issue Tracker

### Applicable Issues

| Source | Area | Relevance | Insight |
|--------|------|-----------|---------|
| nki-samples #69 | Instruction dependency | **HIGH** | WAR/RAW tracking patterns |
| aws-neuron-sdk | VLIW scheduling | **HIGH** | Priority function design |
| aws-neuron-sdk | Memory allocation | **HIGH** | Scratch aliasing strategies |
| Xilinx/llvm-aie | Modulo scheduling | **HIGH** | Cross-round overlap |
| llvm MachineScheduler | List scheduling | **HIGH** | Load-first heuristics |
| llvm RegisterScavenger | Register reuse | **HIGH** | Lifetime-based aliasing |
| NVIDIA CCCL | Architecture tuning | **MEDIUM** | Per-engine slot optimization |

### Deleted (Not Applicable)
- nki-samples #107 (FFT), #51 (attention), #81 (dropout), #67 (MaxPool2D)
- nki-moe #15, #14 (administrative)

---
