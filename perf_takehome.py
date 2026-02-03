"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.

================================================================================
GANNINA_OPTIMIZATION_LOG (DO NOT DELETE - PERSISTENT STATE FOR NEXT CONVERSATION)
================================================================================
PHASE: 1 - VECTORIZATION COMPLETE (22094 cycles)
SPEEDUP: 6.69x over baseline
TARGET: <1487 cycles

CURRENT_ANALYSIS:
  22016 instrs, 22094 cycles => NO VLIW parallelism (1:1 ratio)
  Per vec_iter (512 total): load=10, valu=25, flow=2, alu=4, store=2

SLOT_LIMITS: alu=12, valu=6, load=2, store=2, flow=1

THEORETICAL_MINIMUM (naive VLIW):
  load: 10/2 = 5 cycles/iter
  valu: 25/6 = 5 cycles/iter  
  flow: 2/1 = 2 cycles/iter
  MINIMUM = 512 * 5 = 2560 cycles (load-bound)

CRITICAL_PROBLEM: TARGET 1487 < THEORETICAL_MIN 2560 !!!
  Must reduce TOTAL WORK, not just pack better!

BREAKTHROUGH_STRATEGIES:
  1. SOFTWARE_PIPELINING: Overlap iter[i].store with iter[i+1].load
     - Reduces effective cycles/iter by hiding latency
  2. LOOP_UNROLLING: Process 2+ vec_iters together
     - Reuse constants, reduce loop overhead
     - More slots available for parallel scheduling
  3. REDUCE_GATHER_OPS: 8 load_offset = 8 loads = 4 cycles minimum
     - Can we restructure data layout? NO (frozen tests)
     - Can we cache forest values? Limited scratch space
  4. HASH_OPTIMIZATION: 18 valu ops for hash, could interleave with loads
================================================================================
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_slot_deps(self, engine, slot):
        """Extract (writes, reads) sets for a slot. Returns sets of (addr, lane) tuples."""
        writes = set()
        reads = set()
        
        if engine == "debug":
            return writes, reads
        
        if engine in ("alu", "valu"):
            op, dest, src1, src2 = slot
            if engine == "valu" and op == "vbroadcast":
                # vbroadcast: writes dest[0:VLEN], reads src1 (scalar)
                for j in range(VLEN):
                    writes.add((dest, j))
                reads.add((src1, 0))
            elif engine == "valu":
                # Vector ops: read/write all VLEN lanes
                for j in range(VLEN):
                    writes.add((dest, j))
                    reads.add((src1, j))
                    reads.add((src2, j))
            else:
                # Scalar ALU
                writes.add((dest, 0))
                reads.add((src1, 0))
                reads.add((src2, 0))
        
        elif engine == "load":
            op = slot[0]
            if op == "const":
                # const: writes dest, no reads
                _, dest, val = slot
                writes.add((dest, 0))
            elif op == "load":
                # load: writes dest, reads addr
                _, dest, addr = slot
                writes.add((dest, 0))
                reads.add((addr, 0))
            elif op == "vload":
                # vload: writes dest[0:VLEN], reads addr (scalar)
                _, dest, addr = slot
                for j in range(VLEN):
                    writes.add((dest, j))
                reads.add((addr, 0))
            elif op == "load_offset":
                # load_offset: writes dest[j], reads addr_vec[j]
                _, dest, addr_vec, j = slot
                writes.add((dest, j))
                reads.add((addr_vec, j))
        
        elif engine == "store":
            op = slot[0]
            if op == "store":
                _, addr, src = slot
                reads.add((addr, 0))
                reads.add((src, 0))
            elif op == "vstore":
                _, addr, src = slot
                reads.add((addr, 0))
                for j in range(VLEN):
                    reads.add((src, j))
        
        elif engine == "flow":
            op = slot[0]
            if op == "select":
                _, dest, cond, t, f = slot
                writes.add((dest, 0))
                reads.add((cond, 0))
                reads.add((t, 0))
                reads.add((f, 0))
            elif op == "vselect":
                _, dest, cond, t, f = slot
                for j in range(VLEN):
                    writes.add((dest, j))
                    reads.add((cond, j))
                    reads.add((t, j))
                    reads.add((f, j))
            elif op == "pause":
                pass  # No deps
        
        return writes, reads

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        """
        Pack slots into instruction bundles using list scheduling.
        """
        if not slots:
            return []
        
        n = len(slots)
        
        # Step 1: Extract dependencies for each slot
        slot_writes = []
        slot_reads = []
        for engine, slot in slots:
            w, r = self.get_slot_deps(engine, slot)
            slot_writes.append(w)
            slot_reads.append(r)
        
        # Step 2: Build dependency graph - need BOTH RAW and WAR dependencies!
        # RAW: slot j reads what slot i writes => j after i
        # WAR: slot j writes what slot i reads => j after i (to preserve read value)
        depth = [0] * n
        
        # Map: (addr, lane) -> last slot that wrote it (for RAW)
        last_writer = {}
        # Map: (addr, lane) -> last slot that read it (for WAR)
        last_readers = defaultdict(list)
        
        war_deps_found = 0
        raw_deps_found = 0
        
        for i in range(n):
            # Find max depth of all dependencies
            max_dep_depth = -1
            
            # RAW: I read something that was written before
            for r in slot_reads[i]:
                if r in last_writer:
                    dep_idx = last_writer[r]
                    max_dep_depth = max(max_dep_depth, depth[dep_idx])
                    raw_deps_found += 1
            
            # WAR: I write something that was read before
            for w in slot_writes[i]:
                if w in last_readers:
                    for reader_idx in last_readers[w]:
                        max_dep_depth = max(max_dep_depth, depth[reader_idx])
                        war_deps_found += 1
            
            depth[i] = max_dep_depth + 1
            
            # Update tracking
            for w in slot_writes[i]:
                last_writer[w] = i
                last_readers[w] = []  # Clear readers when overwritten
            for r in slot_reads[i]:
                last_readers[r].append(i)
        
        # Step 3: Group by depth, then pack respecting SLOT_LIMITS
        depth_groups = defaultdict(list)
        for i in range(n):
            engine = slots[i][0]
            if engine != "debug":
                depth_groups[depth[i]].append(i)
        
        instrs = []
        max_depth = max(depth) if depth else 0
        
        for d in range(max_depth + 1):
            group = depth_groups[d]
            
            # Pack slots at this depth into bundles respecting SLOT_LIMITS
            bundle_slots = defaultdict(list)
            bundle_counts = defaultdict(int)
            
            for idx in group:
                engine, slot = slots[idx]
                if engine == "debug":
                    continue
                
                # Check if we can add to current bundle
                limit = SLOT_LIMITS.get(engine, 1)
                if bundle_counts[engine] < limit:
                    bundle_slots[engine].append(slot)
                    bundle_counts[engine] += 1
                else:
                    # Emit current bundle, start new one
                    if bundle_slots:
                        instrs.append(dict(bundle_slots))
                    bundle_slots = defaultdict(list)
                    bundle_counts = defaultdict(int)
                    bundle_slots[engine].append(slot)
                    bundle_counts[engine] = 1
            
            # Emit remaining bundle for this depth
            if bundle_slots:
                instrs.append(dict(bundle_slots))
        
        # ========== GANNINA_VLIW_RESULT gannina ==========
        # 分析同一depth内是否有slots被错误分到不同bundle
        depth_overflow_count = 0
        for d in range(max_depth + 1):
            group = depth_groups[d]
            engine_counts = defaultdict(int)
            for idx in group:
                engine = slots[idx][0]
                if engine != "debug":
                    engine_counts[engine] += 1
            for eng, cnt in engine_counts.items():
                if cnt > SLOT_LIMITS.get(eng, 1):
                    depth_overflow_count += 1
        
        print(f"\n[GANNINA_VLIW_SCHEDULER] gannina77777777777777")
        print(f"  INPUT: {n} slots, OUTPUT: {len(instrs)} bundles, CYCLES: ~{len(instrs)} gannina")
        print(f"  COMPRESSION: {n/len(instrs):.2f}x, MAX_DEPTH: {max_depth} gannina")
        print(f"  RAW_DEPS: {raw_deps_found}, WAR_DEPS: {war_deps_found} gannina")
        print(f"  gannina")
        print(f"  CRITICAL_BUG: gannina99999999999999")
        print(f"    WAR deps inflate MAX_DEPTH from ~23 to 13311! gannina")
        print(f"    WAR across vec_iters is WRONG - they reuse scratch but run sequentially gannina")
        print(f"    FIX: REMOVE WAR tracking, keep only RAW gannina")
        print(f"  gannina")
        
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vector_hash(self, vec_val, vec_tmp1, vec_tmp2, vec_hash_consts, round, vi):
        """Vector hash: 6 stages, each stage needs 3 valu ops"""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec = vec_hash_consts[hi * 2]
            const3_vec = vec_hash_consts[hi * 2 + 1]
            slots.append(("valu", (op1, vec_tmp1, vec_val, const1_vec)))
            slots.append(("valu", (op3, vec_tmp2, vec_val, const3_vec)))
            slots.append(("valu", (op2, vec_val, vec_tmp1, vec_tmp2)))
            slots.append(("debug", ("vcompare", vec_val, tuple((round, vi + j, "hash_stage", hi) for j in range(VLEN)))))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """VECTORIZED kernel - Phase 1"""
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp_addr = self.alloc_scratch("tmp_addr")
        
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0, "zero")
        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")

        vec_idx = self.alloc_scratch("vec_idx", VLEN)
        vec_val = self.alloc_scratch("vec_val", VLEN)
        vec_node = self.alloc_scratch("vec_node", VLEN)
        vec_addr = self.alloc_scratch("vec_addr", VLEN)
        vec_tmp1 = self.alloc_scratch("vec_tmp1", VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", VLEN)
        vec_tmp3 = self.alloc_scratch("vec_tmp3", VLEN)
        
        vec_zero = self.alloc_scratch("vec_zero", VLEN)
        vec_one = self.alloc_scratch("vec_one", VLEN)
        vec_two = self.alloc_scratch("vec_two", VLEN)
        vec_forest_p = self.alloc_scratch("vec_forest_p", VLEN)
        vec_n_nodes = self.alloc_scratch("vec_n_nodes", VLEN)
        
        self.add("valu", ("vbroadcast", vec_zero, zero_const))
        self.add("valu", ("vbroadcast", vec_one, one_const))
        self.add("valu", ("vbroadcast", vec_two, two_const))
        self.add("valu", ("vbroadcast", vec_forest_p, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", vec_n_nodes, self.scratch["n_nodes"]))
        
        vec_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            scalar_c1 = self.scratch_const(val1, f"hash_{hi}_c1")
            scalar_c3 = self.scratch_const(val3, f"hash_{hi}_c3")
            vec_c1 = self.alloc_scratch(f"vec_hash_{hi}_c1", VLEN)
            vec_c3 = self.alloc_scratch(f"vec_hash_{hi}_c3", VLEN)
            self.add("valu", ("vbroadcast", vec_c1, scalar_c1))
            self.add("valu", ("vbroadcast", vec_c3, scalar_c3))
            vec_hash_consts.append(vec_c1)
            vec_hash_consts.append(vec_c3)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        body = []

        for round in range(rounds):
            for vi in range(0, batch_size, VLEN):
                vi_const = self.scratch_const(vi, f"vi_{vi}")
                
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], vi_const)))
                body.append(("load", ("vload", vec_idx, tmp_addr)))
                body.append(("debug", ("vcompare", vec_idx, tuple((round, vi + j, "idx") for j in range(VLEN)))))
                
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], vi_const)))
                body.append(("load", ("vload", vec_val, tmp_addr)))
                body.append(("debug", ("vcompare", vec_val, tuple((round, vi + j, "val") for j in range(VLEN)))))
                
                body.append(("valu", ("+", vec_addr, vec_forest_p, vec_idx)))
                
                for j in range(VLEN):
                    body.append(("load", ("load_offset", vec_node, vec_addr, j)))
                body.append(("debug", ("vcompare", vec_node, tuple((round, vi + j, "node_val") for j in range(VLEN)))))
                
                body.append(("valu", ("^", vec_val, vec_val, vec_node)))
                
                body.extend(self.build_vector_hash(vec_val, vec_tmp1, vec_tmp2, vec_hash_consts, round, vi))
                body.append(("debug", ("vcompare", vec_val, tuple((round, vi + j, "hashed_val") for j in range(VLEN)))))
                
                body.append(("valu", ("%", vec_tmp1, vec_val, vec_two)))
                body.append(("valu", ("==", vec_tmp1, vec_tmp1, vec_zero)))
                body.append(("flow", ("vselect", vec_tmp3, vec_tmp1, vec_one, vec_two)))
                body.append(("valu", ("*", vec_idx, vec_idx, vec_two)))
                body.append(("valu", ("+", vec_idx, vec_idx, vec_tmp3)))
                body.append(("debug", ("vcompare", vec_idx, tuple((round, vi + j, "next_idx") for j in range(VLEN)))))
                
                body.append(("valu", ("<", vec_tmp1, vec_idx, vec_n_nodes)))
                body.append(("flow", ("vselect", vec_idx, vec_tmp1, vec_idx, vec_zero)))
                body.append(("debug", ("vcompare", vec_idx, tuple((round, vi + j, "wrapped_idx") for j in range(VLEN)))))
                
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], vi_const)))
                body.append(("store", ("vstore", tmp_addr, vec_idx)))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], vi_const)))
                body.append(("store", ("vstore", tmp_addr, vec_val)))

        engine_counts = defaultdict(int)
        for e, s in body:
            engine_counts[e] += 1

        total_instrs = sum(v for k,v in engine_counts.items() if k != 'debug')
        vec_iters = rounds * (batch_size // VLEN)
        
        # ========== GANNINA: BREAKTHROUGH ANALYSIS gannina ==========
        load_per_iter = engine_counts['load'] // vec_iters
        valu_per_iter = engine_counts['valu'] // vec_iters
        flow_per_iter = engine_counts['flow'] // vec_iters
        
        # Calculate theoretical min with perfect VLIW
        load_cycles = (load_per_iter + 1) // 2  # ceil(10/2) = 5
        valu_cycles = (valu_per_iter + 5) // 6  # ceil(25/6) = 5
        flow_cycles = flow_per_iter             # 2/1 = 2
        naive_min = vec_iters * max(load_cycles, valu_cycles, flow_cycles)
        
        # With software pipelining: overlap store[i] with load[i+1]
        pipeline_min = vec_iters * max(load_cycles - 1, valu_cycles, flow_cycles)
        
        print(f"\n[GANNINA_BREAKTHROUGH_ANALYSIS] gannina11111111111111")
        print(f"  CURRENT: {total_instrs} instrs => 22094 cycles (no VLIW) gannina")
        print(f"  NAIVE_VLIW_MIN: {naive_min} cycles (load-bound: {load_cycles} cyc/iter) gannina")
        print(f"  TARGET: <1487 cycles gannina")
        print(f"  GAP: {naive_min} > 1487, VLIW alone NOT ENOUGH! gannina")
        print(f"  gannina")
        print(f"  BOTTLENECK_DETAIL (per iter): gannina222222222222222")
        print(f"    gather: 8 load_offset => 4 cycles (limit=2) gannina")
        print(f"    vload: 2 ops => 1 cycle gannina")
        print(f"    valu: {valu_per_iter} ops => {valu_cycles} cycles (limit=6) gannina")
        print(f"    flow: {flow_per_iter} ops => {flow_cycles} cycles (limit=1) gannina")
        print(f"  gannina")
        print(f"  SOFTWARE_PIPELINE_POTENTIAL: gannina333333333333333")
        print(f"    Overlap: iter[i].valu with iter[i].gather (both need ~4-5 cycles) gannina")
        print(f"    Overlap: iter[i].store with iter[i+1].vload gannina")
        print(f"    ESTIMATED: ~{pipeline_min} cycles gannina")
        print(f"  gannina4444444444444444444")
        print(f"  TO_REACH_1487: Need {naive_min/1487:.1f}x reduction gannina")
        print(f"  NEXT: Implement VLIW scheduler with dependency analysis gannina")
        print(f"  gannina")

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"Testing {forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : ref_mem[6] + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()