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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
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
    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        from collections import defaultdict

        U = batch_size // VLEN
        TREE_DEPTH = forest_height + 1

        # ================================================================
        # INIT SECTION - collect all init loads, then pack 2 per cycle
        # ================================================================
        tmp_init = self.alloc_scratch("tmp_init")
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        # Collect init loads instead of emitting them one at a time
        init_loads = []
        for i, v in enumerate(init_vars):
            init_loads.append(("const", tmp_init, i))
            init_loads.append(("load", self.scratch[v], tmp_init))

        # Override scratch_const to collect instead of emit
        _saved_add = self.add
        init_const_loads = []
        def collect_add(engine, slot):
            if engine == "load":
                init_const_loads.append(slot)
            else:
                _saved_add(engine, slot)
        self.add = collect_add
        
        # Pre-create ALL scalar constants during init
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        three_const = self.scratch_const(3)
        vlen_const = self.scratch_const(VLEN)

        # Hash scalar constants
        hash_info = []
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_c = (1 + (1 << val3)) % (2**32)
                sc_mul = self.scratch_const(mul_c)
                sc_add = self.scratch_const(val1)
                hash_info.append(("multiply_add", sc_mul, sc_add))
            else:
                sc_c1 = self.scratch_const(val1)
                sc_c3 = self.scratch_const(val3)
                hash_info.append(("3op", sc_c1, sc_c3))

        # Level-cache offset constants (0..6)
        for offset in range(7):
            self.scratch_const(offset)

        # Restore original add
        self.add = _saved_add
        
        # Save init loads to prepend to body after it's created
        _init_var_loads = []
        for i, v in enumerate(init_vars):
            _init_var_loads.append(("load", ("const", tmp_init, i)))
            _init_var_loads.append(("load", ("load", self.scratch[v], tmp_init)))
        _all_init_loads = _init_var_loads + [("load", cl) for cl in init_const_loads]

        # ================================================================
        # BODY SECTION - collected as list, scheduled by embedded scheduler
        # ================================================================
        body = []
        
        # Prepend init loads so scheduler manages them
        body.extend(_all_init_loads)

        # --- Vector constants ---
        v_zero = self.alloc_scratch("v_zero", VLEN)
        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        v_one = self.alloc_scratch("v_one", VLEN)
        body.append(("valu", ("vbroadcast", v_one, one_const)))
        v_two = self.alloc_scratch("v_two", VLEN)
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        v_three = self.alloc_scratch("v_three", VLEN)
        body.append(("valu", ("vbroadcast", v_three, three_const)))
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        body.append(("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"])))

        # --- Vector hash constants ---
        v_hash_consts = []
        for i, (stype, sc1, sc2) in enumerate(hash_info):
            vc1 = self.alloc_scratch(f"v_hc1_{i}", VLEN)
            vc2 = self.alloc_scratch(f"v_hc2_{i}", VLEN)
            body.append(("valu", ("vbroadcast", vc1, sc1)))
            body.append(("valu", ("vbroadcast", vc2, sc2)))
            v_hash_consts.append((stype, vc1, vc2))

        # --- Per-element vector scratch ---
        v_idx = [self.alloc_scratch(f"vi_{k}", VLEN) for k in range(U)]
        v_val = [self.alloc_scratch(f"vv_{k}", VLEN) for k in range(U)]
        v_nv  = [self.alloc_scratch(f"vn_{k}", VLEN) for k in range(U)]
        v_t1  = [self.alloc_scratch(f"vt1_{k}", VLEN) for k in range(U)]
        v_ad  = [self.alloc_scratch(f"va_{k}", VLEN) for k in range(U)]

        st = self.alloc_scratch("st")
        st2 = self.alloc_scratch("st2")
        v_offset = self.alloc_scratch("v_offset")

        # --- Load initial idx and val vectors ---
        # Compute idx and val addresses simultaneously using separate temporaries
        body.append(("load", ("const", v_offset, 0)))
        for k in range(U):
            body.append(("alu", ("+", st, self.scratch["inp_indices_p"], v_offset)))
            body.append(("alu", ("+", st2, self.scratch["inp_values_p"], v_offset)))
            body.append(("load", ("vload", v_idx[k], st)))
            body.append(("load", ("vload", v_val[k], st2)))
            if k < U - 1:
                body.append(("alu", ("+", v_offset, v_offset, vlen_const)))

        # --- Level cache: pre-load and broadcast levels 0, 1, 2 ---
        lv0 = self.alloc_scratch("lv0")
        vlv0 = self.alloc_scratch("vlv0", VLEN)
        lv_addr = [self.alloc_scratch(f"lv_addr_{i}") for i in range(7)]

        body.append(("alu", ("+", lv_addr[0], self.scratch["forest_values_p"], self.const_map[0])))
        for i in range(2):
            body.append(("alu", ("+", lv_addr[1+i], self.scratch["forest_values_p"], self.const_map[1 + i])))
        for i in range(4):
            body.append(("alu", ("+", lv_addr[3+i], self.scratch["forest_values_p"], self.const_map[3 + i])))

        body.append(("load", ("load", lv0, lv_addr[0])))

        lv1s = [self.alloc_scratch(f"lv1_{i}") for i in range(2)]
        vlv1 = [self.alloc_scratch(f"vlv1_{i}", VLEN) for i in range(2)]
        for i in range(2):
            body.append(("load", ("load", lv1s[i], lv_addr[1+i])))

        lv2s = [self.alloc_scratch(f"lv2s_{i}") for i in range(4)]
        vlv2 = [self.alloc_scratch(f"vlv2_{i}", VLEN) for i in range(4)]
        for i in range(4):
            body.append(("load", ("load", lv2s[i], lv_addr[3+i])))

        body.append(("valu", ("vbroadcast", vlv0, lv0)))
        for i in range(2):
            body.append(("valu", ("vbroadcast", vlv1[i], lv1s[i])))
        for i in range(4):
            body.append(("valu", ("vbroadcast", vlv2[i], lv2s[i])))

        # --- Main loop ---
        for r in range(rounds):
            level = r % TREE_DEPTH

            # STAGE 1: NODE VALUE LOOKUP
            if level == 0:
                for k in range(U):
                    body.append(("valu", ("+", v_nv[k], vlv0, v_zero)))
            elif level == 1:
                for k in range(U):
                    body.append(("valu", ("-", v_ad[k], v_idx[k], v_one)))
                for k in range(U):
                    body.append(("flow", ("vselect", v_nv[k], v_ad[k], vlv1[1], vlv1[0])))
            elif level == 2:
                for k in range(U):
                    body.append(("valu", ("-", v_ad[k], v_idx[k], v_three)))
                    body.append(("valu", ("&", v_t1[k], v_ad[k], v_one)))
                for k in range(U):
                    body.append(("flow", ("vselect", v_t1[k], v_t1[k], vlv2[1], vlv2[0])))
                for k in range(U):
                    body.append(("valu", ("&", v_nv[k], v_ad[k], v_one)))
                for k in range(U):
                    body.append(("flow", ("vselect", v_nv[k], v_nv[k], vlv2[3], vlv2[2])))
                for k in range(U):
                    # v_ad was 0-3, so >>1 yields 0 or 1; &1 is redundant
                    body.append(("valu", (">>", v_ad[k], v_ad[k], v_one)))
                for k in range(U):
                    body.append(("flow", ("vselect", v_nv[k], v_ad[k], v_nv[k], v_t1[k])))
            else:
                for k in range(U):
                    body.append(("valu", ("+", v_ad[k], v_forest_p, v_idx[k])))
                for k in range(U):
                    for lane in range(VLEN):
                        body.append(("load", ("load_offset", v_nv[k], v_ad[k], lane)))

            # STAGE 2: XOR via scalar ALU
            for k in range(U):
                for lane in range(VLEN):
                    body.append(("alu", ("^", v_val[k]+lane, v_val[k]+lane, v_nv[k]+lane)))

            # STAGE 3: HASH
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                stype, c1, c2 = v_hash_consts[hi]
                if stype == "multiply_add":
                    for k in range(U):
                        body.append(("valu", ("multiply_add", v_val[k], v_val[k], c1, c2)))
                else:
                    for k in range(U):
                        body.append(("valu", (op1, v_t1[k], v_val[k], c1)))
                        body.append(("valu", (op3, v_ad[k], v_val[k], c2)))
                        body.append(("valu", (op2, v_val[k], v_t1[k], v_ad[k])))

            # STAGE 4: IDX UPDATE
            if level + 1 >= TREE_DEPTH:
                for k in range(U):
                    body.append(("valu", ("+", v_idx[k], v_zero, v_zero)))
            else:
                for k in range(U):
                    body.append(("valu", ("multiply_add", v_idx[k], v_idx[k], v_two, v_one)))
                for k in range(U):
                    for lane in range(VLEN):
                        body.append(("alu", ("&", v_t1[k]+lane, v_val[k]+lane, v_one)))
                for k in range(U):
                    for lane in range(VLEN):
                        body.append(("alu", ("+", v_idx[k]+lane, v_idx[k]+lane, v_t1[k]+lane)))

        # --- Store results ---
        body.append(("load", ("const", v_offset, 0)))
        for k in range(U):
            body.append(("alu", ("+", st, self.scratch["inp_indices_p"], v_offset)))
            body.append(("store", ("vstore", st, v_idx[k])))
            body.append(("alu", ("+", st, self.scratch["inp_values_p"], v_offset)))
            body.append(("store", ("vstore", st, v_val[k])))
            if k < U - 1:
                body.append(("alu", ("+", v_offset, v_offset, vlen_const)))

        # ================================================================
        # EMBEDDED SCHEDULER - critical path priority, WAR-aware
        # ================================================================
        def get_rw(engine, slot):
            reads = set()
            writes = set()
            if engine == "alu":
                op, dest, a1, a2 = slot
                reads.add(a1); reads.add(a2); writes.add(dest)
            elif engine == "valu":
                op = slot[0]
                if op == "vbroadcast":
                    _, dest, src = slot
                    reads.add(src)
                    for vi in range(VLEN): writes.add(dest + vi)
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    for vi in range(VLEN):
                        reads.add(a+vi); reads.add(b+vi); reads.add(c+vi)
                        writes.add(dest+vi)
                else:
                    _, dest, a1, a2 = slot
                    for vi in range(VLEN):
                        reads.add(a1+vi); reads.add(a2+vi)
                        writes.add(dest+vi)
            elif engine == "load":
                op = slot[0]
                if op == "load":
                    _, dest, addr = slot
                    reads.add(addr); writes.add(dest)
                elif op == "load_offset":
                    _, dest, addr_vec, offset = slot
                    reads.add(addr_vec + offset); writes.add(dest + offset)
                elif op == "vload":
                    _, dest, addr = slot
                    reads.add(addr)
                    for vi in range(VLEN): writes.add(dest + vi)
                elif op == "const":
                    _, dest, val = slot
                    writes.add(dest)
            elif engine == "store":
                op = slot[0]
                if op == "store":
                    _, addr, src = slot
                    reads.add(addr); reads.add(src)
                elif op == "vstore":
                    _, addr, src = slot
                    reads.add(addr)
                    for vi in range(VLEN): reads.add(src + vi)
            elif engine == "flow":
                op = slot[0]
                if op == "select":
                    _, dest, cond, a, b = slot
                    reads.add(cond); reads.add(a); reads.add(b); writes.add(dest)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    for vi in range(VLEN):
                        reads.add(cond+vi); reads.add(a+vi); reads.add(b+vi)
                        writes.add(dest+vi)
                elif op == "add_imm":
                    _, dest, a, imm = slot
                    reads.add(a); writes.add(dest)
            return reads, writes

        n = len(body)
        ops = []
        for i, (eng, slot) in enumerate(body):
            rd, wr = get_rw(eng, slot)
            ops.append((eng, slot, rd, wr))

        # Build dependency graph via per-register tracking
        predecessors = [dict() for _ in range(n)]
        successors = [dict() for _ in range(n)]

        reg_last_writer = {}
        reg_readers_since = defaultdict(list)

        for i in range(n):
            eng, slot, rd, wr = ops[i]

            for r in rd:
                w = reg_last_writer.get(r)
                if w is not None:
                    predecessors[i][w] = 'raw'
                    successors[w][i] = 'raw'

            for r in wr:
                w = reg_last_writer.get(r)
                if w is not None and w not in predecessors[i]:
                    predecessors[i][w] = 'raw'
                    successors[w][i] = 'raw'
                for reader in reg_readers_since.get(r, []):
                    if reader != i and reader not in predecessors[i]:
                        predecessors[i][reader] = 'war'
                        if i not in successors[reader]:
                            successors[reader][i] = 'war'

            for r in wr:
                reg_last_writer[r] = i
                reg_readers_since[r] = []
            for r in rd:
                reg_readers_since[r].append(i)

        # Topological sort
        in_deg = [len(predecessors[i]) for i in range(n)]
        ready = [i for i in range(n) if in_deg[i] == 0]
        topo = []
        while ready:
            node = ready.pop()
            topo.append(node)
            for succ in successors[node]:
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    ready.append(succ)

        # Critical path (backward)
        crit = [0] * n
        for node in reversed(topo):
            mx = 0
            for succ in successors[node]:
                c = crit[succ] + 1
                if c > mx: mx = c
            crit[node] = mx

        sorted_ops = sorted(range(n), key=lambda i: (-crit[i], i))

        schedule = defaultdict(list)
        res_usage = defaultdict(lambda: defaultdict(int))
        reg_avail = defaultdict(int)
        reg_last_rd = defaultdict(int)
        reg_last_wr = {}
        op_time = {}

        slot_limits = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

        for idx in sorted_ops:
            eng, slot, rd, wr = ops[idx]
            t_min = 0
            for pred, dtype in predecessors[idx].items():
                pt = op_time[pred]
                if dtype == 'war':
                    if pt > t_min: t_min = pt
                else:
                    if pt + 1 > t_min: t_min = pt + 1
            for r in rd:
                ra = reg_avail.get(r, 0)
                if ra > t_min: t_min = ra
            for r in wr:
                rl = reg_last_rd.get(r, 0)
                if rl > t_min: t_min = rl
            for r in wr:
                lw = reg_last_wr.get(r)
                if lw is not None and lw + 1 > t_min:
                    t_min = lw + 1
            t = t_min
            while res_usage[t][eng] >= slot_limits[eng]:
                t += 1
            schedule[t].append((eng, slot))
            res_usage[t][eng] += 1
            op_time[idx] = t
            for r in wr:
                reg_avail[r] = t + 1
                reg_last_wr[r] = t
            for r in rd:
                cur = reg_last_rd.get(r, 0)
                if t > cur: reg_last_rd[r] = t

        if schedule:
            max_t = max(schedule.keys())
            for t in range(max_t + 1):
                bundle = {}
                for eng, slot in schedule.get(t, []):
                    if eng not in bundle:
                        bundle[eng] = []
                    bundle[eng].append(slot)
                self.instrs.append(bundle)

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
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

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
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
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
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()