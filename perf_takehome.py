"""
GANNINA Architecture v2 - Wrap-Aware Level Cache
Key Insight: Tree traversal wraps every 10 rounds (tree height)
So round r accesses tree level (r % (height+1))
Caching levels 0-2 saves loads at rounds 0,1,2 AND 11,12,13!

Current: 1992 cycles
Target: <1363 cycles

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.
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

def get_rw(engine, slot):
    reads = set()
    writes = set()
    def add_read(r, length=1):
        for i in range(length):
            reads.add(r + i)
    def add_write(r, length=1):
        for i in range(length):
            writes.add(r + i)

    if engine == "alu":
        op, dest, a1, a2 = slot
        add_read(a1); add_read(a2); add_write(dest)
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            _, dest, src = slot
            add_read(src); add_write(dest, VLEN)
        elif op == "multiply_add":
            _, dest, a, b, c = slot
            add_read(a, VLEN); add_read(b, VLEN); add_read(c, VLEN); add_write(dest, VLEN)
        else:
            _, dest, a1, a2 = slot
            add_read(a1, VLEN); add_read(a2, VLEN); add_write(dest, VLEN)
    elif engine == "load":
        op = slot[0]
        if op == "load":
            _, dest, addr = slot
            add_read(addr); add_write(dest)
        elif op == "load_offset":
            _, dest, addr_vec, offset = slot
            add_read(addr_vec + offset); add_write(dest + offset)
        elif op == "vload":
            _, dest, addr = slot
            add_read(addr); add_write(dest, VLEN)
        elif op == "const":
            _, dest, val = slot
            add_write(dest)
    elif engine == "store":
        op = slot[0]
        if op == "store":
            _, addr, src = slot
            add_read(addr); add_read(src)
        elif op == "vstore":
            _, addr, src = slot
            add_read(addr); add_read(src, VLEN)
    elif engine == "flow":
        op = slot[0]
        if op == "select":
            _, dest, cond, a, b = slot
            add_read(cond); add_read(a); add_read(b); add_write(dest)
        elif op == "vselect":
            _, dest, cond, a, b = slot
            add_read(cond, VLEN); add_read(a, VLEN); add_read(b, VLEN); add_write(dest, VLEN)
        elif op == "add_imm":
            _, dest, a, imm = slot
            add_read(a); add_write(dest)
        elif op in ("halt", "pause"):
            pass
    return reads, writes


class Scheduler:
    def __init__(self):
        self.pending = []

    def add(self, engine, slot, tag=None):
        if engine == "debug":
            return
        reads, writes = get_rw(engine, slot)
        self.pending.append({'engine': engine, 'slot': slot, 'read': reads, 'write': writes, 'tag': tag})

    def flush(self):
        if not self.pending:
            return []
        
        n = len(self.pending)
        for i, op in enumerate(self.pending):
            op['id'] = i
        
        # Build dependency graph
        successors = [[] for _ in range(n)]
        predecessors = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                op_i = self.pending[i]
                op_j = self.pending[j]
                if op_i['write'] & op_j['read']:  # RAW
                    successors[i].append(j)
                    predecessors[j].append(i)
                elif op_i['write'] & op_j['write']:  # WAW
                    successors[i].append(j)
                    predecessors[j].append(i)
                elif op_i['read'] & op_j['write']:  # WAR
                    successors[i].append(j)
                    predecessors[j].append(i)
        
        # Compute critical path lengths (upward rank)
        critical_len = [0] * n
        in_degree = [len(predecessors[i]) for i in range(n)]
        ready = [i for i in range(n) if in_degree[i] == 0]
        topo_order = []
        while ready:
            node = ready.pop()
            topo_order.append(node)
            for succ in successors[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)
        
        for node in reversed(topo_order):
            max_succ = 0
            for succ in successors[node]:
                max_succ = max(max_succ, critical_len[succ] + 1)
            critical_len[node] = max_succ
        
        sorted_ops = sorted(range(n), key=lambda i: (-critical_len[i], i))
        
        schedule = defaultdict(list)
        resource_usage = defaultdict(lambda: defaultdict(int))
        reg_avail = defaultdict(int)
        reg_last_read = defaultdict(int)
        reg_last_write = defaultdict(int)
        op_scheduled_time = {}

        for op_idx in sorted_ops:
            op = self.pending[op_idx]
            t_min = 0
            for pred_idx in predecessors[op_idx]:
                if pred_idx in op_scheduled_time:
                    t_min = max(t_min, op_scheduled_time[pred_idx] + 1)
            for r in op['read']:
                t_min = max(t_min, reg_avail[r])
            for r in op['write']:
                t_min = max(t_min, reg_last_read[r])
            for r in op['write']:
                if r in reg_last_write:
                    t_min = max(t_min, reg_last_write[r] + 1)
            t = t_min
            while resource_usage[t][op['engine']] >= SLOT_LIMITS[op['engine']]:
                t += 1
            schedule[t].append(op)
            resource_usage[t][op['engine']] += 1
            op_scheduled_time[op_idx] = t
            for r in op['write']:
                reg_avail[r] = t + 1
                reg_last_write[r] = t
            for r in op['read']:
                reg_last_read[r] = max(reg_last_read[r], t)

        instrs = []
        if schedule:
            for t in range(max(schedule.keys()) + 1):
                bundle = {}
                for op in schedule[t]:
                    e = op['engine']
                    if e not in bundle:
                        bundle[e] = []
                    bundle[e].append(op['slot'])
                instrs.append(bundle)
        
        self.pending = []
        return instrs


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.scheduler = Scheduler()

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot, tag=None):
        self.scheduler.add(engine, slot, tag)

    def flush(self):
        self.instrs.extend(self.scheduler.flush())

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch: {self.scratch_ptr}/{SCRATCH_SIZE}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        U = 32
        TREE_DEPTH = forest_height + 1  # 11 for height=10

        # Init params
        tmp_init = self.alloc_scratch("tmp_init")
        init_vars = ["rounds","n_nodes","batch_size","forest_height",
                      "forest_values_p","inp_indices_p","inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init, i))
            self.add("load", ("load", self.scratch[v], tmp_init))

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        self.add("valu", ("vbroadcast", v_zero, self.scratch_const(0)))
        v_one = self.alloc_scratch("v_one", VLEN)
        self.add("valu", ("vbroadcast", v_one, self.scratch_const(1)))
        v_two = self.alloc_scratch("v_two", VLEN)
        self.add("valu", ("vbroadcast", v_two, self.scratch_const(2)))
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Hash constants
        v_hash_consts = []
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_const = (1 + (1 << val3)) % (2**32)
                v_mul = self.alloc_scratch(f"v_hmul_{i}", VLEN)
                v_add = self.alloc_scratch(f"v_hadd_{i}", VLEN)
                self.add("valu", ("vbroadcast", v_mul, self.scratch_const(mul_const)))
                self.add("valu", ("vbroadcast", v_add, self.scratch_const(val1)))
                v_hash_consts.append(("multiply_add", v_mul, v_add))
            else:
                v_c1 = self.alloc_scratch(f"v_hc1_{i}", VLEN)
                v_c3 = self.alloc_scratch(f"v_hc3_{i}", VLEN)
                self.add("valu", ("vbroadcast", v_c1, self.scratch_const(val1)))
                self.add("valu", ("vbroadcast", v_c3, self.scratch_const(val3)))
                v_hash_consts.append(("3op", v_c1, v_c3))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Per-element scratch: separate v_nv from v_t1 for mux operations
        v_idx = [self.alloc_scratch(f"vi_{k}", VLEN) for k in range(U)]
        v_val = [self.alloc_scratch(f"vv_{k}", VLEN) for k in range(U)]
        v_nv = [self.alloc_scratch(f"vn_{k}", VLEN) for k in range(U)]
        v_t1 = [self.alloc_scratch(f"vt1_{k}", VLEN) for k in range(U)]
        v_ad = [self.alloc_scratch(f"va_{k}", VLEN) for k in range(U)]
        
        st = self.alloc_scratch("st")

        # Dynamic offset for load/store
        v_offset = self.alloc_scratch("v_offset")
        self.add("load", ("const", v_offset, 0))
        v_vlen = self.scratch_const(VLEN)
        
        # Load initial data
        self.add("load", ("const", v_offset, 0))
        for k in range(U):
            self.add("alu", ("+", st, self.scratch["inp_indices_p"], v_offset))
            self.add("load", ("vload", v_idx[k], st))
            self.add("alu", ("+", st, self.scratch["inp_values_p"], v_offset))
            self.add("load", ("vload", v_val[k], st))
            if k < U - 1:
                self.add("alu", ("+", v_offset, v_offset, v_vlen))
        self.flush()

        # ===== LEVEL CACHE =====
        # Level 0: 1 node
        lv0 = self.alloc_scratch("lv0")
        self.add("alu", ("+", st, self.scratch["forest_values_p"], self.scratch_const(0)))
        self.add("load", ("load", lv0, st))
        vlv0 = self.alloc_scratch("vlv0", VLEN)
        self.add("valu", ("vbroadcast", vlv0, lv0))

        # Level 1: 2 nodes
        lv1s = [self.alloc_scratch(f"lv1_{i}") for i in range(2)]
        vlv1 = [self.alloc_scratch(f"vlv1_{i}", VLEN) for i in range(2)]
        for i in range(2):
            self.add("alu", ("+", st, self.scratch["forest_values_p"], self.scratch_const(1+i)))
            self.add("load", ("load", lv1s[i], st))
            self.add("valu", ("vbroadcast", vlv1[i], lv1s[i]))

        # Level 2: 4 nodes
        lv2s = [self.alloc_scratch(f"lv2s_{i}") for i in range(4)]
        vlv2 = [self.alloc_scratch(f"vlv2_{i}", VLEN) for i in range(4)]
        for i in range(4):
            self.add("alu", ("+", st, self.scratch["forest_values_p"], self.scratch_const(3+i)))
            self.add("load", ("load", lv2s[i], st))
            self.add("valu", ("vbroadcast", vlv2[i], lv2s[i]))

        v_three = self.alloc_scratch("v_three", VLEN)
        self.add("load", ("const", st, 3))
        self.add("valu", ("vbroadcast", v_three, st))

        # ===== GANNINA PHASE 90: SCRATCH BUDGET ===== gannina
        print(f"[GANNINA_PHASE90_SCRATCH] gannina")
        print(f"  SCRATCH: used={self.scratch_ptr}/{SCRATCH_SIZE}, free={SCRATCH_SIZE - self.scratch_ptr}")
        print(f"gannina")
        
        self.flush()

        # ===== GANNINA PHASE 91: WRAP-AWARE LEVEL CACHE WITH INTERLEAVED EMISSION =====
        # KEY OPTIMIZATION: Emit round r's gather alongside round r-1's hash/idx/wrap
        # This lets the scheduler overlap loads with independent compute
        
        for r in range(rounds):
            level = r % TREE_DEPTH
            
            # ===== STAGE 1: NODE VALUE LOOKUP =====
            if level == 0:
                for k in range(U):
                    self.add("valu", ("+", v_nv[k], vlv0, v_zero), tag=f"r{r}_gather")
            elif level == 1:
                for k in range(U):
                    self.add("valu", ("-", v_ad[k], v_idx[k], v_one), tag=f"r{r}_mux")
                for k in range(U):
                    self.add("flow", ("vselect", v_nv[k], v_ad[k], vlv1[1], vlv1[0]), tag=f"r{r}_mux")
            elif level == 2:
                for k in range(U):
                    self.add("valu", ("-", v_ad[k], v_idx[k], v_three), tag=f"r{r}_mux")
                    self.add("valu", ("&", v_t1[k], v_ad[k], v_one), tag=f"r{r}_mux")
                for k in range(U):
                    self.add("flow", ("vselect", v_t1[k], v_t1[k], vlv2[1], vlv2[0]), tag=f"r{r}_mux")
                for k in range(U):
                    self.add("valu", ("&", v_nv[k], v_ad[k], v_one), tag=f"r{r}_mux")
                for k in range(U):
                    self.add("flow", ("vselect", v_nv[k], v_nv[k], vlv2[3], vlv2[2]), tag=f"r{r}_mux")
                for k in range(U):
                    self.add("valu", (">>", v_ad[k], v_ad[k], v_one), tag=f"r{r}_mux")
                    self.add("valu", ("&", v_ad[k], v_ad[k], v_one), tag=f"r{r}_mux")
                for k in range(U):
                    self.add("flow", ("vselect", v_nv[k], v_ad[k], v_nv[k], v_t1[k]), tag=f"r{r}_mux")
            else:
                # Level 3+: gather - emit address calc for ALL elements first
                for k in range(U):
                    self.add("valu", ("+", v_ad[k], v_forest_p, v_idx[k]), tag=f"r{r}_gather")
                # Then emit loads interleaved with previous round's leftover work
                for k in range(U):
                    for lane in range(VLEN):
                        self.add("load", ("load_offset", v_nv[k], v_ad[k], lane), tag=f"r{r}_gather")
            
            # ===== STAGE 2: XOR (use ALU to free VALU) =====
            for k in range(U):
                for lane in range(VLEN):
                    self.add("alu", ("^", v_val[k]+lane, v_val[k]+lane, v_nv[k]+lane), tag=f"r{r}_xor")
            
            # ===== STAGE 3: HASH =====
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                stype, c1, c2 = v_hash_consts[hi]
                if stype == "multiply_add":
                    for k in range(U):
                        self.add("valu", ("multiply_add", v_val[k], v_val[k], c1, c2), tag=f"r{r}_hash{hi}")
                else:
                    for k in range(U):
                        self.add("valu", (op1, v_t1[k], v_val[k], c1), tag=f"r{r}_hash{hi}")
                        self.add("valu", (op3, v_ad[k], v_val[k], c2), tag=f"r{r}_hash{hi}")
                        self.add("valu", (op2, v_val[k], v_t1[k], v_ad[k]), tag=f"r{r}_hash{hi}")
            
            # ===== STAGE 4: IDX UPDATE =====
            if level + 1 >= TREE_DEPTH:
                # Wrap round: idx will be set to 0, so skip the expensive idx computation
                for k in range(U):
                    self.add("valu", ("+", v_idx[k], v_zero, v_zero), tag=f"r{r}_wrap")
            else:
                for k in range(U):
                    self.add("valu", ("multiply_add", v_idx[k], v_idx[k], v_two, v_one), tag=f"r{r}_idx")
                for k in range(U):
                    for lane in range(VLEN):
                        self.add("alu", ("&", v_t1[k]+lane, v_val[k]+lane, v_one), tag=f"r{r}_idx")
                for k in range(U):
                    self.add("valu", ("+", v_idx[k], v_idx[k], v_t1[k]), tag=f"r{r}_idx")

        # ===== GANNINA PHASE 92: OPERATION COUNT SUMMARY ===== gannina
        gather_rounds = sum(1 for r in range(rounds) if r % TREE_DEPTH >= 3)  
        cached_rounds = rounds - gather_rounds
        total_loads_expected = gather_rounds * U * VLEN
        print(f"[GANNINA_PHASE92_WRAPAWARE] gannina")
        print(f"  TREE_DEPTH: {TREE_DEPTH}")
        print(f"  CACHED_ROUNDS: {cached_rounds} (levels 0,1,2)")
        print(f"  GATHER_ROUNDS: {gather_rounds}")
        print(f"  EXPECTED_LOADS: {total_loads_expected}")
        print(f"  MIN_LOAD_CYCLES: {(total_loads_expected + 1) // 2}")
        print(f"gannina")

        # STORE
        self.add("load", ("const", v_offset, 0))
        for k in range(U):
            self.add("alu", ("+", st, self.scratch["inp_indices_p"], v_offset), tag="store")
            self.add("store", ("vstore", st, v_idx[k]), tag="store")
            self.add("alu", ("+", st, self.scratch["inp_values_p"], v_offset), tag="store")
            self.add("store", ("vstore", st, v_val[k]), tag="store")
            if k < U - 1:
                self.add("alu", ("+", v_offset, v_offset, v_vlen), tag="store")

        self.flush()
        self.add("flow", ("pause",))
        self.flush()

        # ===== GANNINA PHASE 93: FINAL DIAGNOSTICS ===== gannina
        total_valu = sum(len(b.get('valu', [])) for b in self.instrs)
        total_load = sum(len(b.get('load', [])) for b in self.instrs)
        total_flow = sum(len(b.get('flow', [])) for b in self.instrs)
        total_alu = sum(len(b.get('alu', [])) for b in self.instrs)
        n_cycles = len(self.instrs)
        
        min_valu = (total_valu + 5) // 6
        min_load = (total_load + 1) // 2
        min_flow = total_flow
        bottleneck = max(min_valu, min_load, min_flow)
        
        print(f"[GANNINA_PHASE93_FINAL] gannina")
        print(f"  SCHEDULED_CYCLES: {n_cycles}")
        print(f"  OPS: valu={total_valu} load={total_load} flow={total_flow} alu={total_alu}")
        print(f"  MIN: valu={min_valu} load={min_load} flow={min_flow}")
        print(f"  THEORETICAL_MIN: {bottleneck}")
        print(f"  OVERHEAD: {n_cycles - bottleneck} ({100*(n_cycles-bottleneck)/bottleneck:.1f}%)")
        print(f"gannina")


BASELINE = 147734

def do_kernel_test(forest_height, rounds, batch_size, seed=123, trace=False, prints=False):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
                      value_trace=value_trace, trace=trace)
    machine.prints = prints
    ref_gen = reference_kernel2(mem, value_trace)
    ref_mems = list(ref_gen)
    machine.run()
    ref_mem = ref_mems[-1]
    inp_values_p = ref_mem[6]
    assert (machine.mem[inp_values_p:inp_values_p+len(inp.values)]
            == ref_mem[inp_values_p:inp_values_p+len(inp.values)]), f"Wrong values"
    inp_indices_p = ref_mem[5]
    assert (machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)]
            == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]), f"Wrong indices"
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
            assert inp.indices == mem[mem[5]:mem[5]+len(inp.indices)]
            assert inp.values == mem[mem[6]:mem[6]+len(inp.values)]

    def test_kernel_cycles(self):
        print(f"Testing forest_height=10, rounds=16, batch_size=256")
        cycles = do_kernel_test(10, 16, 256)
        print(f"CYCLES:  {cycles}")
        speedup = BASELINE / cycles
        print(f"Speedup over baseline:  {speedup}")
        

if __name__ == "__main__":
    unittest.main()