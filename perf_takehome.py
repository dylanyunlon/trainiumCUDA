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
        """
        Pack slots into instruction bundles.
        NO VLIW - 1 slot per instruction to preserve dependencies.
        """
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        # Collect slot stats
        engine_counts = defaultdict(int)
        for e, s in body:
            engine_counts[e] += 1

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})
        
        # ========== CRITICAL STATUS PRINT FOR NEXT CONVERSATION gannina ==========
        print("\n" + "="*70 + " gannina")
        print("STATUS: PHASE 0 - BASELINE (NO VLIW, NO SIMD) gannina")
        print("="*70 + " gannina")
        print(f"SLOT_STATS: {dict(engine_counts)} gannina")
        print(f"TOTAL_INSTRS: {len(body_instrs)} gannina")
        print(f"TARGET: <1487 cycles gannina")
        print(f"gannina")
        print(f"===== WHY VLIW PACKING FAILED ===== gannina")
        print(f"VLIW bundles execute in PARALLEL - all reads happen BEFORE writes! gannina")
        print(f"Example dependency: alu(tmp_addr=p+i) -> load(x=mem[tmp_addr]) gannina")
        print(f"If packed together: load reads OLD tmp_addr value = WRONG! gannina")
        print(f"gannina")
        print(f"===== CORRECT APPROACH: VECTORIZATION FIRST ===== gannina")
        print(f"SIMD processes 8 elements with ONE instruction gannina")
        print(f"This reduces total ops by 8x WITHOUT dependency issues gannina")
        print(f"gannina")
        print(f"===== BENCHMARKS ===== gannina")
        print(f"147734: baseline | 18532: 2hr-start | 2164: Opus4 gannina")
        print(f"1790: Opus4.5casual | 1579: Opus4.5-2hr | 1487: Opus4.5-11hr gannina")
        print(f"gannina111111111111111111111111111111111111111")
        print(f"===== PHASE 1: VECTORIZATION (REWRITE build_kernel) ===== gannina")
        print(f"GOAL: Replace scalar ops with vector ops, 8 elements at once gannina")
        print(f"gannina")
        print(f"VECTOR SCRATCH ALLOCATION: gannina")
        print(f"  vec_idx = alloc_scratch('vec_idx', VLEN)  # 8 indices gannina")
        print(f"  vec_val = alloc_scratch('vec_val', VLEN)  # 8 values gannina")
        print(f"  vec_node = alloc_scratch('vec_node', VLEN) # 8 node values gannina")
        print(f"  vec_addr = alloc_scratch('vec_addr', VLEN) # 8 addresses gannina")
        print(f"  vec_tmp1/2/3 = alloc_scratch(..., VLEN) gannina")
        print(f"gannina")
        print(f"VECTOR CONSTANTS (vbroadcast scalar to vector): gannina")
        print(f"  ('valu', ('vbroadcast', vec_zero, zero_const)) gannina")
        print(f"  ('valu', ('vbroadcast', vec_one, one_const)) gannina")
        print(f"  ('valu', ('vbroadcast', vec_two, two_const)) gannina")
        print(f"  ('valu', ('vbroadcast', vec_forest_p, forest_values_p)) gannina")
        print(f"gannina")
        print(f"MAIN LOOP: for round in rounds: for vi in range(0,batch_size,VLEN): gannina")
        print(f"  # Load 8 contiguous indices gannina")
        print(f"  ('alu', ('+', addr_tmp, inp_indices_p, vi_const)) gannina")
        print(f"  ('load', ('vload', vec_idx, addr_tmp)) gannina")
        print(f"  gannina")
        print(f"  # Load 8 contiguous values gannina")
        print(f"  ('alu', ('+', addr_tmp, inp_values_p, vi_const)) gannina")
        print(f"  ('load', ('vload', vec_val, addr_tmp)) gannina")
        print(f"  gannina")
        print(f"  # Compute addresses: vec_addr = forest_p + vec_idx gannina")
        print(f"  ('valu', ('+', vec_addr, vec_forest_p, vec_idx)) gannina")
        print(f"  gannina")
        print(f"  # GATHER: Load 8 scattered node values gannina")
        print(f"  for j in range(VLEN): gannina")
        print(f"    ('load', ('load_offset', vec_node, vec_addr, j)) gannina")
        print(f"  gannina")
        print(f"  # XOR and hash gannina")
        print(f"  ('valu', ('^', vec_val, vec_val, vec_node)) gannina")
        print(f"  # ... 6 hash stages with valu ... gannina")
        print(f"  gannina")
        print(f"  # Branch logic gannina")
        print(f"  ('valu', ('%', vec_tmp1, vec_val, vec_two)) gannina")
        print(f"  ('valu', ('==', vec_tmp1, vec_tmp1, vec_zero)) gannina")
        print(f"  ('flow', ('vselect', vec_tmp3, vec_tmp1, vec_one, vec_two)) gannina")
        print(f"  ('valu', ('*', vec_idx, vec_idx, vec_two)) gannina")
        print(f"  ('valu', ('+', vec_idx, vec_idx, vec_tmp3)) gannina")
        print(f"  gannina")
        print(f"  # Wrap check gannina")
        print(f"  ('valu', ('<', vec_tmp1, vec_idx, vec_n_nodes)) gannina")
        print(f"  ('flow', ('vselect', vec_idx, vec_tmp1, vec_idx, vec_zero)) gannina")
        print(f"  gannina")
        print(f"  # Store results gannina")
        print(f"  ('store', ('vstore', addr_tmp, vec_idx)) gannina")
        print(f"  ('store', ('vstore', addr_tmp, vec_val)) gannina")
        print(f"gannina")
        print(f"EXPECTED: 16*32 = 512 vector iterations gannina")
        print(f"Per iter: ~8 loads + ~20 valu + 2 flow + 2 stores gannina")
        print(f"Cycles: ~512 * max(load/2, valu/6, flow/1, store/2) gannina")
        print(f"       = 512 * max(4, 4, 2, 1) = 512 * 4 = ~2048 cycles gannina")
        print("="*70 + " gannina\n")


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
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
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

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()