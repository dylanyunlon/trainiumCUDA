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
PHASE: 0 - SCALAR BASELINE
CYCLES: 147734
TARGET: <1487 cycles (100x speedup needed)

FAILED_ATTEMPTS:
- VLIW packing alone FAILED: dependencies between alu->load prevent parallel exec
- Example: alu(tmp_addr=p+i) then load(x=mem[tmp_addr]) - load reads OLD tmp_addr

ARCHITECTURE_KEY_FACTS:
- SLOT_LIMITS: alu=12, valu=6, load=2, store=2, flow=1 per cycle
- VLEN=8 (vector length)
- VLIW: all reads happen BEFORE all writes in same cycle

NEXT_STEP: VECTORIZATION (Phase 1)
- Replace scalar loop with vector loop processing 8 elements at once
- 16 rounds * 32 vector_iters = 512 total vector iterations
- Expected cycles: ~2048 (then apply VLIW packing for further gains)

VECTORIZATION_TRAPS:
- TRAP1: vbroadcast(dest, src) - src must be SCALAR address
- TRAP2: load_offset(dest, addr, j) => scratch[dest+j] = mem[scratch[addr+j]]
- TRAP3: vload/vstore addr parameter is SCALAR (base address)
- TRAP4: Hash constants need vbroadcast to vector first
- TRAP5: For gather, use 8x load_offset, NOT vload (vload is contiguous only)

VECTOR_INSTRUCTION_REFERENCE:
- ('valu', ('vbroadcast', vec_dest, scalar_src)) - broadcast scalar to vector
- ('valu', (op, vec_dest, vec_a, vec_b)) - element-wise vector op
- ('load', ('vload', vec_dest, scalar_addr)) - load 8 contiguous elements
- ('load', ('load_offset', dest, addr, j)) - dest[j] = mem[addr[j]] (for gather)
- ('store', ('vstore', scalar_addr, vec_src)) - store 8 contiguous elements
- ('flow', ('vselect', vec_dest, vec_cond, vec_a, vec_b)) - vector select
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
    """
    GANNINA_STATE: PHASE 0 - SCALAR BASELINE (147734 cycles)
    GANNINA_NEXT: Implement vectorization in build_kernel
    GANNINA_TRAPS:
    - vbroadcast needs SCALAR src address
    - load_offset for gather: dest[j]=mem[addr[j]]
    - vload/vstore addr is SCALAR base
    - Hash constants need vbroadcast first
    """
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

    def build_vector_hash(self, vec_val, vec_tmp1, vec_tmp2, vec_hash_consts, round, vi):
        """Vector hash: 6 stages, each stage needs 3 valu ops"""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # vec_tmp1 = vec_val op1 const1
            # vec_tmp2 = vec_val op3 const3  
            # vec_val = vec_tmp1 op2 vec_tmp2
            const1_vec = vec_hash_consts[hi * 2]
            const3_vec = vec_hash_consts[hi * 2 + 1]
            slots.append(("valu", (op1, vec_tmp1, vec_val, const1_vec)))
            slots.append(("valu", (op3, vec_tmp2, vec_val, const3_vec)))
            slots.append(("valu", (op2, vec_val, vec_tmp1, vec_tmp2)))
            # Debug: compare each lane
            slots.append(("debug", ("vcompare", vec_val, tuple((round, vi + j, "hash_stage", hi) for j in range(VLEN)))))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        GANNINA: PHASE 1 - VECTORIZED implementation
        Process 8 elements at a time using SIMD instructions.
        """
        # Scalar temporaries for address computation
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp_addr = self.alloc_scratch("tmp_addr")
        
        # Load init vars from memory header
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_const = self.scratch_const(0, "zero")
        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")

        # ===== VECTOR SCRATCH ALLOCATION (VLEN=8) =====
        vec_idx = self.alloc_scratch("vec_idx", VLEN)
        vec_val = self.alloc_scratch("vec_val", VLEN)
        vec_node = self.alloc_scratch("vec_node", VLEN)
        vec_addr = self.alloc_scratch("vec_addr", VLEN)
        vec_tmp1 = self.alloc_scratch("vec_tmp1", VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", VLEN)
        vec_tmp3 = self.alloc_scratch("vec_tmp3", VLEN)
        
        # Vector constants (need to broadcast scalar to vector)
        vec_zero = self.alloc_scratch("vec_zero", VLEN)
        vec_one = self.alloc_scratch("vec_one", VLEN)
        vec_two = self.alloc_scratch("vec_two", VLEN)
        vec_forest_p = self.alloc_scratch("vec_forest_p", VLEN)
        vec_n_nodes = self.alloc_scratch("vec_n_nodes", VLEN)
        
        # Broadcast scalar constants to vectors
        self.add("valu", ("vbroadcast", vec_zero, zero_const))
        self.add("valu", ("vbroadcast", vec_one, one_const))
        self.add("valu", ("vbroadcast", vec_two, two_const))
        self.add("valu", ("vbroadcast", vec_forest_p, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", vec_n_nodes, self.scratch["n_nodes"]))
        
        # Hash constants - need vector versions
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

        body = []  # array of slots

        for round in range(rounds):
            for vi in range(0, batch_size, VLEN):
                # Compute addresses for this vector chunk
                vi_const = self.scratch_const(vi, f"vi_{vi}")
                
                # tmp_addr = inp_indices_p + vi (scalar addr for vload)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], vi_const)))
                # vload vec_idx from mem[inp_indices_p + vi : +8]
                body.append(("load", ("vload", vec_idx, tmp_addr)))
                body.append(("debug", ("vcompare", vec_idx, tuple((round, vi + j, "idx") for j in range(VLEN)))))
                
                # tmp_addr = inp_values_p + vi
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], vi_const)))
                # vload vec_val from mem[inp_values_p + vi : +8]
                body.append(("load", ("vload", vec_val, tmp_addr)))
                body.append(("debug", ("vcompare", vec_val, tuple((round, vi + j, "val") for j in range(VLEN)))))
                
                # vec_addr = forest_values_p + vec_idx (element-wise)
                body.append(("valu", ("+", vec_addr, vec_forest_p, vec_idx)))
                
                # GATHER: load node values from non-contiguous addresses
                # load_offset(dest, addr, j) => scratch[dest+j] = mem[scratch[addr+j]]
                for j in range(VLEN):
                    body.append(("load", ("load_offset", vec_node, vec_addr, j)))
                body.append(("debug", ("vcompare", vec_node, tuple((round, vi + j, "node_val") for j in range(VLEN)))))
                
                # vec_val = vec_val ^ vec_node
                body.append(("valu", ("^", vec_val, vec_val, vec_node)))
                
                # Hash: 6 stages
                body.extend(self.build_vector_hash(vec_val, vec_tmp1, vec_tmp2, vec_hash_consts, round, vi))
                body.append(("debug", ("vcompare", vec_val, tuple((round, vi + j, "hashed_val") for j in range(VLEN)))))
                
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", vec_tmp1, vec_val, vec_two)))
                body.append(("valu", ("==", vec_tmp1, vec_tmp1, vec_zero)))
                body.append(("flow", ("vselect", vec_tmp3, vec_tmp1, vec_one, vec_two)))
                body.append(("valu", ("*", vec_idx, vec_idx, vec_two)))
                body.append(("valu", ("+", vec_idx, vec_idx, vec_tmp3)))
                body.append(("debug", ("vcompare", vec_idx, tuple((round, vi + j, "next_idx") for j in range(VLEN)))))
                
                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", vec_tmp1, vec_idx, vec_n_nodes)))
                body.append(("flow", ("vselect", vec_idx, vec_tmp1, vec_idx, vec_zero)))
                body.append(("debug", ("vcompare", vec_idx, tuple((round, vi + j, "wrapped_idx") for j in range(VLEN)))))
                
                # Store results back
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], vi_const)))
                body.append(("store", ("vstore", tmp_addr, vec_idx)))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], vi_const)))
                body.append(("store", ("vstore", tmp_addr, vec_val)))

        # Collect slot stats
        engine_counts = defaultdict(int)
        for e, s in body:
            engine_counts[e] += 1

        # ========== STRATEGIC PRINT FOR PHASE 1 gannina ==========
        total_instrs = sum(v for k,v in engine_counts.items() if k != 'debug')
        vec_iters = rounds * (batch_size // VLEN)
        print(f"\n[GANNINA_PHASE1_VECTOR] rounds={rounds} batch={batch_size} vec_iters={vec_iters} gannina")
        print(f"  VECTOR_INSTRS: valu={engine_counts['valu']} alu={engine_counts['alu']} load={engine_counts['load']} store={engine_counts['store']} flow={engine_counts['flow']} gannina")
        print(f"  TOTAL_INSTRS: {total_instrs} (was 147456 scalar) gannina")
        print(f"  PER_VEC_ITER: valu={engine_counts['valu']//vec_iters} alu={engine_counts['alu']//vec_iters} load={engine_counts['load']//vec_iters} store={engine_counts['store']//vec_iters} flow={engine_counts['flow']//vec_iters} gannina")
        print(f"  NEXT: Apply VLIW packing to reduce cycles further gannina\n")

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