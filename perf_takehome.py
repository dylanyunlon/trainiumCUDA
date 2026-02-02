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
        print(f"\n=== BUILD METHOD ANALYSIS gannina ===")
        print(f"Total slots to pack: {len(slots)} gannina")
        print(f"Current strategy: 1 slot per instruction (no parallelism) gannina")
        print(f"VLIW flag: {vliw} (currently ignored) gannina")
        
        # gannina: Analyze slot distribution
        from collections import Counter
        slot_types = Counter(engine for engine, _ in slots)
        print(f"Slot type counts: {dict(slot_types)} gannina")
        
        # gannina: Calculate optimal packing
        print(f"\n=== OPTIMAL PACKING CALCULATION gannina ===")
        for engine, count in slot_types.items():
            if engine != 'debug':
                max_slots = SLOT_LIMITS[engine]
                min_instrs = (count + max_slots - 1) // max_slots
                print(f"{engine}: {count} ops / {max_slots} slots = {min_instrs} instructions minimum gannina")
        
        print(f"Current output: {len(slots)} instructions (1 op each) gannina")
        max_needed = max((count + SLOT_LIMITS.get(engine, 1) - 1) // SLOT_LIMITS.get(engine, 1) 
                        for engine, count in slot_types.items() if engine != 'debug')
        print(f"Theoretical minimum with perfect packing: {max_needed} instructions gannina")
        print(f"Potential speedup from packing: {len(slots) / max_needed:.2f}x gannina")
        print("gannina build analysis end =========================\n")
        
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
        init_instr_count = len(self.instrs)  # gannina: track init instructions
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

        print(f"Init phase: {len(self.instrs) - init_instr_count} instructions, scratch_ptr={self.scratch_ptr} gannina")

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # gannina: Constant allocation analysis
        print(f"\n=== CONSTANT ALLOCATION gannina11111111111111 ===")
        print(f"Constants allocated so far: {len(self.const_map)} gannina1")
        print(f"Will allocate {batch_size} more constants for i values (0 to {batch_size-1}) gannina1")
        print(f"Total constants expected: {len(self.const_map) + batch_size} gannina1")
        print(f"Scratch space after constants: {self.scratch_ptr + batch_size}/{SCRATCH_SIZE} gannina1")
        print(f"Alternative: Could compute addresses dynamically to save constants gannina1")
        print("gannina constant analysis end =========================\n")

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        print(f"Loop structure: rounds={rounds}, batch_size={batch_size}, total_iters={rounds * batch_size} gannina")
        print(f"Hash stages: {len(HASH_STAGES)}, instructions per hash: {len(HASH_STAGES) * 3} gannina")

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                # print(batch_size, "gannina") 256
                i_const = self.scratch_const(i)
                
                # gannina: Track operations for first iteration only to understand pattern
                if round == 0 and i == 0:
                    print("\n=== FIRST ITERATION OPERATION SEQUENCE gannina ===")
                    op_count = 0
                
                # idx = mem[inp_indices_p + i]
                # print(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)), "gannina")
# ('alu', ('+', 16, 8, 72)) gannina
# ('alu', ('+', 16, 8, 73)) gannina
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Compute addr for idx load gannina"); op_count += 1
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Load idx from memory gannina"); op_count += 1
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Compute addr for val load gannina"); op_count += 1
                body.append(("load", ("load", tmp_val, tmp_addr)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Load val from memory gannina"); op_count += 1
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Compute addr for node_val load gannina"); op_count += 1
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Load node_val from tree gannina"); op_count += 1
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: XOR val with node_val gannina"); op_count += 1
                    print(f"Op {op_count}-{op_count+17}: Hash computation (18 ops) gannina"); op_count += 18
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Modulo to check parity gannina"); op_count += 1
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Compare parity gannina"); op_count += 1
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Select branch offset gannina"); op_count += 1
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Multiply idx by 2 gannina"); op_count += 1
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Add branch offset gannina"); op_count += 1
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Check bounds gannina"); op_count += 1
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Wrap idx if out of bounds gannina"); op_count += 1
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Compute addr for idx store gannina"); op_count += 1
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Store idx to memory gannina"); op_count += 1
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Compute addr for val store gannina"); op_count += 1
                body.append(("store", ("store", tmp_addr, tmp_val)))
                if round == 0 and i == 0:
                    print(f"Op {op_count}: Store val to memory gannina"); op_count += 1
                    print(f"=== Total ops for 1 iteration: {op_count} gannina ===\n")

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})
        
        # gannina: count instruction types
        from collections import Counter
        instr_types = Counter()
        for instr in body_instrs:
            for engine in instr:
                instr_types[engine] += len(instr[engine])
        print(instr_types, "gannina222222222222222222222222222222 count-----")
        
        print("\n=== Kernel Statistics ===")
        print(f"Parameters: height={forest_height}, nodes={n_nodes}, batch={batch_size}, rounds={rounds}")
        print(f"Total iterations: {rounds * batch_size}")
        print(f"Body instructions: {len(body_instrs)}")
        print(f"Instructions per iteration: {len(body_instrs) / (rounds * batch_size):.2f}")
        print(f"Total instructions: {len(self.instrs)}")
        print(f"Constants allocated: {len(self.const_map)} gannina")
        print(f"Scratch space used: {self.scratch_ptr}/{SCRATCH_SIZE} gannina")
        
        # gannina: Key bottleneck analysis
        print(f"\n=== BOTTLENECK ANALYSIS gannina33333333333333333333 ===")
        print(f"ALU ops per iter: {instr_types['alu'] / (rounds * batch_size):.2f}")
        print(f"Load ops per iter: {instr_types['load'] / (rounds * batch_size):.2f}")
        print(f"Store ops per iter: {instr_types['store'] / (rounds * batch_size):.2f}")
        print(f"Flow ops per iter: {instr_types['flow'] / (rounds * batch_size):.2f}")
        print(f"Debug ops per iter: {instr_types['debug'] / (rounds * batch_size):.2f}")
        print(f"Total non-debug per iter: {(instr_types['alu'] + instr_types['load'] + instr_types['store'] + instr_types['flow']) / (rounds * batch_size):.2f}")
        
        # gannina: Hash computation cost
        hash_cost = len(HASH_STAGES) * 3  # 3 ALU ops per stage
        print(f"\nHash cost per iter: {hash_cost} ALU ops (6 stages * 3 ops) gannina")
        print(f"Non-hash ALU per iter: {(instr_types['alu'] - hash_cost * rounds * batch_size) / (rounds * batch_size):.2f} gannina")
        
        # gannina: Memory access pattern
        print(f"\nMemory ops per iter: load={instr_types['load'] / (rounds * batch_size):.2f}, store={instr_types['store'] / (rounds * batch_size):.2f} gannina")
        
        # gannina: Slot utilization (theoretical max)
        print(f"\n=== SLOT UTILIZATION gannina444444444444444444444 ===")
        for engine, count in instr_types.items():
            if engine != 'debug':
                max_slots = SLOT_LIMITS[engine]
                util_pct = (count / len(body_instrs) / max_slots) * 100
                print(f"{engine}: {count / len(body_instrs):.2f} avg/instr (max {max_slots}, util {util_pct:.1f}%) gannina")
        
        print("gannina end =========================\n")
        
        # gannina: VLIW packing opportunity analysis
        print("=== VLIW PACKING OPPORTUNITIES gannina5555555555555555555555555 ===")
        print(f"Current: 1 op per instruction bundle (no parallel execution) gannina")
        print(f"ALU slots available: {SLOT_LIMITS['alu']} per cycle gannina")
        print(f"Load slots available: {SLOT_LIMITS['load']} per cycle gannina")
        print(f"Store slots available: {SLOT_LIMITS['store']} per cycle gannina")
        print(f"Per iteration: ALU=29, Load=3, Store=2 operations gannina")
        print(f"Theoretical min cycles/iter if perfect packing: max(29/{SLOT_LIMITS['alu']}, 3/{SLOT_LIMITS['load']}, 2/{SLOT_LIMITS['store']}) = {max(29/SLOT_LIMITS['alu'], 3/SLOT_LIMITS['load'], 2/SLOT_LIMITS['store']):.2f} gannina")
        print(f"Current cycles/iter: {len(body_instrs) / (rounds * batch_size):.2f} (no parallelism) gannina")
        
        # gannina: Loop structure analysis
        print("\n=== LOOP UNROLLING ANALYSIS gannina666666666666666666666 ===")
        print(f"Current: {rounds} rounds x {batch_size} batch = {rounds * batch_size} total iterations gannina")
        print(f"Each iteration is independent within a round gannina")
        print(f"Unroll opportunity: Can process multiple i values in parallel per round gannina")
        print(f"Memory pattern: Sequential access to inp_indices[i], inp_values[i] gannina")
        print(f"Vectorization potential: VLEN={VLEN}, could process {VLEN} items at once gannina")
        
        # gannina: Address computation pattern
        print("\n=== ADDRESS COMPUTATION PATTERN gannina77777777777777777777777777 ===")
        print(f"Per iteration: 5 address computations (tmp_addr = base + i_const) gannina")
        print(f"Pattern: inp_indices_p+i, inp_values_p+i, forest_values_p+idx, inp_indices_p+i, inp_values_p+i gannina")
        print(f"Optimization: Could precompute or use auto-increment addressing gannina")
        
        # gannina: Hash computation breakdown
        print("\n=== HASH COMPUTATION DETAILS gannina88888888888888888888888 ===")
        print(f"Hash: 6 stages x 3 ops = 18 ALU ops per iteration gannina")
        print(f"Each stage: op1(tmp1), op3(tmp2), op2(result) gannina")
        print(f"Hash operations use 2 scratch temps: tmp1, tmp2 gannina")
        print(f"Hash input: tmp_val (after XOR with node_val) gannina")
        
        # gannina: Critical path dependencies
        print("\n=== DEPENDENCY CHAIN gannina99999999999999999999999999 ===")
        print("1. Load idx -> Load node_val (data dependency) gannina")
        print("2. Load val -> XOR -> Hash -> Store val (data dependency) gannina")
        print("3. Hash -> Modulo -> Compare -> Branch -> idx*2 -> Add -> Bounds -> Store idx (data dependency) gannina")
        print("Key insight: Hash computation blocks everything else gannina")
        
        # gannina: Next steps to try
        print("\n=== NEXT OPTIMIZATION STEPS gannina101010101010101010101010 ===")
        print("STEP 1: Implement VLIW packing in build() method (pack multiple slots per instruction) gannina")
        print("STEP 2: Vectorize memory loads/stores using vload/vstore for batches gannina")
        print("STEP 3: Loop unroll: process multiple i values together gannina")
        print("STEP 4: Reduce address computations: precompute or use base+offset patterns gannina")
        print("STEP 5: Check if hash can be optimized or vectorized gannina")
        print("gannina optimization guide end =========================\n")

        


        print("\n=== CRITICAL NEXT STEPS TRACKER gannina!!!!!!!!!!!!!!!!!!!!!!!!! ===")
        print("STEP 1 [NOT DONE]: Implement VLIW packing in build() method gannina")
        print("  - Goal: Pack multiple slots per instruction bundle gannina")
        print("  - Expected: 19.86x speedup (196608 -> 9899 instrs) gannina")
        print("  - Key: Use SLOT_LIMITS={'alu':12, 'valu':6, 'load':2, 'store':2, 'flow':1} gannina")
        print("STEP 2 [NOT DONE]: Vectorize with vload/vstore gannina")
        print("  - Replace scalar loops with VLEN=8 vector operations gannina")
        print("  - Unroll batch loop: process 8 items at once gannina")
        print("STEP 3 [NOT DONE]: Vectorize hash computation gannina")
        print("  - Check if valu supports all hash ops (XOR, shifts, etc) gannina")
        print("  - Hash is 50% of ALU work (18/36 ops) gannina")
        print("gannina tracker end =========================\n")

        print("\n=== BUILD() METHOD MODIFICATION NEEDED gannina ===")
        print("Current build(): Creates 1 instruction per slot gannina")
        print("Required: Group slots by engine type, pack up to SLOT_LIMITS gannina")
        print("Algorithm: gannina")
        print("  1. Collect all slots, group by engine type gannina")
        print("  2. For each engine, chunk into SLOT_LIMITS[engine] sized groups gannina")
        print("  3. Merge chunks from different engines into single instructions gannina")
        print("  4. Handle dependencies: may need barriers between dependent ops gannina")
        print("Example: gannina")
        print("  {'alu': [op1, op2, ..., op12], 'load': [ld1, ld2]} = 1 instruction gannina")
        print("gannina build strategy end =========================\n")

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
    
    # gannina: Cycle breakdown analysis
    print(f"\n=== CYCLE BREAKDOWN gannina ===")
    total_iters = rounds * batch_size
    print(f"Total cycles: {machine.cycle}")
    print(f"Cycles per iteration: {machine.cycle / total_iters:.2f}")
    print(f"Target: <1487 cycles (Opus 4.5) = {1487 / total_iters:.2f} cycles/iter gannina")
    print(f"Baseline: {BASELINE} cycles = {BASELINE / total_iters:.2f} cycles/iter gannina")
    print(f"Speedup needed: {machine.cycle / 1487:.2f}x gannina")
    
    # gannina: Vectorization feasibility
    print(f"\n=== VECTORIZATION ANALYSIS gannina ===")
    print(f"VLEN = {VLEN}, batch_size = {batch_size} gannina")
    print(f"Can process {VLEN} items in parallel with vload/vstore gannina")
    print(f"Current: Scalar processing (1 item at a time) gannina")
    print(f"Vector potential: {batch_size / VLEN:.0f} vector iterations per round (vs {batch_size} scalar) gannina")
    print(f"Memory access pattern: Sequential (good for vectorization) gannina")
    print(f"Hash vectorization: Need to check if valu supports all hash ops gannina")
    
    # gannina: Next optimization targets
    print(f"\n=== OPTIMIZATION TARGETS gannina ===")
    print("1. VLIW parallelism: Pack multiple ops into same cycle (check SLOT_LIMITS) gannina")
    print("2. Address computation: Precompute i_const addresses or use vectorization gannina")
    print("3. Hash optimization: Hash is 18 ops - can we vectorize or reduce? gannina")
    print("4. Memory access: 3 loads + 2 stores per iter - can batch? gannina")
    print("5. Constant reuse: Check if constants are efficiently cached gannina")
    print(f"6. Current bottleneck: {machine.cycle / total_iters:.2f} cycles/iter with {len(kb.instrs)} total instructions gannina")
    print("7. CRITICAL: Try VLIW packing FIRST - easiest 36x potential improvement gannina")
    print("8. THEN: Try vectorization - 8x potential improvement for memory ops gannina")
    print("gannina analysis end =========================\n")
    
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