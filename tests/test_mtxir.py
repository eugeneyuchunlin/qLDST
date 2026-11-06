"""Tests of the mtxir package: compiled LOAD/STORE circuits are valid Stim
circuits with deterministic detectors, and a chain of teleports carries the
logical qubit to its final location.

Run with: python -m pytest
"""

import numpy as np
import pytest
import stim
from galois import GF

from mtxir import (
    BivariateBicycleCode,
    CircuitCompiler,
    CodeBlock,
    CodeBlockType,
    Instruction,
    MtxIRCompiler,
    SurfaceCode,
)

DISTANCE = 3
ROUNDS = 2
LOGICAL = 6


def compile_program(basis, ops, p=0.0):
    """Compile q0 (starting in BB logical LOGICAL) through ops, a list of
    ("LOAD",) or ("STORE", slot); one instruction every 4 time steps."""
    bb = CodeBlock(BivariateBicycleCode.generate_code(12))
    sc = CodeBlock(SurfaceCode(DISTANCE))
    ir = MtxIRCompiler(initial_memory_code_blocks=[bb], initial_surface_code_blocks=[sc],
                       memory_basis=basis)
    ir.initialize("q0", True, 0, LOGICAL)
    instructions = []
    for k, op in enumerate(ops):
        t = 1 + 4 * k
        instructions.append(Instruction(t, "LOAD", "q0", "SC[0]") if op[0] == "LOAD"
                            else Instruction(t, "STORE", "q0", f"MEM[0][{op[1]}]"))
    ir.compile(instructions)
    circuit = CircuitCompiler([bb], [sc], rounds_per_step=ROUNDS, aux_basis=ir.aux_basis,
                              memory_basis=basis, readout_basis=basis, p=p).compile(ir.MtxIRs)
    return ir, bb, sc, circuit


def logical_value(circuit, columns, basis, shots=32):
    """Per shot, the final value of the basis logical on the given columns: its
    transversal readout times the earlier measurements that fix it (Stim flows)."""
    flat = circuit.flattened()
    last, first_instruction, n = {}, {}, 0
    for i, inst in enumerate(flat):
        if stim.gate_data(inst.name).produces_measurements:
            for target in inst.targets_copy():
                last[target.value] = n
                first_instruction.setdefault(target.value, i)
                n += 1
    # The readout of a column is its last measurement; solve the flow up to it.
    readout_start = max(i for i, inst in enumerate(flat)
                        if stim.gate_data(inst.name).produces_measurements
                        and any(t.value in columns for t in inst.targets_copy()))
    before = stim.Circuit()
    for inst in flat[:readout_start]:
        before.append(inst)
    output = stim.PauliString(circuit.num_qubits)
    for q in columns:
        output[q] = basis
    fixing = before.solve_flow_measurements([stim.Flow(output=output)])[0]
    samples = circuit.compile_sampler().sample(shots).astype(int)
    return (samples[:, fixing].sum(1) + samples[:, [last[q] for q in columns]].sum(1)) % 2


def test_public_api():
    import mtxir
    for name in mtxir.__all__:
        assert hasattr(mtxir, name)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_load_noisy_circuit_is_valid(basis):
    _, _, _, circuit = compile_program(basis, [("LOAD",)], p=1e-3)
    # Raises if any detector or observable is not deterministic.
    dem = circuit.detector_error_model()
    assert dem.num_detectors == circuit.num_detectors > 0
    assert circuit.num_observables > 0


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_noiseless_circuit_has_no_noise(basis):
    _, _, _, circuit = compile_program(basis, [("LOAD",)], p=0.0)
    noise = {"X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}
    assert not any(inst.name in noise for inst in circuit.flattened())
    detectors, observables = circuit.compile_detector_sampler().sample(
        16, separate_observables=True)
    assert not detectors.any() and not observables.any()


CHAINS = [
    [("LOAD",)],
    [("LOAD",), ("STORE", 6)],
    [("LOAD",), ("STORE", 2)],
    [("LOAD",), ("STORE", 2), ("LOAD",)],
]


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("ops", CHAINS, ids=lambda ops: "-".join(
    o[0] + (str(o[1]) if len(o) > 1 else "") for o in ops))
def test_teleport_chain_carries_the_logical(basis, ops):
    """Flipping logical LOGICAL at the start flips q0's final logical, and
    the final logical is deterministic with or without the flip."""
    ir, bb, _, circuit = compile_program(basis, ops)
    block = ir.locations["q0"]
    slot = block.locate("q0")
    to_global = block.local_to_global_col_mapping
    columns = [to_global[q] for q in ir._slot_logicals(block, slot)[basis]]

    # Flip with an error (not a gate, which the reference sample would absorb)
    # before the first TICK.
    flip_support = ir._gross_logicals(bb.code, LOGICAL)[1 if basis == "Z" else 0]
    bb_global = bb.local_to_global_col_mapping
    gate = "X_ERROR(1)" if basis == "Z" else "Z_ERROR(1)"
    tick = next(i for i, inst in enumerate(circuit) if inst.name == "TICK")
    flipped = (circuit[:tick]
               + stim.Circuit(f"{gate} " + " ".join(str(bb_global[q]) for q in flip_support))
               + circuit[tick:])

    v0 = logical_value(circuit, columns, basis)
    v1 = logical_value(flipped, columns, basis)
    assert np.all(v0 == v0[0]) and np.all(v1 == v1[0])
    assert np.all(v0 != v1)


def test_initialize_rejects_occupied_slot():
    bb = CodeBlock(BivariateBicycleCode.generate_code(12))
    sc = CodeBlock(SurfaceCode(DISTANCE))
    ir = MtxIRCompiler(initial_memory_code_blocks=[bb], initial_surface_code_blocks=[sc])
    ir.initialize("q0", True, 0, LOGICAL)
    with pytest.raises(ValueError):
        ir.initialize("q1", True, 0, LOGICAL)


def test_instruction_parses_destination():
    load = Instruction(1, "LOAD", "q0", "SC[0]")
    store = Instruction(5, "STORE", "q0", "MEM[0][3]")
    assert (load.block_index, load.slot_index) == (0, None)
    assert (store.block_index, store.slot_index) == (0, 3)
    with pytest.raises(ValueError):
        Instruction(1, "LOAD", "q0", "MEM[0][3]")


# Parallel LOAD/STORE on one memory block
# ---------------------------------------

# Slot pairs: fully disjoint, and with no shared checks; disjoint, sharing
# checks; sharing one X qubit; overlapping in both bases.
SLOT_PAIRS = [(0, 6), (0, 7), (1, 6), (2, 8), (0, 8), (1, 7)]


def parallel_blocks(n_surface, basis="Z"):
    memory = CodeBlock(BivariateBicycleCode.generate_code(12))
    surfaces = [CodeBlock(SurfaceCode(DISTANCE)) for _ in range(n_surface)]
    ir = MtxIRCompiler(initial_memory_code_blocks=[memory],
                       initial_surface_code_blocks=surfaces, memory_basis=basis)
    return ir, memory, surfaces


def parallel_program(basis, slots, instructions, n_surface=None):
    """Compile instructions with q0, q1, ... starting in the given memory
    slots; returns everything assert_flips_are_isolated needs."""
    n_surface = len(slots) if n_surface is None else n_surface
    ir, memory, surfaces = parallel_blocks(n_surface, basis)
    starts = {}
    for i, slot in enumerate(slots):
        ir.initialize(f"q{i}", True, 0, slot)
        starts[f"q{i}"] = ir._slot_logicals(memory, slot)
    ir.compile(instructions)
    circuit = CircuitCompiler([memory], surfaces, rounds_per_step=ROUNDS,
                              aux_basis=ir.aux_basis, memory_basis=basis,
                              readout_basis=basis).compile(ir.MtxIRs)
    return ir, memory, circuit, starts


def assert_flips_are_isolated(ir, memory, circuit, starts, basis):
    """Flipping one qubit's starting logical flips that qubit's final logical
    and no other's, and every final logical stays deterministic."""
    def value(c, qubit):
        block = ir.locations[qubit]
        slot = block.locate(qubit)
        to_global = block.local_to_global_col_mapping
        return logical_value(c, [to_global[q] for q in ir._slot_logicals(block, slot)[basis]],
                             basis)

    other = "X" if basis == "Z" else "Z"
    gate = "X_ERROR(1)" if basis == "Z" else "Z_ERROR(1)"
    to_global = memory.local_to_global_col_mapping
    tick = next(i for i, inst in enumerate(circuit) if inst.name == "TICK")
    baseline = {q: value(circuit, q) for q in starts}

    for flipped_qubit, logicals in starts.items():
        flipped = (circuit[:tick]
                   + stim.Circuit(f"{gate} " + " ".join(str(to_global[q]) for q in logicals[other]))
                   + circuit[tick:])
        for qubit in starts:
            before, after = baseline[qubit], value(flipped, qubit)
            assert np.all(before == before[0]) and np.all(after == after[0]), qubit
            assert (before[0] != after[0]) == (qubit == flipped_qubit), (flipped_qubit, qubit)


def active_logical_count(layer):
    """k of a layer, ignoring columns the layer does not touch."""
    sx = np.asarray(layer.sx, dtype=np.uint8)
    sz = np.asarray(layer.sz, dtype=np.uint8)
    n_active = int(np.count_nonzero((sx | sz).any(axis=0)))
    matrix = GF(2)(np.hstack([sx, sz]).astype(int))
    return n_active - int(np.linalg.matrix_rank(matrix))


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("slots", SLOT_PAIRS, ids=str)
def test_parallel_loads_same_block(basis, slots):
    ir, memory, circuit, starts = parallel_program(basis, slots, [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(1, "LOAD", "q1", "SC[1]"),
    ])
    circuit.detector_error_model()
    assert_flips_are_isolated(ir, memory, circuit, starts, basis)


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("slots", SLOT_PAIRS, ids=str)
def test_parallel_stores_same_block(basis, slots):
    ir, memory, circuit, starts = parallel_program(basis, slots, [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(1, "LOAD", "q1", "SC[1]"),
        Instruction(5, "STORE", "q0", f"MEM[0][{slots[0]}]"),
        Instruction(5, "STORE", "q1", f"MEM[0][{slots[1]}]"),
    ])
    circuit.detector_error_model()
    assert_flips_are_isolated(ir, memory, circuit, starts, basis)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_three_parallel_loads_same_block(basis):
    ir, memory, circuit, starts = parallel_program(basis, (0, 6, 7), [
        Instruction(1, "LOAD", f"q{i}", f"SC[{i}]") for i in range(3)
    ])
    circuit.detector_error_model()
    assert_flips_are_isolated(ir, memory, circuit, starts, basis)


@pytest.mark.parametrize("slots", SLOT_PAIRS, ids=str)
def test_parallel_loads_remove_one_logical_each(slots):
    """Each simultaneous measurement removes exactly one logical: a missing
    cycle check still commutes, so only the count shows it."""
    ir, memory, surfaces = parallel_blocks(2)
    ir.initialize("q0", True, 0, slots[0])
    ir.initialize("q1", True, 0, slots[1])
    ir.compile([
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(1, "LOAD", "q1", "SC[1]"),
    ])
    initial = active_logical_count(ir.layers[0])
    assert initial == memory.capacity + len(surfaces)
    assert active_logical_count(ir.layers[1]) == initial - 2
    assert active_logical_count(ir.layers[2]) == initial - 2


def test_load_starting_two_steps_later_avoids_opposite_bases():
    """(0, 1) overlaps when one step apart, but not two: the third step of a
    LOAD is one round of the original codes."""
    ir, memory, circuit, starts = parallel_program("Z", (0, 1), [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(3, "LOAD", "q1", "SC[1]"),
    ])
    circuit.detector_error_model()
    assert_flips_are_isolated(ir, memory, circuit, starts, "Z")


def test_staggered_load_is_allowed_when_supports_do_not_overlap():
    """One step apart, X1 is gauged while Z0 is: Z0 and X1 share no qubit,
    so this order is allowed even though (0, 1) is not."""
    ir, memory, circuit, starts = parallel_program("Z", (1, 0), [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(2, "LOAD", "q1", "SC[1]"),
    ])
    circuit.detector_error_model()
    assert_flips_are_isolated(ir, memory, circuit, starts, "Z")


def test_opposite_basis_overlap_is_rejected_for_staggered_loads():
    """One step apart, X0 is gauged while Z1 is: they overlap, so the two
    graphs' vertex checks anticommute on a shared qubit."""
    ir, _, _ = parallel_blocks(2)
    ir.initialize("q0", True, 0, 0)
    ir.initialize("q1", True, 0, 1)
    with pytest.raises(ValueError):
        ir.compile([
            Instruction(1, "LOAD", "q0", "SC[0]"),
            Instruction(2, "LOAD", "q1", "SC[1]"),
        ])


def test_opposite_basis_overlap_is_rejected_for_load_and_store():
    """Z0 is gauged by the LOAD and X3 by the STORE into a fresh slot."""
    ir, _, _ = parallel_blocks(2)
    ir.initialize("q0", True, 0, 0)
    ir.initialize("q1", False, 1)
    with pytest.raises(ValueError):
        ir.compile([
            Instruction(1, "LOAD", "q0", "SC[0]"),
            Instruction(1, "STORE", "q1", "MEM[0][3]"),
        ])


def test_opposite_basis_without_overlap_is_allowed():
    """The same mix passes when Z0 and X7 share no qubit."""
    ir, memory, surfaces = parallel_blocks(2)
    ir.initialize("q0", True, 0, 0)
    ir.initialize("q1", False, 1)
    ir.compile([
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(1, "STORE", "q1", "MEM[0][6]"),
    ])
    CircuitCompiler([memory], surfaces, rounds_per_step=ROUNDS, aux_basis=ir.aux_basis,
                    memory_basis="Z", readout_basis="Z").compile(ir.MtxIRs).detector_error_model()


@pytest.mark.parametrize("load_first", [True, False], ids=["load-first", "store-first"])
def test_store_into_slot_being_loaded_from_is_rejected(load_first):
    """Slot 0 is still in use by the LOAD, whichever is listed first."""
    ir, _, _ = parallel_blocks(2)
    ir.initialize("q0", True, 0, 0)
    ir.initialize("q1", False, 1)
    instructions = [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(1, "STORE", "q1", "MEM[0][0]"),
    ]
    with pytest.raises(ValueError):
        ir.compile(instructions if load_first else instructions[::-1])


@pytest.mark.parametrize("store_first", [True, False], ids=["store-first", "load-first"])
def test_load_into_surface_being_stored_from_is_rejected(store_first):
    """SC[0] is still in use by the STORE, whichever is listed first."""
    ir, _, _ = parallel_blocks(1)
    ir.initialize("q0", True, 0, 0)
    ir.initialize("q1", False, 0)
    instructions = [
        Instruction(1, "STORE", "q1", "MEM[0][6]"),
        Instruction(1, "LOAD", "q0", "SC[0]"),
    ]
    with pytest.raises(ValueError):
        ir.compile(instructions if store_first else instructions[::-1])


def test_store_into_slot_vacated_by_a_finished_load_is_allowed():
    """The same slot is free once the LOAD's three steps are over."""
    ir, memory, circuit, starts = parallel_program("Z", (0,), [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(4, "STORE", "q0", "MEM[0][0]"),
    ])
    circuit.detector_error_model()
    assert_flips_are_isolated(ir, memory, circuit, starts, "Z")


def chained_program(store_slot):
    """q0 moves slot 0 -> SC[0] -> slot 2, then leaves slot 2 at step 7 while
    q1 is stored into store_slot."""
    ir, _, _ = parallel_blocks(2)
    ir.initialize("q0", True, 0, 0)
    ir.initialize("q1", False, 1)
    ir.compile([
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(4, "STORE", "q0", "MEM[0][2]"),
        Instruction(7, "LOAD", "q0", "SC[0]"),
        Instruction(7, "STORE", "q1", f"MEM[0][{store_slot}]"),
    ])


def test_conflict_uses_the_location_a_qubit_has_moved_to():
    with pytest.raises(ValueError, match="two operations"):
        chained_program(2)


def test_slot_a_qubit_has_left_is_not_a_conflict():
    chained_program(0)


def test_code_block_type_is_inferred_from_the_code():
    assert CodeBlock(SurfaceCode(DISTANCE)).type == CodeBlockType.SC
    assert CodeBlock(BivariateBicycleCode.generate_code(12)).type == CodeBlockType.MEM
    assert CodeBlock(SurfaceCode(DISTANCE), CodeBlockType.SC).type == CodeBlockType.SC


def test_slots_count_from_zero():
    block = CodeBlock(BivariateBicycleCode.generate_code(12))
    ir = MtxIRCompiler(initial_memory_code_blocks=[block],
                       initial_surface_code_blocks=[CodeBlock(SurfaceCode(DISTANCE))])
    assert block.first_empty_spot() == 0
    ir.initialize("q0", True, 0, 0)
    assert block.locate("q0") == 0
    assert block.first_empty_spot() == 1
    ir.initialize("q1", True, 0, 11)
    assert block.locate("q1") == 11
    with pytest.raises(ValueError):
        ir.initialize("q2", True, 0, 12)
    assert ir._gross_logicals(block.code, 0)[0] == [
        2, 9, 26, 33, 34, 58, 78, 85, 93, 102, 109, 141]
