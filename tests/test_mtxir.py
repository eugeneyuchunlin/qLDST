"""Tests of the mtxir package: compiled LOAD/STORE circuits are valid Stim
circuits with deterministic detectors, and a chain of teleports carries the
logical qubit to its final location.

Run with: python -m pytest
"""

import numpy as np
import pytest
import stim

from mtxir import (
    BivariateBicycleCode,
    CircuitCompiler,
    CodeBlock,
    Instruction,
    MtxIRCompiler,
    SurfaceCode,
)

DISTANCE = 3
ROUNDS = 2
LOGICAL = 7


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
    [("LOAD",), ("STORE", 7)],
    [("LOAD",), ("STORE", 3)],
    [("LOAD",), ("STORE", 3), ("LOAD",)],
]


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("ops", CHAINS, ids=lambda ops: "-".join(
    o[0] + (str(o[1]) if len(o) > 1 else "") for o in ops))
def test_teleport_chain_carries_the_logical(basis, ops):
    """Flipping logical LOGICAL at the start flips q0's final logical, and
    the final logical is deterministic with or without the flip."""
    ir, bb, _, circuit = compile_program(basis, ops)
    block = ir.locations["q0"]
    slot = block.qubits.index("q0") + 1
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
