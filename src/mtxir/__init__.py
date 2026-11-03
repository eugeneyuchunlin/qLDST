"""mtxir: compile LOAD/STORE programs between a gross (bivariate bicycle)
memory and surface codes into noisy Stim circuits.

Two stages:

1. ``MtxIRCompiler`` (``mtxir.ir``) turns a list of ``Instruction`` into a
   list of ``MtxIRLayer``, one per time step: the checks measured at that
   step, including the gauging (code deformation) layers of each teleport.
2. ``CircuitCompiler`` (``mtxir.compiler``) schedules the layers into a Stim
   circuit with circuit-level noise, detectors and observables.

Example::

    from mtxir import (BivariateBicycleCode, SurfaceCode, CodeBlock,
                       Instruction, MtxIRCompiler, CircuitCompiler)

    memory = CodeBlock(BivariateBicycleCode.generate_code(12))
    surface = CodeBlock(SurfaceCode(5))
    ir = MtxIRCompiler(initial_memory_code_blocks=[memory],
                       initial_surface_code_blocks=[surface],
                       memory_basis="Z")
    ir.initialize("q0", True, 0, 7)          # q0 in logical 7 of MEM[0]
    ir.compile([Instruction(1, "LOAD", "q0", "SC[0]")])

    circuit = CircuitCompiler([memory], [surface], rounds_per_step=12,
                              aux_basis=ir.aux_basis, memory_basis="Z",
                              readout_basis="Z", p=1e-3).compile(ir.MtxIRs)
"""

from .codes import BivariateBicycleCode, SurfaceCode
from .compiler import CircuitCompiler, build_example
from .ir import CodeBlock, Instruction, MemoryInstruction, MtxIRCompiler, MtxIRLayer
from .polynomial import Polynomial

__version__ = "0.1.0"

__all__ = [
    "BivariateBicycleCode",
    "CircuitCompiler",
    "CodeBlock",
    "Instruction",
    "MemoryInstruction",
    "MtxIRCompiler",
    "MtxIRLayer",
    "Polynomial",
    "SurfaceCode",
    "build_example",
]
