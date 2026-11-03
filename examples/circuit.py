from typing import List

from mtxir import Instruction, BivariateBicycleCode, SurfaceCode, CodeBlock, MtxIRCompiler
from mtxir.compiler import CircuitCompiler


if __name__ == '__main__':
    bb = BivariateBicycleCode.generate_code(12)
    surface = SurfaceCode(15)

    mem = [CodeBlock(bb) for i in range(2)]
    sc = [CodeBlock(surface) for i in range(3)]


    instructions : List[Instruction] = [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(1, "LOAD", "q1", "SC[1]"),
        Instruction(3, "LOAD", "q2", "SC[2]"),
        Instruction(4, "STORE", "q0", "MEM[0][1]"),
        Instruction(4, "STORE", "q1", "MEM[1][4]"),
        Instruction(7, "LOAD", "q0", "SC[0]") # reuse SC[0]
    ]

    mtx_ir_compiler = MtxIRCompiler(
            initial_memory_code_blocks=mem, 
            initial_surface_code_blocks=sc
            )

    mtx_ir_compiler.initialize("q0", True, 0, 1)
    mtx_ir_compiler.initialize("q1", True, 1, 1)
    mtx_ir_compiler.initialize("q2", True, 0, 2)


    layers = mtx_ir_compiler.compile(instructions)


    cc = CircuitCompiler(mem, sc, rounds_per_step=15)
    circuit = cc.compile(mtx_ir_compiler.layers)
    circuit.to_file("LD_ST.stim")