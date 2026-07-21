from dataclasses import dataclass, field

import stim


@dataclass(frozen=True)
class MetaInstruction:
    """Duck-types ``stim.CircuitInstruction``'s read interface (``name``,
    ``tag``, ``targets_copy()``, ``gate_args_copy()``) over plain Python
    data, so code that iterates a ``MetaCircuit`` works unchanged whether
    it's actually iterating a ``stim.Circuit`` or a ``MetaCircuit``."""

    name: str
    targets: list = field(default_factory=list)
    args: list = field(default_factory=list)
    tag: str = ""

    def targets_copy(self) -> list:
        return list(self.targets)

    def gate_args_copy(self) -> list:
        return list(self.args)


class MetaCircuit:
    """
    A lightweight stand-in for ``stim.Circuit`` that supports fast
    instruction appending.

    ``stim.Circuit.append()`` is expensive per call (on the order of
    100us, mostly the cost of crossing into the compiled extension and
    validating/growing the circuit's internal state on every single
    instruction). Rebuilding a circuit of tens of thousands of
    instructions one ``append()`` at a time -- as the multi-pass
    underclocking pipeline does (stretch/spacing, movement ticks, emit,
    ...) -- adds up to real minutes of wall time.

    ``MetaCircuit.append()`` mirrors ``stim.Circuit.append()``'s
    signature, but never calls the underlying stim function -- it just
    records the instruction (as a ``MetaInstruction``) in a plain Python
    list, which is effectively free. Call ``to_stim()`` once, when an
    actual ``stim.Circuit`` is needed, to materialize it in a single bulk
    parse instead of one ``append()`` per instruction.
    """

    def __init__(self, circuit: stim.Circuit | None = None):
        self.instructions: list[stim.CircuitInstruction | MetaInstruction] = []
        if circuit is not None:
            for instr in circuit.flattened():
                self.append_instruction(instr)

    def append(self, name, targets=(), arg=(), *, tag=""):
        """Records an instruction like ``stim.Circuit.append()`` does, but
        without calling it -- this is what makes appending fast.

        Like ``stim.Circuit.append()``, also accepts a single already-built
        ``stim.CircuitInstruction`` (or ``MetaInstruction``) as ``name`` --
        in that case it's recorded as-is (see ``append_instruction()``),
        so callers that already have an instruction object don't need to
        branch on whether they're appending to a ``stim.Circuit`` or a
        ``MetaCircuit``.
        """
        if isinstance(name, (stim.CircuitInstruction, MetaInstruction)):
            self.append_instruction(name)
            return
        self.instructions.append(MetaInstruction(name, list(targets), list(arg), tag))

    def append_instruction(
        self, instruction: stim.CircuitInstruction | MetaInstruction
    ):
        """Records an already-built ``stim.CircuitInstruction`` (or
        ``MetaInstruction``) as-is, unmodified. Prefer this over ``append()``
        when re-emitting an instruction you already have (e.g. an original
        gate a noise model interleaves new noise around): it skips
        decomposing the instruction into raw parts only to reconstruct an
        equivalent one later in ``to_stim()``, which matters because
        stim.CircuitInstruction construction itself isn't free, especially
        for instructions with many targets."""
        self.instructions.append(instruction)

    def to_stim(self) -> stim.Circuit:
        """Materializes a real ``stim.Circuit`` from the accumulated
        instructions in one bulk parse, instead of one ``append()`` call
        per instruction. Instructions recorded via ``append_instruction()``
        are already valid ``stim.CircuitInstruction`` objects and are
        serialized directly, with no reconstruction."""
        lines = []
        for instr in self.instructions:
            if isinstance(instr, MetaInstruction):
                instr = stim.CircuitInstruction(
                    instr.name, instr.targets, instr.args, tag=instr.tag
                )
            lines.append(str(instr))
        return stim.Circuit("\n".join(lines))

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)
