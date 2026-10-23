"""Parse and explain einsum equations without evaluating them.

The rules mirror torch.einsum (checked against torch 2.14), so the explanation the
app shows matches what torch computes. tests/test_einsum_core.py pins that claim by
running the generated loop code against torch.einsum.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

ARROW = "->"
_DEFAULT_SIZES = (2, 3, 4, 5, 6)

Role = Literal["batch", "free", "summed"]
Shape = tuple[int, ...]


class EinsumError(ValueError):
    """An equation or shape that torch.einsum would reject, or that this playground can't explain."""


@dataclass(frozen=True)
class Equation:
    inputs: tuple[str, ...]
    output: str
    implicit: bool

    @property
    def explicit_form(self) -> str:
        return ",".join(self.inputs) + ARROW + self.output


@dataclass(frozen=True)
class Index:
    letter: str
    size: int
    role: Role
    operands: tuple[int, ...]


@dataclass(frozen=True)
class Analysis:
    equation: Equation
    shapes: tuple[Shape, ...]
    # Output letters first, in output order, then summed letters in order of first appearance.
    # loop_code() nests its loops in this order.
    indices: tuple[Index, ...]

    @property
    def output_shape(self) -> Shape:
        sizes = {index.letter: index.size for index in self.indices}
        return tuple(sizes[letter] for letter in self.equation.output)

    @property
    def loop_iterations(self) -> int:
        return math.prod(index.size for index in self.indices)


def operand_name(position: int) -> str:
    # Multi-character names can never collide with the single-letter loop variables.
    return f"x{position}"


def parse_equation(text: str) -> Equation:
    compact = "".join(text.split())
    if "..." in compact:
        raise EinsumError("'...' (ellipsis) is not supported here; name every axis with a letter")
    if compact.count(ARROW) > 1:
        raise EinsumError(f"the equation has more than one {ARROW!r}")

    lhs, arrow, rhs = compact.partition(ARROW)
    inputs = tuple(lhs.split(","))
    for position, subscripts in enumerate(inputs):
        _check_letters(subscripts, f"input {operand_name(position)}")

    if not arrow:
        # torch keeps the letters that appear exactly once, sorted by character code,
        # so uppercase letters come before lowercase ones.
        counts = Counter("".join(inputs))
        output = "".join(sorted(letter for letter, count in counts.items() if count == 1))
        return Equation(inputs, output, implicit=True)

    _check_letters(rhs, "the output")
    input_letters = set("".join(inputs))
    for letter, count in Counter(rhs).items():
        if count > 1:
            raise EinsumError(f"output letter {letter!r} appears more than once")
        if letter not in input_letters:
            raise EinsumError(f"output letter {letter!r} does not appear in any input")
    return Equation(inputs, rhs, implicit=False)


def _check_letters(subscripts: str, where: str) -> None:
    for char in subscripts:
        if char in "->":
            raise EinsumError(
                f"{where} has a stray {char!r}; the output arrow is written {ARROW!r}"
            )
        if char not in string.ascii_letters:
            raise EinsumError(f"{where} has {char!r}; subscripts must be letters a-z or A-Z")


def analyze(equation: Equation, shapes: Sequence[Sequence[int]]) -> Analysis:
    shapes = tuple(tuple(shape) for shape in shapes)
    if len(shapes) != len(equation.inputs):
        raise EinsumError(
            f"the equation has {len(equation.inputs)} inputs but {len(shapes)} shapes were given"
        )

    sizes: dict[str, int] = {}
    size_source: dict[str, str] = {}
    operands_of: dict[str, list[int]] = {}
    for position, (subscripts, shape) in enumerate(zip(equation.inputs, shapes)):
        name = operand_name(position)
        if len(subscripts) != len(shape):
            raise EinsumError(
                f"{name} has {len(subscripts)} subscripts ({subscripts!r}) "
                f"but its shape {shape} has {len(shape)} dimensions"
            )

        own_sizes: dict[str, int] = {}
        for letter, size in zip(subscripts, shape):
            # A letter repeated inside one operand walks its diagonal, so torch
            # requires equal sizes there; size 1 does not broadcast.
            if own_sizes.get(letter, size) != size:
                raise EinsumError(
                    f"{letter!r} appears twice in {name} with sizes {own_sizes[letter]} and {size}; "
                    "a repeated letter takes the diagonal, so both sizes must be equal"
                )
            own_sizes[letter] = size

        for letter, size in own_sizes.items():
            seen = sizes.get(letter)
            if seen is None or (seen == 1 and size != 1):
                sizes[letter] = size
                size_source[letter] = name
            elif size not in (1, seen):
                raise EinsumError(
                    f"{letter!r} has size {seen} in {size_source[letter]} but size {size} in {name}; "
                    "sizes for the same letter must match, or one of them must be 1"
                )
            operands_of.setdefault(letter, []).append(position)

    output = equation.output
    input_count = len(equation.inputs)

    def role(letter: str) -> Role:
        if letter not in output:
            return "summed"
        if input_count > 1 and len(operands_of[letter]) == input_count:
            return "batch"
        return "free"

    order = list(output) + [letter for letter in sizes if letter not in output]
    indices = tuple(
        Index(letter, sizes[letter], role(letter), tuple(operands_of[letter])) for letter in order
    )
    return Analysis(equation, shapes, indices)


def loop_code(analysis: Analysis) -> str:
    """Return runnable Python (operands named x0, x1, ...) that computes the same result as torch.einsum."""
    sizes = {index.letter: index.size for index in analysis.indices}
    output = analysis.equation.output

    zeros_args = ", ".join(str(size) for size in analysis.output_shape) or "()"
    lines = [f"out = torch.zeros({zeros_args})"]
    for depth, index in enumerate(analysis.indices):
        lines.append(f"{'    ' * depth}for {index.letter} in range({index.size}):")

    target = f"out[{', '.join(output)}]" if output else "out"
    factors = " * ".join(
        _operand_access(position, subscripts, shape, sizes)
        for position, (subscripts, shape) in enumerate(
            zip(analysis.equation.inputs, analysis.shapes)
        )
    )
    lines.append(f"{'    ' * len(analysis.indices)}{target} += {factors}")
    return "\n".join(lines)


def _operand_access(position: int, subscripts: str, shape: Shape, sizes: dict[str, int]) -> str:
    name = operand_name(position)
    if not subscripts:
        return name
    # A size-1 axis broadcast against a larger size is always read at position 0.
    axes = ", ".join(
        "0" if dim == 1 and sizes[letter] != 1 else letter for letter, dim in zip(subscripts, shape)
    )
    return f"{name}[{axes}]"


def default_shapes(equation: Equation) -> tuple[Shape, ...]:
    """Give each distinct letter its own size (2, 3, 4, ...) so the axes are easy to tell apart."""
    letters = dict.fromkeys("".join(equation.inputs))
    sizes = {
        letter: _DEFAULT_SIZES[position % len(_DEFAULT_SIZES)]
        for position, letter in enumerate(letters)
    }
    return tuple(tuple(sizes[letter] for letter in subscripts) for subscripts in equation.inputs)


def parse_shape(text: str) -> Shape:
    """Parse '2, 3, 4', '(2, 3, 4)', '[2 3 4]' and similar. Empty text means a scalar, shape ()."""
    body = text.strip().strip("()[]")
    dims = []
    for part in re.split(r"[,\s]+", body):
        if not part:
            continue
        if not re.fullmatch(r"[0-9]+", part) or int(part) < 1:
            raise EinsumError(f"{part!r} in shape {text!r} is not a positive integer")
        dims.append(int(part))
    return tuple(dims)
