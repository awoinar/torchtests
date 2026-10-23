"""Streamlit UI for exploring torch.einsum.

Run with `uv run streamlit run app.py`. Plain `python app.py` (an IDE's Run button)
also works: it relaunches itself under `streamlit run`.
"""

from __future__ import annotations

import math
import sys

import streamlit as st
import torch
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web import cli as streamlit_cli

from einsum_core import (
    Analysis,
    EinsumError,
    Shape,
    analyze,
    default_shapes,
    loop_code,
    operand_name,
    parse_equation,
    parse_shape,
)
from presets import PRESETS

# Without a script context, plain Python started this file, so no Streamlit server is
# running and every st.* call would be a no-op. Checking the context (not whether a
# runtime exists) leaves `streamlit run` and AppTest alone, since both provide one.
if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(streamlit_cli.main())

# Caps keep a typo like "2000, 2000" from freezing the page or eating memory.
MAX_ELEMENTS = 1_000_000
MAX_LOOP_ITERATIONS = 50_000_000

VALUE_MODES = ("random integers", "arange", "ones")
ROLE_TEXT = {
    "batch": "batch: kept, and in every input",
    "free": "free: kept in the output",
    "summed": "summed: not in the output, so summed over",
}
PRESETS_BY_NAME = {preset.name: preset for preset in PRESETS}


def _shape_key(position: int) -> str:
    return f"shape_{position}"


def _format_shape(shape: Shape) -> str:
    return ", ".join(str(dim) for dim in shape)


def _reset_shapes() -> None:
    try:
        shapes = default_shapes(parse_equation(st.session_state.equation))
    except EinsumError:
        return  # Keep what was typed; the main script shows the error.
    for position, shape in enumerate(shapes):
        st.session_state[_shape_key(position)] = _format_shape(shape)


def _apply_preset() -> None:
    st.session_state.equation = PRESETS_BY_NAME[st.session_state.preset].equation
    _reset_shapes()


def _make_operand(shape: Shape, mode: str, generator: torch.Generator) -> torch.Tensor:
    if mode == "arange":
        return torch.arange(math.prod(shape)).reshape(shape)
    if mode == "ones":
        return torch.ones(shape, dtype=torch.int64)
    return torch.randint(-3, 4, shape, generator=generator)


def _size_problem(analysis: Analysis) -> str | None:
    tensors = [(operand_name(n), shape) for n, shape in enumerate(analysis.shapes)]
    tensors.append(("out", analysis.output_shape))
    for name, shape in tensors:
        if math.prod(shape) > MAX_ELEMENTS:
            return f"{name} would have {math.prod(shape):,} elements (limit {MAX_ELEMENTS:,})."
    if analysis.loop_iterations > MAX_LOOP_ITERATIONS:
        return (
            f"The naive loop runs {analysis.loop_iterations:,} times "
            f"(limit {MAX_LOOP_ITERATIONS:,})."
        )
    return None


st.set_page_config(page_title="einsum playground", layout="wide")
st.title("einsum playground")
st.caption(
    "Each letter names one axis. Letters missing from the output are summed. "
    "Pick a preset or type your own equation and shapes."
)

if "equation" not in st.session_state:
    st.session_state.preset = PRESETS[0].name
    _apply_preset()

# Controls live on the page, not in st.sidebar: on phones the sidebar collapses into a
# hidden menu, while columns stack so these stay at the top.
preset_column, mode_column, seed_column = st.columns([2, 2, 1])
with preset_column:
    st.selectbox("Preset", list(PRESETS_BY_NAME), key="preset", on_change=_apply_preset)
    preset = PRESETS_BY_NAME[st.session_state.preset]
    if st.session_state.equation == preset.equation:
        st.caption(preset.note)
with mode_column:
    mode = st.radio("Operand values", VALUE_MODES, horizontal=True)
with seed_column:
    seed = st.number_input("Seed", value=0, step=1, disabled=mode != "random integers")

st.text_input(
    "Equation",
    key="equation",
    on_change=_reset_shapes,
    help="For example ij,jk->ik. Leave out '->' to let torch pick the output.",
)

try:
    equation = parse_equation(st.session_state.equation)
except EinsumError as error:
    st.error(f"Equation: {error}")
    st.stop()

fallback_shapes = default_shapes(equation)
shape_columns = st.columns(len(equation.inputs))
for position, (column, subscripts) in enumerate(zip(shape_columns, equation.inputs)):
    # Streamlit drops a widget's state on runs where it isn't drawn, so refill missing boxes.
    st.session_state.setdefault(_shape_key(position), _format_shape(fallback_shapes[position]))
    column.text_input(
        f"{operand_name(position)} shape · {subscripts or 'scalar'}", key=_shape_key(position)
    )

try:
    shapes = []
    for position in range(len(equation.inputs)):
        try:
            shapes.append(parse_shape(st.session_state[_shape_key(position)]))
        except EinsumError as error:
            raise EinsumError(f"{operand_name(position)}: {error}") from error
    analysis = analyze(equation, shapes)
except EinsumError as error:
    st.error(f"Shapes: {error}")
    st.stop()

if equation.implicit:
    st.info(
        "No '->', so torch keeps the letters that appear exactly once, sorted with uppercase "
        f"first: `{equation.explicit_form}`"
    )

shape_metric, loop_metric = st.columns(2)
shape_metric.metric("Output shape", str(analysis.output_shape))
loop_metric.metric(
    "Loop iterations",
    f"{analysis.loop_iterations:,}",
    help="How many multiply-adds the nested loop below does. torch usually finds a cheaper order.",
)

st.subheader("Letters")
st.dataframe(
    [
        {
            "letter": index.letter,
            "size": index.size,
            "role": ROLE_TEXT[index.role],
            "appears in": ", ".join(operand_name(n) for n in index.operands),
        }
        for index in analysis.indices
    ],
    hide_index=True,
)

st.subheader("The same thing as nested loops")
st.code(loop_code(analysis), language="python")

operand_names = ", ".join(operand_name(n) for n in range(len(equation.inputs)))
st.code(
    f'out = torch.einsum("{st.session_state.equation.strip()}", {operand_names})', language="python"
)

st.subheader("Tensors")
problem = _size_problem(analysis)
if problem:
    st.warning(f"{problem} Use smaller shapes to see the values.")
    st.stop()

generator = torch.Generator().manual_seed(int(seed))
operands = [_make_operand(shape, mode, generator) for shape in analysis.shapes]
try:
    # Pass the equation exactly as typed, so torch itself is the judge of what it means.
    result = torch.einsum(st.session_state.equation, *operands)
except RuntimeError as error:
    st.error(f"torch.einsum rejected this: {error}")
    st.stop()

tensor_columns = st.columns(len(operands) + 1)
labelled = [(operand_name(n), tensor) for n, tensor in enumerate(operands)] + [("out", result)]
for column, (name, tensor) in zip(tensor_columns, labelled):
    column.markdown(f"**{name}** · shape `{tuple(tensor.shape)}`")
    column.code(str(tensor), language=None)
