"""Starter equations for the app. Shapes come from einsum_core.default_shapes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    equation: str
    note: str


PRESETS: tuple[Preset, ...] = (
    Preset(
        "Matrix multiply",
        "ij,jk->ik",
        "j is shared and dropped from the output, so each cell is a row of x0 dotted with a column of x1.",
    ),
    Preset(
        "Transpose",
        "ij->ji",
        "No letter is dropped, so nothing is summed. The axes are only reordered.",
    ),
    Preset(
        "Sum of all elements",
        "ij->",
        "Every letter is dropped from the output, so every element is summed.",
    ),
    Preset("Row sums", "ij->i", "j is dropped, so each row is summed across its columns."),
    Preset("Column sums", "ij->j", "i is dropped, so each column is summed down its rows."),
    Preset(
        "Trace",
        "ii->",
        "A letter repeated inside one operand walks its diagonal. Dropping it sums the diagonal.",
    ),
    Preset("Diagonal", "ii->i", "Keeping the repeated letter returns the diagonal itself."),
    Preset("Dot product", "i,i->", "Matching elements are multiplied, then i is summed away."),
    Preset(
        "Outer product",
        "i,j->ij",
        "No shared letters and nothing dropped, so every pair of elements is multiplied.",
    ),
    Preset(
        "Elementwise product",
        "ij,ij->ij",
        "Shared letters that stay in the output multiply matching elements (Hadamard product).",
    ),
    Preset("Matrix-vector product", "ij,j->i", "Each row of x0 is dotted with the vector x1."),
    Preset(
        "Batch matrix multiply",
        "bij,bjk->bik",
        "b is in every input and in the output, so it is a batch axis: one matmul per b.",
    ),
    Preset("Bilinear form", "i,ij,j->", "Three operands at once: x0ᵀ · x1 · x2, a single number."),
    Preset(
        "Attention scores",
        "bhqd,bhkd->bhqk",
        "Multi-head attention's query-key scores. b batch, h heads, q queries, k keys, d head size. "
        "Summing over d dots every query with every key.",
    ),
    Preset(
        "Implicit output",
        "ij,jk",
        "No '->'. torch keeps the letters that appear exactly once, sorted, giving ij,jk->ik.",
    ),
)
