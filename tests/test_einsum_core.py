import re

import pytest
import torch

from einsum_core import (
    EinsumError,
    Equation,
    Index,
    analyze,
    default_shapes,
    loop_code,
    operand_name,
    parse_equation,
    parse_shape,
)
from presets import PRESETS


class TestParseEquation:
    def test_keeps_explicit_output_as_written(self):
        assert parse_equation("bij,bjk->bik") == Equation(("bij", "bjk"), "bik", implicit=False)

    def test_ignores_whitespace(self):
        assert parse_equation(" ij , jk -> ik ") == parse_equation("ij,jk->ik")

    @pytest.mark.parametrize(
        "equation, output",
        [("ij,jk", "ik"), ("ii", ""), ("i,i", ""), ("ba", "ab"), ("aB", "Ba")],
    )
    def test_implicit_output_keeps_letters_seen_once_sorted_by_character_code(
        self, equation, output
    ):
        parsed = parse_equation(equation)
        assert parsed.implicit
        assert parsed.output == output

    def test_implicit_output_order_matches_torch_for_mixed_case(self):
        result = torch.einsum("aB", torch.zeros(2, 3))
        assert result.shape == (3, 2)  # "Ba": uppercase sorts first

    def test_empty_subscripts_are_a_scalar_operand(self):
        assert parse_equation("i,->i").inputs == ("i", "")

    @pytest.mark.parametrize(
        "equation, message",
        [
            ("...ij->ij", "'...' (ellipsis) is not supported here"),
            ("ij->ji->ij", "the equation has more than one '->'"),
            ("i1->i", "input x0 has '1'; subscripts must be letters a-z or A-Z"),
            ("ij,j2", "input x1 has '2'"),
            ("ij-ji", "input x0 has a stray '-'; the output arrow is written '->'"),
            ("ij->i!", "the output has '!'"),
            ("ij->ii", "output letter 'i' appears more than once"),
            ("ij->k", "output letter 'k' does not appear in any input"),
        ],
    )
    def test_rejects_invalid_equation_with_specific_message(self, equation, message):
        with pytest.raises(EinsumError, match=re.escape(message)):
            parse_equation(equation)


class TestAnalyze:
    def test_classifies_batch_free_and_summed_letters(self):
        analysis = analyze(parse_equation("bij,bjk->bik"), [(2, 3, 4), (2, 4, 5)])
        assert analysis.indices == (
            Index("b", 2, "batch", (0, 1)),
            Index("i", 3, "free", (0,)),
            Index("k", 5, "free", (1,)),
            Index("j", 4, "summed", (0, 1)),
        )
        assert analysis.output_shape == (2, 3, 5)
        assert analysis.loop_iterations == 2 * 3 * 4 * 5

    def test_single_operand_letters_are_free_not_batch(self):
        analysis = analyze(parse_equation("ij->ji"), [(2, 3)])
        assert [index.role for index in analysis.indices] == ["free", "free"]

    def test_scalar_output_has_empty_shape(self):
        analysis = analyze(parse_equation("i,i->"), [(3,), (3,)])
        assert analysis.output_shape == ()
        assert analysis.loop_iterations == 3

    @pytest.mark.parametrize("shapes", [[(2, 1), (3, 4)], [(2, 3), (1, 4)]])
    def test_size_one_broadcasts_across_operands(self, shapes):
        analysis = analyze(parse_equation("ij,jk->ik"), shapes)
        operands = [torch.ones(shape) for shape in shapes]
        assert analysis.output_shape == torch.einsum("ij,jk->ik", *operands).shape

    @pytest.mark.parametrize(
        "equation, shapes, message",
        [
            ("ij,jk->ik", [(2, 3)], "the equation has 2 inputs but 1 shapes were given"),
            (
                "ijk->ik",
                [(2, 3)],
                "x0 has 3 subscripts ('ijk') but its shape (2, 3) has 2 dimensions",
            ),
            (
                "ij,jk->ik",
                [(2, 3), (4, 5)],
                "'j' has size 3 in x0 but size 4 in x1; sizes for the same letter must match",
            ),
            ("ii->i", [(2, 3)], "'i' appears twice in x0 with sizes 2 and 3"),
            # Unlike across operands, size 1 does not broadcast along a diagonal.
            ("ii->i", [(1, 3)], "'i' appears twice in x0 with sizes 1 and 3"),
        ],
    )
    def test_rejects_shapes_torch_would_reject(self, equation, shapes, message):
        with pytest.raises(EinsumError, match=re.escape(message)):
            analyze(parse_equation(equation), shapes)
        with pytest.raises(RuntimeError):
            torch.einsum(equation, *[torch.ones(shape) for shape in shapes])


class TestLoopCode:
    def test_nests_output_letters_then_summed_letters(self):
        analysis = analyze(parse_equation("ij,jk->ik"), [(2, 3), (3, 4)])
        assert loop_code(analysis) == (
            "out = torch.zeros(2, 4)\n"
            "for i in range(2):\n"
            "    for k in range(4):\n"
            "        for j in range(3):\n"
            "            out[i, k] += x0[i, j] * x1[j, k]"
        )

    def test_scalar_output_uses_zero_dim_tensor(self):
        analysis = analyze(parse_equation("i,i->"), [(3,), (3,)])
        assert loop_code(analysis) == (
            "out = torch.zeros(())\nfor i in range(3):\n    out += x0[i] * x1[i]"
        )

    def test_broadcast_axis_is_read_at_zero(self):
        analysis = analyze(parse_equation("ij,jk->ik"), [(2, 1), (3, 4)])
        assert loop_code(analysis).endswith("out[i, k] += x0[i, 0] * x1[j, k]")


EQUIVALENCE_CASES = [(preset.equation, None) for preset in PRESETS] + [
    ("ij,jk->ik", [(2, 1), (3, 4)]),
    ("i,->i", [(3,), ()]),
    ("ii,i->i", [(1, 1), (3,)]),
    ("aB", None),
    ("Ab,bC->CA", None),
    ("ijk->kji", None),
]


@pytest.mark.parametrize("equation, shapes", EQUIVALENCE_CASES)
def test_loop_code_computes_the_same_result_as_torch(equation, shapes):
    parsed = parse_equation(equation)
    shapes = shapes if shapes is not None else default_shapes(parsed)
    generator = torch.Generator().manual_seed(0)
    operands = [
        torch.randint(-3, 4, shape, generator=generator).to(torch.float64) for shape in shapes
    ]
    expected = torch.einsum(equation, *operands)

    analysis = analyze(parsed, shapes)
    namespace = {"torch": torch} | {operand_name(n): x for n, x in enumerate(operands)}
    # Running the exact text the app displays is the point of this test.
    exec(loop_code(analysis), namespace)  # noqa: S102

    assert analysis.output_shape == expected.shape
    torch.testing.assert_close(namespace["out"], expected, check_dtype=False)


class TestDefaultShapes:
    def test_gives_each_letter_its_own_size(self):
        assert default_shapes(parse_equation("bij,bjk->bik")) == ((2, 3, 4), (2, 4, 5))

    def test_repeated_letter_gets_equal_sizes(self):
        assert default_shapes(parse_equation("ii->")) == ((2, 2),)

    def test_scalar_operand_gets_empty_shape(self):
        assert default_shapes(parse_equation("i,->i")) == ((2,), ())


class TestParseShape:
    @pytest.mark.parametrize(
        "text, shape",
        [
            ("2, 3, 4", (2, 3, 4)),
            ("(2,3)", (2, 3)),
            ("[4]", (4,)),
            ("2 3", (2, 3)),
            ("", ()),
            ("  ", ()),
            ("()", ()),
        ],
    )
    def test_accepts_common_spellings(self, text, shape):
        assert parse_shape(text) == shape

    @pytest.mark.parametrize("text, bad", [("2,a", "a"), ("0", "0"), ("-1", "-1"), ("2.5", "2.5")])
    def test_rejects_anything_but_positive_integers(self, text, bad):
        with pytest.raises(EinsumError, match=re.escape(f"{bad!r} in shape {text!r}")):
            parse_shape(text)
