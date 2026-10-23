from pathlib import Path

from streamlit.testing.v1 import AppTest

from presets import PRESETS

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    return app


def test_first_preset_renders_without_errors():
    app = _run_app()
    assert not app.exception
    assert not app.error
    assert app.text_input(key="equation").value == "ij,jk->ik"
    assert app.metric[0].value == "(2, 4)"


def test_controls_and_preset_note_are_on_the_page_not_in_the_sidebar():
    app = _run_app()
    assert not app.sidebar.children
    assert PRESETS[0].note in [caption.value for caption in app.main.caption]


def test_choosing_a_preset_sets_equation_and_matching_shapes():
    app = _run_app()
    app.selectbox(key="preset").set_value("Batch matrix multiply").run()
    assert app.text_input(key="equation").value == "bij,bjk->bik"
    assert app.text_input(key="shape_0").value == "2, 3, 4"
    assert app.text_input(key="shape_1").value == "2, 4, 5"
    assert app.metric[0].value == "(2, 3, 5)"


def test_typing_an_equation_refills_shapes():
    app = _run_app()
    app.text_input(key="equation").set_value("i,j,k->ijk").run()
    assert not app.error
    assert [app.text_input(key=f"shape_{n}").value for n in range(3)] == ["2", "3", "4"]
    assert app.metric[0].value == "(2, 3, 4)"


def test_invalid_equation_shows_error():
    app = _run_app()
    app.text_input(key="equation").set_value("ij->k").run()
    assert app.error[0].value == "Equation: output letter 'k' does not appear in any input"


def test_mismatched_shape_shows_error_naming_both_operands():
    app = _run_app()
    app.text_input(key="shape_1").set_value("5, 4").run()
    assert app.error[0].value.startswith("Shapes: 'j' has size 3 in x0 but size 5 in x1")


def test_oversized_shapes_skip_building_tensors():
    app = _run_app()
    app.text_input(key="shape_0").set_value("2000, 3").run()
    app.text_input(key="shape_1").set_value("3, 2000").run()
    assert not app.exception
    assert "would have 4,000,000 elements" in app.warning[0].value
