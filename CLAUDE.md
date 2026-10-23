# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app for exploring `torch.einsum`. For an equation and operand shapes, it shows
the output shape, each letter's role, the equivalent nested Python loops, and torch's result.

## Commands

Everything runs through `uv` (Python 3.12, pinned in `.python-version`).

```bash
uv sync                                    # install (CPU-only torch)
uv run streamlit run app.py                # app on http://localhost:8501
uv run python app.py                       # same; app.py relaunches itself under streamlit run
uv run pytest                              # all tests, about 3 s
uv run pytest tests/test_einsum_core.py::TestAnalyze::test_classifies_batch_free_and_summed_letters
uv run pytest -k loop_code                 # tests whose names match
uv run ruff check . && uv run ruff format --check .   # line length 100
```

## Container and deploy

`Dockerfile` builds the app image: Python 3.12, dependencies from `uv.lock` (CPU-only torch),
Streamlit on port 8501. `.dockerignore` is an allowlist: a new source file must be added both
there and to the `COPY` line, or the image won't contain it.

On every push to `main`, `.github/workflows/build.yml` builds the image for `linux/arm64` and
pushes it to `ghcr.io/awoinar/torchtests`, tagged `latest` and `sha-<short commit>`. Its
`deploy` job then logs into the server `bebop` over SSH, which runs `/opt/torchtests/deploy.sh`
(pull, restart, prune), and fails unless the server reports the pushed commit. So a push to
`main` goes live on https://tt.bebop.ro, which is public with no login. The key, secrets and
rollback are described in the `hetzner` repo (`torchtests.md`).

## Architecture

The code is split so that the explanation logic has no Streamlit or torch import:

- `einsum_core.py` (pure Python): `parse_equation` returns an `Equation`.
  `analyze(equation, shapes)` returns an `Analysis`: each letter's size and role
  (`batch`, `free`, `summed`), the output shape, and the loop count. `loop_code(analysis)`
  returns runnable nested-loop Python. Every rule violation raises `EinsumError`, and the app
  shows its message as-is, so keep messages specific (name the letter and the operand).
- `presets.py`: starter equations. Their shapes always come from `default_shapes`.
- `app.py`: the Streamlit script. It validates with `einsum_core` first. Then it builds
  integer tensors and calls `torch.einsum` with the equation exactly as typed.

Rules that hold across these files:

- `einsum_core` copies torch.einsum's rules, checked against torch 2.14:
  - A size-1 axis broadcasts across operands, but not along a letter repeated inside one
    operand (`ii` needs equal sizes).
  - With no `->`, the output keeps letters that appear exactly once, sorted by character
    code, so uppercase comes first (`aB` gives `Ba`).
  - Whitespace is ignored.

  Before changing a rule, confirm the behavior by running torch.
- `test_loop_code_computes_the_same_result_as_torch` runs the generated loop code with
  `exec` and compares it to `torch.einsum`. It covers every preset plus edge cases, so a new
  preset is tested automatically. This test is the proof that the displayed explanation is
  correct.
- Operands are named `x0`, `x1`, ... everywhere (loop code, error messages, UI labels) via
  `operand_name`. Multi-character names can't clash with the single-letter loop variables.
- The order of `Analysis.indices` is the loop nesting order: output letters in output
  order, then summed letters in order of first appearance.
- `...` (ellipsis) is rejected on purpose rather than half-supported. Adding it means
  changing `parse_equation`, `analyze`, `loop_code`, and `default_shapes` together.

How `app.py` is laid out and how its state flows:

- The user runs it with PyCharm's Run button, which runs plain `python app.py`. The block
  at the top that calls `get_script_run_ctx` relaunches the file under `streamlit run`.
  Keep it.
- The user views the app on a phone. Keep all controls and the preset note on the page, in
  the top row of `st.columns`. Don't use `st.sidebar`: on narrow screens it collapses into a
  hidden menu.

- Widgets have keys: `equation`, `preset`, `shape_0`, `shape_1`, .... Picking a preset or
  committing a new equation runs an `on_change` callback. It rewrites the shape boxes from
  `default_shapes`. Editing a shape box does not, so users can try mismatches and broadcasting.
- Streamlit deletes a keyed widget's state on any run where the widget isn't drawn, for
  example after an equation error calls `st.stop()`. So the shape keys are refilled with
  `setdefault` before the shape boxes are drawn.
- `MAX_ELEMENTS` and `MAX_LOOP_ITERATIONS` only gate building the tensors. The letter
  table and loop code still render for huge shapes.
- `tests/test_app.py` runs the real script headless with `streamlit.testing.v1.AppTest`.

## Environment

- Torch comes from the `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`), set
  up in `pyproject.toml` with `explicit = true`. This machine has no NVIDIA GPU, and PyPI's
  Linux torch wheel pulls several GB of `nvidia-*` CUDA packages. After changing
  dependencies, check that `uv.lock` has no `nvidia-` packages.
- ROS 2 Jazzy is sourced in the user's shell, so `/opt/ros/jazzy/...` is on `PYTHONPATH`.
  pytest runs with `--disable-plugin-autoload` so ROS's pytest plugins aren't loaded.
- Dependencies are pinned with `==` in `pyproject.toml`. Pin new ones the same way.
