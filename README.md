# einsum playground

A Streamlit app for learning `torch.einsum`. Pick a preset or type an equation and the
operand shapes. The app shows:

- the output shape
- each letter's size and role: batch, free (kept), or summed
- the same computation written as nested Python loops
- the operand values and torch's actual result

## Run

```bash
uv sync
uv run streamlit run app.py
```

`uv run python app.py` (or an IDE's Run button) also works: the script relaunches itself
under `streamlit run`. To open it on a phone on the same network, use the "Network URL"
that Streamlit prints.

Torch comes from PyTorch's CPU-only wheel index (set in `pyproject.toml`), so no CUDA
libraries are downloaded.

## Test and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Limits

- `...` (ellipsis) is not supported. Name every axis with a letter.
- Tensors are only built when each one has at most 1,000,000 elements and the naive loop
  runs at most 50,000,000 times. Larger shapes still show the letter table and loop code.
