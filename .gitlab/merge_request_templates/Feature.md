## Summary

[What REST/GIS/PyTorch feature does this MR add and why does it matter?]

## Changes

[Bullet list of the main updates.]

- Add REST endpoint ___ or extend existing route ___
- Implement GIS processing step ___
- Train/update PyTorch model ___ or inference logic ___
- Update configuration/tests/docs ___

## Demo

[cURL snippets, screenshots of GIS output, sample tensors, etc. If none, write N/A.]

## How to Test

[Explain how reviewers can verify the feature.]

1. [Set up](../../README.md) with `uv sync`.
2. Start required services/scripts: `uv run uvicorn main:app --reload`, `docker compose up ___`, etc.
3. Run the feature manually:
    - `curl -X POST http://localhost:8000/...`
    - `uv run python scripts/train_<name>.py`
    - CLI or notebook command ___
4. Run automated checks:
    - `uv run pytest tests/<feature>`
    - `uv run ruff check`
5. Confirm outputs (REST responses, GIS layers, PyTorch metrics) look correct.

## Checklist

- [ ] Tests for the new behavior exist and pass.
- [ ] `uv run pytest` passes.
- [ ] `uv run ruff check` passes.
- [ ] API/request/response changes are documented (README or OpenAPI).
- [ ] Updated GIS data or PyTorch artifacts (if any) are stored/linked.

## Dependencies

[Other MRs, datasets, or artifacts needed. If none, say N/A.]

- Requires MR #___ / dataset ___ / model artifact ___

## Notes

[Follow-ups, temporary limits, or manual steps. If none, write N/A.]

## Issues

- Closes #_
- Related to #_
