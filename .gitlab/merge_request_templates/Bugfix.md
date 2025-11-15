## Summary

[What does this merge request fix?]

## Root Cause

[Why did the bug happen? Reference the REST endpoint, GIS data flow, or PyTorch code involved.]

## Changes

[List the concrete fixes.]

- Fix ___ in REST handler / service
- Correct GIS data processing for ___
- Adjust PyTorch model/training/inference step ___
- Update tests or configs in ___

## Steps to Reproduce

[Explain how someone else can see the bug.]

1. [Set up](../../README.md) with `uv sync`.
2. Start the API (`uv run uvicorn main:app --reload`) or relevant script.
3. Run the failing request/command: `curl ___`, `uv run python scripts/___`, etc.
4. Observe the incorrect behavior/output.

## How to Test

[Show how to prove it is fixed.]

1. Repeat the steps above and confirm the bug is gone.
2. Run automated checks:
    - `uv run pytest tests/<area>`
    - `uv run ruff check`
    - Any GIS data validation or PyTorch notebook/script if required
3. Spot-check REST responses or maps produced by the change.

## Demo

[Screenshots, console output, curl responses, or brief notes. If not needed, say N/A.]

## Checklist

- [ ] Added/updated tests cover the regression.
- [ ] `uv run pytest` passes.
- [ ] `uv run ruff check` passes.
- [ ] REST contract (request/response fields) is unchanged or documented.
- [ ] GIS datasets or PyTorch artifacts updated if they changed.

## Dependencies

[Other work this relies on. If none, write N/A.]

- Requires MR #___ / dataset ___ / model artifact ___

## Notes

[Edge cases, manual steps, or follow-ups. If none, write N/A.]

## Issues

- Fixes #_
- Related to #_
