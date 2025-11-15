## Summary

[What infrastructure/tooling/config change does this MR introduce?  
Why is it needed for our GIS + REST + PyTorch backend?]

## Scope of Change

[Briefly list what you touched: CI jobs, Docker image, deployment config, UV/Ruff settings, storage, etc.]

## Changes

[Bullet list of edits.]

- Update GitLab CI step ___
- Adjust Dockerfile / compose service ___
- Change `uv` / `ruff` settings ___
- Add/remove environment variable or secret ___
- Improve deployment or data prep script ___

## Rationale

[Why this approach helps the project.]

## How to Test

[Steps reviewers can run.]

- `uv sync`
- `uv run pytest`
- `uv run ruff check`
- `docker compose up ___` or `docker build .`
- Any other command/script relevant to the change (list it)

## Checklist

- [ ] GitLab pipeline (or equivalent local run) passes.
- [ ] `uv run pytest` passes when applicable.
- [ ] `uv run ruff check` passes when applicable.
- [ ] New env vars/secrets documented and shared.
- [ ] Docs updated if commands/config changed.

## Dependencies

[Other MRs, services, datasets needed. If none, write N/A.]

- Requires MR #___ / service ___ / dataset ___

## Notes

[Manual steps, rollout plan, or follow-ups. If none, write N/A.]

## Issues

- Closes #_
- Related to #_
