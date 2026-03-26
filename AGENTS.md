# WhereWild Back-End Copilot Instructions

## Scope

This repo is the Python backend for WhereWild.

- Core libraries: `util/`
- Data/one-off pipelines: `scripts/`
- API entrypoint: `main.py`
- Large runtime data: `data/` (synced from Backblaze B2, not fully tracked in git. Sometimes mounted from B2 if developer does not have local data.)

## Stack

- Python 3.12+
- FastAPI + Uvicorn
- Pandas / NumPy / PyArrow
- GDAL-backed Docker workflow for GIS dependencies
- Docker image installs dependencies from `requirements.txt` (pip)
- `uv` is retained for host linting convenience (`ruff`), not as the default runtime/test path

## Ground Rules

- Avoid using bold for emphasis in Markdown, even for section labels or warnings.
- Keep changes focused and incremental; avoid broad rewrites.
- Prefer clear, direct logic over indirection.
- Keep functions/files concise; split large files into focused helpers/components when needed.
- Maintain low code smell: avoid redundant wrappers, unclear naming, and comments that explain obvious logic.
- Do not hardcode paths that are environment-specific.
- Preserve existing CLI/script behavior unless explicitly requested.
- When adding/adjusting scripts, keep them compatible with the `pd` alias module style (`python -m scripts.<name>`).

## Setup and Run

### Canonical workflow (agent default: one-off Docker commands)

- For agents, default to non-interactive one-off commands via `docker compose exec -T ...`.
- Use `./gt.sh` only when an interactive shell is explicitly needed for debugging.

### Interactive workflow (developer shell)

- Start GDAL shell: `./gt.sh`
- `gt.sh` starts/uses the `gdal` service and runs `b2-mount` before opening the shell.
- Inside container, use helper commands from `docker/aliases.sh`:
  - API: `api`, `api-fg`, `api-stop`
  - Docs: `docs`, `docs-stop`
  - Scripts: `pd <script>`, `pdb <script>`, `pdbs <script>`, `pdbc <script ...>`
  - Tests: `pt [pytest args...]`
  - B2: `b2-mount`, `b2-umount`, `b2-pull-all`, `b2-pull-sync`, `b2-pull`, `b2-push`, `b2-push-all`, `b2-overwrite-remote`

### Non-interactive host commands for agents

When an agent should run a one-off command from the host shell (without entering `./gt.sh`), use:

- `docker compose up -d gdal`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; api-fg'`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; api-stop'`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; b2-pull <remote/path>'`

Agent guardrail:
- Do not run data-processing scripts (`pd`, `pdb`, `pdbc`) unless the user explicitly asks.

### Host tooling (limited use)

Do not use host Python runtime/tests as the default path (GDAL mismatch risk).
Use host `uv` only for linting unless a task explicitly requires otherwise.

- Host lint: `uv run ruff check .`

## Data and B2

- Shared source of truth is B2 bucket `wherewild-data`.
- Use read-only/mount flows by default; only use write helpers when intentionally publishing data.
- `gt.sh` usually handles initial mount via `b2-mount`.

## Validation Checklist

Before finishing code changes:

1. If API behavior changed, sanity-check with `api-fg` and stop it with `api-stop`.
2. Run tests with `pt` (inside `./gt.sh` or one-off via `docker compose exec -T gdal ...`).
3. If you intentionally used host linting, run `uv run ruff check .`.
4. Keep docs in sync when behavior or workflows change (`README.md`, docs pages).

## Testing Instructions

Preferred testing flow for agents (non-interactive one-off commands):

- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt'`

Speed guidance:

- During active iteration, prefer default `pt` (changed-mode/testmon) for fast feedback.
- Before FINAL wrap up, consider a full validation run with `pt --no-cache`, but prefer to avoid this unless necessary to avoid making the user wait.
- If a full `--no-cache` run is too costly for the current step (as is in many cases), explicitly tell the user and ask them to run it before merge/final sign-off.

Interactive reference (use only when explicitly needed for debugging):

- `./gt.sh` then run `pt` inside the container shell.

Common test commands:

- `pt` (default run: remote mode + changed-mode/testmon)
- `pt --no-cache` (full run; clears cache and disables changed-only mode)
- `pt --no-cov` (run tests without coverage)
- `pt --local` (force local `/workspace/data`)
- `pt --remote` (force `/workspace/.b2-mount`, requires mount)
- `pt tests/api/test_health.py -q` (targeted test run)

One-off host equivalent:

- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt'`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt --no-cache'`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt --no-cov'`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt --local'`
- `docker compose exec -T gdal bash -lc '. /etc/wherewild_aliases.sh; pt --remote'`

## Known Mismatches To Avoid

- Do not assume host `uv`/`pytest` is the default project runtime path.
- Do not suggest host `pytest` as standard unless the task explicitly calls for it.
- Do not assume `uv` manages container dependencies; Docker image uses `requirements.txt`.
- Do not document commands as standard unless they exist in this branch (`gt.sh`, `docker/aliases.sh`, `README.md`).
- Prefer Docker alias commands in instructions unless a change explicitly targets host tooling.

## Common Pitfalls

- Mixing host and container paths (`/workspace/...` is container path).
- Assuming B2 sync happens automatically in all run modes.
- Editing data-processing scripts without validating downstream expectations in `util/`.
- Introducing heavy abstractions where simple functions are sufficient.
