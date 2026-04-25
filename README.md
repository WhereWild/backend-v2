# WhereWild Backend v2

## Getting started

```bash
./gt.sh
```

Builds the container if needed and drops you into a shell. You may have to install Docker and/or uv, and possibly get a token to pull the image.

## Inside the container

| Command | Description |
|---|---|
| `api` | Start API in background |
| `api-fg` | Start API in foreground with reload |
| `api-stop` | Stop API |
| `pt` | Run tests with coverage |
| `pl` | Lint (ruff) |
| `pp` | Lint + test (pipeline approximation) |
| `ww-help` | Show this command list |

API runs on `http://localhost:8000`
