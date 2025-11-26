from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="WhereWild API", version="0.1.0")


@app.get("/health", summary="Simple liveness probe")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
