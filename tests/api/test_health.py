"""Tests for GET /health"""
from datetime import datetime, timezone


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_has_status_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_health_has_iso_timestamp(client):
    body = client.get("/health").json()
    ts = body["timestamp"]
    # Must parse as a valid ISO-8601 datetime
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None, "timestamp should be timezone-aware"


def test_health_timestamp_is_recent(client):
    body = client.get("/health").json()
    dt = datetime.fromisoformat(body["timestamp"])
    now = datetime.now(timezone.utc)
    delta = abs((now - dt).total_seconds())
    assert delta < 10, f"timestamp is {delta:.1f}s away from now — clock skew?"
