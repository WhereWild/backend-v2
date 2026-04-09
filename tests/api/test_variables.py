"""Tests for GET /variables"""


def test_variables_returns_200(client):
    r = client.get("/variables")
    assert r.status_code == 200


def test_variables_returns_list(client):
    body = client.get("/variables").json()
    assert isinstance(body, list)


def test_variables_count_reasonable(client):
    body = client.get("/variables").json()
    assert len(body) >= 10, "Expected at least 10 environmental variables"


def test_variables_each_has_required_fields(client):
    body = client.get("/variables").json()
    for var in body:
        assert "id" in var, f"Variable missing 'id': {var}"
        assert "name" in var, f"Variable missing 'name': {var}"
        assert "value_type" in var, f"Variable missing 'value_type': {var}"


def test_variables_value_types_are_known(client):
    body = client.get("/variables").json()
    known_types = {"numeric", "categorical", "circular"}
    for var in body:
        vtype = (var.get("value_type") or "").lower()
        assert vtype in known_types, f"Unexpected value_type '{vtype}' for {var.get('id')}"


def test_variables_ids_are_unique(client):
    body = client.get("/variables").json()
    ids = [v["id"] for v in body]
    assert len(ids) == len(set(ids)), "Duplicate variable IDs found"


def test_variables_metric_unit_system_accepted(client):
    r = client.get("/variables?unit_system=metric")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_variables_imperial_unit_system_accepted(client):
    r = client.get("/variables?unit_system=imperial")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
