"""Tests for the database persistence layer (/predictions/*)."""
import pytest
from fastapi.testclient import TestClient


# ─── Helpers ─────────────────────────────────────────────────────────────────

SINGLE_FEMALE = {"sex": "female", "age": 29, "fare": 71.28}
SINGLE_MALE   = {"sex": "male",   "age": 35, "fare": 8.05}

BULK_PAYLOAD  = [SINGLE_FEMALE, SINGLE_MALE, {"sex": "female", "age": 5, "fare": 22.0}]


def _make_prediction(client: TestClient, payload=SINGLE_FEMALE):
    res = client.post("/predict/single", json=payload)
    assert res.status_code == 200
    return res.json()


# ─── /predictions/ (list) ────────────────────────────────────────────────────

class TestListPredictions:

    def test_list_returns_200(self, client):
        res = client.get("/predictions/")
        assert res.status_code == 200

    def test_list_schema(self, client):
        res = client.get("/predictions/")
        body = res.json()
        assert "total" in body
        assert "records" in body
        assert isinstance(body["records"], list)

    def test_prediction_is_persisted_after_single_call(self, client):
        before = client.get("/predictions/").json()["total"]
        _make_prediction(client)
        after = client.get("/predictions/").json()["total"]
        assert after == before + 1

    def test_bulk_persists_all_records(self, client):
        before = client.get("/predictions/").json()["total"]
        client.post("/predict/bulk", json=BULK_PAYLOAD)
        after = client.get("/predictions/").json()["total"]
        assert after == before + len(BULK_PAYLOAD)

    def test_filter_by_survived(self, client):
        _make_prediction(client, SINGLE_FEMALE)   # likely survived=1
        res = client.get("/predictions/?survived=1")
        assert res.status_code == 200
        for r in res.json()["records"]:
            assert r["survived"] == 1

    def test_filter_by_source_api(self, client):
        _make_prediction(client)
        res = client.get("/predictions/?source=api")
        assert res.status_code == 200
        for r in res.json()["records"]:
            assert r["source"] == "api"

    def test_record_fields_present(self, client):
        _make_prediction(client)
        record = client.get("/predictions/").json()["records"][0]
        expected = {"id", "sex", "age", "fare", "survived",
                    "survival_probability", "model_used", "source", "created_at"}
        assert expected.issubset(record.keys())

    def test_pagination_limit(self, client):
        res = client.get("/predictions/?limit=2")
        assert res.status_code == 200
        assert len(res.json()["records"]) <= 2


# ─── /predictions/{id} (get single) ─────────────────────────────────────────

class TestGetPrediction:

    def test_get_by_id_returns_correct_record(self, client):
        _make_prediction(client)
        record_id = client.get("/predictions/").json()["records"][0]["id"]
        res = client.get(f"/predictions/{record_id}")
        assert res.status_code == 200
        assert res.json()["id"] == record_id

    def test_unknown_id_returns_404(self, client):
        res = client.get("/predictions/999999")
        assert res.status_code == 404


# ─── /predictions/{id} (delete) ──────────────────────────────────────────────

class TestDeletePrediction:

    def test_delete_removes_record(self, client):
        _make_prediction(client)
        record_id = client.get("/predictions/").json()["records"][0]["id"]
        del_res = client.delete(f"/predictions/{record_id}")
        assert del_res.status_code == 204
        get_res = client.get(f"/predictions/{record_id}")
        assert get_res.status_code == 404

    def test_delete_unknown_id_returns_404(self, client):
        res = client.delete("/predictions/999999")
        assert res.status_code == 404


# ─── /predictions/stats/summary ──────────────────────────────────────────────

class TestPredictionStats:

    def test_stats_returns_200(self, client):
        res = client.get("/predictions/stats/summary")
        assert res.status_code == 200

    def test_stats_schema(self, client):
        res = client.get("/predictions/stats/summary")
        body = res.json()
        assert "total_predictions" in body
        assert "survived" in body
        assert "not_survived" in body
        assert "survival_rate" in body
        assert "by_source" in body

    def test_stats_counts_match_list(self, client):
        stats = client.get("/predictions/stats/summary").json()
        total = client.get("/predictions/").json()["total"]
        assert stats["total_predictions"] == total

    def test_survival_rate_between_0_and_1(self, client):
        _make_prediction(client)
        rate = client.get("/predictions/stats/summary").json()["survival_rate"]
        assert 0.0 <= rate <= 1.0
