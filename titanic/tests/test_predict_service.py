"""Tests for the JSON prediction API (/predict/*)."""


# ─── /predict/single ─────────────────────────────────────────────────────────

class TestPredictSingle:

    def test_female_high_fare_survives(self, client):
        res = client.post("/predict/single", json={"sex": "female", "age": 29, "fare": 71.28})
        assert res.status_code == 200
        body = res.json()
        assert body["survived"] == 1
        assert 0.0 <= body["survival_probability"] <= 1.0
        assert "model_used" in body

    def test_male_low_fare_does_not_survive(self, client):
        res = client.post("/predict/single", json={"sex": "male", "age": 35, "fare": 8.05})
        assert res.status_code == 200
        body = res.json()
        assert body["survived"] == 0
        assert 0.0 <= body["survival_probability"] <= 1.0

    def test_response_schema(self, client):
        res = client.post("/predict/single", json={"sex": "female", "age": 22, "fare": 15.0})
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"survived", "survival_probability", "model_used"}
        assert body["survived"] in (0, 1)

    def test_invalid_sex_returns_422(self, client):
        res = client.post("/predict/single", json={"sex": "unknown", "age": 29, "fare": 32.0})
        assert res.status_code == 422

    def test_negative_age_returns_422(self, client):
        res = client.post("/predict/single", json={"sex": "female", "age": -1, "fare": 32.0})
        assert res.status_code == 422

    def test_missing_field_returns_422(self, client):
        res = client.post("/predict/single", json={"sex": "female", "age": 29})
        assert res.status_code == 422


# ─── /predict/bulk ───────────────────────────────────────────────────────────

class TestPredictBulk:

    def test_bulk_returns_all_records(self, client):
        payload = [
            {"sex": "female", "age": 29, "fare": 71.28},
            {"sex": "male",   "age": 35, "fare": 8.05},
            {"sex": "female", "age": 5,  "fare": 22.00},
        ]
        res = client.post("/predict/bulk", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["total_records"] == 3
        assert len(body["predictions"]) == 3
        assert 0.0 <= body["survival_rate"] <= 1.0

    def test_bulk_each_record_has_schema(self, client):
        payload = [{"sex": "male", "age": 40, "fare": 10.0}]
        res = client.post("/predict/bulk", json=payload)
        pred = res.json()["predictions"][0]
        assert set(pred.keys()) == {"survived", "survival_probability", "model_used"}

    def test_bulk_survival_rate_matches_predictions(self, client):
        payload = [
            {"sex": "female", "age": 28, "fare": 50.0},
            {"sex": "male",   "age": 45, "fare": 7.0},
        ]
        res  = client.post("/predict/bulk", json=payload)
        body = res.json()
        calculated = sum(p["survived"] for p in body["predictions"]) / body["total_records"]
        assert abs(body["survival_rate"] - calculated) < 1e-4

    def test_empty_list_returns_422(self, client):
        res = client.post("/predict/bulk", json=[])
        assert res.status_code == 422

    def test_invalid_record_in_bulk_returns_422(self, client):
        payload = [{"sex": "alien", "age": 29, "fare": 10.0}]
        res = client.post("/predict/bulk", json=payload)
        assert res.status_code == 422
