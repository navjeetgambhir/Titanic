"""Tests for the explainability agent (/explain/single and /ui/explain/single)."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


VALID_PAYLOAD = {"sex": "female", "age": 29.0, "fare": 71.28}
VALID_MALE    = {"sex": "male",   "age": 40.0, "fare": 8.05}

_FAKE_EXPLANATION = "This passenger was predicted to survive because of her gender and fare."


def _mock_run_agent(*args, **kwargs):
    """Replace the real Claude API call in tests."""
    return _FAKE_EXPLANATION


# ─── /explain/single (JSON API) ──────────────────────────────────────────────

class TestExplainSingleAPI:

    def test_explain_returns_200(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            res = client.post("/explain/single", json=VALID_PAYLOAD)
        assert res.status_code == 200

    def test_explain_schema(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_PAYLOAD).json()
        assert "survived" in body
        assert "survival_probability" in body
        assert "model_used" in body
        assert "feature_contributions" in body
        assert "explanation" in body

    def test_survived_is_int(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_PAYLOAD).json()
        assert body["survived"] in (0, 1)

    def test_probability_between_0_and_1(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_PAYLOAD).json()
        assert 0.0 <= body["survival_probability"] <= 1.0

    def test_three_feature_contributions(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_PAYLOAD).json()
        assert len(body["feature_contributions"]) == 3

    def test_feature_contribution_schema(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            fc = client.post("/explain/single", json=VALID_PAYLOAD).json()["feature_contributions"][0]
        assert "feature" in fc
        assert "shap_value" in fc
        assert "direction" in fc
        assert fc["direction"] in ("increases_survival", "decreases_survival", "neutral")

    def test_explanation_is_string(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_PAYLOAD).json()
        assert isinstance(body["explanation"], str)
        assert len(body["explanation"]) > 10

    def test_mocked_explanation_content(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_PAYLOAD).json()
        assert body["explanation"] == _FAKE_EXPLANATION

    def test_male_passenger(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            body = client.post("/explain/single", json=VALID_MALE).json()
        assert body["survived"] in (0, 1)

    def test_invalid_sex_returns_422(self, client: TestClient):
        res = client.post("/explain/single", json={"sex": "alien", "age": 29, "fare": 30})
        assert res.status_code == 422

    def test_negative_age_returns_422(self, client: TestClient):
        res = client.post("/explain/single", json={"sex": "female", "age": -1, "fare": 30})
        assert res.status_code == 422


# ─── /ui/explain/single (HTMX) ───────────────────────────────────────────────

class TestExplainUI:

    def test_explain_ui_page_returns_200(self, client: TestClient):
        res = client.get("/ui/explain")
        assert res.status_code == 200
        assert b"Explainer" in res.content

    def test_explain_ui_post_returns_html(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            res = client.post("/ui/explain/single",
                              data={"sex": "female", "age": "29", "fare": "71.28"})
        assert res.status_code == 200
        assert b"html" in res.headers["content-type"].encode() or b"<" in res.content

    def test_explain_ui_contains_survived_badge(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            res = client.post("/ui/explain/single",
                              data={"sex": "female", "age": "29", "fare": "71.28"})
        body = res.text
        assert "Survived" in body or "Did Not Survive" in body

    def test_explain_ui_contains_ai_explanation(self, client: TestClient):
        with patch("routes.frontend_service._run_explain_agent", side_effect=_mock_run_agent):
            res = client.post("/ui/explain/single",
                              data={"sex": "female", "age": "29", "fare": "71.28"})
        assert _FAKE_EXPLANATION in res.text

    def test_explain_ui_missing_age_uses_median(self, client: TestClient):
        with patch("routes.explain_service._run_explain_agent", side_effect=_mock_run_agent):
            res = client.post("/ui/explain/single",
                              data={"sex": "male", "fare": "10"})
        assert res.status_code == 200
