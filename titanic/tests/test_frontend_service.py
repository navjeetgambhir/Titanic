"""Tests for the HTMX UI routes (/ui/*)."""


# ─── /ui/ ────────────────────────────────────────────────────────────────────

class TestUIIndex:

    def test_index_returns_200(self, client):
        res = client.get("/ui/")
        assert res.status_code == 200

    def test_index_is_html(self, client):
        res = client.get("/ui/")
        assert "text/html" in res.headers["content-type"]

    def test_index_contains_form(self, client):
        res = client.get("/ui/")
        assert b"Single Passenger Prediction" in res.content
        assert b"Bulk Prediction" in res.content

    def test_root_redirects_to_ui(self, client):
        res = client.get("/", follow_redirects=False)
        assert res.status_code in (301, 302, 307, 308)
        assert "/ui" in res.headers["location"]


# ─── /ui/predict/single ──────────────────────────────────────────────────────

class TestUISingle:

    def test_female_returns_survived_html(self, client):
        res = client.post("/ui/predict/single",
                          data={"sex": "female", "age": "29", "fare": "71.28"})
        assert res.status_code == 200
        assert b"Survived" in res.content

    def test_male_returns_not_survived_html(self, client):
        res = client.post("/ui/predict/single",
                          data={"sex": "male", "age": "55", "fare": "7.0"})
        assert res.status_code == 200
        assert b"Not Survive" in res.content

    def test_missing_age_uses_median(self, client):
        res = client.post("/ui/predict/single",
                          data={"sex": "female", "fare": "30.0"})
        assert res.status_code == 200
        assert b"Survived" in res.content or b"Not Survive" in res.content

    def test_invalid_sex_returns_error_html(self, client):
        res = client.post("/ui/predict/single",
                          data={"sex": "robot", "age": "25", "fare": "20.0"})
        assert res.status_code == 200          # HTMX gets 200 with error partial
        assert b"error" in res.content.lower() or b"Invalid" in res.content or b"\u26a0" in res.content


# ─── /ui/predict/bulk ────────────────────────────────────────────────────────

class TestUIBulk:

    def test_valid_csv_returns_table(self, client):
        csv = "female,29,71.28\nmale,35,8.05\nfemale,5,22.00"
        res = client.post("/ui/predict/bulk", data={"csv_data": csv})
        assert res.status_code == 200
        assert b"Total" in res.content
        assert b"Survival Rate" in res.content

    def test_blank_age_fare_uses_median(self, client):
        csv = "female,,\nmale,,"
        res = client.post("/ui/predict/bulk", data={"csv_data": csv})
        assert res.status_code == 200
        assert b"Total" in res.content

    def test_invalid_sex_returns_error_partial(self, client):
        csv = "alien,29,30.0"
        res = client.post("/ui/predict/bulk", data={"csv_data": csv})
        assert res.status_code == 200
        assert b"male" in res.content.lower() or b"error" in res.content.lower()

    def test_non_numeric_age_returns_error(self, client):
        csv = "female,abc,30.0"
        res = client.post("/ui/predict/bulk", data={"csv_data": csv})
        assert res.status_code == 200
        assert b"numbers" in res.content or b"error" in res.content.lower()

    def test_empty_csv_returns_error(self, client):
        res = client.post("/ui/predict/bulk", data={"csv_data": "   "})
        assert res.status_code == 200
        assert b"No valid rows" in res.content or b"error" in res.content.lower()
