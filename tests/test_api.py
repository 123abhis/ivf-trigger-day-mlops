# from fastapi.testclient import TestClient
# from app.api import app

# client = TestClient(app)

# def test_health():
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json()["status"] == "ok"

# def test_predict():
#     payload = {
#         "age": 30,
#         "amh_ng_ml": 3.2,
#         "cycle_day": 10,
#         "avg_follicle_size_mm": 18.5,
#         "follicle_count": 12,
#         "estradiol_pg_ml": 2200,
#         "progesterone_ng_ml": 0.9,
#         "bmi": 22.4,
#         "basal_lh_miu_ml": 6.1,
#         "afc": 14,
#         "cluster_id": 2
#     }

#     response = client.post("/predict", json=payload)
#     assert response.status_code == 200
#     assert "prediction" in response.json()





from urllib import response
from fastapi.testclient import TestClient
from app.api import app
from unittest.mock import patch
import numpy as np

client = TestClient(app)

@patch("app.load_model.get_model")
def test_health(mock_model):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@patch("app.load_model.get_model")
def test_predict(mock_model):
    mock_model.return_value.predict.return_value = np.array([1])

    payload = {
        "age": 30,
        "amh_ng_ml": 3.2,
        "cycle_day": 10,
        "avg_follicle_size_mm": 18.5,
        "follicle_count": 12,
        "estradiol_pg_ml": 2200,
        "progesterone_ng_ml": 0.9,
        "bmi": 22.4,
        "basal_lh_miu_ml": 6.1,
        "afc": 14
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    print(response.json())
