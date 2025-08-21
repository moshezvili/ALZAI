"""
Test suite for the FastAPI serving endpoint.
"""
import requests
import json
import time
import pytest


class TestAPI:
    """Test cases for the ML serving API."""
    
    base_url = "http://localhost:8000"
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_single_prediction(self):
        """Test single patient prediction."""
        # High-risk patient data with all required fields
        patient_data = {
            "patient_id": "TEST001",
            "year": 2024,
            "age": 75,
            "gender": "M",
            "bmi": 32.5,
            "systolic_bp": 180,
            "diastolic_bp": 95,
            "cholesterol": 280,
            "glucose": 140,
            "smoking_status": "Current",
            "num_visits": 8,
            "medications_count": 5,
            "lab_abnormal_flag": True,
            "primary_diagnosis": "I10",
            "additional_diagnoses": "E11,I25"
        }
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/predict", json=patient_data)
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        
        assert "patient_id" in data
        assert "probability" in data
        assert "prediction" in data
        assert "timestamp" in data
        
        # Validate probability range
        prob = data["probability"]
        assert 0 <= prob <= 1
        
        print(f"✅ Single prediction: {prob:.4f} probability")
        print(f"⏱️ Response time: {(end_time - start_time) * 1000:.1f}ms")
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        patients = [
            {
                "patient_id": "TEST002",
                "year": 2024,
                "age": 45,
                "gender": "F",
                "bmi": 24.0,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "cholesterol": 180,
                "glucose": 90,
                "smoking_status": "Never",
                "num_visits": 2,
                "medications_count": 0,
                "lab_abnormal_flag": False,
                "primary_diagnosis": "Z00",
                "additional_diagnoses": ""
            },
            {
                "patient_id": "TEST003",
                "year": 2024,
                "age": 65,
                "gender": "M",
                "bmi": 28.0,
                "systolic_bp": 160,
                "diastolic_bp": 90,
                "cholesterol": 240,
                "glucose": 120,
                "smoking_status": "Former",
                "num_visits": 5,
                "medications_count": 3,
                "lab_abnormal_flag": True,
                "primary_diagnosis": "I10",
                "additional_diagnoses": "E11"
            }
        ]
        
        response = requests.post(f"{self.base_url}/predict/batch", json={"patients": patients})
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        
        for pred in data["predictions"]:
            assert "patient_id" in pred
            assert "probability" in pred
            assert "prediction" in pred
            assert 0 <= pred["probability"] <= 1
        
        print(f"✅ Batch prediction: {len(data['predictions'])} patients processed")
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = requests.get(f"{self.base_url}/model/info")
        assert response.status_code == 200
        data = response.json()
        
        assert "model_type" in data
        assert "threshold" in data
        assert "algorithm" in data
        assert "training_config" in data
        
        print(f"✅ Model info: {data['model_type']} ({data['algorithm']})")


if __name__ == "__main__":
    # Run the tests
    tester = TestAPI()
    
    print("🧪 Testing ML Serving API")
    print("=" * 50)
    
    try:
        tester.test_health_check()
        print("✅ Health check passed")
        
        tester.test_single_prediction()
        print("✅ Single prediction passed")
        
        tester.test_batch_prediction()
        print("✅ Batch prediction passed")
        
        tester.test_model_info()
        print("✅ Model info passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Make sure the API server is running:")
        print("   python src/serving/api.py")
