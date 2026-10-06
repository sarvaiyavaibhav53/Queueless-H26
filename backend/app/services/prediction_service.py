import os
import joblib
import pandas as pd

class PredictionServiceError(Exception):
    pass

class WaitTimePredictionService:
    def __init__(self):
        self.model = None
        self.features_list = [
            "office_id", "service_id", "hour", "day_of_week", "people_ahead",
            "queue_length", "active_counters", "arrival_rate_15m", "counter_load",
            "historical_avg_service_minutes", "service_time_std", "no_show_rate"
        ]
        
    def load_model(self, model_path: str = None):
        if not model_path:
            # default relative to backend if running from backend root
            model_path = os.path.join(os.path.dirname(__file__), "../../../ml/models/random_forest_wait_time_pipeline.joblib")
            
        if not os.path.exists(model_path):
            raise PredictionServiceError(f"Model file not found at {model_path}")
            
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise PredictionServiceError(f"Failed to load model: {str(e)}")

    def predict_wait_time(self, data: dict) -> float:
        if self.model is None:
            raise PredictionServiceError("Model is not loaded.")
            
        try:
            # Construct DataFrame in exact feature order
            input_df = pd.DataFrame([[data[f] for f in self.features_list]], columns=self.features_list)
            prediction = self.model.predict(input_df)[0]
            return float(prediction)
        except Exception as e:
            raise PredictionServiceError(f"Prediction failed: {str(e)}")

# Singleton instance
prediction_service = WaitTimePredictionService()
