import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "../invoice_flagging/models/predict_flag_invoice.pkl"
SCALER_PATH = Path(__file__).resolve().parent / "../invoice_flagging/models/scaler.pkl"
FEATURES = ['invoice_quantity', 'invoice_dollars', 'Freight', 'total_item_quantity', 'total_item_dollars']


def load_model(model_path: Path = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path: Path = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):
    input_df = pd.DataFrame(input_data)
    missing_columns = set(FEATURES) - set(input_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}")

    model = load_model()
    scaler = load_scaler()

    features_df = input_df[FEATURES].copy()
    scaled_features = scaler.transform(features_df)

    input_df['Predicted_Flag'] = model.predict(scaled_features)
    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10, 5, 3, 12],
        "invoice_dollars": [1200, 800, 200, 1450],
        "Freight": [50, 30, 10, 75],
        "total_item_quantity": [10, 5, 3, 12],
        "total_item_dollars": [1200, 800, 200, 1450]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)
