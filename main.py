from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Only the strongly/weakly correlated features are user-facing
class PhoneSpecs(BaseModel):
    ram: int
    battery_power: int
    px_height: int
    px_width: int

# Defaults for near-zero-correlation features, based on training data medians
default_values = {
    "blue": 0,
    "clock_speed": 1.5,
    "dual_sim": 1,
    "fc": 3,
    "four_g": 1,
    "int_memory": 32,
    "m_dep": 0.5,
    "mobile_wt": 141,
    "n_cores": 4,
    "pc": 10,
    "sc_h": 12,
    "sc_w": 5,
    "talk_time": 11,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

price_range_labels = {
    0: "Cheap Phone",
    1: "Lower Midrange",
    2: "Upper Midrange",
    3: "Flagship Phone"
}

# The exact column order your model was trained on
feature_order = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]

@app.post("/predict")
def phone_price_prediction(specs: PhoneSpecs):
    # Start with the defaults, then overwrite with the user's actual input
    full_row = default_values.copy()
    full_row.update(specs.model_dump())

    # Build the DataFrame in the exact column order the model expects
    input_df = pd.DataFrame([full_row])[feature_order]

    scaled_df = scaler.transform(input_df)
    prediction = model.predict(scaled_df)
    predicted_class = int(prediction[0])

    return {
        "price range class": predicted_class,
        "phone category": price_range_labels[predicted_class]
    }