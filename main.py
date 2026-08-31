from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
app = FastAPI()
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
class PhoneSpecs(BaseModel):
    battery_power: int
    blue:int
    clock_speed:float
    dual_sim:int
    fc:int
    four_g:int
    int_memory:int
    m_dep:float
    mobile_wt:int
    n_cores:int
    pc:int
    px_height:int
    px_width:int
    ram:int
    sc_h:int
    sc_w:int
    talk_time:int
    three_g:int
    touch_screen:int
    wifi:int
    

@app.post("/predict")
def phone_price_prediction(specs:PhoneSpecs):
    input_df = pd.DataFrame([specs.model_dump()])
    scaled_df= scaler.transform(input_df)
    prediction = model.predict(scaled_df)
    return f"predicted price range is:\n{int(prediction[0])}"


