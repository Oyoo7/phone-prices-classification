import joblib
import pandas as pd
model = joblib.load('model.pkl')
scaler =joblib.load('scaler.pkl')
print(model)
print(scaler)
sample_phone = {
    'battery_power': 1500,
    'blue': 1,
    'clock_speed': 2.2,
    'dual_sim': 1,
    'fc': 5,
    'four_g': 1,
    'int_memory': 32,
    'm_dep': 0.5,
    'mobile_wt': 150,
    'n_cores': 4,
    'pc': 8,
    'px_height': 800,
    'px_width': 1200,
    'ram': 2000,
    'sc_h': 12,
    'sc_w': 7,
    'talk_time': 10,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1
}
def phone_price_range(phonespecs: dict):
    input_df = pd.DataFrame([phonespecs])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return int (prediction[0])

print(phone_price_range(sample_phone))