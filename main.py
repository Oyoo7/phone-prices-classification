from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib
import pandas as pd
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
app = FastAPI(
     title = "Phone class predictor by Raphael Oyoo",
     description= ("This model predicts phone price range based on specs."
                  "RAM is the phones active memory used to run apps simultaneously."
                  "Battery_power is the phones battery capacity."
                  "px_height and width are the phones screen resolutions"
                  "The model was trained with an old generation phones datasets which may appear unrealistic in todays time."
                  "Maximum entries allowed for Ram is 4GB, batery power 2000 Mah, Px_height 1960, px_width 1998 as thats what allowable with the dataset used on the model"
                   ))
     
     

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
# error handler for input values exceeding our trained data limits

@app.exception_handler(RequestValidationError)
async def custom_validation_handler(request, exc):
    field_ranges = {
        "active_memory_MB": "256 and 4000",
        "battery_capacity_MAH": "400 and 2000",
        "height_resolution_pixel": "0 and 1960",
        "Length_resolution_pixel": "400 and 1998"
    }

    errors = []
    for err in exc.errors():
        field = err["loc"][-1]
        range_text = field_ranges.get(field, "the allowed range")
        errors.append(f"{field} must be between {range_text}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input-Please enter a value within the threshhold.",
            "issues": errors
        }
    )

# Only the strongly/weakly correlated features are user-facing
class PhoneSpecs(BaseModel):
    active_memory_MB: int = Field(
        ..., ge =256, le= 4000,
        description= "Ram in megabytes should be between 256 MB to 4000 MB"
    )
    battery_capacity_MAH: int= Field(
        ...,ge= 400, le= 2000,
        description = "Battery capacity has to be between 400 and 2000 Mah"
    )
    height_resolution_pixel: int = Field(
        ...,ge = 0 ,le = 1960,
        description = " screen height resolution has to be between 0 and 1960 pixel"
    )
    Length_resolution_pixel: int =Field(
        ..., ge = 400, le = 1998,
        description = "Width resolution has to be between 400 and 1998 pixel"
    )

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
    try:
        #map our input features to model features
            user_inputs= {
                "ram":specs.active_memory_MB,
                "battery_power": specs.battery_capacity_MAH,
                "px_height": specs.height_resolution_pixel,
                "px_width":specs.Length_resolution_pixel
            }
            # Start with the defaults, then overwrite with the user's actual input
            full_row = default_values.copy()
            full_row.update(user_inputs)
        
            # Build the DataFrame in the exact column order the model expects
            input_df = pd.DataFrame([full_row])[feature_order]
        
            scaled_df = scaler.transform(input_df)
            prediction = model.predict(scaled_df)
            predicted_class = int(prediction[0])
        
            #probability calculation
            probability  = model.predict_proba(scaled_df)[0]
            confidence = round(float(probability[predicted_class])* 100,2)
        
            return {
                "price range class": predicted_class,
                "phone category": price_range_labels[predicted_class],
                "confidence level of prediction (percentage)": confidence
            }
    except Exception as Error:
         return{
              "error":"something went wrong while generating your prediction",
              "details" :str(Error)
         }
        
        


    
#api endpoint health check
@app.get("/health_check")
def apihealthcheckreport():
    return{"status":"server healthy"}