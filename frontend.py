import streamlit as st
import requests 
#Title and description
st.title("Phone Price Range Predictor")
st.write("Enter your phone specs below for price category prediction")

# Disclaimer about training data limitations
st.warning(
    "This model was trained on an older dataset where RAM values range up to "
    "roughly 4GB and battery capacity upto 2000 Mah. Predictions are only "
    "reliable within the input ranges enforced below. "
    "Entering specs matching a modern device specs is intentionally "
    "not possible here, since the model was never trained on data in that range."
)
# Illustrative price reference box
st.markdown("### Price Category Reference in US Dollars (Illustrative)")
st.markdown(
    """
    | Category | Approx. Price Range |
    |---|---|
    | Cheap Phone | 50 – 150 |
    | Lower Midrange | 150 – 350 |
    | Upper Midrange | 350 – 600 |
    | Flagship Phone | 600+ |
    """
)
# Input fields matching our API's PhoneSpecs model
ram_capacity = st.number_input("RAM (MB)", min_value=256, max_value=4000, value=2000)
battery_capacity = st.number_input("Battery Capacity (mAh)", min_value=400, max_value=2000, value=1000)
height_resolution = st.number_input("Screen Height (pixels)", min_value=0, max_value=1960, value=800)
Length_resolution_pixel = st.number_input("Screen Width (pixels)", min_value=400, max_value=1998, value=1200)
#mapping our frontend inputs to backend inputs
if st.button("Predict price category"):
    payload = {
        "active_memory_MB": ram_capacity,
        "battery_capacity_MAH": battery_capacity,
        "height_resolution_pixel":height_resolution,
        "Length_resolution_pixel": Length_resolution_pixel
    }
    #sending payload and response receipt

    response = requests.post("https://phone-prices-classification.onrender.com/predict", json=payload)
    if response.status_code  == 200:
      result = response.json()
      st.success(f"predicted category by Oyoo's regression model is : {result['phone category']}")
      st.write(f"The confidence level of this prediction is:  {result['confidence level of prediction (percentage)']} percent")
    else:
      st.error("Something went wrong-Please check your input values")
