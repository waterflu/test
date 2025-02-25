import streamlit as st
import pandas as pd
import numpy as np

def forecast_spend(daily_spend, trend_factor, adjustment):
    projected_spend = np.sum(daily_spend) * trend_factor + adjustment
    return projected_spend

# Streamlit UI
st.title("Weekly Ad Spend Forecaster")

st.write("Enter your daily spend data and assumptions to estimate weekly spend.")

# Input section
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_spend = []

for day in days:
    value = st.number_input(f"{day} Spend (£)", min_value=0.0, format="%.2f")
    daily_spend.append(value)

trend_factor = st.slider("Week-on-Week Trend Factor (%)", min_value=-50, max_value=50, value=0) / 100 + 1
adjustment = st.number_input("Manual Adjustment (£)", value=0.0, format="%.2f")

# Forecast calculation
if st.button("Calculate Forecast"):
    estimated_spend = forecast_spend(daily_spend, trend_factor, adjustment)
    st.metric(label="Estimated Weekly Spend (£)", value=round(estimated_spend, 2))
