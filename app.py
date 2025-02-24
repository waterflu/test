import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

# Streamlit UI setup
st.title("Google Ads tROAS Spend Predictor")
st.write("Estimate how spend changes when adjusting target ROAS.")

# Sample data (replace with real data)
data = {
    'tROAS': [12.96, 12.47, 12.01, 11.94, 11.64, 12.53],
    'Spend': [537570, 551493, 601826, 632184, 605314, 560778]
}
df = pd.DataFrame(data)

# Log transformation
df['log_tROAS'] = np.log(df['tROAS'])
df['log_Spend'] = np.log(df['Spend'])

# Fit log-linear model
X = sm.add_constant(df['log_tROAS'])
y = df['log_Spend']
model = sm.OLS(y, X).fit()

# Prediction function
def predict_spend(new_tROAS):
    new_log_tROAS = np.log(new_tROAS)
    pred_log_spend = model.params[0] + model.params[1] * new_log_tROAS
    return np.exp(pred_log_spend)

# User input for tROAS
new_tROAS = st.slider("Select new tROAS value", min_value=1.0, max_value=15.0, step=0.01, value=2.5)
predicted_spend = predict_spend(new_tROAS)

# Display results
st.write(f"### Predicted Spend: £{predicted_spend:,.2f}")

# Show historical data
st.write("#### Historical Data")
st.dataframe(df[['tROAS', 'Spend']])
