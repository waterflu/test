import numpy as np
import pandas as pd
import statsmodels.api as sm

# -------------------------------
# Step 1: Input your historical data
# Replace these sample values with your actual data.
# For example, 'tROAS' represents your target ROAS and 'Spend' is the associated spend.
data = {
    'tROAS': [12.96, 12.47, 12.01, 11.94, 11.64],
    'Spend': [537570, 551493, 601826, 632184, 605314]
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Create log-transformed columns
df['log_tROAS'] = np.log(df['tROAS'])
df['log_Spend'] = np.log(df['Spend'])

# -------------------------------
# Step 3: Fit the log-linear model
# The model is: log(Spend) = beta0 + beta1 * log(tROAS) + error
X = sm.add_constant(df['log_tROAS'])
y = df['log_Spend']
model = sm.OLS(y, X).fit()
print(model.summary())

# -------------------------------
# Step 4: Create a function to predict Spend based on new tROAS values
def predict_spend(new_tROAS):
    """
    Predicts the spend for a given tROAS using the fitted log-linear model.
    
    Parameters:
        new_tROAS (float or array-like): New tROAS value(s) to predict spend for.
    
    Returns:
        Predicted spend value(s).
    """
    # Calculate the log(tROAS) for the new value(s)
    new_log_tROAS = np.log(new_tROAS)
    
    # Calculate predicted log(Spend) using the model: beta0 + beta1 * log(tROAS)
    pred_log_spend = model.params[0] + model.params[1] * new_log_tROAS
    
    # Transform back from log(Spend) to Spend
    pred_spend = np.exp(pred_log_spend)
    return pred_spend

# -------------------------------
# Example usage:
new_tROAS = 12.68  # Change this to the new tROAS you want to test
predicted_spend = predict_spend(new_tROAS)
print(f"Predicted spend for tROAS = {new_tROAS}: {predicted_spend:.2f}")