import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xlsxwriter
import torch.nn.functional as F


# Load the data
descriptors_path = 'descriptors.csv'
targets_path = 'compiled_data.csv'

descriptors_df = pd.read_csv(descriptors_path)
targets_df = pd.read_csv(targets_path)

# Keep only numeric columns
descriptors_numeric = descriptors_df.select_dtypes(include=['number'])
targets_numeric = targets_df.select_dtypes(include=['number'])

# Merge the numeric dataframes on the common label column
numeric_data = pd.merge(descriptors_numeric, targets_numeric, left_on='Label', right_on='mol_num')
numeric_data = numeric_data.drop(columns=['Label', 'mol_num'])

# Separate features and targets
X = numeric_data.iloc[:, :-30]  # Assuming the last 30 columns are targets
y = numeric_data.iloc[:, -30:]

# Apply variance threshold
selector = VarianceThreshold()
X_high_variance = selector.fit_transform(X)

# Convert to numpy arrays
X = X_high_variance
y = y.values

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train = scaler_X.transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the individual model class
class SingleTargetNet(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(SingleTargetNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        self.fc_skip = nn.Linear(512, 256)  
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)
        
        # Skip connection
        x2 += self.fc_skip(x1)
        
        x3 = self.fc3(x2)
        return x3

# Initialize Excel writer
output_path = 'individual_model_predictions_with_plots.xlsx'
writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
workbook = writer.book

# Prepare DataFrames for train, validation, and test predictions
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

r2_scores, rmse_scores, mae_scores = [], [], []

def create_excel_chart(sheet_name, target_index, worksheet, df, start_row, start_col):
    chart = workbook.add_chart({'type': 'scatter'})
    
    observed_col = f'Observed_{target_index}'
    predicted_col = f'Predicted_{target_index}'
    
    chart.add_series({
        'name': f'Target {target_index}',
        'categories': [sheet_name, start_row+1, df.columns.get_loc(observed_col), start_row+df.shape[0], df.columns.get_loc(observed_col)],
        'values': [sheet_name, start_row+1, df.columns.get_loc(predicted_col), start_row+df.shape[0], df.columns.get_loc(predicted_col)],
        'marker': {'type': 'circle', 'size': 5},
        'trendline': {
            'type': 'linear',
            'display_equation': True,
            'display_r_squared': True,
        }
    })
    chart.set_title({'name': f'Parity Plot for Target {target_index}'})
    chart.set_x_axis({'name': 'Observed'})
    chart.set_y_axis({'name': 'Predicted'})
    chart.set_legend({'none': True})
    
    # Make axes square with the same unit ranges on x and y axis
    min_val = min(df[observed_col].min(), df[predicted_col].min())
    max_val = max(df[observed_col].max(), df[predicted_col].max())
    chart.set_x_axis({'min': min_val, 'max': max_val})
    chart.set_y_axis({'min': min_val, 'max': max_val})
    
    worksheet.insert_chart(start_row + df.shape[0] + 2, start_col, chart)
    
    # Calculate metrics
    observed = df[observed_col]
    predicted = df[predicted_col]
    r2 = r2_score(observed, predicted)
    rmse = mean_squared_error(observed, predicted, squared=False)
    mae = mean_absolute_error(observed, predicted)
    
    # Write metrics to Excel
    metrics_start_row = start_row + df.shape[0] + 22
    worksheet.write(metrics_start_row, start_col, f'Target {target_index}')
    worksheet.write(metrics_start_row + 1, start_col + 1, 'R²')
    worksheet.write(metrics_start_row + 1, start_col + 2, r2)
    worksheet.write(metrics_start_row + 2, start_col + 1, 'RMSE')
    worksheet.write(metrics_start_row + 2, start_col + 2, rmse)
    worksheet.write(metrics_start_row + 3, start_col + 1, 'MAE')
    worksheet.write(metrics_start_row + 3, start_col + 2, mae)
    
    return r2, rmse, mae

for target_index in range(y_train.shape[1]):
    # Load the saved model
    model = SingleTargetNet(X_train.shape[1])
    model.load_state_dict(torch.load(f'best_model_target_{target_index}.pth'))
    
    # Make predictions on the train, validation, and test sets
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor).numpy()
        y_val_pred = model(X_val_tensor).numpy()
        y_test_pred = model(X_test_tensor).numpy()

    # Inverse transform the predictions and targets to their original scale
    y_train_pred_orig = scaler_y.inverse_transform(np.concatenate([np.zeros((y_train_pred.shape[0], target_index)), y_train_pred, np.zeros((y_train_pred.shape[0], y_train.shape[1] - target_index - 1))], axis=1))[:, target_index]
    y_val_pred_orig = scaler_y.inverse_transform(np.concatenate([np.zeros((y_val_pred.shape[0], target_index)), y_val_pred, np.zeros((y_val_pred.shape[0], y_val.shape[1] - target_index - 1))], axis=1))[:, target_index]
    y_test_pred_orig = scaler_y.inverse_transform(np.concatenate([np.zeros((y_test_pred.shape[0], target_index)), y_test_pred, np.zeros((y_test_pred.shape[0], y_test.shape[1] - target_index - 1))], axis=1))[:, target_index]

    y_train_orig = scaler_y.inverse_transform(y_train)[:, target_index]
    y_val_orig = scaler_y.inverse_transform(y_val)[:, target_index]
    y_test_orig = scaler_y.inverse_transform(y_test)[:, target_index]

    # Create dataframes for the predictions and actual values
    train_df[f'Observed_{target_index}'] = y_train_orig
    train_df[f'Predicted_{target_index}'] = y_train_pred_orig

    val_df[f'Observed_{target_index}'] = y_val_orig
    val_df[f'Predicted_{target_index}'] = y_val_pred_orig

    test_df[f'Observed_{target_index}'] = y_test_orig
    test_df[f'Predicted_{target_index}'] = y_test_pred_orig

# Write dataframes to Excel sheets
train_df.to_excel(writer, sheet_name='Train', index=False)
val_df.to_excel(writer, sheet_name='Validation', index=False)
test_df.to_excel(writer, sheet_name='Test', index=False)

# Create and insert parity plots for train, validation, and test sets
for target_index in range(y_train.shape[1]):
    r2, rmse, mae = create_excel_chart('Train', target_index, writer.sheets['Train'], train_df, start_row=0, start_col=target_index*9)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    create_excel_chart('Validation', target_index, writer.sheets['Validation'], val_df, start_row=0, start_col=target_index*9)
    create_excel_chart('Test', target_index, writer.sheets['Test'], test_df, start_row=0, start_col=target_index*9)

# Save and close the Excel file
writer.save()

# Calculate and print the average R², RMSE, and MAE for the validation set
avg_r2 = np.mean(r2_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)

print(f"Average R² for Validation Set: {avg_r2}")
print(f"Average RMSE for Validation Set: {avg_rmse}")
print(f"Average MAE for Validation Set: {avg_mae}")

print(f"Predictions and plots written to {output_path}")
