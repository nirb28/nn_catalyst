import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xlsxwriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW


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

# Check the shapes of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Apply variance threshold
selector = VarianceThreshold()
X_high_variance = selector.fit_transform(X)

# Convert to numpy arrays
X = X_high_variance
y = y.values

# Check the shapes of X and y after variance thresholding
print(f"Shape of X after variance thresholding: {X.shape}")
print(f"Shape of y after variance thresholding: {y.shape}")

# Ensure there are enough samples to split
if len(X) < 3:
    raise ValueError("Not enough samples to split into training, validation, and test sets.")

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

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the individual model class
class SingleTargetNet(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(SingleTargetNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        self.fc_skip = nn.Linear(1024, 512)  
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

# Function to train and evaluate individual models
def train_and_evaluate(target_index):
    model = SingleTargetNet(X_train.shape[1])
    
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    best_val_loss = np.inf
    patience_counter = 0
    num_epochs = 150
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets[:, target_index])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Target {target_index} - Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets[:, target_index])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Target {target_index} - Validation Loss: {val_loss}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_target_{target_index}.pth')
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f'Target {target_index} - Early stopping triggered')
                break
    
    model.load_state_dict(torch.load(f'best_model_target_{target_index}.pth'))
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets[:, target_index])
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Target {target_index} - Test Loss: {test_loss}')
    
    return model, test_loss

# Train and evaluate individual models for each target
test_losses = []
models = []
for target_index in range(y_train.shape[1]):
    model, test_loss = train_and_evaluate(target_index)
    models.append(model)
    test_losses.append(test_loss)

# Print average test loss
avg_test_loss = np.mean(test_losses)
print(f'Average Test Loss for Individual Models: {avg_test_loss}')

# Initialize Excel writer
output_path = 'individual_model_predictions_with_plots.xlsx'
writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
workbook = writer.book

# Create necessary Excel sheets
train_sheet = workbook.add_worksheet('Train')
val_sheet = workbook.add_worksheet('Validation')
test_sheet = workbook.add_worksheet('Test')

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
    model = models[target_index]
    
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

    # Create and insert parity plots for train, validation, and test sets
    r2, rmse, mae = create_excel_chart('Train', target_index, train_sheet, train_df, start_row=0, start_col=target_index*9)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    create_excel_chart('Validation', target_index, val_sheet, val_df, start_row=0, start_col=target_index*9)
    create_excel_chart('Test', target_index, test_sheet, test_df, start_row=0, start_col=target_index*9)

# Write dataframes to Excel sheets
train_df.to_excel(writer, sheet_name='Train', index=False)
val_df.to_excel(writer, sheet_name='Validation', index=False)
test_df.to_excel(writer, sheet_name='Test', index=False)

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
