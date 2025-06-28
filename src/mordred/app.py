import streamlit as st
import pandas as pd
import torch
from mordred import Calculator, descriptors
from rdkit import Chem
import numpy as np

def generate_mordred_descriptors(smiles_list, columns=None):
    """
    Generate Mordred descriptors for a list of SMILES strings.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings to calculate descriptors for
    columns : list, optional
        List of specific columns to include
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing molecular descriptors for each SMILES string
    """
    # Create a calculator with all available descriptors
    calc = Calculator(descriptors)
    
    # If columns are specified, filter the descriptors
    if columns:
        # Filter descriptors to match the specified columns
        filtered_descriptors = [desc for desc in calc.descriptors if str(desc) in columns]
        calc = Calculator(filtered_descriptors)
    
    # Prepare results
    results = []
    
    # Calculate descriptors for each SMILES string
    for smiles in smiles_list:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # Calculate descriptors
            try:
                desc_values = calc(mol)
                # Convert to dictionary, adding SMILES as first column
                desc_dict = {'SMILES': smiles, **dict(desc_values)}
                results.append(desc_dict)
            except Exception as e:
                print(f"Error calculating descriptors for {smiles}: {e}")
        else:
            print(f"Invalid SMILES string: {smiles}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure all specified columns are present, fill with NaN if missing
    # if columns:
    #     for col in columns:
    #         if col not in df.columns:
    #             df[col] = np.nan
        
    #     # Reorder columns to match the original specification
    #     df = df[['SMILES'] + [col for col in columns if col != 'SMILES']]
    
    return df

def read_descriptors(file_path):
    """
    Read the descriptors from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file containing column names
    
    Returns:
    --------
    list
        List of column names
    """
    with open(file_path, 'r') as f:
        # Read the first line and split by tab
        columns = f.readline().strip().split('\t')
    return columns


# # Function to make predictions
# def predict(model, descriptors_df):
#     with torch.no_grad():
#         X = torch.tensor(descriptors_df.values, dtype=torch.float32)
#         predictions = model(X)
#     return predictions

# # Load the model
# model = load_model()

# Streamlit UI
st.title('SMILES to Descriptors and Prediction')

# Input SMILES string
smiles_input = st.text_input('Enter SMILES string:', 'CC(=O)OC1=CC=CC=C1C(=O)O')

def process_smiles(smiles_list):
    """
    Process the input SMILES string.
    
    Returns:
    --------
    str
        Processed SMILES string
    """
    # Remove leading/trailing whitespace
    smiles_list = [smiles.strip() for smiles in smiles_list]
    descriptors_df = generate_mordred_descriptors(smiles_list)
    
    features_file_path = 'src/mordred/descriptors.txt'
    features = read_descriptors(features_file_path)
    
    descriptors_df = generate_mordred_descriptors(smiles_list, features)
    non_numeric_columns = descriptors_df.select_dtypes(exclude=[np.number]).columns
    print("Non-numeric columns in descriptors_df:", non_numeric_columns.tolist())
    filled_descriptors_df = descriptors_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return filled_descriptors_df

def predict(model, X):
    predictions = []
    for target_index in range(0,29):
        predictions.append(eval_model(X, target_index+1))
        
    # Stack predictions into array
    predictions = np.hstack(predictions)
    return predictions

def eval_model(X_data, target_num):
    X_scaled = scaler.transform(X_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    checkpoint_path=f'{CHECKPOINTS_FOLDER}/{target_num}'
    pathlist = Path(checkpoint_path).glob('**/*.ckpt')
    for path in pathlist:
        # because path is object not string
        model = SingleTargetNet.load_from_checkpoint(str(path))
        model.eval()
        model.cpu()
        with torch.no_grad():
                y_pred = model(X_tensor)
        return y_pred.detach().numpy()

def load_model():
    """
    Load the model from the checkpoint file.
    
    Returns:
    --------
    torch.nn.Module
        Model loaded from the checkpoint file
    """
    # Define the path to the checkpoint file
    checkpoint_path = f'{CHECKPOINTS_FOLDER}/model.ckpt'

    # Load the model from the checkpoint file
    model = SingleTargetNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

# Define the path to the pickle file
pickle_file_path = f'{CHECKPOINTS_FOLDER}/scaler_X.pkl'

# Load the StandardScaler from the pickle file
with open(pickle_file_path, 'rb') as file:
    scaler = joblib.load(file)

# Now you can use the scaler
# Example: scaler.transform(data)

# Standardize features
X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

if smiles_input:
    # Convert SMILES to descriptors
    smiles_list = [smiles_input]
    filled_descriptors_df = process_smiles(smiles_list)

    # Display descriptors
    st.write('Descriptors:')
    st.write(str(filled_descriptors_df))
    
    # # Perform prediction
    predictions = predict(model, descriptors_df)
    
    # # Display predictions
    st.write('Predictions for 29 targets:')
    st.write(predictions.numpy())

# Run the Streamlit app
if __name__ == '__main__':
    smiles_list = ['CCO', 'CCN', 'CCOCC']
    process_smiles(smiles_list)

    