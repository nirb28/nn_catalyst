import pandas as pd
from pathlib import Path


def process_dataframe(dataframe_path):
    """
    Reads a dataframe from a given path, preserves specified columns,
    and returns a new dataframe.

    Args:
        dataframe_path (str): The path to the input dataframe file (e.g., CSV).

    Returns:
        pandas.DataFrame: A new dataframe with only the preserved columns.
    """
    # Read the original dataframe
    Refined_Data = pd.read_csv(dataframe_path)

    # Define the columns to preserve
    columns_to_preserve = ["file_name",
                           "ion_pot",
                           "elec_aff",
                           "homo_n",
                           "lumo_n",
                           "smiles",
                           "selfies"]

    # Create the new dataframe with only the specified columns
    Paper_All_Data = Refined_Data[columns_to_preserve]

    return Paper_All_Data


def adjust_dft_values(df):
    """
    Adjusts the 'ion_pot' and 'elec_aff' columns of a dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The dataframe with adjusted values.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_adjusted = df.copy()

    # Add +4.44 to the "ion_pot" column
    df_adjusted["ion_pot"] = df_adjusted["ion_pot"] + 4.44
    print("Added +4.44 to 'ion_pot' column.")

    # Add -4.44 from the "elec_aff" column
    df_adjusted["elec_aff"] = df_adjusted["elec_aff"] - 4.44
    print("Subtracted -4.44 from 'elec_aff' column.")

    return df_adjusted


# Define the input path for the dataset
input_path = Path.cwd() / "All_Data" / "Refined_Dataset.csv"


# Step 1: Create the initial dataframe with preserved columns
Paper_All_Data = process_dataframe(input_path)
print("\nOriginal 'Paper_All_Data' dataframe created successfully:")
print(Paper_All_Data.head())

# Step 2: Adjust the values in the 'ion_pot' and 'elec_aff' columns
Final_Paper_Data = adjust_dft_values(Paper_All_Data)
print("\nDataframe after adjusting values:")
print(Final_Paper_Data.head())

# Step 3: Save the final dataframe to a new CSV file
output_path = Path.cwd() / "All_Data" / "Paper_All_Data_Adjusted.csv"
Final_Paper_Data.to_csv(output_path, index=False)

print(f"\nFinal dataframe saved successfully to:\n{output_path}")
