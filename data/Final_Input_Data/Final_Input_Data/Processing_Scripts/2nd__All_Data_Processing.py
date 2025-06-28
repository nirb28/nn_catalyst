from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import selfies as sf


def merge_dataframes_with_nan(
    *dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge multiple DataFrames and identify complete/incomplete records.

    Args:
        *dataframes: DataFrames to merge using 'file_name' as key.
                    First DataFrame serves as the base for merging.

    Returns:
        Tuple containing:
            - merged_df: Combined DataFrame with all records
            - missing_df: DataFrame containing rows with missing values
            - complete_df: DataFrame containing only complete records
    """
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on="file_name",
            how="outer"
        )

    missing_df = merged_df[
        merged_df.isnull().any(axis=1)
    ]
    complete_df = merged_df.dropna()

    return merged_df, missing_df, complete_df


def add_selfies_and_smiles(
    df: pd.DataFrame,
    selfies_folder: Path) -> pd.DataFrame:
    """
    Add SELFIES and SMILES columns to DataFrame.

    Args:
        df: DataFrame containing 'file_name' column for identifying SELFIES files
        selfies_folder: Path to directory containing SELFIES text files

    Returns:
        pd.DataFrame: Copy of input DataFrame with added 'selfies' and 'smiles' columns
    """
    selfies_list: List[Optional[str]] = []
    smiles_list: List[Optional[str]] = []

    for file_name in df["file_name"]:
        selfies_file = selfies_folder / f"{file_name}_selfies.txt"

        if selfies_file.exists():
            with open(selfies_file, "r") as f:
                selfies_content = f.read().strip()
                selfies_list.append(selfies_content)
                try:
                    smiles_list.append(
                        sf.decoder(selfies_content)
                    )
                except Exception:
                    smiles_list.append(None)
        else:
            selfies_list.append(None)
            smiles_list.append(None)

    df = df.copy()
    df["selfies"] = selfies_list
    df["smiles"] = smiles_list

    return df


# --- Path definitions ---
Path_to_current_file = Path.cwd()
Path_to_data_folder = Path_to_current_file / "Preprocessed_Data"
Path_to_data_n = Path_to_data_folder / "data_n.csv"
Path_to_data_o = Path_to_data_folder / "data_o.csv"
Path_to_data_r = Path_to_data_folder / "data_r.csv"
Path_to_data_ddg_ox = Path_to_data_folder / "data_ddg_ox.csv"
Path_to_data_ddg_red = Path_to_data_folder / "data_ddg_red.csv"

# --- Load data into DataFrames ---
data_n = pd.read_csv(Path_to_data_n)
data_o = pd.read_csv(Path_to_data_o)
data_r = pd.read_csv(Path_to_data_r)
data_ddg_ox = pd.read_csv(Path_to_data_ddg_ox)
data_ddg_red = pd.read_csv(Path_to_data_ddg_red)

# --- Merge DataFrames ---
merged_df, missing_df, complete_df = merge_dataframes_with_nan(
    data_n,
    data_o,
    data_r,
    data_ddg_ox,
    data_ddg_red
)

# --- Add SELFIES and SMILES ---
selfies_folder = Path.cwd() / "SELFIES"

merged_df = add_selfies_and_smiles(merged_df, selfies_folder)
missing_df = add_selfies_and_smiles(missing_df, selfies_folder)
complete_df = add_selfies_and_smiles(complete_df, selfies_folder)

# --- Process missing data ---
rows_dropped_from_complete = complete_df[complete_df["selfies"].isnull()]
missing_df = pd.concat(
    [missing_df, rows_dropped_from_complete]
).drop_duplicates().reset_index(drop=True)
complete_df = complete_df.dropna(subset=["selfies"])

# --- Save results ---
Path_to_input_data = Path.cwd() / "All_Data"
Path_to_input_data.mkdir(parents=True, exist_ok=True)

merged_df.to_csv(Path_to_input_data / "Fully_Merged_Dataset.csv", index=False)
missing_df.to_csv(Path_to_input_data / "Incomplete_Dataset.csv", index=False)
complete_df.to_csv(Path_to_input_data / "Complete_Dataset.csv", index=False)

print("Merged DataFrame, Missing DataFrame, and Complete DataFrame with SELFIES and SMILES have been saved.")