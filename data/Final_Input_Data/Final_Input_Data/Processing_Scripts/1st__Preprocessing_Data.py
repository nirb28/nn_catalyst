import pandas as pd
from pathlib import Path

# Paths to data files
Path_to_current_file       = Path.cwd()
Path_to_raw_data_folder    = Path_to_current_file / "Raw_Data"
Path_to_raw_data_n         = Path_to_raw_data_folder / "data_n.csv"
Path_to_raw_data_o         = Path_to_raw_data_folder / "data_o.csv"
Path_to_raw_data_r         = Path_to_raw_data_folder / "data_r.csv"
Path_to_raw_data_ddg_ox    = Path_to_raw_data_folder / "ddg_ox.csv"
Path_to_raw_data_ddg_red   = Path_to_raw_data_folder / "ddg_red.csv"

# Load data into DataFrames
data_n         = pd.read_csv(Path_to_raw_data_n)
data_o         = pd.read_csv(Path_to_raw_data_o)
data_r         = pd.read_csv(Path_to_raw_data_r)
data_ddg_ox    = pd.read_csv(Path_to_raw_data_ddg_ox)
data_ddg_red   = pd.read_csv(Path_to_raw_data_ddg_red)

# data_o transformation
data_o['file_name'] = data_o['file_name'].str.extract(r'rxn_(\d+)')[0].astype(int)

def preserve_columns_o(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only the specified columns for the 'o' dataset and reset index.

    Args:
        df: Original DataFrame with detailed columns.

    Returns:
        DataFrame filtered to the desired columns with a reset index.
    """
    # --- Columns to preserve ---
    columns_to_keep = [
        "file_name", "homo_spin_up", "lumo_spin_up",
        "homo_spin_down", "lumo_spin_down",
        "max_charge_pos", "max_charge_neg",
        "max_spin", "dipole", "gibbs", "elec_en"
    ]
    # -----------------------------
    filtered_df = df[columns_to_keep]
    return filtered_df.reset_index(drop=True)

# Preserve only the desired columns
def rename_columns_o(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append '_o' suffix to measurement columns in the 'o' dataset.

    Args:
        df: DataFrame with original column names.

    Returns:
        DataFrame with renamed columns.
    """
    # --- Columns mapping for rename ---
    columns_to_rename = {
        "homo_spin_up": "homo_spin_up_o",
        "lumo_spin_up": "lumo_spin_up_o",
        "homo_spin_down": "homo_spin_down_o",
        "lumo_spin_down": "lumo_spin_down_o",
        "max_charge_pos": "max_charge_pos_o",
        "max_charge_neg": "max_charge_neg_o",
        "max_spin": "max_spin_o",
        "dipole": "dipole_o",
        "gibbs": "gibbs_o",
        "elec_en": "elec_en_o"
    }
    # -----------------------------------
    return df.rename(columns=columns_to_rename)

# Rename and save 'o' data
updated_data_o       = preserve_columns_o(data_o)
renamed_data_o       = rename_columns_o(updated_data_o)
Path_to_processed_o  = Path_to_current_file / "Preprocessed_Data"
Path_to_processed_o.mkdir(parents=True, exist_ok=True)
renamed_data_o.to_csv(Path_to_processed_o / "data_o.csv", index=False)

# data_r transformation

def preserve_columns_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain specified columns for the 'r' dataset and reset index.
    """
    # --- Columns to preserve ---
    columns_to_keep = [
        "file_name", "homo_spin_up", "lumo_spin_up",
        "homo_spin_down", "lumo_spin_down",
        "max_charge_pos", "max_charge_neg",
        "max_spin", "dipole", "gibbs", "elec_en"
    ]
    # -----------------------------
    filtered_df = df[columns_to_keep]
    return filtered_df.reset_index(drop=True)

# Preserve and rename 'r' data
def rename_columns_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append '_r' suffix to measurement columns in the 'r' dataset.
    """
    # --- Columns mapping for rename ---
    columns_to_rename = {
        "homo_spin_up": "homo_spin_up_r",
        "lumo_spin_up": "lumo_spin_up_r",
        "homo_spin_down": "homo_spin_down_r",
        "lumo_spin_down": "lumo_spin_down_r",
        "max_charge_pos": "max_charge_pos_r",
        "max_charge_neg": "max_charge_neg_r",
        "max_spin": "max_spin_r",
        "dipole": "dipole_r",
        "gibbs": "gibbs_r",
        "elec_en": "elec_en_r"
    }
    # -----------------------------------
    return df.rename(columns=columns_to_rename)

updated_data_r       = preserve_columns_r(data_r)
renamed_data_r       = rename_columns_r(updated_data_r)
Path_to_processed_r  = Path_to_current_file / "Preprocessed_Data"
Path_to_processed_r.mkdir(parents=True, exist_ok=True)
renamed_data_r.to_csv(Path_to_processed_r / "data_r.csv", index=False)

# data_n transformation

def preserve_columns_n(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain specified columns for the 'n' dataset and reset index.
    """
    # --- Columns to preserve ---
    columns_to_keep = [
        "file_name", "homo", "lumo",
        "max_charge_pos", "max_charge_neg",
        "dipole", "gibbs", "elec_en"
    ]
    # -----------------------------
    filtered_df = df[columns_to_keep]
    return filtered_df.reset_index(drop=True)

# Preserve and rename 'n' data
def rename_columns_n(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename 'n' dataset columns with '_n' suffix.
    """
    # --- Columns mapping for rename ---
    columns_to_rename = {
        "homo": "homo_n",
        "lumo": "lumo_n",
        "max_charge_pos": "max_charge_pos_n",
        "max_charge_neg": "max_charge_neg_n",
        "dipole": "dipole_n",
        "gibbs": "gibbs_n",
        "elec_en": "elec_en_n"
    }
    # -----------------------------------
    return df.rename(columns=columns_to_rename)

updated_data_n       = preserve_columns_n(data_n)
renamed_data_n       = rename_columns_n(updated_data_n)
Path_to_processed_n  = Path_to_current_file / "Preprocessed_Data"
Path_to_processed_n.mkdir(parents=True, exist_ok=True)
renamed_data_n.to_csv(Path_to_processed_n / "data_n.csv", index=False)

# ddg_ox transformation

def preserve_columns_ddg_ox(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain 'name' and 'delta_delta_g_ox' columns and reset index.
    """
    # --- Columns to preserve ---
    columns_to_keep = ["name", "delta_delta_g_ox"]
    # -----------------------------
    filtered_df = df[columns_to_keep]
    return filtered_df.reset_index(drop=True)

# Preserve and rename 'ddg_ox' data
def rename_columns_redox(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename redox columns to standardized names 'file_name', 'ddg_ox', 'ddg_red'.
    """
    # --- Columns mapping for rename ---
    columns_to_rename = {
        "name": "file_name",
        "delta_delta_g_ox": "ion_pot",
        "delta_delta_g_red": "elec_aff"
    }
    # -----------------------------------
    return df.rename(columns=columns_to_rename)

updated_data_ddg_ox = preserve_columns_ddg_ox(data_ddg_ox)
renamed_data_ddg_ox = rename_columns_redox(updated_data_ddg_ox)
Path_to_processed_ddg_ox = Path_to_current_file / "Preprocessed_Data"
Path_to_processed_ddg_ox.mkdir(parents=True, exist_ok=True)
renamed_data_ddg_ox.to_csv(Path_to_processed_ddg_ox / "data_ddg_ox.csv", index=False)

# ddg_red transformation

def preserve_columns_ddg_red(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain 'name' and 'delta_delta_g_red' columns and reset index.
    """
    # --- Columns to preserve ---
    columns_to_keep = ["name", "delta_delta_g_red"]
    # -----------------------------
    filtered_df = df[columns_to_keep]
    return filtered_df.reset_index(drop=True)

# Preserve and rename 'ddg_red' data
def rename_columns_redox(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename redox columns to standardized names 'file_name', 'ddg_ox', 'ddg_red'.
    """
    # --- Columns mapping for rename ---
    columns_to_rename = {
        "name": "file_name",
        "delta_delta_g_ox": "ion_pot",
        "delta_delta_g_red": "elec_aff"
    }
    # -----------------------------------
    return df.rename(columns=columns_to_rename)

updated_data_ddg_red = preserve_columns_ddg_red(data_ddg_red)
renamed_data_ddg_red = rename_columns_redox(updated_data_ddg_red)
Path_to_processed_ddg_red = Path_to_current_file / "Preprocessed_Data"
Path_to_processed_ddg_red.mkdir(parents=True, exist_ok=True)
renamed_data_ddg_red.to_csv(Path_to_processed_ddg_red / "data_ddg_red.csv", index=False)
