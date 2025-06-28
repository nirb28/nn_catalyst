from pathlib import Path
from typing import Optional
import pandas as pd
from typing import Tuple
from transformers import AutoTokenizer

def Tokenizator(
    csv_file_path: Path,
    tokenizer: AutoTokenizer,
    smiles_col: str = "smiles",
    selfies_col: str = "selfies",
    output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Tokenize SMILES and SELFIES columns in a CSV file and add token lists and lengths.

    Args:
        csv_file_path: Path to the input CSV containing SMILES and SELFIES columns
        tokenizer: A Hugging Face tokenizer to apply to each string
        smiles_col: Column name for SMILES strings
        selfies_col: Column name for SELFIES strings
        output_path: Optional path to save the updated CSV

    Returns:
        pd.DataFrame: DataFrame with added tokenization columns

    Raises:
        FileNotFoundError: If csv_file_path does not exist
        ValueError: If specified columns are missing in the CSV
    """
    # --- Path and file definitions ---
    csv_input = Path(csv_file_path)
    # ------------------------------------

    # --- Validation checks ---
    if not csv_input.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_input}")
    # ------------------------------------

    # --- Data loading ---
    df = pd.read_csv(csv_input)
    print(f"Loaded CSV from {csv_input}, shape: {df.shape}")
    # ------------------------------------

    # --- Column existence checks ---
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in CSV file.")
    if selfies_col not in df.columns:
        raise ValueError(f"Column '{selfies_col}' not found in CSV file.")
    # ------------------------------------

    # --- Tokenization ---
    df['SMILES_Tokenized'] = df[smiles_col].apply(
        lambda x: tokenizer.tokenize(str(x))
    )
    df['SMILES_Token_len'] = df['SMILES_Tokenized'].apply(len)

    df['SELFIES_Tokenized'] = df[selfies_col].apply(
        lambda x: tokenizer.tokenize(str(x))
    )
    df['SELFIES_Token_len'] = df['SELFIES_Tokenized'].apply(len)
    print("Tokenization complete.")
    # ------------------------------------

    # --- Save if path provided ---
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Updated CSV saved to {output_path}")
    # ------------------------------------

    return df

def split_tokenized_dataset(
        tokenized_df: pd.DataFrame,
        selfie_threshold: int = 200,
        smile_threshold: int = 150,
        output_dir: Path = Path.cwd() / "All_Data"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split tokenized DataFrame based on SELFIES and SMILES token length thresholds.

    Args:
        tokenized_df: DataFrame containing tokenized SELFIES and SMILES
        selfie_threshold: Maximum allowed SELFIES token length
        smile_threshold: Maximum allowed SMILES token length
        output_dir: Directory to save split datasets

    Returns:
        Tuple containing:
            - Refined DataFrame: Entries meeting both threshold conditions
            - Unsuited DataFrame: Entries exceeding either threshold

    Raises:
        ValueError: If required columns are missing in the DataFrame
        FileNotFoundError: If output directory creation fails
    """
    # --- Validation checks ---
    required_columns = ['SELFIES_Token_len', 'SMILES_Token_len']
    missing_columns = [
        col for col in required_columns
        if col not in tokenized_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing_columns}"
        )
    # ------------------------------------

    # --- Create output directory ---
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Failed to create output directory: {output_dir}"
        )
    # ------------------------------------

    # --- Apply filtering conditions ---
    condition = (
            (tokenized_df['SELFIES_Token_len'] <= selfie_threshold) &
            (tokenized_df['SMILES_Token_len'] <= smile_threshold)
    )

    refined_df = tokenized_df[condition].copy()
    unsuited_df = tokenized_df[~condition].copy()
    # ------------------------------------

    # --- Print statistics ---
    total_count = len(tokenized_df)
    refined_count = len(refined_df)
    unsuited_count = len(unsuited_df)

    print("\nDataset splitting statistics:")
    print(f"Total entries: {total_count}")
    print(f"Refined entries: {refined_count} ({refined_count / total_count * 100:.2f}%)")
    print(f"Unsuited entries: {unsuited_count} ({unsuited_count / total_count * 100:.2f}%)")
    # ------------------------------------

    # --- Save split datasets ---
    refined_path = output_dir / "Refined_Dataset.csv"
    unsuited_path = output_dir / "Unsuited_Dataset.csv"

    refined_df.to_csv(refined_path, index=False)
    unsuited_df.to_csv(unsuited_path, index=False)

    print("\nSaved datasets:")
    print(f"Refined dataset: {refined_path}")
    print(f"Unsuited dataset: {unsuited_path}")
    # ------------------------------------

    return refined_df, unsuited_df

Path_to_input_data = Path.cwd() / "All_Data"
Path_to_input_data.mkdir(parents=True, exist_ok=True)
Path_to_Complete_Dataset = Path_to_input_data / "Complete_Dataset.csv"
Tokenizer_Path = (Path.cwd() / "Input_Models" / "chemBERTa_77M_MTR").as_posix()

tokenizer = AutoTokenizer.from_pretrained(
    Tokenizer_Path)

tokenized_df = Tokenizator(
    csv_file_path=Path_to_Complete_Dataset,
    tokenizer=tokenizer,
    smiles_col="smiles",
    selfies_col="selfies",
    output_path=None
)


refined_df, unsuited_df = split_tokenized_dataset(
    tokenized_df=tokenized_df,
    selfie_threshold=165,
    smile_threshold=130
)
# Save the refined DataFrame to a CSV file
refined_df.to_csv(
    Path_to_input_data / "Refined_Dataset.csv",
    index=False
)
# Save the unsuited DataFrame to a CSV file
unsuited_df.to_csv(
    Path_to_input_data / "Unsuited_Dataset.csv",
    index=False
)
print("Refined and unsuited datasets have been saved.")


