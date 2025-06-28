import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_data_partitions(
    complete_dataset_path: Path,
    output_base_dir: Path,
    num_partitions: int = 20,
    test_size_fraction: float = 0.1,
    val_size_fraction_of_train_val: float = 2/9) -> None:
    """
    Create multiple train/validation/test splits from a complete dataset.

    This function loads the full dataset, then for each partition index from 1 to num_partitions:
      1. Splits off test_size_fraction of data for testing.
      2. Splits the remaining data into training and validation sets, with
         validation size val_size_fraction_of_train_val of the train_val set.
      3. Saves each split to its own CSV under a Partition_{i} directory.

    Args:
        complete_dataset_path: Path to the CSV of the full dataset.
        output_base_dir: Directory under which Partition_1..Partition_N will be created.
        num_partitions: Number of different random splits to create (default: 20).
        test_size_fraction: Fraction of data to reserve for testing (default: 0.1).
        val_size_fraction_of_train_val: Fraction of train_val to reserve for validation (default: 2/9).

    Returns:
        None: Writes CSV files for each split to disk and prints statuses.
    """
    # --- Validate and prepare directories ---
    if not complete_dataset_path.exists():
        raise FileNotFoundError(f"Complete dataset not found: {complete_dataset_path}")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using complete dataset: {complete_dataset_path}")
    # ------------------------------------------

    # --- Load the dataset ---
    complete_df = pd.read_csv(complete_dataset_path)
    print(f"Loaded full dataset, shape: {complete_df.shape}")
    # ------------------------

    # --- Generate partitions ---
    for i in range(1, num_partitions + 1):
        partition_dir = output_base_dir / f"Partition_{i}"
        partition_dir.mkdir(exist_ok=True)

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            complete_df,
            test_size=test_size_fraction,
            random_state=i
        )
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_fraction_of_train_val,
            random_state=i
        )

        # --- Paths for current partition ---
        train_path = partition_dir / "Train_Dataset.csv"
        val_path   = partition_dir / "Val_Dataset.csv"
        test_path  = partition_dir / "Test_Dataset.csv"
        # -----------------------------------

        # --- Save CSVs ---
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Partition {i} created:")
        print(f"  Train -> {train_path}")
        print(f"  Val   -> {val_path}")
        print(f"  Test  -> {test_path}\n")
    # ----------------------------

    print("All partitions have been created successfully!")



# --- Top-level path definitions ---
script_dir            = Path.cwd()
complete_data_file    = script_dir / "All_Data" / "Paper_All_Data_Adjusted.csv"
partitions_output_dir = script_dir / "Input_Data"
# -----------------------------------

create_data_partitions(
    complete_dataset_path=complete_data_file,
    output_base_dir=partitions_output_dir
)
