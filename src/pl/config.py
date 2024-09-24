# Training hyperparameters
INPUT_SIZE = 1479
NUM_CLASSES = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "cpu" #"gpu"
DEVICES = [0]
PRECISION = 64