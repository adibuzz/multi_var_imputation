import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def generate_folds(csv_path, n_folds=5, test_size=0.1, val_size=0.2, seed=432, simulation_name='mimic'):
    N = pd.read_csv(csv_path)["ID"].nunique()
    indices = np.arange(N)
    np.random.seed(seed)

    for fold in range(n_folds):
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed + fold)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=seed + fold)
        fold_dir = f"{simulation_name}_fold_idx_{fold}/"
        os.makedirs(fold_dir, exist_ok=True)
        np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)
        np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)

if __name__ == "__main__":
    generate_folds("Item3.csv")
