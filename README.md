# Multi-Variable Imputation with GRU-ODE

A machine learning project for analyzing medical time series data from the MIMIC (Medical Information Mart for Intensive Care) dataset using Gated Recurrent Unit Ordinary Differential Equations (GRU-ODE) with Bayesian jumps.

## Overview

This project implements advanced neural ODE techniques for modeling irregularly sampled multivariate clinical time series data. The core innovation is the Neural Negative Feedback ODE with Bayesian Jumps (NNFOwithBayesianJumps) model, which combines:

- **GRU-ODE**: Continuous-time gated recurrent units for modeling patient physiological trajectories
- **Bayesian Updates**: Discrete updates based on actual clinical observations
- **Covariate Integration**: Static patient characteristics (age, gender, etc.) 
- **Classification Head**: Predicts clinical outcomes or events

The model handles missing data naturally through its continuous-time formulation and provides uncertainty estimates through the Bayesian framework.

## Project Structure

```
multi_var_imputation/
├── demo_tables/              # Sample MIMIC CSV files
│   ├── admissions.csv
│   ├── chartevents.csv
│   ├── d_icd_diagnoses.csv
│   ├── d_items.csv
│   ├── d_labitems.csv
│   ├── diagnoses_icd.csv
│   ├── icustays.csv
│   └── patients.csv
├── full_tables/              # Directory for full MIMIC dataset (empty)
├── patient_measurements_48h/ # Processed patient data
├── demo_patient_measurements_48h/ # Demo processed data
├── older-unused-files/       # Archived code
├── checker-files-from-Claude-Code/ # Code analysis artifacts
├── test-files-by-Claude-Code/    # Test files
├── spark-warehouse/          # Spark processing files
├── spark_conf/               # Spark configuration
├── Readmitted_patients/      # Readmission analysis
├── create_patient_parquets.py # Data preprocessing script
├── data_utils.py             # Dataset loading and utilities
├── generate_folds.py         # Data split generation
├── logger.py                 # TensorBoard logging utilities
├── models.py                 # Neural network architectures (GRU-ODE variants)
├── plot_MIMIC.py             # Plotting utilities
├── plot_patient_ehr.py       # Enhanced patient EHR plotting
├── plot_item.ipynb           # Jupyter notebook for item exploration
├── requirements.txt          # Python dependencies
├── test_imblearn.py          # Imbalanced learning tests
├── test_mimic.py             # MIMIC-specific tests
├── train_mimic.py            # Main training script
└── ultra_simple_test.py      # Simple validation test
```

## Key Features

- **GRU-ODE with Bayesian Jumps**: Novel architecture combining continuous dynamics with discrete observation updates
- **Handles Irregular Sampling**: Naturally works with unevenly spaced clinical measurements
- **Missing Data Robustness**: Built-in handling of sparsely observed variables
- **Uncertainty Quantification**: Provides probabilistic predictions through Bayesian framework
- **Covariate Integration**: Incorporates static patient characteristics
- **Flexible Solvers**: Supports Euler, midpoint, and Dopri5 ODE solvers
- **Path Return Capability**: Can return full trajectory paths for analysis
- **Smoother Mode**: Optional auxiliary loss for improved representation learning

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd multi_var_imputation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (version appropriate for your system):
   ```bash
   # Example for CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Install torchdiffeq** (required for ODE solvers):
   ```bash
   pip install torchdiffeq
   ```

## Data Preparation

This project expects MIMIC-III formatted data. The `demo_tables/` directory contains sample data files for testing and development.

To prepare your own data:

1. **Obtain MIMIC-III data** from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (requires credentialing)
2. **Place the CSV files** in the `full_tables/` directory
3. **Run the preprocessing script**:
   ```bash
   python create_patient_parquets.py
   ```
   This script converts raw CSV files to efficient Parquet format and creates the `Item3.csv` file expected by the training script.

4. **Generate data splits** (optional):
   ```bash
   python generate_folds.py
   ```
   This creates train/validation/test splits in directories like `mimic_fold_idx_0/`, `mimic_fold_idx_1/`, etc.

## Usage

### Training the Model

Run the main training script:
```bash
python train_mimic.py
```

This will:
1. Load processed patient data from `Item3.csv`
2. Use GRU-ODE with Bayesian jumps to model patient trajectories
3. Train for 700 epochs by default
4. Save model checkpoints and logs to `./Logs/mimic/`
5. Save final model to `mimic_fold_idx_1/mimic_MAX.pt` (best model) and `mimic_fold_idx_1/mimic.pt` (periodic saves)

### Configuration

Modify `train_mimic.py` to adjust:
- **Model architecture** (hidden sizes, dropout rates, etc.)
- **Training parameters** (learning rate, batch size, epochs)
- **Data paths** (csv file locations)
- **Loss weights** (lambda parameter balancing reconstruction and classification)
- **ODE solver** (euler, midpoint, dopri5)
- **Device** (cpu/cuda)

### Evaluation

The training script automatically computes and logs:
- Training/validation loss
- AUC (Area Under ROC Curve) for classification
- Log-likelihood of observations
- Reconstruction MSE
- Correlation between predictions and ground truth

Results are logged to TensorBoard and can be viewed with:
```bash
tensorboard --logdir=./Logs
```

## Model Architecture

The core model (`NNFOwithBayesianJumps` in `models.py`) consists of:

1. **Covariate Encoder**: Maps static patient features to initial hidden state
2. **Parameter Generator (p_network)**: Maps hidden state to ODE function parameters
3. **GRU-ODE Core**: Continuous-time dynamics modeled with either:
   - Standard GRU-ODE cell
   - Full GRU-ODE cell (with reset gate)
   - Autonomous variants (no explicit time dependence)
4. **Observation Updater**: Discrete Bayesian updates when measurements arrive
5. **Classification Head**: Maps final hidden state to prediction

## Dependencies

See `requirements.txt` for the full list, but key packages include:
- PyTorch (≥1.0)
- NumPy
- Pandas
- PySpark
- scikit-learn
- torchdiffeq
- TensorBoard
- tqdm

## Results and Outputs

During training, the script produces:
- **TensorBoard logs**: In `./Logs/[simulation_name]/`
- **Model checkpoints**: Periodic saves during training
- **Best model**: Saved as `mimic_fold_idx_[fold]/[simulation_name]_MAX.pt`
- **Final model**: Saved as `mimic_fold_idx_[fold]/[simulation_name].pt`
- **Parameter files**: `.npy` files containing training configuration (in fold directory)

## Plotting and Visualization

Several visualization tools are included:
- `plot_patient_ehr.py`: Create enhanced plots for individual patients
- `plot_MIMIC.py`: General MIMIC data plotting utilities
- `plot_item.ipynb`: Jupyter notebook for exploring item distributions

## Notes

1. **GPU Usage**: The training script defaults to CUDA if available. Modify `device = torch.device("cuda")` in `train_mimic.py` to use CPU.

2. **Data Requirements**: The model expects data in a specific format with columns:
   - `ID`: Patient identifier
   - `Time`: Observation time
   - `Value_*`: Measurement values
   - `Mask_*`: Observation indicators (1.0 = observed, 0.0 = missing)

3. **Memory Usage**: Training with large batches may require significant GPU memory. Adjust batch size in `train_mimic.py` if needed.

4. **Reproducibility**: Random seeds are set in various places for reproducibility, but full reproducibility across runs may require additional controls.

## References

This implementation builds upon:
- Neural Ordinary Differential Equations (Chen et al., 2018)
- GRU-D: Bidirectional GRU with time decay (Che et al., 2018)
- Latent ODEs for irregularly-sampled time series (Rubanova et al., 2019)
- Bayesian deep learning techniques

## License

[Specify your license here]

## Acknowledgments

- MIMIC-III dataset contributors
- Open-source deep learning community
- Claude Code for development assistance