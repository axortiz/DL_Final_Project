# Boxing Match Predictor

## Project Description

The **Boxing Match Predictor** is a machine learning project aimed at forecasting the outcomes of boxing matches. Utilizing both neural network architectures (Multi-Layer Perceptron - MLP) and gradient boosting techniques (LightGBM), this project analyzes boxer statistics and historical match data to provide accurate predictions. It supports various training methodologies, including standard training, k-fold cross-validation, and LightGBM-specific training, offering flexibility and robustness in model development.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Train with MLP](#train-with-mlp)
  - [Train with K-Fold Cross-Validation](#train-with-k-fold-cross-validation)
  - [Train with LightGBM](#train-with-lightgbm)
- [Inference](#inference)
  - [Using `inference.py`](#using-inferencepy)
  - [Using `inference_lgb.py`](#using-inferencelgbpy)
- [Usage Examples](#usage-examples)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/boxing-match-predictor.git
   cd boxing-match-predictor   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt   ```

   *If `requirements.txt` is not present, you can install the necessary packages manually:*
   ```bash
   pip install torch torchvision numpy pandas scikit-learn lightgbm matplotlib   ```

## Data Preparation

Ensure that your data files are placed in the `data_src/` directory:

- **Training Data:** `data_src/train.csv`
- **Validation Data:** `data_src/validation.csv`
- **Combined Data for K-Fold and LightGBM:** `data_src/combined_data.csv`
- **Inference Data:** `data_src/inference_data.csv`

The data should include boxer statistics and match outcomes, properly formatted and preprocessed.

## Training

### Train with MLP

To train the standard Multi-Layer Perceptron (MLP) model, run:

```bash
python train.py
```

This script will:

- Load and normalize the training and validation data.
- Initialize the MLP model.
- Train the model for a specified number of epochs.
- Save the best model as `best_model.pth`.

### Train with K-Fold Cross-Validation

To train the model using 5-fold cross-validation, run:

```bash
python train_k_fold.py
```

This script will:

- Perform 5-fold stratified cross-validation on the combined dataset.
- Train the MLP model on each fold.
- Save the best model across all folds as `best_model_kfold.pth`.
- Generate loss plots for each fold and the average performance.

### Train with LightGBM

To train a LightGBM model with 5-fold cross-validation, run:

```bash
python train_k_lgb.py
```

This script will:

- Perform 5-fold stratified cross-validation on the combined dataset.
- Train the LightGBM model on each fold.
- Save the best LightGBM model as `best_model_lgb.txt`.
- Generate loss plots for each fold and the average performance.

## Inference

### Using `inference.py`

The `inference.py` script allows you to perform inference using either the standard MLP model or the K-Fold trained model.

**Running Inference:**

```bash
python inference.py --model_type [kfold|mlp] [--model_path PATH] [--data_path PATH]
```

**Arguments:**

- `--model_type`: Type of model to use for inference. Choose between `kfold` and `mlp`.
- `--model_path`: (Optional) Path to the model file. Defaults to `best_model_kfold.pth` for `kfold` or `best_model.pth` for `mlp`.
- `--data_path`: (Optional) Path to the inference data CSV file. Defaults to `data_src/inference_data.csv`.

**Examples:**

1. **Using the K-Fold Model with Default Path:**

   ```bash
   python inference.py --model_type kfold
   ```

2. **Using the MLP Model with Default Path:**

   ```bash
   python inference.py --model_type mlp
   ```

3. **Specifying a Custom Model Path:**

   ```bash
   python inference.py --model_type mlp --model_path path/to/custom_model.pth
   ```

4. **Specifying a Custom Data Path:**

   ```bash
   python inference.py --model_type kfold --data_path path/to/custom_inference_data.csv
   ```

### Using `inference_lgb.py`

The `inference_lgb.py` script allows you to perform inference using the trained LightGBM model.

**Running Inference:**

```bash
python inference_lgb.py
```

**Description:**

- This script loads the LightGBM model from `best_model_lgb.txt` by default.
- It processes the inference data from `data_src/inference_data.csv`.
- It outputs the predicted winners along with confidence scores and probability distributions.

**Customization:**

To use different model or data paths, modify the `MODEL_PATH` and `INFERENCE_DATA_PATH` variables in the script or adjust the script to accept command-line arguments similarly to `inference.py`.

## Usage Examples

1. **Train the MLP Model:**

   ```bash
   python train.py
   ```

2. **Train with 5-Fold Cross-Validation:**

   ```bash
   python train_k_fold.py
   ```

3. **Train with LightGBM:**

   ```bash
   python train_k_lgb.py
   ```

4. **Run Inference with MLP Model:**

   ```bash
   python inference.py --model_type mlp
   ```

5. **Run Inference with K-Fold Model:**

   ```bash
   python inference.py --model_type kfold
   ```

6. **Run Inference with LightGBM:**

   ```bash
   python inference_lgb.py
   ```

## Dependencies

Ensure you have the following Python packages installed:

- Python 3.6+
- numpy
- pandas
- scikit-learn
- torch
- lightgbm
- matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

*Sample `requirements.txt`:*

```
numpy
pandas
scikit-learn
torch
lightgbm
matplotlib
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 