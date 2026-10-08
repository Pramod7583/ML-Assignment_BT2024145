# Multi-Stage Geothermal Power Plant & Reservoir Optimization

**Student Roll Number:** BT2024145  
**Repository:** `ML-Assignment-1-BT2024145`

---

## 1. Project Overview

This project applies machine learning regression techniques to two geothermal engineering problems:

1. **Phase 1 – Steam Turbine Optimization (`var1`)**
2. **Phase 2 – Subterranean Thermal Reservoir Mapping (`var2`)**

For both phases, polynomial features are combined with feature standardization and **Ridge Regression** to model nonlinear relationships between the input variables and the target variable `y`.

The implementation automatically evaluates multiple polynomial degrees and Ridge regularization parameters (`alpha`) using repeated cross-validation, selects the best combination, performs an additional 80/20 hold-out evaluation, and finally trains the selected model on the complete training dataset to generate predictions for the test dataset.

---

## 2. Project Structure

```text
ML-Assignment-1-BT2024145/
│
├── BT2024145_train_var1.csv
├── BT2024145_test_var1.csv
├── BT2024145_train_var2.csv
├── BT2024145_test_var2.csv
│
├── phase1var1.py
├── phase2var2.py
│
├── cv_results_var1.csv
├── cv_results_var2.csv
│
├── BT2024145_pred_var1.csv
├── BT2024145_pred_var2.csv
│
└── README.md
```

### File Description

| File | Description |
|---|---|
| `BT2024145_train_var1.csv` | Training dataset for Phase 1 |
| `BT2024145_test_var1.csv` | Test dataset for Phase 1 |
| `BT2024145_train_var2.csv` | Training dataset for Phase 2 |
| `BT2024145_test_var2.csv` | Test dataset for Phase 2 |
| `phase1var1.py` | Machine learning pipeline for Phase 1 |
| `phase2var2.py` | Machine learning pipeline for Phase 2 |
| `cv_results_var1.csv` | Cross-validation results for Phase 1 |
| `cv_results_var2.csv` | Cross-validation results for Phase 2 |
| `BT2024145_pred_var1.csv` | Final predictions for Phase 1 test data |
| `BT2024145_pred_var2.csv` | Final predictions for Phase 2 test data |
| `README.md` | Project documentation |

---

# 3. Phase 1 – Steam Turbine Optimization

## Objective

Phase 1 predicts the target value `y` from the six operational input parameters in the `var1` dataset.

The training and test files are automatically loaded using the student roll number and variable number. The script uses:

```text
BT2024145_train_var1.csv
BT2024145_test_var1.csv
```

The target column is `y`, while all remaining columns are used as input features.

## Model

The Phase 1 pipeline uses:

- Polynomial Features
- StandardScaler
- Ridge Regression

The model is constructed as:

```text
PolynomialFeatures
        ↓
StandardScaler
        ↓
Ridge Regression
```

The implementation evaluates polynomial degrees from **1 to 10** and Ridge `alpha` values from:

```text
1e-6
1e-3
1e-2
1e-1
1
3
10
30
100
```

This search is implemented directly in the Phase 1 script.

---

# 4. Phase 2 – Subterranean Thermal Reservoir Mapping

## Objective

Phase 2 predicts the target value `y` from the spatial variables in the `var2` dataset.

The script loads:

```text
BT2024145_train_var2.csv
BT2024145_test_var2.csv
```

The target column `y` is separated from the input features.

## Model

Phase 2 uses the same machine learning pipeline:

```text
PolynomialFeatures
        ↓
StandardScaler
        ↓
Ridge Regression
```

For Phase 2, polynomial degrees from **1 to 14** are evaluated.

The same set of Ridge regularization parameters is tested:

```text
1e-6
1e-3
1e-2
1e-1
1
3
10
30
100
```

The model therefore searches across multiple combinations of polynomial complexity and regularization strength.

---

# 5. Model Selection

The project does not simply choose a polynomial degree manually.

For every combination of:

- Polynomial degree
- Ridge `alpha`

the model is evaluated using **Repeated 5-Fold Cross-Validation**.

The cross-validation configuration is:

```text
Number of folds = 5
Number of repetitions = 3
Random state = 0
```

This results in a more robust estimate of model performance than using a single train/validation split.

The combination with the lowest cross-validation MSE is selected as the final model.

---

# 6. Polynomial Feature Generation

Polynomial regression allows the model to capture nonlinear relationships between the input variables.

For example, with a single feature, polynomial expansion can produce:

```text
x
x²
x³
x⁴
...
```

With multiple features, polynomial expansion also creates interaction terms between features.

The implementation uses:

```python
PolynomialFeatures(deg, include_bias=False)
```

This allows the model to represent increasingly complex relationships as the polynomial degree increases.

---

# 7. Feature Scaling

After generating polynomial features, the resulting features are standardized using:

```python
StandardScaler()
```

The complete pipeline is:

```python
make_pipeline(
    PolynomialFeatures(deg, include_bias=False),
    StandardScaler(),
    Ridge(alpha=alpha)
)
```

Feature scaling is particularly useful for Ridge Regression because the regularization penalty depends on the magnitude of the model coefficients.

---

# 8. Ridge Regression

Ridge Regression is used to reduce overfitting when polynomial features become numerous.

The Ridge model introduces a regularization parameter:

```text
alpha
```

A larger `alpha` applies stronger regularization, while a very small `alpha` makes the model behave more similarly to ordinary polynomial regression.

The project evaluates multiple `alpha` values automatically rather than selecting one manually.

---

# 9. Cross-Validation Results

The cross-validation results are saved to separate CSV files:

```text
cv_results_var1.csv
cv_results_var2.csv
```

Each result contains:

```text
degree
alpha
cv_mse
```

The scripts generate these files automatically after evaluating all degree/alpha combinations. 

The best model is the combination with the **minimum cross-validation MSE**.

The scripts also report the best plain-polynomial degree for reference by considering the smallest tested `alpha`.

---

# 10. Hold-Out Evaluation

After selecting the best degree and `alpha`, an additional **80/20 hold-out evaluation** is performed.

The training data is randomly shuffled using:

```text
random_state = 0
```

Then:

```text
80% → Training
20% → Hold-out evaluation
```

The model is evaluated using:

- Mean Squared Error (MSE)
- R² Score

The corresponding implementation is present in both Phase 1 and Phase 2 scripts. 

---

# 11. Final Model Training

After model selection and hold-out evaluation, the selected model is retrained using **the complete training dataset**.

The final model is then used to predict the target values of the corresponding test dataset.

The prediction files generated are:

```text
BT2024145_pred_var1.csv
BT2024145_pred_var2.csv
```

---

# 12. Evaluation Metrics

## Mean Squared Error (MSE)

MSE measures the average squared difference between the actual and predicted values.

```text
MSE = (1/n) Σ(y_actual - y_predicted)²
```

A lower MSE indicates better prediction performance.

The project uses MSE as the primary metric for selecting the best degree and Ridge `alpha`.

---

## R² Score

The R² score measures how much of the variation in the target variable is explained by the model.

```text
R² = 1 - (SS_res / SS_tot)
```

A value closer to `1` indicates a stronger fit.

The R² score is reported for the 80/20 hold-out evaluation.

---

# 13. Requirements

The project requires Python 3 and the following Python libraries:

```text
numpy
pandas
scikit-learn
```

Install the required dependencies using:

```bash
pip install numpy pandas scikit-learn
```

If a `requirements.txt` file is included in the repository, the dependencies can instead be installed using:

```bash
pip install -r requirements.txt
```

---

# 14. How to Run the Project

## Step 1 – Open the Repository

Open a terminal in the project directory:

```bash
cd ML-Assignment-1-BT2024145
```

---

## Step 2 – Install Dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install numpy pandas scikit-learn
```

---

## Step 3 – Run Phase 1

Run the Steam Turbine Optimization model using:

```bash
python phase1var1.py
```

The script will:

1. Load the Phase 1 training and test datasets.
2. Generate polynomial features.
3. Standardize the features.
4. Evaluate polynomial degrees from 1–10.
5. Evaluate the specified Ridge `alpha` values.
6. Perform repeated 5-fold cross-validation.
7. Select the degree/alpha combination with the lowest CV MSE.
8. Perform an 80/20 hold-out evaluation.
9. Retrain the selected model on all training data.
10. Generate the final test predictions.

The output files include:

```text
cv_results_var1.csv
BT2024145_pred_var1.csv
```

---

## Step 4 – Run Phase 2

Run the Thermal Reservoir Mapping model using:

```bash
python phase2var2.py
```

The script will:

1. Load the Phase 2 training and test datasets.
2. Generate polynomial features.
3. Standardize the features.
4. Evaluate polynomial degrees from 1–14.
5. Evaluate the specified Ridge `alpha` values.
6. Perform repeated 5-fold cross-validation.
7. Select the degree/alpha combination with the lowest CV MSE.
8. Perform an 80/20 hold-out evaluation.
9. Retrain the selected model on all training data.
10. Generate the final test predictions.

The output files include:

```text
cv_results_var2.csv
BT2024145_pred_var2.csv
```

---

# 15. Running with a Custom Data Directory

Both Python scripts accept an optional command-line argument specifying the directory containing the datasets.

By default, the scripts use the current directory:

```python
DATA = sys.argv[1] if len(sys.argv) > 1 else "."
```

Therefore, the normal commands are:

```bash
python phase1var1.py
python phase2var2.py
```

If the datasets are stored in another directory, you can provide that directory as an argument:

```bash
python phase1var1.py /path/to/data
```

and:

```bash
python phase2var2.py /path/to/data
```

The scripts then read the corresponding training and test CSV files from that directory. 

---

# 16. Output Files

After successful execution, the repository will contain the generated results:

```text
cv_results_var1.csv
cv_results_var2.csv

BT2024145_pred_var1.csv
BT2024145_pred_var2.csv
```

### Cross-Validation Files

The files:

```text
cv_results_var1.csv
cv_results_var2.csv
```

contain the tested:

```text
degree
alpha
cv_mse
```

values for each model configuration.

### Prediction Files

The files:

```text
BT2024145_pred_var1.csv
BT2024145_pred_var2.csv
```

contain the original test features together with the predicted `y` values. 

---

# 17. Complete Run Commands

For convenience, the complete workflow is:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 1
python phase1var1.py

# Run Phase 2
python phase2var2.py
```

After running both scripts, the prediction and cross-validation result files will be generated automatically.

---

# 18. Summary

This project implements a systematic machine learning workflow for two geothermal optimization tasks.

The main approach consists of:

```text
Input Dataset
      ↓
Polynomial Feature Generation
      ↓
Feature Standardization
      ↓
Ridge Regression
      ↓
Repeated 5-Fold Cross-Validation
      ↓
Degree + Alpha Selection
      ↓
80/20 Hold-Out Evaluation
      ↓
Retraining on Full Training Data
      ↓
Test Set Prediction
```

The approach allows the model complexity and regularization strength to be selected systematically instead of manually choosing a polynomial degree.

The final prediction files are:

```text
BT2024145_pred_var1.csv
BT2024145_pred_var2.csv
```

and the model-selection results are:

```text
cv_results_var1.csv
cv_results_var2.csv
```