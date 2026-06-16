# California Housing Price Prediction: Regression Analysis Benchmark

## Project Overview
This project performs an end-to-end regression evaluation benchmarking five supervised machine learning algorithms to predict median house values across California districts. Performance is verified utilizing Mean Squared Error (MSE), Mean Absolute Error (MAE), and the R-squared Coefficient ($R^2$).

---

## 1. Data Preprocessing & Theoretical Justifications
* **Missing Values Assessment:** The standard California Housing dataset contains zero structural missing fields, guaranteeing clean row evaluations.
* **Feature Scaling (Standardization):** Features were transformed using standard normal scaling ($z = (x - \mu) / \sigma$). 
* **Justification:** Algorithms like Linear Regression and Support Vector Machines (SVR) are highly sensitive to feature magnitudes. Features such as `AveRooms` (average rooms per household) vary on a vastly different scale compared to `Population`. Scaling prevents features with large magnitudes from dominating distance and weight calculations, accelerating gradient descent convergence.

---

## 2. Regression Algorithms & Architectural Suitability

### 1. Linear Regression
* **How it Works:** Models a linear relationship between input features and the continuous target by finding coefficients that minimize the Sum of Squared Residuals (SSR).
* **Suitability:** Establishes a highly interpretable structural baseline to determine if housing price variations can be explained linearly.

### 2. Decision Tree Regressor
* **How it Works:** Splits the dataset recursively into non-overlapping regions based on feature thresholds that maximize variance reduction.
* **Suitability:** Capable of capturing localized, step-wise non-linear relationships without needing complex hyperparameter tuning.

### 3. Random Forest Regressor
* **How it Works:** An ensemble bagging technique that trains multiple deep decision trees in parallel on bootstrap samples and averages their outputs.
* **Suitability:** Reduces the high variance and overfitting tendencies of standalone decision trees, handling complex feature interactions across geographic blocks gracefully.

### 4. Gradient Boosting Regressor
* **How it Works:** An ensemble boosting technique that fits decision trees sequentially. Each new tree is trained to predict the pseudo-residuals (errors) of the preceding combination.
* **Suitability:** Known for state-of-the-art predictive accuracy on tabular datasets by optimizing complex non-linear cost functions step-by-step.

### 5. Support Vector Regressor (SVR)
* **How it Works:** Uses a kernel function (RBF) to map features into higher-dimensional spaces, fitting a regression line inside a bounded error tube ($\epsilon$) while maximizing spatial margin distances.
* **Suitability:** Effective at managing localized geographical feature weights securely without being overly sensitive to extreme housing price outliers.

---

## 3. Benchmark Evaluation Summary

| Algorithm | Mean Squared Error (MSE) | Mean Absolute Error (MAE) | R-squared ($R^2$) Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.5559 | 0.5332 | 0.5758 |
| **Decision Tree** | 0.3809 | 0.4208 | 0.7093 |
| **Random Forest** | 0.2660 | 0.3391 | 0.7970 |
| **Gradient Boosting** | 0.2940 | 0.3713 | 0.7756 |
| **Support Vector Regressor (SVR)** | 0.3571 | 0.3953 | 0.7275 |

### Operational Conclusions
* **Best-Performing Algorithm:** **Random Forest Regressor** ($R^2 \approx 0.7970$, lowest MSE $\approx 0.2660$). It effectively handles the complex geographical interactions (latitude/longitude) and density parameters inherent in the California dataset through ensemble averaging, significantly minimizing overfitting.
* **Worst-Performing Algorithm:** **Linear Regression** ($R^2 \approx 0.5758$, highest MSE $\approx 0.5559$). Housing valuations are influenced by complex, non-linear socioeconomic factors and geographical locations that cannot be adequately modeled by a straight hyperplane.

---

## Workspace Setup Guide
1. Clone this repository to your local Mac directory.
2. Ensure you have scikit-learn and visualization suites ready:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
