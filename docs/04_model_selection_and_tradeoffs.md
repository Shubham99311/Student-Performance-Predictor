# 04. Model Selection & Architectural Trade-offs

Model selection is the process of training, tuning, and evaluating multiple algorithm candidates to select the architecture that yields optimal performance on unseen test data.

---

## 1. Regression Models Benchmark Inventory

Ten regressors spanning linear models, distance-based estimators, non-linear decision trees, and ensemble boosting frameworks were evaluated in `src/components/model_trainer.py`:

| Model Family | Algorithm Evaluated | Scikit-Learn / Library Class |
| :--- | :--- | :--- |
| **Linear Models** | Ordinary Least Squares (OLS) | `LinearRegression` |
| | Lasso Regression (L1 Regularization) | `Lasso` |
| | Ridge Regression (L2 Regularization) | `Ridge` |
| **Distance-Based** | $K$-Nearest Neighbors | `KNeighborsRegressor` |
| **Tree-Based** | Decision Tree | `DecisionTreeRegressor` |
| **Ensemble (Bagging)** | Random Forest | `RandomForestRegressor` |
| **Ensemble (Boosting)**| Gradient Boosting | `GradientBoostingRegressor` |
| | XGBoost | `XGBRegressor` |
| | CatBoost | `CatBoostRegressor` |
| | AdaBoost | `AdaBoostRegressor` |

---

## 2. Evaluation Metrics Framework

Models were evaluated using three primary regression metrics:

### A. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **Interpretation**: Average magnitude of errors in score points. MAE treats all errors linearly.

### B. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Interpretation**: Penalizes larger prediction errors more heavily due to the squaring operation. Useful when large blunders are particularly undesirable.

### C. Coefficient of Determination ($R^2$ Score)
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
* **Interpretation**: Proportion of target variance explained by predictor features. An $R^2 = 1.0$ indicates perfect prediction; $R^2 = 0.0$ indicates performance equivalent to predicting the mean $\bar{y}$.

---

## 3. Hyperparameter Grid Tuning (`GridSearchCV`)

Hyperparameters were systematically optimized across candidates using 3-fold cross-validation (`cv=3`):

```python
params = {
    "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
    "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
    "Gradient Boosting": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Linear Regression": {},
    "Lasso": {'alpha': [0.1, 1.0, 10.0]},
    "Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "K-Neighbors Regressor": {'n_neighbors': [5, 7, 9, 11]},
    "XGBRegressor": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "CatBoost Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor": {
        'learning_rate': [0.1, 0.01, 0.5, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}
```

---

## 4. Model Performance Benchmark Results

The validation results on the unseen test split (200 records) yielded the following ranking:

| Rank | Model Name | Test $R^2$ Score | Test MAE | Test RMSE |
| :---: | :--- | :---: | :---: | :---: |
| **1 (Winner)** | **Lasso Regression** ($\alpha = 0.1$) | **0.8820** | **3.27** | **4.29** |
| 2 | Ridge Regression ($\alpha = 1.0$) | 0.8805 | 3.31 | 4.32 |
| 3 | Linear Regression | 0.8803 | 3.32 | 4.33 |
| 4 | CatBoost Regressor | 0.8520 | 3.75 | 4.81 |
| 5 | Random Forest Regressor | 0.8490 | 3.82 | 4.86 |
| 6 | Gradient Boosting Regressor | 0.8475 | 3.85 | 4.88 |
| 7 | XGBRegressor | 0.8380 | 4.01 | 5.03 |
| 8 | AdaBoost Regressor | 0.8320 | 4.10 | 5.12 |
| 9 | K-Neighbors Regressor | 0.7820 | 4.65 | 5.84 |
| 10 | Decision Tree Regressor | 0.7450 | 5.20 | 6.32 |

---

## 5. Master Interview Question: Why did Lasso/Linear Models out-perform Tree Ensembles (Random Forest / XGBoost)?

Interviewers frequently ask candidates to explain **why** a simpler linear model beat complex state-of-the-art tree ensembles like XGBoost or CatBoost.

Here are the **3 fundamental reasons**:

### Reason 1: High Linear Signal in Continuous Features
The continuous predictor features (`reading_score` and `writing_score`) exhibit strong linear relationships ($r = 0.82$ and $r = 0.80$) with `math_score`. Linear regression models fit a hyper-plane:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

When the true data generating mechanism is predominantly linear, linear models represent the **optimal structural bias**.

### Reason 2: Decision Trees Construct Axis-Aligned Step Functions
Decision trees split feature space using orthogonal cuts ($x_i \le c$). When approximating a smooth linear diagonal relationship ($y \approx x$), decision trees require dozens of step-wise staircase splits. This step-function approximation introduces variance and fails to extrapolate smoothly beyond bin thresholds.

```
Linear Model Fit: Smooth Diagonal Line (Low Variance, Fits true trend)
    y |     /
      |    /
      |   /
      +------- x

Decision Tree Fit: Staircase Step Function (High Variance, Bounded bins)
    y |   _┌──
      | _┌┘
      |┌┘
      +------- x
```

### Reason 3: Multicollinearity & L1 Shrinkage (Bias-Variance Trade-off)
`reading_score` and `writing_score` are highly collinear ($r = 0.95$). OLS Ordinary Least Squares can suffer from inflated coefficient variance. 

**Lasso Regression** penalizes coefficient magnitudes with an L1 norm penalty:

$$\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \|\beta\|_1 \right\}$$

Lasso shrinks noisy correlated coefficients towards zero, reducing model variance, preventing overfitting, and improving test set generalization over both unregularized Linear Regression and over-parameterized Tree Ensembles on a dataset of 1,000 samples.
