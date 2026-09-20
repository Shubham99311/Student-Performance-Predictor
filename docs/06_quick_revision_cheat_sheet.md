# 06. Night-Before-Interview Quick Revision Cheat Sheet

> **Use this file for a 5-minute rapid review right before your interview.**

---

## ⚡ 1-Minute Elevator Pitch

> *"I built an end-to-end Machine Learning regression solution that predicts student mathematics scores (`math_score`) based on demographic attributes, socio-economic proxies (`lunch` type), and subject assessment scores (`reading_score`, `writing_score`). I designed a modular production architecture separating data ingestion, transformation pipelines (`ColumnTransformer`), model evaluation across 10 regression algorithms, and Flask API deployment. The best-performing model was **Lasso Regression**, achieving an **$R^2$ score of 0.882** (MAE: **3.27**, RMSE: **4.29**), outperforming tree-based ensembles due to strong linear signals in the continuous predictors and effective L1 regularization handling high multicollinearity ($r = 0.95$) between reading and writing scores."*

---

## 📊 Key Metrics & Leaderboard Table

* **Target Variable ($y$)**: `math_score` (Continuous scale 0–100, Mean: 66.09, Std Dev: 15.16)
* **Dataset Size**: 1,000 rows, 8 features (0 missing values, 0 duplicate rows)
* **Train/Test Split**: 80/20 (800 training records, 200 testing records, `random_state=42`)

| Metric | Champion Model (**Lasso Regression**) | Runner-Up (**Ridge Regression**) | Tree Ensemble Baseline (**Random Forest**) |
| :--- | :---: | :---: | :---: |
| **$R^2$ Score** | **0.8820** | 0.8805 | 0.8490 |
| **MAE** | **3.27** points | 3.31 points | 3.82 points |
| **RMSE** | **4.29** points | 4.32 points | 4.86 points |

---

## 🔍 Key EDA & Domain Insights

1. **Multicollinearity**: `reading_score` and `writing_score` have a **0.95 Pearson correlation** ($r = 0.82$ and $r = 0.80$ with `math_score`).
2. **Socio-Economic Impact**: Standard lunch students scored **11.1 points higher** in math than free/reduced lunch students (70.0 vs 58.9).
3. **Test Prep Course**: Completing test prep boosted writing scores by **+10 points** and math scores by **+5.7 points**.
4. **Gender Dynamics**: Male students averaged higher in math (68.7 vs 63.6); Female students averaged higher in reading (72.6 vs 65.5) and writing (72.5 vs 63.4).
5. **Target Leakage Warning**: `total_score` and `average` features engineered during EDA were **excluded** from $X$ during model training because they explicitly contain the target `math_score`.

---

## 🛠️ Data Preprocessing & Pipeline Mechanics

* **Numerical Features** (`reading_score`, `writing_score`):
  * `SimpleImputer(strategy="median")` (Robust to outliers) $\rightarrow$ `StandardScaler()` ($Z = \frac{X-\mu}{\sigma}$).
* **Categorical Features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`):
  * `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder()` $\rightarrow$ `StandardScaler(with_mean=False)`.
* **🔥 Critical Interview Question**: Why pass `with_mean=False` to `StandardScaler` on One-Hot encoded features?
  * *Answer*: One-Hot Encoding produces a **sparse matrix**. Subtracting the column mean ($\mu$) turns all implicit zeros into non-zero dense values, converting sparse storage into dense storage and causing severe memory spikes. `with_mean=False` scales variance without destroying sparse matrix efficiency.

---

## 💡 Why Lasso Won over XGBoost / Random Forest

1. **True Linear Relationship**: `reading_score` and `writing_score` have strong linear correlations with `math_score`. Linear models fit smooth continuous hyperplanes, whereas decision trees fit staircase step-functions ($x_i \le c$) that approximate linear diagonals poorly.
2. **High Multicollinearity**: Reading and writing scores ($r = 0.95$) inflate OLS parameter variance. Lasso's L1 regularization penalty ($\alpha \sum |\beta_j|$) performs coefficient shrinkage, stabilizing model variance.
3. **Sample Size ($N = 1000$)**: Tree ensembles (Random Forest / XGBoost) overfitted the noise in continuous features due to excess model complexity. Lasso maintained low variance and superior test set generalization.

---

## ⚙️ Engineering & MLOps Infrastructure

* **Serialization**: Used `dill` instead of standard `pickle` (`dill.dump(obj, file)`) because `dill` supports complex nested scikit-learn pipeline objects and custom transformers.
* **Production Serving**: Flask API (`app.py`) serving form inputs converted into DataFrames via a `CustomData` class and passed to `PredictPipeline`.
* **Exception Handling**: Custom exception class (`CustomException` in `src/exception.py`) extracting script filename, line number, and error message from `sys.exc_info()`.
* **Logging**: Centralized timestamped log files (`logs/MM_DD_YYYY_HH_MM_SS.log`) created via `src/logger.py`.
* **Packaging**: `setup.py` configured with `find_packages()` and `requirements.txt` containing `-e .` for editable local package installation.
