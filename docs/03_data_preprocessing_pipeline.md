# 03. Data Preprocessing Pipeline

Preprocessing transforms messy, raw input data into a clean, normalized matrix ready for machine learning algorithms. In this project, data preprocessing is implemented in `src/components/data_ingestion.py` and `src/components/data_transformation.py`.

---

## 1. Data Ingestion Architecture (`data_ingestion.py`)

The ingestion module handles dataset loading, column header standardization, train-test partitioning, and initial storage in the `artifacts/` folder.

```python
# Key Ingestion Steps
1. Load dataset: pd.read_csv('data/stud.csv')
2. Header Standardization: Replace spaces and slashes with underscores:
   df.columns = [col.strip().replace(' ', '_').replace('/', '_') for col in df.columns]
3. Raw Preservation: Save df to artifacts/raw.csv
4. Partitioning: Train-Test Split (80% train, 20% test, random_state=42)
5. Save Splits: artifacts/train.csv and artifacts/test.csv
```

### Why Partition Before Preprocessing?
Partitioning the dataset *before* fitting any imputer, scaler, or encoder is mandatory to prevent **Data Leakage**. If statistics (such as mean, median, standard deviation, or One-Hot category levels) are computed on the entire dataset prior to splitting, information from the test set leaks into the training pipeline, leading to overly optimistic evaluation metrics.

---

## 2. Feature Pipeline Design (`data_transformation.py`)

Preprocessing uses Scikit-Learn's `ColumnTransformer` to route numerical and categorical features into independent execution pipelines.

```
Input DataFrame (X)
│
├── Numerical Features: ["reading_score", "writing_score"]
│   └── Pipeline:
│       ├── SimpleImputer(strategy="median")
│       └── StandardScaler()
│
└── Categorical Features: ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
    └── Pipeline:
        ├── SimpleImputer(strategy="most_frequent")
        ├── OneHotEncoder()
        └── StandardScaler(with_mean=False)
```

---

## 3. Mathematical & Algorithmic Component Analysis

### A. Numerical Pipeline Breakdown

1. **`SimpleImputer(strategy="median")`**:
   * Replaces any missing values in numerical columns with the median of the training split.
   * *Why Median over Mean?* The median is robust to outliers and skewed distributions, whereas the sample mean can be heavily pulled by extreme score values.

2. **`StandardScaler()`**:
   * Scales numerical features to zero mean ($\mu = 0$) and unit standard deviation ($\sigma = 1$):

     $$Z = \frac{X - \mu}{\sigma}$$

   * *Why Standard Scaling is Necessary*:
     * **Gradient Descent & Regularization**: Regularized linear models (Lasso/Ridge) add a penalty term based on coefficient magnitude ($\lambda \|\beta\|$). If feature scales differ, larger features dominate the penalty penalty term regardless of true importance.
     * **Distance-based Algorithms**: Algorithms like $K$-Nearest Neighbors ($K$-NN) rely on Euclidean distance calculations:

       $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

       Without scaling, features with larger numerical ranges dominate distance calculations.

### B. Categorical Pipeline Breakdown

1. **`SimpleImputer(strategy="most_frequent")`**:
   * Fills missing categorical values with the mode (most common category in training data).

2. **`OneHotEncoder()`**:
   * Converts categorical strings into binary indicator vectors (dummy variables).
   * Maps $N$ discrete categories into $N$ binary columns (e.g., `lunch` $\rightarrow$ `lunch_free/reduced`, `lunch_standard`).

3. **`StandardScaler(with_mean=False)` (Crucial Interview Concept!)**:
   * *Question*: Why pass `with_mean=False` to `StandardScaler` inside the categorical pipeline?
   * *Answer*: One-Hot Encoding produces a **sparse matrix** (`scipy.sparse.csr_matrix`) consisting mostly of zeros to save memory. 
   * Standard scaling centers data by subtracting the column mean ($\mu$). Subtracting a non-zero mean from zero entries changes every zero to $-\mu$, turning the matrix into a **dense matrix**.
   * Converting a large sparse matrix to a dense matrix causes exponential memory consumption spikes. Passing `with_mean=False` skips mean subtraction and only scales variance ($\frac{X}{\sigma}$), preserving sparse matrix efficiency!

---

## 4. Fitting, Transforming & Combining Arrays

```python
# Code logic in initiate_data_transformation()
input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

# Re-combine preprocessed X matrix with target vector y
train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

# Save preprocessor object
save_object(
    file_path=self.data_transformation_config.preprocessor_obj_file_path,
    obj=preprocessing_obj
)
```

### Key Technical Rule for Inference:
* **Training Split**: Use `.fit_transform()` to compute parameters ($\mu, \sigma$, categories) and transform $X_{\text{train}}$.
* **Testing Split & Production Serving**: Use `.transform()` ONLY. Re-fitting on test or production input data violates the principle of fixed parameter inference and introduces data leakage.
