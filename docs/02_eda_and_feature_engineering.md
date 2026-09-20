# 02. Exploratory Data Analysis (EDA) & Feature Engineering

Exploratory Data Analysis (EDA) is the crucial phase where raw data is inspected, visualized, and analyzed to discover patterns, detect anomalies, test hypotheses, and verify assumptions before building machine learning models.

---

## 1. Dataset Overview & Data Quality Integrity

The dataset (`data/stud.csv`) consists of **1,000 student records** with **8 attributes**.

### Data Sanity & Completeness Checks
```python
# Output from notebook 1_EDA.ipynb
- Shape: 1000 rows × 8 columns
- Missing / Null Values: 0 across all features
- Duplicate Rows: 0 duplicate entries
```

### Feature Inventory & Data Types

| Feature Name | Type | Class / Range | Description |
| :--- | :--- | :--- | :--- |
| `gender` | Categorical (Nominal) | `female` (518), `male` (482) | Gender identity of the student |
| `race_ethnicity` | Categorical (Nominal) | `group A` to `group E` | Ethnic grouping classification |
| `parental_level_of_education` | Categorical (Ordinal) | 6 categories | Highest educational attainment of parents |
| `lunch` | Categorical (Binary) | `standard` (645), `free/reduced` (355) | Type of lunch subsidization (Socio-economic proxy) |
| `test_preparation_course` | Categorical (Binary) | `none` (642), `completed` (358) | Completion status of test prep course |
| `reading_score` | Numerical (Discrete/Continuous) | 17 to 100 | Reading assessment score |
| `writing_score` | Numerical (Discrete/Continuous) | 10 to 100 | Writing assessment score |
| **`math_score` (Target)** | Numerical (Discrete/Continuous) | 0 to 100 | Mathematics assessment score |

---

## 2. Statistical Summary & Score Distributions

### Descriptive Statistics

| Score Metric | Count | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`math_score`** | 1000 | **66.09** | 15.16 | 0 | 57.0 | 66.0 | 77.0 | 100 |
| **`reading_score`** | 1000 | **69.17** | 14.60 | 17 | 59.0 | 70.0 | 79.0 | 100 |
| **`writing_score`** | 1000 | **68.05** | 15.20 | 10 | 57.75 | 69.0 | 79.0 | 100 |

### Insights from Distribution Parameters:
1. **Symmetry & Normality**: All three score distributions exhibit near-bell curve shapes with mean values around 66–69 and median values around 66–70.
2. **Spread Consistency**: The standard deviation across all three domains is consistently ~15 points, indicating similar relative variability.
3. **Outlier Detection**: `math_score` has a minimum value of 0 (a single extreme lower outlier), whereas reading and writing minimums sit at 17 and 10 respectively.

---

## 3. Deep Dive: Categorical & Demographic Impact Analysis

### A. Socio-Economic Status (`lunch`)
* **Finding**: Students with **standard lunch** outperform students with **free/reduced lunch** across all three subjects by an average of **11–12 points**.
* **Statistical Averages**:
  * Standard Lunch Mean Math Score: **70.03**
  * Free/Reduced Lunch Mean Math Score: **58.92**
* **Business/EDA Takeaway**: `lunch` status serves as a powerful proxy for household income and socio-economic support. Nutritional security and resources directly correlate with academic test outcomes.

### B. Test Preparation Course (`test_preparation_course`)
* **Finding**: Students who completed the prep course scored significantly higher than those who did not.
* **Statistical Averages**:
  * Completed Prep Course Mean Math Score: **69.70**
  * No Prep Course Mean Math Score: **64.04**
  * Writing score delta is even larger: **74.42 (Completed)** vs **64.50 (None)** (+10 point jump).
* **Business/EDA Takeaway**: Test preparation has a stronger direct effect on writing and reading mechanics than on pure math, though it produces statistically significant uplifts in all areas.

### C. Parental Level of Education (`parental_level_of_education`)
* **Trend**: Higher parental educational attainment directly mirrors higher student score medians.
* **Rank Ordering of Student Math Averages by Parental Education**:
  1. Master's Degree: **69.75**
  2. Bachelor's Degree: **69.39**
  3. Associate's Degree: **67.88**
  4. Some College: **67.13**
  5. High School: **62.14**
  6. Some High School: **63.50**
* **Takeaway**: Parental education provides both an academic environment and resources that foster higher baseline student achievement.

### D. Gender Performance Breakdown (`gender`)
* **Math Score**: Male students (Mean: **68.73**) outperform Female students (Mean: **63.63**) by ~5.1 points.
* **Reading & Writing Scores**: Female students dominate verbal subjects:
  * Female Reading Mean: **72.60** vs Male Reading Mean: **65.47** (+7.1 points).
  * Female Writing Mean: **72.46** vs Male Writing Mean: **63.38** (+9.1 points).
* **Takeaway**: Gender shows cross-subject trade-offs. Incorporating gender allows the regression model to calibrate subject-specific baseline adjustments.

---

## 4. Correlation Analysis & Multicollinearity

A correlation heatmap was calculated across the numerical assessment variables:

```
               math_score  reading_score  writing_score
math_score        1.00          0.82           0.80
reading_score     0.82          1.00           0.95
writing_score     0.80          0.95           1.00
```

### Critical Statistical Insights for Interviews:
1. **Strong Predictor Signal**: `reading_score` ($r = 0.82$) and `writing_score` ($r = 0.80$) exhibit strong positive linear correlation with `math_score`. Students who excel in literacy and comprehension also excel in quantitative problem solving.
2. **Extreme Multicollinearity**: `reading_score` and `writing_score` have a Pearson correlation of **0.95**.
   * *Interview Watchout*: Extremely high collinearity between predictors can inflate the variance of linear regression coefficients ($\text{VIF} > 10$).
   * *Solution in Pipeline*: Regularized models like **Lasso (L1)** and **Ridge (L2)** shrink coefficient estimates to stabilize estimation under high multicollinearity.

---

## 5. Feature Engineering Conducted during EDA

During the exploratory phase (`notebooks/1_EDA.ipynb`), two composite features were engineered to study overall academic performance:

$$\text{total\_score} = \text{math\_score} + \text{reading\_score} + \text{writing\_score}$$

$$\text{average} = \frac{\text{total\_score}}{3}$$

### Why these features were excluded from model input features ($X$):
* **Data Leakage Risk**: `total_score` and `average` both explicitly contain `math_score` (the target variable) in their mathematical definition.
* Including `total_score` or `average` in $X$ would cause target leakage, allowing a simple linear model to achieve an $R^2 = 1.0$ through trivial algebraic substitution ($\text{math\_score} = 3 \times \text{average} - \text{reading\_score} - \text{writing\_score}$).
* **Key Interview Insight**: Always separate features engineered for *EDA exploration & reporting* from features engineered for *model training input features*.
