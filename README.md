# Student Performance Prediction Pipeline

An end-to-end Machine Learning Regression project that predicts a student's **Mathematics Score** based on demographic and academic variables.

🚀 **Live Deployed Web App**: [https://8a58a43c0d9f33.lhr.life/predictdata](https://8a58a43c0d9f33.lhr.life/predictdata)  
⚡ **Framework**: Python | Flask | Scikit-Learn | CatBoost | XGBoost | Pandas

## Project Structure
```
student-performance-prediction/
│
├── data/
│   └── stud.csv                      # Raw dataset
│
├── notebooks/
│   ├── 1_EDA.ipynb                   # Exploratory Data Analysis
│   └── 2_Model_Training.ipynb        # Model Training and comparison
│
├── src/
│   ├── __init__.py
│   ├── exception.py                  # Custom traceback handling
│   ├── logger.py                     # Execution logging system
│   ├── utils.py                      # Reusable utilities (saving, grid-evaluation)
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py        # Data ingestion and train-test splitting
│   │   ├── data_transformation.py   # Preprocessing (StandardScaler + OneHotEncoder)
│   │   └── model_trainer.py         # Multi-model evaluation and tuning
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── prediction_pipeline.py    # Custom input mapping and prediction serving
│
├── artifacts/                        # Model & preprocessing artifacts (generated)
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── templates/
│   └── index.html                    # Glassmorphism HTML Form Interface
│
├── app.py                            # Flask application entry point
├── requirements.txt                  # Python dependencies
├── setup.py                          # Packaging setup file
└── README.md
```

## Setup & Installation

1. **Clone the project & Navigate to folder**:
   ```bash
   cd student-performance-prediction
   ```

2. **Install dependencies**:
   This runs local packaging setup using `setup.py` automatically:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Flow

### Step 1: Run the Training Pipeline
Run the data ingestion module to split the dataset, execute column transformations, run hyperparameter grid tuning across 10 regression models, and save the best model and preprocessor objects to the `artifacts/` folder:
```bash
python -m src.components.data_ingestion
```
*Note: The best-performing model found on the testing dataset is **Lasso** with a validation **$R^2$ Score of 0.882**.*

### Step 2: Start the Web App Server
To serve predictions through the Flask web interface:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/predictdata
```

## Dataset Features
- **Target Variable**:
  - `math_score` (Continuous Numerical)
- **Input Variables**:
  - `gender` (`male`, `female`)
  - `race_ethnicity` (`group A` - `group E`)
  - `parental_level_of_education` (Associate's, Bachelor's, High school, Master's, etc.)
  - `lunch` (`standard`, `free/reduced`)
  - `test_preparation_course` (`none`, `completed`)
  - `reading_score` (Continuous Numerical, 0 - 100)
  - `writing_score` (Continuous Numerical, 0 - 100)

## Performance Summary
A comparison of the standard algorithms tested (Linear Regression, Lasso, Ridge, K-Neighbors, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost) selected **Lasso** as the final model based on validation metrics:
*   **$R^2$ Score**: ~0.8820
*   **MAE**: ~3.27
*   **RMSE**: ~4.29
