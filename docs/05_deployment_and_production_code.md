# 05. Deployment & Production Software Architecture

To transition a machine learning model from a Jupyter Notebook experiment into a production-grade application, the code must be structured with error handling, logging, modular classes, and web serving capabilities.

---

## 1. Inference Pipeline Architecture (`prediction_pipeline.py`)

The inference architecture separates input formatting (`CustomData`) from model loading and prediction execution (`PredictPipeline`).

```python
# src/pipeline/prediction_pipeline.py

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            
            # Load serialized objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform input features & generate prediction
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
```

### Input Data Contract (`CustomData` Class)
Web forms and API payloads deliver key-value pairs or strings. `CustomData` encapsulates these inputs, enforcing structural consistency and mapping raw request values into a pandas DataFrame matching model expectations:

```python
class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
```

---

## 2. Object Serialization (`dill` vs `pickle`)

In `src/utils.py`, model and preprocessor objects are saved and loaded using the **`dill`** library:

```python
def save_object(file_path, obj):
    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)

def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)
```

### Why `dill` over standard `pickle`?
* Standard Python `pickle` can fail when serializing complex, nested Python objects, lambdas, or custom transformer functions defined inside scripts.
* `dill` extends Python’s `pickle` module to serialize almost anything in Python, including lambda functions, nested classes, and complex scikit-learn preprocessing pipelines.

---

## 3. Web Serving Layer (`app.py`)

The serving layer uses **Flask** to handle incoming HTTP requests:

```python
# app.py
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html', results=None)
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results[0], inputs=request.form)
```

---

## 4. Production Exception & Logging Framework

### A. Custom Exception Tracking (`src/exception.py`)
Standard Python error tracebacks print full console stacks that are difficult to parse in log aggregators. `CustomException` inspects `sys.exc_info()` to capture the exact script name, line number, and error string:

```python
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
```

### B. Timestamped Centralized Logging (`src/logger.py`)
Log files are generated dynamically with execution timestamps (`MM_DD_YYYY_HH_MM_SS.log`) inside a dedicated `logs/` directory:

```python
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

---

## 5. Python Packaging & Editable Installs (`setup.py`)

`setup.py` enables the project to be packaged as a local library, allowing modules to be imported across scripts without modifying `sys.path`:

```python
# setup.py
from setuptools import find_packages, setup

setup(
    name='student_performance_prediction',
    version='0.0.1',
    author='Utkarsh Pareek',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
```

* Including `-e .` in `requirements.txt` triggers `pip install -e .`, installing the root directory in **editable mode** so changes in `src/` are instantly reflected throughout the environment.
