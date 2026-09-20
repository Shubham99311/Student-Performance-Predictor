import os
import sys
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    '''
    Saves an object to a specified file path using dill.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    '''
    Loads an object from a specified file path using dill.
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    '''
    Trains multiple models, performing grid search hyperparameter tuning,
    and returns a dictionary map of model names to R2 scores on test data.
    '''
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            # Perform Grid Search CV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Re-fit the model with the best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
