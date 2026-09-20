from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html', results=None) 

# Route to process form submissions and show predictions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html', results=None)
    else:
        # Extract inputs and build CustomData object
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Convert to Pandas DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)
        
        print("Starting prediction process...")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction completed successfully: ", results[0])
        
        # Render page with result and preserve inputs
        return render_template('index.html', results=results[0], inputs=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
