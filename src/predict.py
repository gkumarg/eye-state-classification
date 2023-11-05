import pickle

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import os


# to load locally
# def load(filename: str):
#     # Get the directory of the current script
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     # Get the parent directory
#     parent_dir = os.path.dirname(script_dir)
#     with open(os.path.join(parent_dir, 'models', filename), 'rb') as f:
#         return pickle.load(f)
    
def load(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
       
model = load('xgb_model.bin')

app = Flask('sleep-state-predictor')

@app.route('/predict', methods=['POST'])
def predict():
    eeg = request.get_json()
    X = np.array(list(eeg.values())).reshape(1, -1)
    y_pred = model.predict_proba(X)[0, 1]
    eye_closed = y_pred > 0.5
    
    result = {
        "get_eye_closed_probability": float(y_pred),
        "eye_closed": bool(eye_closed),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9595)

# eeg = {'AF3': 4311.79,
#  'F7': 4029.74,
#  'F3': 4289.23,
#  'FC5': 4169.74,
#  'T7': 4381.54,
#  'P7': 4650.26,
#  'O1': 4096.41,
#  'O2': 4631.79,
#  'P8': 4216.92,
#  'T8': 4236.41,
#  'FC6': 4194.36,
#  'F4': 4287.18,
#  'F8': 4617.44,
#  'AF4': 4369.74}