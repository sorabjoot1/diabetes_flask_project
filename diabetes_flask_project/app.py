from pathlib import Path
import subprocess
import sys

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_leve",
    "blood_glucose_levell",
]


def load_model():
    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, str(BASE_DIR / "train_model.py")],
            check=True,
            cwd=str(BASE_DIR),
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = {
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "smoking_history": request.form["smoking_history"],
            "bmi": float(request.form["bmi"]),
            "HbA1c_leve": float(request.form.get("HbA1c_leve", request.form.get("HbA1c_level"))),
            "blood_glucose_levell": float(request.form.get("blood_glucose_levell", request.form.get("blood_glucose_level"))),
        }

        input_df = pd.DataFrame([form_data], columns=FEATURE_COLUMNS)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1]) * 100

        if prediction == 1:
            result = "Diabetes Detected"
            message = f"Estimated diabetes probability: {probability:.2f}%"
            result_class = "danger"
        else:
            result = "Diabetes Not Detected"
            message = f"Estimated diabetes probability: {probability:.2f}%"
            result_class = "success"

        return render_template(
            "index.html",
            result=result,
            message=message,
            result_class=result_class,
            form_data=form_data,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Please enter valid values. Error: {str(e)}",
            form_data=request.form.to_dict(),
        )


if __name__ == "__main__":
    app.run(debug=True)
