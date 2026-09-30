# Diabetes Prediction Flask ML Project

## Dataset
- Rows: 100,000
- Input features: 8
- Target: diabetes
- Model: Logistic Regression
- Test accuracy on this training run: 0.9605 (96.05%)

## Project structure

diabetes_flask_project/
├── app.py
├── train_model.py
├── model.pkl
├── diabetes_prediction_dataset.csv
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── .vscode/
    └── settings.json

## Run in VS Code

1. Open this project folder in VS Code.
2. Use Python 3.12 for this project. The workspace is configured to use:

   C:\Users\Sarb\AppData\Local\Python\pythoncore-3.12-64\python.exe

3. Install packages:

   python -m pip install -r requirements.txt

4. If you want to retrain the model:

   python train_model.py

5. Start Flask:

   python app.py

6. Open the address shown in the terminal, normally:

   http://127.0.0.1:5000

## Important
- This project works reliably on Python 3.12.
- Python 3.14 can fail on Windows because the scikit-learn DLL is blocked by system security policy.
- The saved model.pkl contains both preprocessing and Logistic Regression, so app.py can directly accept the form values.

This project is for educational purposes and is not a medical diagnosis.
