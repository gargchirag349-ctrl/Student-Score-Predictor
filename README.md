# Student Score Predictor

A simple Gradio web application for the Student Score Predictor project.

## Files
- `app.py` — trains Linear Regression and Decision Tree Regressor, evaluates them, and provides a prediction interface.
- `StudentPerformanceFactors.csv` — dataset used by the application.
- `requirements.txt` — Python libraries required to run the application.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Deploy on Render (Free Web Service)
1. Upload all three files to the **root** of your GitHub repository.
2. In Render, connect the repository and select **Python 3**.
3. Leave Root Directory blank.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `python app.py`
6. Select the Free instance.
7. Click Deploy.

The application uses Render's `PORT` environment variable automatically.
