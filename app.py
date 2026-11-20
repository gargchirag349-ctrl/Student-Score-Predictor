import os
import html
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Keep StudentPerformanceFactors.csv in the same GitHub folder as this file.
DATASET_PATH = os.path.join(os.path.dirname(__file__), "StudentPerformanceFactors.csv")

FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Family_Income",
    "Parental_Education_Level",
    "School_Type",
]
TARGET = "Exam_Score"

# Load and prepare the same seven project features used in the original notebook.
df = pd.read_csv(DATASET_PATH)
df = df[FEATURES + [TARGET]].copy()

# The selected dataset columns contain some missing values.
# Remove incomplete rows so the model can train without errors.
df = df.dropna(subset=FEATURES + [TARGET])

X_raw = df[FEATURES].copy()
y = df[TARGET].astype(float)

# Use the same one-hot encoding approach as the notebook.
X = pd.get_dummies(X_raw, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
model_columns = X.columns


def predict_exam_score(
    hours,
    attendance,
    previous_score,
    sleep_hours,
    family_income,
    parental_education,
    school_type,
):
    new_student = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Previous_Scores": previous_score,
        "Sleep_Hours": sleep_hours,
        "Family_Income": family_income,
        "Parental_Education_Level": parental_education,
        "School_Type": school_type,
    }])

    # Apply exactly the same encoding and column order used during training.
    encoded_student = pd.get_dummies(new_student, drop_first=True, dtype=int)
    encoded_student = encoded_student.reindex(columns=model_columns, fill_value=0)

    score = round(float(model.predict(encoded_student)[0]), 2)
    score = max(0, min(100, score))

    if score >= 85:
        grade, color, message = (
            "A+ (Excellent)", "#10b981", "Top tier academic trajectory!"
        )
    elif score >= 70:
        grade, color, message = (
            "B (Good)", "#3b82f6",
            "Solid performance. A bit more study time can push this to an A."
        )
    elif score >= 55:
        grade, color, message = (
            "C (Average)", "#f59e0b",
            "Passing, but attendance or study habits may need a boost."
        )
    else:
        grade, color, message = (
            "D (At Risk)", "#ef4444",
            "This estimate suggests that extra academic support may help."
        )

    safe_grade = html.escape(grade)
    safe_message = html.escape(message)

    return f"""
    <div style="background:linear-gradient(145deg,#1e1b4b,#0f172a);
                border:1px solid #6366f1;border-radius:16px;padding:28px;
                text-align:center;box-shadow:0 10px 25px rgba(99,102,241,.25);">
      <p style="color:#94a3b8;font-size:14px;text-transform:uppercase;
                letter-spacing:1.5px;margin-bottom:8px;">
        Predicted Final Exam Score
      </p>
      <h1 style="font-size:56px;font-weight:800;margin:0;
                 background:linear-gradient(90deg,#818cf8,#38bdf8);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        {score} <span style="font-size:24px;-webkit-text-fill-color:#64748b;">/ 100</span>
      </h1>
      <div style="display:inline-block;background:{color}22;color:{color};
                  border:1px solid {color};padding:6px 16px;border-radius:999px;
                  font-weight:600;font-size:14px;margin-top:14px;">
        Grade: {safe_grade}
      </div>
      <p style="color:#cbd5e1;font-size:14px;margin-top:16px;margin-bottom:0;">
        {safe_message}
      </p>
      <p style="color:#94a3b8;font-size:12px;margin-top:16px;">
        This is an estimate from the trained model, not a guaranteed exam result.
      </p>
    </div>
    """


custom_css = """
body, .gradio-container {
    background-color: #090d16 !important;
    font-family: 'Inter', sans-serif;
}
.header-banner {
    background: linear-gradient(90deg, #312e81 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #4338ca;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
}
.header-banner h1 {
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.card-box {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 16px !important;
    padding: 18px !important;
}
#predict-btn {
    background: linear-gradient(90deg, #6366f1, #0ea5e9) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 12px !important;
    border-radius: 12px !important;
}
#predict-btn:hover {
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
}
footer { display: none !important; }
"""

default_output = predict_exam_score(23, 85, 75, 7, "Medium", "College", "Public")

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    title="Student Score Predictor",
) as demo:
    gr.HTML("""
        <div class="header-banner">
            <h1>🎓 Student Score Predictor</h1>
            <p style="color:#c7d2fe;margin:8px 0 0;">
                AI-Based Academic Performance Prediction
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="card-box"):
            gr.Markdown("### 📚 Academic Habits")
            hours = gr.Slider(0, 45, value=23, step=1, label="Hours Studied (per week)")
            attendance = gr.Slider(50, 100, value=85, step=1, label="Attendance (%)")
            previous_score = gr.Slider(40, 100, value=75, step=1, label="Previous Exam Score")
            sleep_hours = gr.Slider(4, 12, value=7, step=1, label="Sleep Hours (per night)")

        with gr.Column(scale=1, elem_classes="card-box"):
            gr.Markdown("### 🏫 Background & Environment")
            family_income = gr.Radio(
                ["Low", "Medium", "High"], value="Medium", label="Family Income"
            )
            parental_education = gr.Dropdown(
                ["High School", "College", "Postgraduate"],
                value="College",
                label="Parental Education Level",
            )
            school_type = gr.Radio(
                ["Public", "Private"], value="Public", label="School Type"
            )

        with gr.Column(scale=1):
            predict_btn = gr.Button("✨ Predict Exam Score", elem_id="predict-btn")
            output_card = gr.HTML(value=default_output)

    inputs_list = [
        hours, attendance, previous_score, sleep_hours,
        family_income, parental_education, school_type
    ]

    predict_btn.click(
        fn=predict_exam_score, inputs=inputs_list, outputs=output_card
    )
    for component in inputs_list:
        component.change(
            fn=predict_exam_score, inputs=inputs_list, outputs=output_card
        )

    gr.Markdown(
        f"**Model:** Linear Regression  |  **MAE:** {mae:.2f}  |  "
        f"**RMSE:** {rmse:.2f}  |  **Training records:** {len(df):,}"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
