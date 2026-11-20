# Student Score Predictor
# A beginner-friendly Gradio application using Linear Regression
# and Decision Tree Regression.

import os
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# 1. Load the dataset kept in the same folder as this file.
DATA_FILE = os.path.join(os.path.dirname(__file__), "StudentPerformanceFactors.csv")
df = pd.read_csv(DATA_FILE)

# 2. Select the project features and target column.
features = ["Hours_Studied", "Attendance", "Previous_Scores"]
target = "Exam_Score"

X = df[features]
y = df[target]

# 3. Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Train both required machine-learning models.
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# 5. Calculate MAE and RMSE for both models.
linear_predictions = linear_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = sqrt(mean_squared_error(y_test, linear_predictions))

tree_mae = mean_absolute_error(y_test, tree_predictions)
tree_rmse = sqrt(mean_squared_error(y_test, tree_predictions))

metrics_text = (
    "MODEL EVALUATION (20% TEST DATA)\n\n"
    f"Linear Regression\n"
    f"MAE: {linear_mae:.2f}\n"
    f"RMSE: {linear_rmse:.2f}\n\n"
    f"Decision Tree Regressor\n"
    f"MAE: {tree_mae:.2f}\n"
    f"RMSE: {tree_rmse:.2f}\n\n"
    "Lower MAE and RMSE generally indicate better prediction accuracy."
)

# 6. Function to predict a student's exam score.
def predict_score(hours_studied, attendance, previous_scores, model_name):
    if hours_studied is None or attendance is None or previous_scores is None:
        return "Please enter all three values."

    if hours_studied < 0 or attendance < 0 or attendance > 100:
        return "Enter valid values. Attendance must be between 0 and 100."

    if previous_scores < 0 or previous_scores > 100:
        return "Previous scores must be between 0 and 100."

    student_data = pd.DataFrame(
        [[hours_studied, attendance, previous_scores]],
        columns=features
    )

    if model_name == "Linear Regression":
        prediction = linear_model.predict(student_data)[0]
    else:
        prediction = tree_model.predict(student_data)[0]

    return f"Predicted Exam Score: {prediction:.2f}"

# 7. Create the web application.
with gr.Blocks(title="Student Score Predictor") as demo:
    gr.Markdown(
        """
        # Student Score Predictor
        ### Predict a student's exam score using Machine Learning.

        Enter the student's academic details and choose a model to get a prediction.
        """
    )

    with gr.Row():
        hours_input = gr.Number(
            label="Hours Studied",
            value=5,
            minimum=0
        )
        attendance_input = gr.Number(
            label="Attendance (%)",
            value=80,
            minimum=0,
            maximum=100
        )
        previous_input = gr.Number(
            label="Previous Scores",
            value=70,
            minimum=0,
            maximum=100
        )

    model_input = gr.Dropdown(
        choices=["Linear Regression", "Decision Tree Regressor"],
        value="Linear Regression",
        label="Choose Machine Learning Model"
    )

    predict_button = gr.Button("Predict Exam Score", variant="primary")
    prediction_output = gr.Textbox(label="Prediction", interactive=False)

    predict_button.click(
        fn=predict_score,
        inputs=[
            hours_input,
            attendance_input,
            previous_input,
            model_input
        ],
        outputs=prediction_output
    )

    gr.Markdown("## Model Performance")
    gr.Textbox(
        value=metrics_text,
        label="MAE and RMSE Results",
        lines=12,
        interactive=False
    )

    gr.Markdown(
        """
        **Project note:** This is an educational prediction model. Its result is an
        estimate based on patterns in the dataset, not a guaranteed exam result.
        """
    )

if __name__ == "__main__":
    # Render provides the PORT environment variable when deploying a web service.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
