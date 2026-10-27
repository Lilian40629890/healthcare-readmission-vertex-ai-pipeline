"""Vertex AI inference pipeline: train the selected Logistic Regression model and
run batch predictions on the unlabeled test set.

Transcribed from the project's final presentation (slides 23-25). Bucket paths
are read from environment variables instead of being hard-coded.
"""

import os

from kfp import compiler
from kfp.dsl import Dataset, Input, Model, Output, component, pipeline

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket")
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

X_TRAIN_URI = f"gs://{BUCKET_NAME}/healthcare_X_train_final.csv"
Y_TRAIN_URI = f"gs://{BUCKET_NAME}/healthcare_y_train_final.csv"
X_TEST_URI = f"gs://{BUCKET_NAME}/healthcare_X_test_final.csv"


@component(
    packages_to_install=["pandas", "scikit-learn", "joblib", "gcsfs", "fsspec"],
    base_image="python:3.10",
)
def train_logistic_model_for_inference(
    x_train_uri: str,
    y_train_uri: str,
    model_output: Output[Model],
):
    import os

    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Read training data from GCS
    X_train = pd.read_csv(x_train_uri)
    y_train = pd.read_csv(y_train_uri)

    # If y has one column, convert it to Series
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_train columns:", list(X_train.columns))

    # Build Logistic Regression pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=2000)),
    ])

    # Train model
    model.fit(X_train, y_train)

    # Quick training performance check
    train_preds = model.predict(X_train)
    print("Training accuracy:", accuracy_score(y_train, train_preds))
    print("Classification report:")
    print(classification_report(y_train, train_preds))

    # Save model artifact
    os.makedirs(model_output.path, exist_ok=True)
    model_path = os.path.join(model_output.path, "model.joblib")
    joblib.dump(model, model_path)

    print("Model saved to:", model_path)


@component(
    packages_to_install=["pandas", "scikit-learn", "joblib", "gcsfs", "fsspec"],
    base_image="python:3.10",
)
def batch_predict_logistic_model(
    model_output: Input[Model],
    x_test_uri: str,
    predictions_output: Output[Dataset],
):
    import os

    import joblib
    import pandas as pd

    # Load test data
    X_test = pd.read_csv(x_test_uri)

    print("X_test shape:", X_test.shape)
    print("X_test columns:", list(X_test.columns))

    # Load trained model
    model_path = os.path.join(model_output.path, "model.joblib")
    model = joblib.load(model_path)

    print("Loaded model from:", model_path)

    # Predict
    predictions = model.predict(X_test)

    # If model supports probability, also output predicted probability
    try:
        probabilities = model.predict_proba(X_test)
        positive_probability = probabilities[:, 1]
    except Exception:
        positive_probability = [None] * len(predictions)

    # Create output dataframe
    results = pd.DataFrame({
        "row_id": range(len(predictions)),
        "prediction": predictions,
        "positive_probability": positive_probability,
    })

    print("Prediction results preview:")
    print(results.head())

    # Save predictions
    os.makedirs(predictions_output.path, exist_ok=True)
    output_path = os.path.join(predictions_output.path, "predictions.csv")
    results.to_csv(output_path, index=False)

    print("Predictions saved to:", output_path)


@pipeline(
    name="logistic-inference-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def logistic_inference_pipeline(
    x_train_uri: str,
    y_train_uri: str,
    x_test_uri: str,
):
    train_task = train_logistic_model_for_inference(
        x_train_uri=x_train_uri,
        y_train_uri=y_train_uri,
    )

    predict_task = batch_predict_logistic_model(
        model_output=train_task.outputs["model_output"],
        x_test_uri=x_test_uri,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=logistic_inference_pipeline,
        package_path="logistic_inference_pipeline.json",
    )
    print("Pipeline compiled successfully.")

    if os.environ.get("RUN_ON_VERTEX") == "1":
        from google.cloud import aiplatform

        aiplatform.init(
            project=os.environ["PROJECT_ID"],
            location=os.environ.get("REGION", "us-central1"),
            staging_bucket=f"gs://{BUCKET_NAME}",
        )

        job = aiplatform.PipelineJob(
            display_name="logistic-inference-pipeline-run",
            template_path="logistic_inference_pipeline.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "x_train_uri": X_TRAIN_URI,
                "y_train_uri": Y_TRAIN_URI,
                "x_test_uri": X_TEST_URI,
            },
        )

        job.run()
