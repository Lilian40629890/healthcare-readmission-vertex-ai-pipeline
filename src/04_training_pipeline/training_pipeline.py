"""Vertex AI training pipeline: train a Logistic Regression model and evaluate it
on the validation set (Accuracy, F1, PR-AUC).

Transcribed from the project's final presentation (slides 20-21). Bucket paths
are read from environment variables instead of being hard-coded.
"""

import os

from kfp import compiler, dsl
from kfp.dsl import Input, Metrics, Model, Output, component

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket")
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

X_TRAIN_URI = f"gs://{BUCKET_NAME}/healthcare_X_train_final.csv"
Y_TRAIN_URI = f"gs://{BUCKET_NAME}/healthcare_y_train_final.csv"
X_VAL_URI = f"gs://{BUCKET_NAME}/healthcare_X_validation.csv"
Y_VAL_URI = f"gs://{BUCKET_NAME}/healthcare_y_validation.csv"


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "joblib", "gcsfs"],
)
def train_logistic_model(
    x_train_uri: str,
    y_train_uri: str,
    model_output: Output[Model],
):
    import os

    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = pd.read_csv(x_train_uri)
    y_train_df = pd.read_csv(y_train_uri)

    X_train = X_train.loc[:, ~X_train.columns.str.contains("^Unnamed")]

    y_train = y_train_df.iloc[:, 0].values.ravel()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    os.makedirs(model_output.path, exist_ok=True)

    joblib.dump(model, os.path.join(model_output.path, "model.joblib"))

    print("Training completed successfully.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "joblib", "gcsfs"],
)
def evaluate_logistic_model(
    model_input: Input[Model],
    x_val_uri: str,
    y_val_uri: str,
    metrics: Output[Metrics],
):
    import os

    import joblib
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
    )

    X_val = pd.read_csv(x_val_uri)
    y_val_df = pd.read_csv(y_val_uri)

    X_val = X_val.loc[:, ~X_val.columns.str.contains("^Unnamed")]
    y_val = y_val_df.iloc[:, 0].values.ravel()

    model = joblib.load(os.path.join(model_input.path, "model.joblib"))

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Area under precision/recall curve:", pr_auc)
    print("Confusion Matrix:")
    print(cm)

    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("f1_score", float(f1))
    metrics.log_metric("area_under_precision_recall_curve", float(pr_auc))


@dsl.pipeline(
    name="logistic-training-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def logistic_training_pipeline(
    x_train_uri: str = X_TRAIN_URI,
    y_train_uri: str = Y_TRAIN_URI,
    x_val_uri: str = X_VAL_URI,
    y_val_uri: str = Y_VAL_URI,
):
    train_task = train_logistic_model(
        x_train_uri=x_train_uri,
        y_train_uri=y_train_uri,
    )

    evaluate_task = evaluate_logistic_model(
        model_input=train_task.outputs["model_output"],
        x_val_uri=x_val_uri,
        y_val_uri=y_val_uri,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=logistic_training_pipeline,
        package_path="logistic_training_pipeline.json",
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
            display_name="logistic-training-pipeline-run",
            template_path="logistic_training_pipeline.json",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "x_train_uri": X_TRAIN_URI,
                "y_train_uri": Y_TRAIN_URI,
                "x_val_uri": X_VAL_URI,
                "y_val_uri": Y_VAL_URI,
            },
        )

        job.run()
