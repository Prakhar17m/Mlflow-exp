import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

df = pd.read_csv("data/train.csv")

df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

target = "count"
drop_cols = ["datetime", "count", "casual", "registered"]
X = df.drop(columns=drop_cols)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}


model = RandomForestRegressor(**params)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

plt.figure()
plt.scatter(y_test, pred)
plt.xlabel("Actual count")
plt.ylabel("Predicted count")
plt.title("Actual vs Predicted")
plot_path = "actual_vs_pred.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()


# local tacking folder

mlflow.set_experiment("bike-demand-tracking")

with mlflow.start_run(run_name="randomforest-baseline"):

    # log parameters
    mlflow.log_params(params)

    # log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # artifacts
    mlflow.log_artifact(plot_path)

    # log model
    mlflow.sklearn.log_model(model, name="model")

    # set tags
    mlflow.set_tag("dataset", "bike sharing train csv file")
    mlflow.set_tag("model type", "random forest regressor")

    # autolog

    print("==== Baseline Results ====")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
