# Databricks notebook source
# MAGIC %md ## Iris Train - Repos
# MAGIC * Train and register a model.

# COMMAND ----------

dbutils.widgets.text("Registered model", "")
registered_model = dbutils.widgets.get("Registered model")
if registered_model == "": registered_model = None 

# COMMAND ----------

import mlflow
from sklearn import svm, datasets

with mlflow.start_run() as run:
    print("run_id:",run.info.run_id)
    print("experiment_id:",run.info.experiment_id)
    iris = datasets.load_iris()
    mlflow.log_metric("degree", 5)
    model = svm.SVC(C=2.0, degree=5, kernel="rbf")
    model.fit(iris.data, iris.target)
    mlflow.sklearn.log_model(model, "model", registered_model_name=registered_model)
