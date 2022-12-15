# Databricks notebook source
# MAGIC %md ## Iris Train - Git Repos example
# MAGIC * Test Databricks Git Repos notebooks and MLflow interaction
# MAGIC * Simple rain and register a model.

# COMMAND ----------

dbutils.widgets.text("Registered model", "")
registered_model = dbutils.widgets.get("Registered model")
if registered_model == "": registered_model = None 
print("registered_model:",registered_model)

# COMMAND ----------

import time
ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

# COMMAND ----------

import mlflow
from sklearn import svm, datasets

with mlflow.start_run(run_name=ts) as run:
    print("run_id:",run.info.run_id)
    print("experiment_id:",run.info.experiment_id)
    iris = datasets.load_iris()
    mlflow.log_metric("degree", 5)
    model = svm.SVC(C=2.0, degree=5, kernel="rbf")
    model.fit(iris.data, iris.target)
    mlflow.sklearn.log_model(model, "model", registered_model_name=registered_model)
