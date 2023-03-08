# Databricks notebook source
# MAGIC %md ## Iris Train - Git Repos example
# MAGIC * Test Databricks Git Repos notebooks and MLflow interaction
# MAGIC * Simple train and register a model

# COMMAND ----------

dbutils.widgets.text("1. Experiment", "")
experiment_name = dbutils.widgets.get("1. Experiment")
if experiment_name == "": 
    experiment_name = None 
print("experiment:", experiment_name)

dbutils.widgets.text("2. Registered model", "")
registered_model = dbutils.widgets.get("2. Registered model")
if registered_model == "": 
    registered_model = None 
print("registered_model:", registered_model)

dbutils.widgets.text("3. Info", "")
info = dbutils.widgets.get("3. Info")
print("info:", info)

# COMMAND ----------

if experiment_name:
    import mlflow
    mlflow.set_experiment(experiment_name)

# COMMAND ----------

import time
now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
now

# COMMAND ----------

import os
import mlflow
from sklearn import svm, datasets

with mlflow.start_run(run_name=now) as run:
    print("run_id:", run.info.run_id)
    print("experiment_id:", run.info.experiment_id)
    iris = datasets.load_iris()
    mlflow.log_metric("degree", 5)
    model = svm.SVC(C=2.0, degree=5, kernel="rbf")
    model.fit(iris.data, iris.target)
    mlflow.log_param("C", 2.0)
    mlflow.log_param("degree", 5)
    mlflow.set_tag("timestamp", now)
    mlflow.set_tag("info", info)
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.databricks_runtime", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
    mlflow.sklearn.log_model(model, "model", registered_model_name=registered_model)
