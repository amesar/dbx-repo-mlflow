# Databricks notebook source
# MAGIC %md ## Iris Train - Repos - 02
# MAGIC * Train and register a model.

# COMMAND ----------

import mlflow

dbutils.widgets.text("Registered model", "")
registered_model = dbutils.widgets.get("Registered model")
if registered_model == "": registered_model = None 

# COMMAND ----------

# MAGIC %md 
# MAGIC RestException: INVALID_PARAMETER_VALUE: MLflow experiment creation is not permitted in a repo. Use the default experiment for a notebook in a repo or create an MLflow experiment in the workspace.

# COMMAND ----------

exp_name = "/Repos/andre.mesarovic@databricks.com/dbx-repo-mlflow/foo"
mlflow.set_experiment(exp_name)

# COMMAND ----------

exp_name = "/Users/andre.mesarovic@databricks.com/experiments/dbx-repo-mlflow"
mlflow.set_experiment(exp_name)

# COMMAND ----------

from sklearn import svm, datasets

with mlflow.start_run() as run:
    print("run_id:",run.info.run_id)
    print("experiment_id:",run.info.experiment_id)
    iris = datasets.load_iris()
    mlflow.log_metric("degree", 5)
    model = svm.SVC(C=2.0, degree=5, kernel="rbf")
    model.fit(iris.data, iris.target)
    mlflow.sklearn.log_model(model, "model", registered_model_name=registered_model)

# COMMAND ----------


