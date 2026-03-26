#!/bin/bash
pkill -f mlflow
sleep 2
nohup /home/ubuntu/mlflow-env/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://cosmo-gas-vision-storage/mlflow-artifacts --host 0.0.0.0 --port 5000 --allowed-hosts "*" --cors-allowed-origins "*" --serve-artifacts > mlflow.log 2>&1 < /dev/null &
disown
exit 0
