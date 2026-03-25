FROM python:3.10-slim

# Accept the Run ID from the pipeline
ARG RUN_ID
ENV MLFLOW_RUN_ID=$RUN_ID

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Requirement: simulated command to "download" model from MLflow
RUN echo "Fetching model artifacts for Run ID: ${MLFLOW_RUN_ID}" > /app/load_report.txt

CMD ["python", "-c", "print('Deployment successful for Run: ' + '$MLFLOW_RUN_ID')"]