FROM python:3.10-slim

# Accept the Run ID from the pipeline
ARG RUN_ID
ENV MLFLOW_RUN_ID=$RUN_ID

WORKDIR /app

# Copy requirements if exists
COPY requirements.txt .
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; else echo "No requirements.txt found"; fi

# Copy the check_threshold script for verification
COPY check_threshold.py .

RUN echo "Fetching model artifacts for Run ID: ${MLFLOW_RUN_ID}" > /app/load_report.txt
RUN echo "Model download simulation complete" >> /app/load_report.txt

# run when container starts
CMD ["python", "-c", "print('Deployment successful for Run: ' + '$MLFLOW_RUN_ID'); print('Model is ready for serving')"]
