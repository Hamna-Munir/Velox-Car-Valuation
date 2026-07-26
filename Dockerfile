FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY velox/ velox/
COPY data/ data/
COPY app.py train_model.py ./
COPY .streamlit/ .streamlit/

# Train once at build time so the container starts instantly (skip if you'd
# rather train on first request — see velox.model.load_models).
RUN python train_model.py

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
