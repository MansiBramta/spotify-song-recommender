FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY hybrid_recommedation_system/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY hybrid_recommedation_system/hybrid_recommedation_system/ ./
COPY hybrid_recommedation_system/data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]