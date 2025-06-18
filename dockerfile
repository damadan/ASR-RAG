# Dockerfile
FROM python:3.10-slim

# Устанавливаем системные зависимости для pyannote, ffmpeg и pydub
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        libsndfile1 \
        git && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY . .

# Экспорт порта Streamlit
EXPOSE 8501

# Переменные окружения для Streamlit
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_PORT=8501

# Точка входа
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
