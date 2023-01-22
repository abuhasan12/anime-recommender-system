FROM python:3.9

WORKDIR /anime-recommender-system

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/anime-recommender-system"

EXPOSE 8080

LABEL maintainer="Abu Hasan" \
      version="1.0"

CMD ["python", "app/api.py"]