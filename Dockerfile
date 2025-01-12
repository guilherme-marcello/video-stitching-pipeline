FROM python:3.11-slim

WORKDIR /app

COPY . /app

ENV PYTHONUNBUFFERED=1

# install dependencies
RUN pip install --no-cache-dir numpy scipy matplotlib

# run the main script
ENTRYPOINT ["python", "main.py"]