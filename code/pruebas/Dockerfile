FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY knn_iris.onnx .
COPY gnb_iris.onnx .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


#FROM python:3.12

#WORKDIR /app

#COPY requirements.txt .
#COPY main.py .
#COPY knn_iris.onnx .

#RUN pip install --no-cache-dir -r requirements.txt

#EXPOSE 8000

# CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
