# builder image 

FROM python:3.11-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip freeze | grep -E "flask|streamlit|pandas|scikit-learn|requests|joblib" > prod-requirements.txt

FROM python:3.11-slim as prod
WORKDIR /app
COPY --from=builder /build/prod-requirements.txt .
RUN pip install --no-cache-dir -r prod-requirements.txt

COPY ui.py app.py ./
COPY model/ model/

EXPOSE 5000 8501
CMD python app.py & streamlit run ui.py --server.port=8501 --server.address=0.0.0.0