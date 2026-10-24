FROM python:3.12-slim

WORKDIR /app

# cpu-only torch keeps the image small; the api only embeds short questions, so a gpu isn't needed here
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY banklens ./banklens
COPY api ./api
COPY app ./app
COPY eval ./eval

# the index is built on my own machine (with the gpu) and mounted in, so the container starts fast
VOLUME ["/app/data"]
EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
