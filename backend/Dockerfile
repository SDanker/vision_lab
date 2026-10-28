FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Santiago \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata libgomp1 procps ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Pesos preentrenados (ImageNet) de todo el catálogo incluidos en la imagen: entrenar no depende de internet.
# Keras 2.15 los guarda en ~/.keras (si no existe, cae en /tmp/.keras), así que se descargan con HOME=/opt
# en /opt/.keras y luego ~/.keras del usuario de la app apunta a esa carpeta.
COPY app/ml/architectures.py /tmp/architectures.py
RUN mkdir -p /opt/.keras && cd /tmp && HOME=/opt python -c "\
from tensorflow import keras; from architectures import ARCHITECTURES as A; \
[getattr(keras.applications, s['keras'])(include_top=False, weights='imagenet', input_shape=(224,224,3), **s['kwargs']) for s in A.values()]; \
keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(160,160,3))" \
    && echo "Pesos preentrenados en la imagen: $(ls /opt/.keras/models | wc -l) archivos" \
    || echo "ADVERTENCIA: no se pudieron pre-descargar los pesos; se descargarán al primer entrenamiento."

COPY . .
RUN chmod +x /app/entrypoint.sh \
    && useradd -m -u 1000 app \
    && ln -s /opt/.keras /home/app/.keras \
    && mkdir -p /data/work && chown -R app:app /data /app /opt/.keras
USER app

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
