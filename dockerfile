# ============================================================
# BASE
# ============================================================
FROM python:3.10-slim

# ============================================================
# SISTEMA + PAQUETES NECESARIOS PARA DVC Y MLflow
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# DIRECTORIO DE TRABAJO
# ============================================================
WORKDIR /app

# ============================================================
# COPIAR ARCHIVOS FUNDAMENTALES
# ============================================================
COPY requirements.txt .
COPY dvc.yaml dvc.lock params.yaml ./
COPY .dvc/ .dvc/

# ============================================================
# COPIAR EL CÓDIGO DEL PROYECTO
# ============================================================
COPY scripts/ scripts/
COPY serving/ serving/

# Data y models solo si los usas directamente SIN DVC
COPY data/ data/
COPY models/ models/

# Si deseas notebooks dentro del contenedor, habilita esta línea:
# COPY notebooks/ notebooks/

# ============================================================
# INSTALAR DEPENDENCIAS DEL PROYECTO
# ============================================================
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install dvc[gs,s3,ssh] mlflow

# ============================================================
# EXPONER PUERTOS PARA MLflow
# ============================================================
EXPOSE 5000

# ============================================================
# ENTRYPOINT INTERACTIVO
# Puedes cambiarlo por el comando que quieras correr.
# ============================================================
CMD ["/bin/bash"]
