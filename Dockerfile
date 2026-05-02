# Stage 2b training container — bakes source + minimum deps for SageMaker.
#
# Image expectations:
#   - Base provides CUDA 12.x runtime compatible with ml.g5.xlarge (NVIDIA A10G)
#   - PyTorch with CUDA 12.x wheels
#   - SageMaker entrypoint = python -u /opt/ml/code/experiments/nerf/pipeline.py
#   - SageMaker spot checkpointing path = /opt/ml/checkpoints/ (auto-synced to S3)
#
# Build:
#   bash scripts/build_and_push_ecr.sh
#
# Notes:
#   - Source is baked at /opt/ml/code/ so SAGEMAKER_CODE_S3_URI staging is optional.
#   - Heavy doc-generation deps (pandoc, weasyprint, fpdf, dvc) are intentionally
#     excluded — not needed for training.
#   - Sherwood data is NOT baked in. The pipeline's dummy-data fallback covers
#     the memory-smoke run; for the inner-tier sweep, mount or download from
#     S3 at job start.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/ml/code \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/ml/code

# Slim training-only dep set; see pyproject.toml for the full project deps.
RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "h5py>=3.8" \
    "mlflow>=3.1.4" \
    "boto3>=1.41" \
    "python-dotenv>=1.2" \
    "scipy>=1.10"

# Source layout. Keep narrow — only what training needs.
COPY src/data/ ./src/data/
COPY src/models/ ./src/models/
COPY experiments/nerf/ ./experiments/nerf/
COPY pyproject.toml ./pyproject.toml

# SageMaker convention: checkpoints under /opt/ml/checkpoints sync to S3
RUN mkdir -p /opt/ml/checkpoints

ENTRYPOINT ["python", "-u", "/opt/ml/code/experiments/nerf/pipeline.py"]
