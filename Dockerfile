# Italian Real Estate Pipeline - Main Application Container
# GPU-enabled container with TensorFlow support for synthetic data generation
#
# Build: docker build -t italian-real-estate .
# Run:   docker-compose up -d
#
# Author: Leonardo Pacciani-Mori
# License: MIT

# =============================================================================
# Base Image: NVIDIA CUDA with cuDNN for GPU-accelerated TensorFlow
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    # Database client libraries
    libpq-dev \
    # Build tools
    gcc \
    g++ \
    # MongoDB shell for health checks
    gnupg \
    curl \
    ca-certificates \
    unzip \
    tzdata \
    # Utilities
    git \
    && rm -rf /var/lib/apt/lists/*

# Install MongoDB shell (mongosh) for health checks
RUN curl -fsSL https://pgp.mongodb.com/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg \
    && echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-7.0.list \
    && apt-get update \
    && apt-get install -y mongodb-mongosh \
    && rm -rf /var/lib/apt/lists/*

# Install PostgreSQL client for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome and matching ChromeDriver for Selenium
RUN curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-linux.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

RUN CHROME_VERSION="$(google-chrome --version | awk '{print $3}')" \
    && CHROME_MAJOR="$(echo ${CHROME_VERSION} | cut -d. -f1)" \
    && export CHROME_MAJOR \
    && DRIVER_URL="$(python3 -c 'import json, os, urllib.request; major=os.environ["CHROME_MAJOR"]; url="https://googlechromelabs.github.io/chrome-for-testing/latest-versions-per-milestone-with-downloads.json"; data=json.load(urllib.request.urlopen(url)); linux=[d["url"] for d in data["milestones"][major]["downloads"]["chromedriver"] if d["platform"]=="linux64"]; print(linux[0])')" \
    && curl -sS -o /tmp/chromedriver.zip "${DRIVER_URL}" \
    && unzip /tmp/chromedriver.zip -d /tmp/chromedriver \
    && mv /tmp/chromedriver/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver /tmp/chromedriver.zip

# =============================================================================
# Python Setup
# =============================================================================
# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Application Setup
# =============================================================================
WORKDIR /app

# Copy project files needed for installation/runtime
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY dags/ dags/

# Install Python dependencies (full project)
# Note: TensorFlow will automatically use GPU when available in this CUDA container
RUN pip install --no-cache-dir .

# Create data directory
RUN mkdir -p data

# =============================================================================
# Environment Configuration
# =============================================================================
ENV PYTHONPATH=/app/src
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Ensure a valid timezone configuration exists for pendulum/airflow logging.
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo "${TZ}" > /etc/timezone

# Selenium/Chrome configuration
ENV CHROME_BIN=/usr/bin/google-chrome
ENV CHROMEDRIVER_PATH=/usr/local/bin/chromedriver

# =============================================================================
# Entrypoint
# =============================================================================
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command: launch the interactive TUI
CMD ["python", "scripts/pipeline_tui.py"]
