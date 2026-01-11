#!/bin/bash
# Italian Real Estate Pipeline - Container Entrypoint
#
# This script runs when the container starts and ensures all database
# services are available before starting the application.
#
# Author: Leonardo Pacciani-Mori
# License: MIT

set -e

echo "=============================================="
echo "  Italian Real Estate Pipeline"
echo "  Container Starting..."
echo "=============================================="
echo ""

# =============================================================================
# Wait for MongoDB
# =============================================================================
echo "Waiting for MongoDB at ${MONGODB_HOST:-mongodb}:${MONGODB_PORT:-27017}..."

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if mongosh --host "${MONGODB_HOST:-mongodb}" --port "${MONGODB_PORT:-27017}" --eval "db.adminCommand('ping')" &>/dev/null; then
        echo "MongoDB is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for MongoDB... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: MongoDB did not become available in time"
    exit 1
fi

# =============================================================================
# Wait for PostgreSQL
# =============================================================================
echo "Waiting for PostgreSQL at ${POSTGRES_HOST:-postgres}:${POSTGRES_PORT:-5432}..."

RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if pg_isready -h "${POSTGRES_HOST:-postgres}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-listing_website}" &>/dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for PostgreSQL... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: PostgreSQL did not become available in time"
    exit 1
fi

# =============================================================================
# Display GPU Status
# =============================================================================
echo ""
echo "Checking GPU availability..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'  GPU available: {len(gpus)} device(s)')
        for gpu in gpus:
            print(f'    - {gpu.name}')
    else:
        print('  No GPU detected - using CPU')
except Exception as e:
    print(f'  GPU check failed: {e}')
" 2>/dev/null || echo "  Could not check GPU status"

# =============================================================================
# Start Application
# =============================================================================
echo ""
echo "=============================================="
echo "  Starting application..."
echo "=============================================="
echo ""

# Execute the command passed to the container
exec "$@"
