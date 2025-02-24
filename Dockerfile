FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app

# Create start script
RUN echo '#!/bin/bash\n\
    streamlit run app/main.py --server.port=$STREAMLIT_SERVER_PORT --server.address=0.0.0.0' > /app/start.sh && \
    chmod +x /app/start.sh

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

# Expose the port Streamlit will run on
EXPOSE ${STREAMLIT_SERVER_PORT}

# Command to run the application
ENTRYPOINT ["/app/start.sh"] 