# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set the HF Spaces required user and working directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Output directly to the terminal stdout to prevent missing logs
ENV PYTHONUNBUFFERED=1

# Install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main server & models
COPY --chown=user . .

# Hugging Face Spaces uses 7860 by default
EXPOSE 7860

# Start server safely
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
