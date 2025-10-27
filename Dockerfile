# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for faster builds)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY nip_sugestions.py .
COPY preprocess.py .
COPY therapist.py .
COPY lstm_forecast.py .
COPY nn_classifier.py .
COPY app.py .
COPY apprun.py .
# Expose Flask’s default port
EXPOSE 5000

# Default command when the container runs
CMD ["python", "backend/backend/app.py"]

