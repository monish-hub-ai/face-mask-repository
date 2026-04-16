FROM python:3.10-slim



# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (default 5000, but often overridden by PaaS environment variables)
EXPOSE 5000

# Start gunicorn, binding to 0.0.0.0 and dynamically reading the PORT env variable
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 app:app
