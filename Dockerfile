# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Ensure dependencies are installed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the local requirements.txt file to the container
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the required Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set the entry point to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "tender_eval_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
