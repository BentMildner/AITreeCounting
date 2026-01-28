FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System-Abhängigkeiten (OpenCV + Streamlit + Leafmap)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# Ports
EXPOSE 5000
EXPOSE 8503

# Beide Prozesse starten
CMD ["bash", "-c", "python src/prototype/backend/app.py & streamlit run src/prototype/frontend/streamlit_app.py --server.address=0.0.0.0"]
