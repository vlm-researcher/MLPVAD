FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Set working directory and Python path for src/ imports
WORKDIR /app
ENV PYTHONPATH=/app/src

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install opencv-python
RUN pip install -e ./LAVIS

EXPOSE 8000

CMD ["uvicorn", "infer_video:app", "--host", "0.0.0.0", "--port", "8000"]
