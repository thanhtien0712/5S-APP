FROM python:3.9-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư mục làm việc
WORKDIR /code

# Copy requirements và cài đặt
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Tạo thư mục quyền user bình thường (HuggingFace bắt buộc)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy toàn bộ code vào
COPY --chown=user . $HOME/app

# Tạo các thư mục cần thiết
RUN mkdir -p dataset/images/train static/results

# Chạy ứng dụng FastAPI thông qua Uvicorn, port mặc định của Hugging Face là 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]