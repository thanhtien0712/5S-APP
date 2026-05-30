@echo off
echo ==================================================
echo DANG THIET LAP MOI TRUONG PYTHON...
echo ==================================================

IF NOT EXIST "venv" (
    echo 1. Dang tao Moi truong ao Virtual Environment...
    python -m venv venv
)

echo 2. Kich hoat moi truong ao...
call venv\Scripts\activate

echo 3. Cai dat thu vien (Neu chua co)...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ==================================================
echo KHOI DONG SERVER FASTAPI...
echo (Ban co the truy cap tu dien thoai hoac may tinh khac)
echo (Thong qua dia chi: http://<IP_MAY_TINH>:8000)
echo ==================================================
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause