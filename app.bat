@echo off
cd /d "%~dp0"

if not exist "models\fraud_model.pkl" (
    pip install -r requirements.txt
    python src\train.py
)

netstat -ano | findstr :8000 >nul
if errorlevel 1 (
    python src/api.py
    timeout /t 2 >nul
)

python -m streamlit run src/streamlit_app.py
