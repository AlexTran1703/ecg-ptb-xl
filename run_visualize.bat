@echo off

call .\.venv\Scripts\activate.bat
cd source\visualize_data
python -m main
pause