Для запуска кода нужно  
Скачать данные  
dvc pull 

Создать файл .env  
Внести в него строку  
HUGGINGFACE_TOKEN=<ваш токен>  

выполнить в bash uv sync

Запуск обучения uv run train.py  
Запуск инференса uv run infer.py
