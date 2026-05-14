Для запуска кода нужно  
Скачать данные  
dvc pull 

Создать файл .env  
Внести в него строку  
HUGGINGFACE_TOKEN=<ваш токен>  

Выполнить в bash uv sync

Запуск обучения uv run uspppm/train.py  
Запуск инференса uv run uspppm/inference.py
