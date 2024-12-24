# Как с этим работать

## Структура проекта

Состоит из Бэкенда -- music_predictor_backend и фронтеда -- music_predictor_stramlit.

### Бэкенда структура

```angular2html
1. dto - модели для api
2. repository - папка для работы с моделями
3. routes - пути api
4. services - бизнес-логика приложения. 
Преобразования данных пользователя в данные приложения
5. settings - содержит глобальную конфигурацию бэкенда
```

### Streamlit структура

```angular2html
1. dto - модели для api
2. services - бизнес-логика приложения. Кнопочки 
3. settings - содержит глобальную конфигурацию 
```

## Docker

```bash
docker compose --profile monitoring up -d --build
```


Без системы мониторинга 

```bash
docker compose up -d --build
```

## Нативно

### Запуск 


```bash
poetry shell
poetry install
python music_predictor_backend/main.py
streamlit run music_predictor_stramlit/client.py
```

### Редактирование 

#### Добавление либы
```bash
poetry add lib
```

#### Подготовка кода к коммиту
```bash
black music_predictor_backend
black music_predictor_streamlit

pre-commit 
```