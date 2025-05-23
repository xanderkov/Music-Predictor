# Обзор кода: обработка аудио и классификация музыкальных жанров


![Схема нового пайплайна](https://github.com/xanderkov/Music-Predictor/blob/dl_exp_2/notebooks/Bagantsova/music_tagging/pipeline_desc.png)


## 1. Загрузка данных
Мы используем два источника данных:
1. **Локальные аудиофайлы** — загружаем WAV-файлы из папок, где каждая папка представляет музыкальный жанр.
2. **Готовый датасет** (`luli0034/music-tags-to-spectrogram`) — загружаем уже подготовленные спектрограммы.

## 2. Предварительная обработка аудио
- Аудиофайлы разбиваются на отрезки **по 4 секунды** с перекрытием **в 2 секунды**.
- Из аудио создаются **мел-спектрограммы**, которые затем **сохраняются в сжатом формате `.npz`**.
- Все сохраненные файлы хранятся в кэше (`spectrogram_cache` и `image_cache`), а в дальнейшем модель использует **только пути к файлам**, а не сами данные в оперативной памяти. Это позволяет **значительно экономить RAM**.

## 3. Обработка изображений спектрограмм
- Спектрограммы загружаются, разделяются на небольшие части и сохраняются в `.npz` файлах.
- Используется **параллельная обработка**, чтобы ускорить процесс.

## 4. Архитектура нейросети
Модель **SpectrogramCNN** — это **сверточная нейросеть**, состоящая из:
- **Трех блоков сверточных слоев** с `BatchNorm` и `ReLU`,
- **Макспулинга** для уменьшения размерности,
- **Дропаута** для регуляризации,
- **Глобального усредненного пулинга**,
- **Полносвязного классификатора**.

## 5. Функция ошибки
Мы используем **Asymmetric Focal Loss**, которая помогает лучше работать с несбалансированными классами. Эта функция ошибки:
- Уменьшает вес частых классов (чтобы модель не переучивалась на них),
- Усиливает влияние редких классов.

## 6. Обучение модели
- Обучение идет **с визуализацией** (графики потерь и точности).
- Используется **Adam** с L2-регуляризацией (`weight_decay`).
- Каждые 5 эпох обновляются графики, чтобы отслеживать процесс.

## 7. Оценка модели
- После обучения модель тестируется, и мы получаем **отчет о классификации** (`classification_report`).
- Также вычисляется **точность** (`accuracy_score`).
- В время обучения использую также метрику доли правильных ответов из тех, что должны быть предсказаны как 1 - таким образом, визуально ошибка не дает ввести в заблуждение, так как предсказывать нулевые значение, которых очень много, ума не надо, хотим все-таки добиться именно угадывания класса.

## 8. Классификация жанров
Мы **вручную выделяем** самые популярные жанры, чтобы уменьшить количество классов и улучшить качество классификации.


# Основные изменения на стороне бэкенда

- Реализована начальная версия бэкенда для обработки текстовых данных.
- Перемещены и исправлены скрипты EDA, очень сильно улучшена их организационная структура.
- Учтены комментарии после код-ревью.
- Добавлен Streamlit-фронтенд для работы с текстовыми данными.
- Внедрен этап обучения модели даже при отсутствии данных.


# Классификации жанров песен с PyTorch и трансформерами Bert

Задача классификации текстов на основе библиотеки PyTorch и трансформеров от Hugging Face (RobertaForSequenceClassification и BertForSequenceClassification), включая подготовку данных, обучение модели, ее оценку и предсказание меток для новых данных.


## Описание функционала и используемых библиотек:
-	Определение устройства для вычислений. Если доступна, используется GPU, иначе CPU.
-	Предобработка данных. Загрузка файлов, визуализация длин текстов, преобразование жанров в числовые метки (label), токенизация текстов, формирование загрузчиков данных (DataLoader)
-	Создание модели. Загрузка моделей RoBERTa и BERT, создание оптимизатора Adam, опциональная заморозка слоев модели для возможности дообучения
-	Обучение модели на тренировочных данных (функция train_model из модуля functions)
-	Оценка модели на тестовом наборе данных (функция evaluate, которая возвращает потерю, точность, F1-меру), построение матрицы ошибок (confusion matrix) для визуализации результатов
-	Пример предсказаний (функция predict из модуля functions) - результаты добавляются в датафрейм с сохранением в CSV-файл
-	Используемые модули/библиотеки. Для работы с данными (Pandas), визуализации (Seaborn, Matplotlib), обработки текстов (Transformers), обучения моделей (PyTorch) и вычисления метрик (Scikit-learn).
## Описание структуры проекта и файлов:
-	song_lyrics_20250309.ipynb - основной файл
-	functions.py - модуль с основными функциями для работы в song_lyrics_20250309.ipynb
## Результаты экспериментов:
-	Улучшение качества по метрике accuracy на 10% (с 0,62 до 0,68).
 
-	Комментарии:
*	Обучение происходило всего на 3 эпохах, с увеличением количества эпох предположительно улучшение до 0,73. 
*	Не было детальной предобработки текстов (кастомной токенизации, лемматизации и т.д.)
*	Прежние эксперименты проводились с простыми Dense-моделями и использованием библиотеки TensorFlow, лучшие результаты были на 8 эпохе, при этом обучение происходило заметно быстрее.

