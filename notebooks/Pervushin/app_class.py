# streamlit run app_classificate.py --server.maxUploadSize 3000 --server.maxMessageSize 500


import streamlit as st

import pandas as pd
import numpy as np

import re
import sys
from itertools import chain
import nltk
from io import BytesIO
import datetime

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt

pd.set_option("styler.render.max_elements", 10000000)

SS = nltk.stem.SnowballStemmer('english') # стемизатор
WORD = re.compile(r'\w+')


@st.cache_data()
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8') # , nrows=50
    return df

@st.cache_data
def df_to_csv(df_csv):
    df_csv[df_csv.columns[1:]] = df_csv[df_csv.columns[1:]].round(2)#.applymap(lambda x: str(x).replace('.', ','))
    return df_csv.to_csv().encode('utf-8')

@st.cache_data
def df_to_excel(df_excel):
    in_memory_fp = BytesIO()
    df_excel.to_excel(in_memory_fp, index = False)
    in_memory_fp.seek(0)
    return in_memory_fp.read()


def del_session_elements(): # keys
    # for key in st.session_state.keys():
    #     if key in keys:
    #         del st.session_state[key]
    st.write(st.session_state.keys())
    for key in st.session_state.keys():
        del st.session_state[key]


def find_col_candidates(df, reverse = False):
    tuples = []
    for col in df.select_dtypes(include=['object']).columns: # df.select_dtypes(include=['object']).columns
        tuples.append((col, round(df[col].nunique() / len(df), 4)))
    sorted_tuples = sorted(tuples, key=lambda x: x[-1], reverse=reverse)[0]

    return sorted_tuples[0]


# Функция выделения категорий для классификации по количеству примеров в категории
def get_classification_cat(s, cat_len_thr):
    '''
    Функция выделения категорий для классификации по количеству примеров в категории
    s - серия с данными (категориями),
    cat_len_thr - трешхолд: если меньше единицы, воспринимается как процент данных от всей выборки, иначе количество примеров в категории
    Возвращает количество примеров по каждой категории согласно трешхолду
    '''
    s_vc = s.value_counts()
    if cat_len_thr < 1:
        share = cat_len_thr
    else:
        share = s_vc.loc[lambda x: x > cat_len_thr].sum() / len(s)
    j = 0
    for i in s_vc:
        j += i
        if j >= len(s) * share:
            print(f'Нижний порог количества примеров в категории {i}, всего примеров {j}')
            break
    s_res = s_vc.loc[lambda x: x >= i]
    return s_res


# Функция для токенизации и фильтрации текстов
def _reg_tokenize(text):
    """
    Функция токенизации текста на основе регулярных выражений
    """
    words = WORD.findall(text)
    return words

def get_tokens(text):
    """
    Функция токенизации текста, а также фильтрации и нормализации токенов
    Параметры:
        text : str, текст
    Возвращает:
        tokens : list, список отфильтрованных токенов
    """
    text = text.replace(r'([^\w\s]+)', ' \\1 ').strip().lower() # вклинивание пробелов между словами и знаками препинания, приведение к нижнему регистру
    # print(text)
    tokens = _reg_tokenize(text) # токенизация
    # print(tokens)
    tokens = [element for element in list(chain(*[re.split(r'\W+', element) for element in tokens])) if element != ''] # разделение составных элементов слов #####
    # print(tokens)
    tokens = list(chain(*[re.findall(r'\d+|\D+', element) if element.isalnum() else element for element in tokens])) # разбиение токенов, состоящих из букв и цифр
    # print(tokens)
    return tokens





# Функция компиляции и обучения модели нейронной сети
def compile_train_model(model, 
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        optimizer='adam',
                        epochs=50,
                        batch_size=128,
                        figsize=(20, 5)):

    # Компиляция модели
    model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    model.summary()

    # Обучение модели с заданными параметрами
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val))

    # Вывод графиков точности и ошибки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе')
    ax1.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Ошибка на обучающем наборе')
    ax2.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    # plt.show()
    st.pyplot(fig)


# Функция вывода результатов оценки модели на заданных данных
def eval_model(model, x, y_true,
            class_labels=[],
            cm_round=3,
            title='',
            figsize=(10, 10)):
    # Вычисление предсказания сети
    y_pred = model.predict(x)
    # Построение матрицы ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                        np.argmax(y_pred, axis=1),
                        normalize='true')
    cm = np.around(cm, cm_round)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'{title}: матрица ошибок', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Стирание ненужной цветовой шкалы
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)
    # plt.show()
    st.pyplot(fig)    

    # st.write('-'*100)
    # st.write(f'Нейросеть: {title}')

    # for cls in range(len(class_labels)):
    #     # Определяется индекс класса с максимальным значением предсказания (уверенности)
    #     cls_pred = np.argmax(cm[cls])
    #     # Формируется сообщение о верности или неверности предсказания
    #     msg = 'ВЕРНО' if cls_pred == cls else 'НЕВЕРНО'
    #     # Выводится текстовая информация о предсказанном классе и значении уверенности
    #     st.write('Класс: {:<20} {:3.0f}% к классу {:<20} - {}'.format(class_labels[cls],
    #                                                                         100. * cm[cls, cls_pred],
    #                                                                         class_labels[cls_pred],
    #                                                                         msg))
    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    st.text('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))


# Совместная функция обучения и оценки модели нейронной сети
def compile_train_eval_model(model, 
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            class_labels,
                            title='',
                            optimizer='adam',
                            epochs=50,
                            batch_size=128,
                            graph_size=(20, 5),
                            cm_size=(10, 10)):

    # Компиляция и обучение модели на заданных параметрах
    # В качестве проверочных используются тестовые данные
    compile_train_model(model, 
                        x_train, y_train,
                        x_test, y_test,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        figsize=graph_size)

    # Вывод результатов оценки работы модели на тестовых данных
    eval_model(model, x_test, y_test, 
            class_labels=class_labels, 
            title=title,
            figsize=cm_size)



st.header('Классификация') #, divider='grey'
st.markdown('*Обучение моделей и предсказание категорий для данных*')


st.divider()
st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True) 
st.markdown('#### 1. Предобработка данных')

agree = st.checkbox('Начать предобработку данных') # , on_change = del_session_elements

if agree:
    # Загрузка данных
    uploaded_file = st.file_uploader("", type="csv", key='file_uploader')

    if uploaded_file is not None:
        data_load_state = st.text('Идет загрузка ...')
        df_1 = load_data(uploaded_file)
        data_load_state.text(f'1. Данные загружены. Размер: {round(sys.getsizeof(df_1) / (1024 * 1024), 2)}Mb, {df_1.shape[0]} x {df_1.shape[1]}')
        st.session_state['df_1'] = df_1

    # df_1 = pd.read_csv('id10_to_train.csv')
    # st.session_state['df_1'] = df_1

    if 'df_1' in st.session_state:
        df_1 = st.session_state['df_1']

        with st.form("form1"):
            st.write("Выбор столбцов для классификации")
            group_cols = st.multiselect('Выберите столбцы для анализа', list(df_1.columns)) #, default = ['name', 'name.1', '1.0', '2.0', '3.0']
            col1, col2 = st.columns(2)
            with col1:
                pass
            with col2:
                # st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
            
            if submitted:
                # st.write('You selected:', group_cols)
                df_2 = df_1[group_cols]
                st.dataframe(df_2.head(3), use_container_width = True)
                st.session_state['df_2'] = df_2

    if 'df_2' in st.session_state:
        df_2 = st.session_state['df_2']
        st.text(f'2. Выбраны столбцы для классификации: {list(df_2.columns)}. Размер: {round(sys.getsizeof(df_2) / (1024 * 1024), 2)}Mb, {df_2.shape[0]} x {df_2.shape[1]}')


        # Удаление ненормализуемых категорий
        with st.form("form2"):
            st.write("Выбор/исключение категорий")

            cat_col = st.selectbox("Выберите столбец с категорией", options = list(df_2.columns), index = list(df_2.columns).index(find_col_candidates(df_2)), placeholder = find_col_candidates(df_2))

            col1, col2 = st.columns(2)
            with col1:
                str_contains = st.text_input("Введите подстроку поиска категории для удаления", 'classic')
            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
                
            if submitted:
                df_3 = df_2[~(df_2[cat_col].str.contains(str_contains))]
                # st.dataframe(df_3, use_container_width = True)
                st.session_state['cat_col'] = cat_col
                st.session_state['df_3'] = df_3

    if 'cat_col' in st.session_state:
        cat_col = st.session_state['cat_col']

    if 'df_3' in st.session_state:
        df_3 = st.session_state['df_3']
        st.text(f'3. Выбран уровень категорий для классификации. Размер: {round(sys.getsizeof(df_3) / (1024 * 1024), 2)}Mb, {df_3.shape[0]} x {df_3.shape[1]}')


        # Отбор классов по трешхолду
        with st.form("form3"):
            st.write("Отбор категорий по количеству данных")
            col1, col2 = st.columns(2)
            with col1:
                n_examples = st.number_input('Выберите пороговое значение', min_value=0.1, max_value=1000.00, value=0.98, placeholder="Введите число", help = 'Если значение меньше 1, оно указывает на долю от всех данных (отбираются категории, суммарное количество данных которых соответсвует указанной доле); если значение равно или больше 1, то обозначается количество примеров в категории (отбираются категории, где количество примеров не менее указанного)')
            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
                
            if submitted:
                s_1 = get_classification_cat(df_3[cat_col], n_examples)
                st.dataframe(s_1, use_container_width = True)
                st.session_state['s_1'] = s_1


    if 's_1' in st.session_state:
        s_1 = st.session_state['s_1']


        # Получение датафрейма для классификации
        df_4 = df_3[df_3[cat_col].isin(s_1.index)]
        st.text(f'4. Отобраны категории по количеству данных. Размер: {round(sys.getsizeof(df_4) / (1024 * 1024), 2)}Mb, {df_4.shape[0]} x {df_4.shape[1]}')


        # # Преобразования текстов
        with st.form("form4"):
            st.write("Нормализация текстов")
            item_col = st.selectbox("Выберите столбец с текстами", options = list(df_2.columns), index = list(df_2.columns).index(find_col_candidates(df_2, reverse=True)) , placeholder = find_col_candidates(df_2, reverse=True))
            col1, col2 = st.columns(2)
            with col1:
                # text_tokens = st.checkbox("Лемматизация (слова и цифры)", value = True)
                # words_tokens = st.checkbox("Лемматизация (только слова)")
                text_stem_tokens = st.checkbox("Стемминг (слова и цифры)")
                words_stem_tokens = st.checkbox("Стемминг (только слова)")
            with col2:
                n_items = st.number_input('Выберите количество первых слов каждой записи', min_value=1, value=10, placeholder="Введите число")
                submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
            




            if submitted:
                df_5 = df_4.copy()

                if text_stem_tokens or words_stem_tokens:

                    df_5['tokens'] = df_5[item_col].apply(get_tokens) # получение списка токенов
                    st.text(f"Всего уникальных токенов: {df_5['tokens'].explode().nunique()}")

                

                    df_5['stem_tokens'] = df_5['tokens'].apply(lambda x: list(map(SS.stem, x)))
                    st.text(f"Всего уникальных основ: {df_5['stem_tokens'].explode().nunique()}")

                if text_stem_tokens:
                    df_5['stem_tokens_n'] = df_5['stem_tokens'].apply(lambda x: x[:n_items])
                    st.text(f"Уникальных основ (слова и цифры): {df_5['stem_tokens_n'].explode().nunique()}")
                    df_5['text_stem_tokens'] = df_5['stem_tokens_n'].apply(lambda x: ' '.join(x))

                if words_stem_tokens:
                    df_5['words_stem_tokens_n'] = df_5['stem_tokens'].apply(lambda x: list(filter(str.isalpha, x))[:n_items])
                    st.text(f"Уникальных основ (только слова): {df_5['words_stem_tokens_n'].explode().nunique()}")
                    df_5['words_stem_tokens'] = df_5['words_stem_tokens_n'].apply(lambda x: ' '.join(x))

                df_5 = df_5[df_5.columns.drop(list(df_5.filter(['tokens', 'stem_tokens', 'tokens_n', 'words_tokens_n', 'stem_tokens_n', 'words_stem_tokens_n'])))]

                st.dataframe(df_5, use_container_width = True)
                st.session_state['df_5'] = df_5

    if 'df_5' in st.session_state:
        df_5 = st.session_state['df_5']
        st.text(f'5. Тексты нормализованы. Размер: {round(sys.getsizeof(df_5) / (1024 * 1024), 2)}Mb, {df_5.shape[0]} x {df_5.shape[1]}')


        df_5.to_csv('df_c.csv', index = False)
        st.text('6. Подготовка данных для обучения моделей завершена. Данные сохранены в файл')

st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)  
st.divider()
st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True) 
st.markdown('#### 2. Обучение модели')

agree2 = st.checkbox('Начать обучение моделей', on_change = del_session_elements)

if agree2:

    # # Загрузка данных
    # uploaded_file2 = st.file_uploader("", type="csv", key='file_uploader2')

    # if uploaded_file2 is not None:
    #     data_load_state = st.text('Идет загрузка ...')
    #     df_class = load_data(uploaded_file2)
    #     data_load_state.text(f'1. Данные загружены. Размер: {round(sys.getsizeof(df_class) / (1024 * 1024), 2)}Mb, {df_class.shape[0]} x {df_class.shape[1]}')
    #     st.session_state['df_class'] = df_class

    df_class = pd.read_csv('df_c.csv')
    st.session_state['df_class'] = df_class
        
    if 'df_class' in st.session_state:
        df_class = st.session_state['df_class']
        

        # Выбор нормализованных данных для обучения
        with st.form("form5"):
            st.write("Выбор нормализованных данных для обучения")
            st.dataframe(df_class.head())
            col1, col2 = st.columns(2)
            with col1:
                norm_col = st.selectbox("Выберите столбец нужного типа нормализации", options = [col for col in df_class if col.endswith('tokens')], index = 0)
            with col2:
                cat_col = st.selectbox("Выберите столбец с уровнем категории", options = list(df_class.columns), index = list(df_class.columns).index(find_col_candidates(df_class)), placeholder = find_col_candidates(df_class))
            with col1:
                VOCAB_SIZE = st.number_input('Выберите размер словаря', min_value=2000, max_value=30000, value=15000, step = 1000, placeholder="Введите число")
            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
                
            if submitted:
                df_class_2 = df_class[~df_class[norm_col].isna()]
                # st.dataframe(df_3, use_container_width = True)
                st.session_state['norm_col'] = norm_col
                st.session_state['cat_col'] = cat_col
                st.session_state['VOCAB_SIZE'] = VOCAB_SIZE
                st.session_state['df_class_2'] = df_class_2

    if 'df_class_2' in st.session_state:
        df_class_2 = st.session_state['df_class_2']
        st.text(f'2. Примеры {norm_col} и метки классов {cat_col} выбраны для обучения. Размер: {round(sys.getsizeof(df_class_2) / (1024 * 1024), 2)}Mb, {df_class_2.shape[0]} x {df_class_2.shape[1]}')

        # ========== Получение X ========== 

        text_data = df_class_2[norm_col].astype(str).tolist()

        # Обучение токенизатора
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff', lower=True, split=' ', oov_token='неизвестное_слово', char_level=False)
        tokenizer.fit_on_texts(text_data)

        # Сохранение токенизатора в JSON файл
        tokenizer_json = tokenizer.to_json()
        with open('tknz.json', 'w', encoding='utf-8') as f:
            f.write(tokenizer_json)

        st.text(f'3. Токенизатор обучен, веса сохранены')

        # Преобразование текстов в матрицу
        X = tokenizer.texts_to_matrix(text_data)
        st.text(f'4. Тексты примеров для обучения преобразованы в матрицу')

        # ========== Получение y ==========

        class_data = df_class_2[cat_col].tolist()

        # Кодирование меток классов индексами (числами)
        encoder = LabelEncoder()
        class_labels = encoder.fit_transform(class_data)

        CLASS_LIST = encoder.classes_

        CLASS_COUNT = len(CLASS_LIST)

        np.save('cls_lst', CLASS_LIST)

        y = utils.to_categorical(class_labels, CLASS_COUNT)
        st.text(f'5. Метки классов категоризированы')


        with st.form("form6"):
            st.write("Выбор моделей и параметров для обучения")
            MODEL = st.multiselect('Выберите модель', ['BoW + dense'], default=['BoW + dense'],)
            
            col1, col2 = st.columns(2)
            with col1:
                TEST_SIZE = st.slider('Выберите размер тестовой выборки', 0.05, 0.45, 0.2, 0.05)
                
            with col2:
                BATCH_SIZE = st.select_slider('Выберите размер батчей', value  = 4096, options=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

            with col1:
                EPOCHS = st.number_input('Выберите количество эпох обучения', min_value=1, max_value=300, value=7, step=1, placeholder="Введите число")

            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
            
            if submitted:
                # st.write('You selected:', group_cols)
                st.session_state['TEST_SIZE'] = TEST_SIZE
                st.session_state['BATCH_SIZE'] = BATCH_SIZE
                st.session_state['EPOCHS'] = EPOCHS
                st.session_state['MODEL'] = MODEL

        if 'MODEL' in st.session_state:
            MODEL = st.session_state['MODEL']
        if 'TEST_SIZE' in st.session_state:
            TEST_SIZE = st.session_state['TEST_SIZE']
        if 'BATCH_SIZE' in st.session_state:
            BATCH_SIZE = st.session_state['BATCH_SIZE']
        if 'EPOCHS' in st.session_state:
            EPOCHS = st.session_state['EPOCHS']


            # Разделение данных на обучающую и проверочную выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=TEST_SIZE, 
                                                        random_state=42,
                                                        stratify=y)
            
            st.text(f'6. Данные разделены на обучающую и проверочную выборки')
            
            # st.write(X_train.shape, X_test.shape)
            # st.write(y_train.shape, y_test.shape)

            # st.write(f"Количество строк в y_train по классам: {np.bincount(np.argmax(y_train, axis=1))}")
            # st.write(f"Количество строк в y_test по классам: {np.bincount(np.argmax(y_test, axis=1))}")

            model_text_bow_dense = Sequential()
            model_text_bow_dense.add(Dense(100, input_dim=VOCAB_SIZE, activation="relu"))
            model_text_bow_dense.add(Dropout(0.4))
            model_text_bow_dense.add(Dense(100, activation='relu'))
            model_text_bow_dense.add(Dropout(0.4))
            model_text_bow_dense.add(Dense(100, activation='relu'))
            model_text_bow_dense.add(Dropout(0.4))
            model_text_bow_dense.add(Dense(CLASS_COUNT, activation='softmax'))

            st.text(f'7. Выбраны слои нейронной сети')

            # Входные данные подаются в виде векторов bag of words
            compile_train_eval_model(model_text_bow_dense,
                                    X_train, y_train,
                                    X_test, y_test,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    cm_size=(16, 16),
                                    class_labels=CLASS_LIST,
                                    title=MODEL)
            

            model_text_bow_dense.save('mdl.h5')
            st.text(f'8. Модель обучена, веса модели сохранены')


st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)     
st.divider()
st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True) 
st.markdown('#### 3. Предсказание модели')


# session_keys = ['df_', 'diff_cols', 'mean_cols', 'df_filter', 'df_report']
agree3 = st.checkbox('Предсказать категорию для текстов', on_change = del_session_elements) # , args=[session_keys]

if agree3:
    # Загрузка токенизатора из JSON файла
    with open('tknz.json', 'r', encoding='utf-8') as f:
        loaded_tokenizer_json = f.read()
        tokenizer_1 = tokenizer_from_json(loaded_tokenizer_json)
        st.session_state['tokenizer_1'] = tokenizer_1
    if 'tokenizer_1' in st.session_state:
        tokenizer_1 = st.session_state['tokenizer_1']
        st.text(f'1. Веса токенизатора загружены')

        # Загрузка модели из h5 файла
        model_1 = load_model('mdl.h5')
        st.session_state['model_1'] = model_1


        
        
        if 'model_1' in st.session_state:
            model_1 = st.session_state['model_1']
            st.text(f'2. Обученная модель загружена')
            PRED_LIST = np.load('cls_lst.npy')

            with st.container(border = True):

                st.write('Выбор данных для предсказания')

                radio = st.radio(
                "Выберите способ ввода новых данных для предсказания",
                ["Предсказание по одному объекту", "Предсказание по нескольким объектам"],
                captions=[
                    "Ввести текст",
                    "Загрузить таблицу с текстами"
                ], index = None)

                if radio == "Предсказание по одному объекту":
                    # Классификация номенклатурных наименований
                    with st.form("form6"):
                        st.write("Классификация новых текстов")
                        col1, col2 = st.columns(2)
                        with col1:
                            item_list = st.text_area('Введите текст',
                                                    "First things first I'ma say all the words inside myhead I'm fired up and tired of the way that things have been, oh-ooh The way that things have been, oh-ooh Second thing second Don't you tell me what you think that I can be I'm the one at the sail, I'm the master of my sea, oh The master of my sea, oh I was broken from a young age Taking my sulking to the masses Writing my poems for the few That looked at me took to me, shook to me, feeling me Singing from heartache from the pain Take up my message from the veins Speaking my lesson from the brain Seeing the beauty through the... Pain! You made me a, you made me a believer, believer Pain! You break me down and build me up, believer, believer All the words inside my Believer You break me down you build me up, believer Pain All the words inside my Believer You break me down you build me up, believer Pain Third things third Send a prayer to the ones up above All the hate that you've heard has turned your spirit to a dove, oh-ooh Your spirit up above, oh-ooh I was choking in the crowd Living my brain up in the cloud Falling like ashes to the ground Hoping my feelings, they would drown But they never did, ever lived Ebbing and flowing, inhibited, limited Till it broke open and it rained down It rained down, like... Pain! You made me a, you made me a believer, believer Pain! You break me down and build me up, believer, believer All the words inside my Believer You break me down you build me up, believer Pain All the words inside my Believer You break me down you build me up, believer Pain")
                        with col2:
                            st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                            submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
                            
                        if submitted:
                            x_test1 = tokenizer_1.texts_to_matrix([item_list])
                            y_test1 = model_1.predict(x_test1)
                            class_pred = np.argmax(y_test1, axis = 1)
                            item_class = dict(zip([item_list], PRED_LIST[class_pred]))
                            # st.write(item_class)

                            df_report = pd.DataFrame(list(item_class.items()), columns=['lyrics', 'genre'])
                            st.dataframe(df_report)
                            st.session_state['df_report'] = df_report

                elif radio == "Предсказание по нескольким объектам":
                    # Загрузка данных
                    uploaded_file3 = st.file_uploader("", type="csv", key='file_uploader3')

                    if uploaded_file3 is not None:
                        data_load_state = st.text('Идет загрузка ...')
                        df_pred = load_data(uploaded_file3)
                        data_load_state.text(f'1. Данные загружены. Размер: {round(sys.getsizeof(df_pred) / (1024 * 1024), 2)}Mb, {df_pred.shape[0]} x {df_pred.shape[1]}')
                        

                        st.session_state['df_pred'] = df_pred

                else:
                    st.write('Пока способ не выбран')
                    


            if 'df_pred' in st.session_state:
                df_pred = st.session_state['df_pred']

                with st.form("form7"):
                    st.write("Выбор столбца с текстами для предсказания")
                    pred_col = st.selectbox('Выберите столбец', list(df_pred.columns)) #, default = ['name', 'name.1', '1.0', '2.0', '3.0']
                    col1, col2 = st.columns(2)
                    with col1:
                        pass
                    with col2:
                        # st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                        submitted = st.form_submit_button("Выбрать", type = 'primary', use_container_width = True)
                    
                    if submitted:
                        item_list = df_pred[pred_col].to_list()

                        x_test1 = tokenizer_1.texts_to_matrix(item_list)
                        y_test1 = model_1.predict(x_test1)
                        class_pred = np.argmax(y_test1, axis = 1)
                        item_class = dict(zip(item_list, PRED_LIST[class_pred]))

                        df_genre = pd.DataFrame(list(item_class.items()), columns=[pred_col, 'genre'])

                        df_report = pd.merge(df_pred, df_genre, how="inner", on=pred_col)
                        st.dataframe(df_report)
                        st.session_state['df_report'] = df_report
                        

            if 'df_report' in st.session_state:
                today = datetime.datetime.today()
                df_report = st.session_state['df_report']
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_to_csv(df_report)
                    st.download_button(label="Скачать как CSV", data=csv, file_name=f"report_{today:%Y%m%d}.csv", mime='text/csv', use_container_width = True)
                with col2:
                    excel = df_to_excel(df_report)
                    st.download_button(label="Скачать как Excel", data=excel, file_name=f"report_{today:%Y%m%d}.xlsx", use_container_width = True)