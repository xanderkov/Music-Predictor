import re
import sys
from itertools import chain

import nltk
import pandas as pd
import streamlit as st
from loguru import logger

from music_predictor_streamlit.settings.settings import config


class TextEDA:
    def __init__(self):
        self._back_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/text"
        self._url_upload_data = self._back_url + "/upload_dataset"
        self._ss = nltk.stem.SnowballStemmer("english")  # стемизатор
        self._word = re.compile(r"\w+")

    def make_eda(self):
        st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
        st.markdown("#### 1. Предобработка данных")
        cat_col = None
        self._upload_data()
        if "df_1" in st.session_state:
            self._choose_classification_columns()
        if "df_2" in st.session_state:
            self._choose_categorical_columns()
        if "cat_col" in st.session_state:
            cat_col = st.session_state["cat_col"]
        if "df_3" in st.session_state:
            self._choose_class_by_threshold(cat_col)
        if "s_1" in st.session_state and "df_3" in st.session_state:
            self._transform_text(cat_col)
        if "df_5" in st.session_state:
            df_5 = st.session_state["df_5"]
            st.text(
                f"5. Тексты нормализованы. Размер: {round(sys.getsizeof(df_5) / (1024 * 1024), 2)}Mb, {df_5.shape[0]} x {df_5.shape[1]}"
            )

            df_5.to_csv("df_c.csv", index=False)
            st.text(
                "6. Подготовка данных для обучения моделей завершена. Данные сохранены в файл"
            )

    def _upload_data(self):
        logger.info("Uploading data...")
        uploaded_file = st.file_uploader("", type="csv", key="file_uploader")
        if uploaded_file is not None:
            data_load_state = st.text("Идет загрузка ...")
            df_1 = load_data(uploaded_file)
            data_load_state.text(
                f"1. Данные загружены. Размер: {round(sys.getsizeof(df_1) / (1024 * 1024), 2)}Mb, {df_1.shape[0]} x {df_1.shape[1]}"
            )
            st.session_state["df_1"] = df_1
        # response = requests.post(self._url_upload_data, files=uploaded_file)
        # if response.status_code == 200:
        #     df_1 = response.json()  # Pandas чето должен сделать
        #     data_load_state.text(
        #         f"1. Данные загружены. Размер: {round(sys.getsizeof(df_1) / (1024 * 1024), 2)}Mb, {df_1.shape[0]} x {df_1.shape[1]}"
        #     )
        #     st.session_state["df_1"] = df_1
        # else:
        #     logger.error(f"Status code: {response.status_code}. Text: {response.text}")
        #

    def _choose_classification_columns(self):
        df_1 = st.session_state["df_1"]

        with st.form("form1"):
            st.write("Выбор столбцов для классификации")
            group_cols = st.multiselect(
                "Выберите столбцы для анализа", list(df_1.columns)
            )  # , default = ['name', 'name.1', '1.0', '2.0', '3.0']
            col1, col2 = st.columns(2)
            with col1:
                pass
            with col2:
                # st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Выбрать", type="primary", use_container_width=True
                )

            if submitted:
                # st.write('You selected:', group_cols)
                df_2 = df_1[group_cols]
                st.dataframe(df_2.head(3), use_container_width=True)
                st.session_state["df_2"] = df_2

    def _choose_categorical_columns(self):
        df_2 = st.session_state["df_2"]
        st.text(
            f"2. Выбраны столбцы для классификации: {list(df_2.columns)}. Размер: {round(sys.getsizeof(df_2) / (1024 * 1024), 2)}Mb, {df_2.shape[0]} x {df_2.shape[1]}"
        )

        # Удаление ненормализуемых категорий
        with st.form("form2"):
            st.write("Выбор/исключение категорий")

            cat_col = st.selectbox(
                "Выберите столбец с категорией",
                options=list(df_2.columns),
                index=list(df_2.columns).index(find_col_candidates(df_2)),
                placeholder=find_col_candidates(df_2),
            )

            col1, col2 = st.columns(2)
            with col1:
                str_contains = st.text_input(
                    "Введите подстроку поиска категории для удаления", "classic"
                )
            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Выбрать", type="primary", use_container_width=True
                )

            if submitted:
                df_3 = df_2[~(df_2[cat_col].str.contains(str_contains))]
                # st.dataframe(df_3, use_container_width = True)
                st.session_state["cat_col"] = cat_col
                st.session_state["df_3"] = df_3

    def _choose_class_by_threshold(self, cat_col):
        df_3 = st.session_state["df_3"]
        st.text(
            f"3. Выбран уровень категорий для классификации. Размер: {round(sys.getsizeof(df_3) / (1024 * 1024), 2)}Mb, {df_3.shape[0]} x {df_3.shape[1]}"
        )

        # Отбор классов по трешхолду
        with st.form("form3"):
            st.write("Отбор категорий по количеству данных")
            col1, col2 = st.columns(2)
            with col1:
                n_examples = st.number_input(
                    "Выберите пороговое значение",
                    min_value=0.1,
                    max_value=1000.00,
                    value=0.98,
                    placeholder="Введите число",
                    help="Если значение меньше 1, оно указывает на долю от всех данных (отбираются категории, суммарное количество данных которых соответсвует указанной доле); если значение равно или больше 1, то обозначается количество примеров в категории (отбираются категории, где количество примеров не менее указанного)",
                )
            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Выбрать", type="primary", use_container_width=True
                )

            if submitted:
                s_1 = get_classification_cat(df_3[cat_col], n_examples)
                st.dataframe(s_1, use_container_width=True)
                st.session_state["s_1"] = s_1

    def _transform_text(self, cat_col):
        s_1 = st.session_state["s_1"]
        df_3 = st.session_state["df_3"]
        df_2 = st.session_state["df_2"]

        # Получение датафрейма для классификации
        df_4 = df_3[df_3[cat_col].isin(s_1.index)]
        st.text(
            f"4. Отобраны категории по количеству данных. Размер: {round(sys.getsizeof(df_4) / (1024 * 1024), 2)}Mb, {df_4.shape[0]} x {df_4.shape[1]}"
        )

        # # Преобразования текстов
        with st.form("form4"):
            st.write("Нормализация текстов")
            item_col = st.selectbox(
                "Выберите столбец с текстами",
                options=list(df_2.columns),
                index=list(df_2.columns).index(find_col_candidates(df_2, reverse=True)),
                placeholder=find_col_candidates(df_2, reverse=True),
            )
            col1, col2 = st.columns(2)
            with col1:
                # text_tokens = st.checkbox("Лемматизация (слова и цифры)", value = True)
                # words_tokens = st.checkbox("Лемматизация (только слова)")
                text_stem_tokens = st.checkbox("Стемминг (слова и цифры)")
                words_stem_tokens = st.checkbox("Стемминг (только слова)")
            with col2:
                n_items = st.number_input(
                    "Выберите количество первых слов каждой записи",
                    min_value=1,
                    value=10,
                    placeholder="Введите число",
                )
                submitted = st.form_submit_button(
                    "Выбрать", type="primary", use_container_width=True
                )

            if submitted:
                df_5 = df_4.copy()
                self._use_staming_tokens(
                    df_5, text_stem_tokens, words_stem_tokens, item_col, n_items
                )

    def _use_staming_tokens(
        self,
        df_5: pd.DataFrame,
        text_stem_tokens: bool,
        words_stem_tokens: bool,
        item_col: str,
        n_items: int,
    ):
        if text_stem_tokens or words_stem_tokens:
            df_5["tokens"] = df_5[item_col].apply(
                self._get_tokens
            )  # получение списка токенов
            st.text(f"Всего уникальных токенов: {df_5['tokens'].explode().nunique()}")

            df_5["stem_tokens"] = df_5["tokens"].apply(
                lambda x: list(map(self._ss.stem, x))
            )
            st.text(
                f"Всего уникальных основ: {df_5['stem_tokens'].explode().nunique()}"
            )

        if text_stem_tokens:
            df_5["stem_tokens_n"] = df_5["stem_tokens"].apply(lambda x: x[:n_items])
            st.text(
                f"Уникальных основ (слова и цифры): {df_5['stem_tokens_n'].explode().nunique()}"
            )
            df_5["text_stem_tokens"] = df_5["stem_tokens_n"].apply(
                lambda x: " ".join(x)
            )

        if words_stem_tokens:
            df_5["words_stem_tokens_n"] = df_5["stem_tokens"].apply(
                lambda x: list(filter(str.isalpha, x))[:n_items]
            )
            st.text(
                f"Уникальных основ (только слова): {df_5['words_stem_tokens_n'].explode().nunique()}"
            )
            df_5["words_stem_tokens"] = df_5["words_stem_tokens_n"].apply(
                lambda x: " ".join(x)
            )

        df_5 = df_5[
            df_5.columns.drop(
                list(
                    df_5.filter(
                        [
                            "tokens",
                            "stem_tokens",
                            "tokens_n",
                            "words_tokens_n",
                            "stem_tokens_n",
                            "words_stem_tokens_n",
                        ]
                    )
                )
            )
        ]

        st.dataframe(df_5, use_container_width=True)
        st.session_state["df_5"] = df_5

    def _reg_tokenize(self, text):
        """
        Функция токенизации текста на основе регулярных выражений
        """
        words = self._word.findall(text)
        return words

    def _get_tokens(self, text):
        """
        Функция токенизации текста, а также фильтрации и нормализации токенов
        Параметры:
            text : str, текст
        Возвращает:
            tokens : list, список отфильтрованных токенов
        """
        text = (
            text.replace(r"([^\w\s]+)", " \\1 ").strip().lower()
        )  # вклинивание пробелов между словами и знаками препинания, приведение к нижнему регистру
        # print(text)
        tokens = self._reg_tokenize(text)  # токенизация
        # print(tokens)
        tokens = [
            element
            for element in list(
                chain(*[re.split(r"\W+", element) for element in tokens])
            )
            if element != ""
        ]  # разделение составных элементов слов #####
        # print(tokens)
        tokens = list(
            chain(
                *[
                    re.findall(r"\d+|\D+", element) if element.isalnum() else element
                    for element in tokens
                ]
            )
        )  # разбиение токенов, состоящих из букв и цифр
        # print(tokens)
        return tokens


def find_col_candidates(df, reverse=False):
    tuples = []
    for col in df.select_dtypes(
        include=["object"]
    ).columns:  # df.select_dtypes(include=['object']).columns
        tuples.append((col, round(df[col].nunique() / len(df), 4)))
    sorted_tuples = sorted(tuples, key=lambda x: x[-1], reverse=reverse)[0]

    return sorted_tuples[0]


def get_classification_cat(s, cat_len_thr):
    """
    Функция выделения категорий для классификации по количеству примеров в категории
    s - серия с данными (категориями),
    cat_len_thr - трешхолд: если меньше единицы, воспринимается как процент данных от всей выборки, иначе количество примеров в категории
    Возвращает количество примеров по каждой категории согласно трешхолду
    """
    s_vc = s.value_counts()
    if cat_len_thr < 1:
        share = cat_len_thr
    else:
        share = s_vc.loc[lambda x: x > cat_len_thr].sum() / len(s)
    j = 0
    for i in s_vc:
        j += i
        if j >= len(s) * share:
            print(
                f"Нижний порог количества примеров в категории {i}, всего примеров {j}"
            )
            break
    s_res = s_vc.loc[lambda x: x >= i]
    return s_res


@st.cache_data()
def load_data(file):
    df = pd.read_csv(file, encoding="utf-8")  # , nrows=50
    return df
