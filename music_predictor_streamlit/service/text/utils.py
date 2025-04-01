import pandas as pd
import streamlit as st


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


def del_session_elements():  # keys
    # for key in st.session_state.keys():
    #     if key in keys:
    #         del st.session_state[key]
    st.write(st.session_state.keys())
    for key in st.session_state.keys():
        del st.session_state[key]
