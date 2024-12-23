
from music_predictor_streamlit.app import setup_logger
from music_predictor_streamlit.service.introduction import make_introduction
from music_predictor_streamlit.service.service import Service


def main():
    setup_logger()
    service = Service()
    service.start_service()
    # model = None
    # st.title("Music Genre Classification")
    # 

    # 
    # 
    # if json_file is not None and zip_file is not None:
    #     df = load_data(json_file, zip_file)
    #     st.write("Загруженные данные:")
    #     st.dataframe(df)
    # 
    #     # Аналитика
    #     st.subheader("Анализ жанров")
    #     plot_genre_distribution(df)
    # 
    # if st.button("Создать модель"):
    #     model = create_model()
    #     st.success("Модель создана и сохранена!")
    # 
    # st.subheader("Создание модели")
    # n_estimators = st.number_input("Количество деревьев (n_estimators)", min_value=1, max_value=1000, value=100)
    # max_depth = st.number_input("Максимальная глубина дерева (max_depth)", min_value=1, max_value=50, value=None)
    # 
    # st.subheader("Инференс на базе изображения")
    # image_file = st.file_uploader("Загрузите изображение для предсказания", type=["jpg", "png", "jpeg"])
    # if image_file is not None:
    #     image = Image.open(image_file)
    #     st.image(image, caption='Загруженное изображение.', use_column_width=True)
    #     model = joblib.load('music_genre_model.pkl')
    # 
    # if st.button("Предсказать жанр") and model:
    #     prediction = predict_image(model, image_file)
    #     st.write(f"Предсказанный жанр: {prediction[0]}")


if __name__ == "__main__":
    main()