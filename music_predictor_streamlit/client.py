from loguru import logger

from music_predictor_streamlit.app import setup_logger
from music_predictor_streamlit.service.introduction import make_introduction
from music_predictor_streamlit.service.service import Service


def main():
    setup_logger()
    logger.info("Reindex service")
    service = Service()
    service.start_service()
    # model = None
    # st.title("Music Genre Classification")
    #

    #
    #
    #

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
