from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_labels(ds_train, ds_test, genre_map, common_genres):
    """
    Wrapper function to convert and transform genres for multilabel classification.
    """
    def convert_genres(vectors, genre_map):
        converted = []
        for genres in vectors:
            mapped_genres = list(set(genre_map.get(g, None) for g in genres))
            converted.append(mapped_genres)
        return converted

    def transform_genres(genres_list, common_genres):
        simple_genres_list = []
        for genres in genres_list:
            simple_genres = [genre for genre in genres if genre in common_genres]
            if simple_genres:
                simple_genres_list.append(simple_genres)
            else:
                simple_genres_list.append([])
        return simple_genres_list

    all_genres = [genre[1].split(" ") for genre in ds_train]
    all_genres_test = [genre[1].split(" ") for genre in ds_test]

    converted_train_genres = convert_genres(all_genres, genre_map)
    converted_test_genres = convert_genres(all_genres_test, genre_map)

    transformed_train_genres = transform_genres(converted_train_genres, common_genres)
    transformed_test_genres = transform_genres(converted_test_genres, common_genres)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(transformed_train_genres)
    y_test = mlb.transform(transformed_test_genres)

    return y_train, y_test, mlb.classes_
