from __future__ import print_function
import pandas as pd


def process_genre(data):
    result = {}
    allGenres = set()

    for index, row in data.iterrows():
        allGenres = allGenres.union(row["genres"].split("|"))

    print ('*** Unique genre:', allGenres)

    for index, row in data.iterrows():
        # if index % 1000 == 1 and index != 1:
        #     print('{} rows processed'.format(index - 1))
        resEntry = {}
        for genre in allGenres:
            resEntry[genre] = 1 if genre in row["genres"] else 0
        result[row["movie_imdb_link"]] = resEntry
    # print(result)

    outData = pd.DataFrame(result).transpose()
    outData.index.name = "movie_imdb_link"
    outData = outData.reset_index()

    # merge data with outData
    data = pd.merge(data, outData, on='movie_imdb_link').drop('genres', axis=1)
    return data


def process_language(data):
    print('*** Unique language:', data.language.unique())
    data['is_english'] = (data['language'] == 'English').astype(int)
    data = data.drop('language', axis=1)
    return data


def process_content_rating(df):
    print('*** Unique content rating: ', df.content_rating.unique())
    df = pd.get_dummies(df, columns=['content_rating'])
    df = df.drop('content_rating_Not Rated', axis=1)  # this can be derived from other content_rating_ columns
    return df
