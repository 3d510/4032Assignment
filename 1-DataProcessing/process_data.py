from __future__ import print_function
import pandas as pd
import sys
import os
import glob

from scrape_cpi_data import scrape_cpi
from process_categorical import process_genre, process_language, process_content_rating


def process_money_value(df):
    cpi_by_year = pd.read_csv(os.path.join(outputSubdir, 'cpidata.csv'))
    cpi_map = {}
    for index, row in cpi_by_year.iterrows():
        cpi_map[row['year']] = row['cpi']
    for index, row in df.iterrows():
        new_budget = row['budget'] * 1.0 * cpi_map[2017] / cpi_map[row['title_year']]
        df.loc[index, 'budget'] = new_budget
        # new_gross = row['gross'] * 1.0 * cpi_map[2017] / cpi_map[row['title_year']]
        # df.loc[index, 'gross'] = new_gross
    df = df.drop(['title_year'], axis=1)
    return df


####### Main Program #######
inputSubdir = 'InputData'
outputSubdir = 'OutputData'
if not os.path.exists(outputSubdir):
    os.makedirs(outputSubdir)

### read raw csv file ###
print ('Start reading data....')

filename = "movie_metadata.csv"
if len(sys.argv) > 1:
    filename = sys.argv[1]  # used for new set of data

df = pd.read_csv(os.path.join(inputSubdir, filename))
columns = ['movie_imdb_link', 'num_critic_for_reviews', 'duration', 'director_facebook_likes',
           'actor_3_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes',
           'genres', 'cast_total_facebook_likes', 'language', 'content_rating', 'budget',
           'title_year', 'movie_facebook_likes', 'imdb_score']
df = df.loc[:, columns]

### remove duplicates and null value ###
print ('\nStart removing duplicates and null values....')
df = df.drop_duplicates('movie_imdb_link', keep='first')
df = df.dropna()
print (df.info())

### process categorical attributes ###
print ('\nStart processing categorical attributes....')
df = process_genre(df)
df = process_language(df)
df = process_content_rating(df)


### convert all budget and gross to the money value of year 2017 ###
print ('\nStart processing budget values....')

if not glob.glob(os.path.join(outputSubdir, 'cpidata.csv')):
    scrape_cpi()
df = process_money_value(df)

### write processed data to file ###
print ('\nSummary of data....')
df.columns = [column.lower() for column in df.columns]
df = df.drop(['movie_imdb_link'], axis=1)

# move imdb_score to last column
columns = list(df.columns)
columns.remove('imdb_score')
columns.append('imdb_score')
df = df[columns]

print (df.info())
print ('\nWriting processed data to file....')
df.to_csv(os.path.join(outputSubdir, 'movie_metadata_processed.csv'), index=False)
print ('\nDone')
