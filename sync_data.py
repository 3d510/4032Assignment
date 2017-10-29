from __future__ import print_function
import fnmatch
import os
import pandas

posts_path = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(".")
    for f in fnmatch.filter(files, 'movie_metadata_processed.csv')]

posts = pandas.read_csv(os.path.join("1-DataProcessing", "OutputData", 'movie_metadata_processed.csv'))
for post_path in posts_path:
    print(post_path)
    posts.to_csv(post_path, index=False, encoding='utf-8')
