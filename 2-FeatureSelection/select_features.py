from __future__ import print_function
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt
import os

movies = pd.read_csv(os.path.join("InputData", "movie_metadata_processed.csv"))
X = movies.drop('imdb_score', axis=1).values
y = movies['imdb_score'].values
columns = movies.drop('imdb_score', axis=1).columns

lasso = Lasso(alpha=0.015, fit_intercept=False, max_iter=100000, positive=True)
lasso_coef = lasso.fit(X, y).coef_

selected_features = sorted([[lasso_coef[i], columns[i]] for i in range(len(columns)) if lasso_coef[i] > 0])
print (selected_features)

plt.plot(range(len(columns)), lasso_coef)
plt.xticks(range(len(columns)), columns, rotation=60)
plt.ylabel('Coefficients')
plt.savefig(os.path.join('OutputData', 'feature_selection.png'))
plt.show()

columns = [selected[1] for selected in selected_features]
columns.append('imdb_score')
df = movies[columns]
df.to_csv(os.path.join('OutputData', 'movie_metadata_feature_selected.csv'), index=False, encoding='utf-8')

