print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR,LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

df = pd.read_csv('InputData/movie_metadata_processed.csv', header=0, sep=',')   # columns names if no header
#df = pd.read_csv('test.csv', header=0, sep=',')   # columns names if no header
#df = (df-df.mean())/df.std()

X = (df.drop('imdb_score',axis=1)).as_matrix()
y = df['imdb_score'].as_matrix()

#X = df[['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes']].as_matrix()
#y = df['cast_total_facebook_likes'].as_matrix()


X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.2, shuffle=False
)

# Create linear regression object
regr = linear_model.LinearRegression()
#regr = linear_model.LogisticRegression()
# regr = SVR(C=10)
#regr = MLPRegressor(max_iter=5000)
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print('Mean: %.2f' % np.mean(y))
# Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, diabetes_y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())

#plt.show()
