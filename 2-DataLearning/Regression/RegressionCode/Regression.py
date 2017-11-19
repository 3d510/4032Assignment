import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Read data frame from csv file
df = pd.read_csv('InputData/movie_metadata_processed.csv', header=0, sep=',')

# Seperate column 'imdb_score' from other columns
X = (df.drop('imdb_score',axis=1)).as_matrix()
y = df['imdb_score'].as_matrix()

# Split train and test data to 7:3
X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.3, shuffle=True)

# Create linear regression object
linearRegr = linear_model.LinearRegression()
# Train the model using the training sets
linearRegr.fit(X_train, y_train)
# Make predictions using the testing set
linear_y_pred = linearRegr.predict(X_test)
# The root mean squared error
print("Linear Regression RMSE: %.3f"
      % np.sqrt(mean_squared_error(y_test, linear_y_pred)))
# The r2 score
print('Linear Regression r2 score: %.3f' % r2_score(y_test, linear_y_pred))

# Do the same for Random Forest Regression
for i in range(5,26,5):
    ranForestRegr = RandomForestRegressor(n_estimators=i)
    ranForestRegr.fit(X_train, y_train)
    ranForest_y_pred = ranForestRegr.predict(X_test)
    print("Random Forest Regression ("+str(i)+" trees) RMSE: %.3f"
          % np.sqrt(mean_squared_error(y_test, ranForest_y_pred)))
    print("Random Forest Regression ("+str(i)+" trees) r2 score: %.3f" % r2_score(y_test, ranForest_y_pred))
