# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):

    m = RandomForestClassifier() # random forest model
    y = df.loc[:,'SalePrice'] # dependent variable
    X = df.loc[:,df.columns != 'SalePrice'] # independent variables

    rfe = RFE(m, n_features_to_select=None)# , verbose=1)
    rfe = rfe.fit(X, y)
    rank = rfe.ranking_ # Get list of rankings = 1 means the column is selected

    # Extract column name based on ratings from RFE algo
    # Store indices of columns that have rating == 1
    feature_indices = []
    for i in xrange(len(rank)):
        if rank[i] == 1:
            feature_indices += [i]
#     print feature_indices
    ind = df.iloc[:,feature_indices].columns # Index object
    return ind.values.tolist() # return as list

f = rf_rfe(data)
print type(f)
print f
