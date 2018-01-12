# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):

    m = RandomForestClassifier() # Model
    y = df.loc[:,'SalePrice'] # Dependent variable
    X = df.loc[:, df.columns != 'SalePrice'] # Independent Variable

    select_model = SelectFromModel(m, prefit=False) # Set the model
    sel = select_model.fit(X, y) # fit the model
    rank =  sel.get_support() # Get selected features

    # Iterate through the indices and figure out which column has been selected
    feature_indices = []
    for i in xrange(len(rank)):
        if rank[i] == True:
            feature_indices += [i]

    ind = df.iloc[:,feature_indices].columns # Index object
    return ind.values.tolist() # return as list

print ("#### Testing ####")
f = select_from_model(data)
print type(f)
print f
