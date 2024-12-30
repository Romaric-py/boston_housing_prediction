import numpy as np
import joblib
from mlearn.linear_model import LinearRegression
from mlearn.preprocessing import Scaler


boston_variables = [
    ('crim', 'Per capita crime rate by town'),
    ('zn', 'Proportion of residential land zoned for lots over 25,000 sq.ft.'),
    ('indus', 'Proportion of non-retail business acres per town'),
    (   'chas',
        'Charles River dummy variable (1 if tract bounds river; 0 otherwise)'),
    ('nox', 'Nitric oxides concentration (parts per 10 million)'),
    ('rm', 'Average number of rooms per dwelling'),
    ('age', 'Proportion of owner-occupied units built prior to 1940'),
    ('dis', 'Weighted distances to five Boston employment centres'),
    ('rad', 'Index of accessibility to radial highways'),
    ('tax', 'Full-value property-tax rate per $10,000'),
    ('ptratio', 'Pupil-teacher ratio by town'),
    (   'b',
        '1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by '
        'town'),
    ('lstat', '% lower status of the population')
]

scaler = Scaler()
scaler.load_from_json('boston_scaler.json')

custom_model = LinearRegression()
custom_model.load_from_json('boston_custom_model.json')

model = joblib.load('boston_model.joblib')

selected_model = custom_model

# ***************************************************** #

def to_array(data, cols_names):
    array = np.array([float(data[col]) for col in cols_names])
    return array.reshape(1, -1)

def preprocess(X):
    return scaler.transform(X)

def predict(data):
    cols_names = [x[0] for x in boston_variables]
    array = to_array(data, cols_names)
    array = preprocess(array)
    
    price = selected_model.predict(array)[0]
    return {'price': round(price, 2)}