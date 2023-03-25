import pandas as pd
import numpy as np
import streamlit as st

data = pd.read_csv('demand.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[['Production', 'Sales ', 'gdp', 'disbusment']]
y = data['demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

data_model = LinearRegression()
data_model.fit(X_train, y_train)

y_pred = data_model.predict(X_test)

@st.cache.data
def predict_demand(production, sales, gdp, disbusment):
    x = np.zeros(len(X.columns))
    x[0] = production
    x[1] = sales
    x[2] = gdp
    x[3] = disbusment
    return data_model.predict([x])[0]

st.title('Cement Demand Prediction')
st.write('This is a simple app to predict cement demand in the US')

production = st.number_input('Production', min_value=0.0, max_value=100000.0, value=0.0)
sales = st.number_input('Sales', min_value=0.0, max_value=100000.0, value=0.0)
gdp = st.number_input('GDP', min_value=0.0, max_value=100000.0, value=0.0)
disbusment = st.number_input('Disbusment', min_value=0.0, max_value=100000.0, value=0.0)

if st.button('Predict'):
    result = predict_demand(production, sales, gdp, disbusment)
    st.success('The demand is {}'.format(result))

st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
