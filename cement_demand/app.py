import streamlit as st
from model import predict_demand

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
