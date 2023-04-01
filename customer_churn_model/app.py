import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

link = 'https://raw.githubusercontent.com/sid-almeida/datascience/main/customer_churn_model/churn_model.csv'

# Carreguei os dados
data_model = pd.read_csv(link)

# Dividi os dados em treino e teste
X = data_model.drop('churn', axis=1)
y = data_model['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Padronizei os dados
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Criei o modelo
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Realizei as previsões
predictions = logmodel.predict(X_test)

# Avaliei o modelo
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Criei uma função para prever o churn
def churn_prediction(account_weeks, cust_serv_calls, day_mins, day_calls, monthly_charge, overage_fee, roam_mins):
    X = np.array([account_weeks, cust_serv_calls, day_mins, day_calls, monthly_charge, overage_fee, roam_mins]).reshape(1, -1)
    X = scaler.transform(X)
    prediction = logmodel.predict(X)
    if prediction == 1:
        st.write('O cliente irá cancelar o serviço')
    else:
        st.write('O cliente não irá cancelar o serviço')

st.title('Previsão de Churn dos Clientes')
st.write('Este é um aplicativo simples que utiliza Machine-Learning para Prever o Churn de Clientes')

account_weeks = st.number_input('Tempo no Serviço (Semanas)', min_value=0, max_value=1000, value=0)
cust_serv_calls = st.number_input('Ligações SAC', min_value=0, max_value=10, value=0)
day_mins = st.number_input('Minutos de ligação / Dia', min_value=0, max_value=1000, value=0)
day_calls = st.number_input('Ligações / Dia', min_value=0, max_value=1000, value=0)
monthly_charge = st.number_input('Mensalidade', min_value=0, max_value=1000, value=0)
overage_fee = st.number_input('Taxa de Excesso', min_value=0, max_value=1000, value=0)
roam_mins = st.number_input('Minutos Fora de Área', min_value=0, max_value=1000, value=0)

if st.button('Prever'):
    churn_prediction(account_weeks, cust_serv_calls, day_mins, day_calls, monthly_charge, overage_fee, roam_mins)
else:
    st.write('Pressione o Botão Para Prever')

st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
