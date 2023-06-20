import pandas as pd
import numpy as np
import streamlit as st

# Importei o dataset
url = 'https://raw.githubusercontent.com/sid-almeida/datascience/main/heart_failure_mortality_prediction/data_model.csv'
data = pd.read_csv(url)

# Dividi o dataset em variáveis independentes e variável dependente
X = data.drop('morte', axis=1)
y = data['morte']

# Dividi o dataset em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Treinei o modelo de regressão logística
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Criei uma função para prever a morte de um paciente
def prever_morte(creatinina_sérica, fração_injeção, idade, período_acompanhamento, sódio_sérico):
    X = np.array([creatinina_sérica, fração_injeção, idade, período_acompanhamento, sódio_sérico]).reshape(1, -1)
    res = model.predict(X)
    if res == 1:
        st.write('O paciênte tem alta probabilidade de morte')
    else:
        st.write('O paciente tem baixa probabilidade de morte')

# Criei uma função para prever a probabilidade de morte de um paciente
def probabilidade_morte(creatinina_sérica, fração_injeção, idade, período_acompanhamento, sódio_sérico):
    X = np.array([creatinina_sérica, fração_injeção, idade, período_acompanhamento, sódio_sérico]).reshape(1, -1)
    res = model.predict_proba(X)
    st.write('A probabilidade de morte do paciente é de: ', res[0][1]*100, "%")

#Criei o App
st.title('Previsão de Mortalidade por Insuficiência Cardíaca')
st.write('Esse app foi desenvolvido para prever a mortalidade de pacientes com insuficiência cardíaca utilizando o modelo de Regressão Logística')

creatinina_sérica = st.number_input('Exame de Creatinina Sérica', min_value=0.0, max_value=10.0, value=0.0)
fração_injeção = st.number_input('Exame de Fração de Injeção', min_value=0.0, max_value=150.0, value=0.0)
idade = st.number_input('Idade do Paciênte', min_value=0.0, max_value=110.0, value=0.0)
período_acompanhamento = st.number_input('Período de Acompanhamento (Dias)', min_value=0.0, max_value=1000.0, value=0.0)
sódio_sérico = st.number_input('Exame de Sódio Sérico', min_value=0.0, max_value=250.0, value=0.0)



if st.button('Predict'):
    result = prever_morte(creatinina_sérica, fração_injeção, idade, período_acompanhamento, sódio_sérico)
    result = probabilidade_morte(creatinina_sérica, fração_injeção, idade, período_acompanhamento, sódio_sérico)

st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
