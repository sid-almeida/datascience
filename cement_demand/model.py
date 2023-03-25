import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('demand.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X = data[['Production', 'Sales ', 'gdp', 'disbusment']]
y = data['demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

data_model = LinearRegression()
data_model.fit(X_train, y_train)

y_pred = data_model.predict(X_test)

# Imprimindo os coeficientes
print('Coeficientes: \n', data_model.coef_)
# Imprimindo o erro quadrático médio
print('Erro quadrático médio: %.2f'
      % metrics.mean_squared_error(y_test, y_pred))
# Imprimindo o coeficiente de determinação: 1 é perfeito previsão
print('Coeficiente de determinação: %.2f'
      % metrics.r2_score(y_test, y_pred))

def predict_demand(production, sales, gdp, disbusment):
    x = np.zeros(len(X.columns))
    x[0] = production
    x[1] = sales
    x[2] = gdp
    x[3] = disbusment
    return data_model.predict([x])[0]

