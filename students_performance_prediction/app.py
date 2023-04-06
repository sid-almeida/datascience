import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Load the data
data_model = pd.read_csv('https://raw.githubusercontent.com/sid-almeida/datascience/main/students_performance_prediction/data_model.csv')

# Split the data into X and y
X = data_model.drop('final_score', axis=1)
y = data_model['final_score']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Model valuation
y_pred = regressor.predict(X_test)
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2: ', r2_score(y_test, y_pred))


# Create the function
def predict_final_score(math_score, reading_score, writing_score):
    final_score = regressor.predict([[math_score, reading_score, writing_score]])
    return print(f'A nota final do aluno foi {round(final_score[0])} e sua situação é {np.where(final_score[0] >= 60, "Aprovado", "Reprovado")}')


st.title('Student Situation Prediction')
st.write('This is a simple web app that utilizes Machine-Learning to predict possible student situations based on their performance in Math, Reading and Writing.')

math_score = st.number_input("Student's Math Score: ", min_value=0, max_value=100, value=0)
reading_score = st.number_input("Student's Reading Score: ", min_value=0, max_value=100, value=0)
writing_score = st.number_input("Student's Writing Score: ", min_value=0, max_value=100, value=0)

# Function streamlit
def score_prediction(math_score, reading_score, writing_score):
    final_score = regressor.predict([[math_score, reading_score, writing_score]])
    if final_score >= 60:
        st.write(f'Final Score: {round(final_score[0])}')
        st.write('Status: Approved')
    else:
        st.write(f'Final Score: {round(final_score[0])}')
        st.write('The student is: Not approved')

# Buttons
if st.button('Predict'):
    score_prediction(math_score, reading_score, writing_score)
else:
    st.write('Please insert the student scores')


st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
