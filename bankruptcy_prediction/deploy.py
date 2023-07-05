import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the model from the file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Verifiquei se existe um arquivo de dados no diretório do sistema
# Cache do dataset
if os.path.isfile('data.csv'):
    df = pd.read_csv('data.csv')
else:
    df = pd.read_csv('https://raw.githubusercontent.com/saaelmeida/UniChurn/main/data.csv')
    # Salvei o arquivo de dados no diretório do sistema
    df.to_csv('https://raw.githubusercontent.com/sid-almeida/datascience/main/bankruptcy_prediction/data.csv', index=False)

# Create a sidebar header
with st.sidebar:
    st.image("https://github.com/sid-almeida/datascience/blob/main/bankruptcy_prediction/Brainize%20Tech.png?raw=true", width=250)
    st.title("UniChurn")
    choice = st.radio("**Navegação:**", ("About", "Batch Prediction"))
    st.info('**Note:** Please be aware that this application is intended solely for educational purposes. It is strongly advised against utilizing this tool for making any financial decisions.')


if choice == "About":
    # Create a title and sub-title
    st.write("""
    # Business Bankruptcy Prediction App
    This app predicts the probability of bankruptcy of a company!
    """)
    st.write('---')
    st.write('**About the App:**')
    st.write('Utilizing a Random Forest Classification Algorithm, the aforementioned approach employs a meticulously trained model encompassing 96 distinct financial ratios as input features. Its primary objective is to ascertain the likelihood of bankruptcy for Taiwanese enterprises.')
    st.info('**Note:** Please be aware that this application is intended solely for educational purposes. It is strongly advised against utilizing this tool for making any financial decisions.')
    st.write('---')
    st.write('**About the Data:**')
    st.write('The dataset utilized in this analysis was obtained from the Taiwan Economic Journal, covering the period from 1999 to 2009. For further details and access to additional information, please click on the following link:')
    st.write('**Data:** [**DataSet**](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)')
    st.write('---')

if choice == 'Batch Prediction':
    # Create a title and sub-title
    st.write("""
    # Business Bankruptcy Prediction App
    This app predicts the probability of bankruptcy of a company!
    """)
    st.write('---')
    st.info('**Guide:** Please, upload the dataset with the predicting features.')
    st.write('---')
    # Create a file uploader to upload the dataset of predicting features
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)
        st.write('---')
        st.write('**Dataset:**')
        st.write(df_pred)
        # Create a button to predict the probability of bankruptcy
        if st.button("Predict"):
            # Predict the probability of bankruptcy using the model.pkl file and create new column for the probability of bankruptcy
            df_pred['Bankruptcy Probability (%)'] = model.predict_proba(df_pred)[:, 1] * 100
            # Create a csv file for the predicted probability of bankruptcy
            df_pred.to_csv('predicted.csv', index=False)
            # Create a success message
            st.success('The probability of bankruptcy was predicted successfully!')
            # Create a button to download the predicted probability of bankruptcy
            st.write('---')
            st.write('**Predicted Dataset:**')
            st.write(df_pred)
            # Create a button to download the dataset with the predicted probability of bankruptcy
            if st.download_button(label='Download Predicted Dataset', data=df_pred.to_csv(index=False), file_name='predicted.csv', mime='text/csv'):
                pass
        else:
            st.write('---')
            st.info('Click the button to predict the probability of bankruptcy!')
    else:
        st.write('---')



st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
