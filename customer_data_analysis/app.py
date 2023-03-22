# Importando dependências
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Criando título
st.title('Análise de Dados de Clientes')

# Criando subtítulo
st.markdown('Este é um dashboard para análise de dados de clientes')

# Criando sidebar
st.sidebar.title('Análise de Dados de Clientes')

# Criando subsidebar
st.sidebar.markdown('Este é um dashboard para análise de dados de clientes')

# Criando menu de navegação
st.sidebar.subheader('Navegação')

# Criando menu de navegação
pagina = st.sidebar.radio('Selecione uma página', ('Home', 'Gráficos', 'Sobre'))

# Criando página Home
if pagina == 'Home':
    st.image('https://media.giphy.com/media/3o7TKsO8UHmuYbI1aQ/giphy.gif')
    st.markdown('## Bem-vindo(a) ao dashboard de análise de dados de clientes')
    st.markdown('### Aqui você pode visualizar os dados de clientes e criar gráficos')
    st.markdown('### Para começar, selecione a página Gráficos no menu de navegação')

# Criando página Gráficos
elif pagina == 'Gráficos':
    # Criando subheader
    st.subheader('Análise de Dados de Clientes')

    # Criando subsubheader
    st.markdown('### Nesta página você pode criar gráficos com os dados de clientes')

    # Criando menu de navegação
    st.sidebar.subheader('Gráficos')

    # Criando menu de navegação
    grafico = st.sidebar.selectbox('Selecione um gráfico', ('Histograma', 'Boxplot', 'Scatterplot'))

    # Criando página Histograma
    if grafico == 'Histograma':
        # Criando subsubheader
        st.subheader('Histograma')

        # Criando menu de navegação
        st.sidebar.subheader('Filtros')

        # Criando menu de navegação
        coluna = st.sidebar.selectbox('Selecione uma coluna numérica', ('age', 'income'))

        # Criando menu de navegação
        bins = st.sidebar.slider('Selecione o número de bins', 1, 100, 10)

        # Criando menu de navegação
        cor = st.sidebar.radio('Selecione uma cor', ('Azul', 'Vermelho', 'Verde', 'Amarelo'))

        # Criando menu de navegação
        if cor == 'Azul':
            cor = 'blue'
        elif cor == 'Vermelho':
            cor = 'red'
        elif cor == 'Verde':
            cor = 'green'
        else:
            cor = 'yellow'

        # Carregando dados
        df = pd.read_csv('new_customers.csv')

        # Criando gráfico
        plt.figure(figsize=(10, 6))
        plt.hist(df[coluna], bins=bins, color=cor)
        plt.title('Histograma')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        plt.show()

        # Plotando gráfico
        st.pyplot()

    # Criando página Boxplot
    elif grafico == 'Boxplot':
        # Criando subsubheader
        st.subheader('Boxplot')

        # Criando menu de navegação
        st.sidebar.subheader('Filtros')

        # Criando menu de navegação
        coluna = st.sidebar.selectbox('Selecione uma coluna numérica', ('age', 'income'))

        # Criando menu de navegação
        cor = st.sidebar.radio('Selecione uma cor', ('Azul', 'Vermelho', 'Verde', 'Amarelo'))

        # Criando menu de navegação
        if cor == 'Azul':
            cor = 'blue'
        elif cor == 'Vermelho':
            cor = 'red'
        elif cor == 'Verde':
            cor = 'green'
        else:
            cor = 'yellow'

        # Carregando dados
        df = pd.read_csv('new_customers.csv')

        # Criando gráfico
        plt.figure(figsize=(10, 6))
        sns.boxplot(df[coluna], color=cor)
        plt.title('Boxplot')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        plt.show()

        # Plotando gráfico
        st.pyplot()

    # Criando página Scatterplot
    else:
        # Criando subsubheader
        st.subheader('Scatterplot')

        # Criando menu de navegação
        st.sidebar.subheader('Filtros')

        # Criando menu de navegação
        x = st.sidebar.selectbox('Selecione uma coluna numérica para o eixo x', ('age', 'income'))

        # Criando menu de navegação
        y = st.sidebar.selectbox('Selecione uma coluna numérica para o eixo y', ('age', 'income'))

        # Criando menu de navegação
        cor = st.sidebar.radio('Selecione uma cor', ('Azul', 'Vermelho', 'Verde', 'Amarelo'))

        # Criando menu de navegação
        if cor == 'Azul':
            cor = 'blue'
        elif cor == 'Vermelho':
            cor = 'red'
        elif cor == 'Verde':
            cor = 'green'
        else:
            cor = 'yellow'

        # Carregando dados
        df = pd.read_csv('new_customers.csv')

        # Criando gráfico
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x], df[y], color=cor)
        plt.title('Scatterplot')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

        # Plotando gráfico
        st.pyplot()

# Criando página Sobre
else:
    # Criando subheader
    st.subheader('Sobre')

    # Criando subsubheader
    st.markdown('### Este dashboard foi desenvolvido por [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')

