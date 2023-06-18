import streamlit as st
import pandas as pd
import os
import sweetviz as sv
import codecs
import streamlit.components.v1 as components

def st_display_sweetviz(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)

with st.sidebar:
    imglink = 'https://github.com/sid-almeida/datascience/blob/main/UniChurn/Brainize%20Tech(1).png?raw=true'
    st.image(imglink, width=250)
    st.title("UniChurn")
    choice = st.radio("**Navegação:**", ("Upload", "Análise", "Machine Learning", "Previsão", "Previsão de Conjunto de Dados"))
    st.info("Esta aplicação permite a análise de dados de uma universidade fictícia, com o objetivo de prever a evasão de alunos."
            " Além disso, ela utiliza Machine Learning para prever o estado futuro de alunos.")

if os.path.exists("data.csv"):
    dataframe = pd.read_csv("data.csv")

if choice == "Upload":
    st.header("Upload de dados (Treino / Teste)")
    st.subheader("Faça o upload do arquivo .csv para análise e modelagem.")
    file = st.file_uploader("Upload do arquivo", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head(10))
        st.success("Upload realizado com sucesso!")
        st.balloons()
        st.markdown("##")
        data.to_csv("data.csv", index=False)
        st.success("Arquivo salvo com sucesso!")
        st.balloons()
    else:
        st.warning("Por favor, faça o upload do arquivo .csv.")

if choice == "Análise":
    st.header("Análise de dados")
    st.subheader("Análise exploratória dos dados com SweetViz.")
    if os.path.exists("data.csv"):
        dataframe = pd.read_csv("data.csv")
        if dataframe is not None:
            report = sv.analyze(dataframe)
            # st.write(report.show_html(), unsafe_allow_html=True)
            st.success("Análise realizada com sucesso!")
            report.show_html(open_browser=False)
            with open("SWEETVIZ_REPORT.html", "w") as html_file:
                html_file.write(report._page_html)
            st_display_sweetviz("SWEETVIZ_REPORT.html")
            st.balloons()
        else:
            st.warning("Por favor, faça o upload do arquivo .csv.")
    else:
        st.warning("Por favor, faça o upload do arquivo .csv.")

if choice == "Machine Learning":
    dataframe = pd.read_csv("data.csv")
    st.header("Treino de modelos de Machine Learning")
    st.subheader("Treino de modelos de Machine Learning para prever a evasão de alunos.")
    problema = st.selectbox("Selecione o problema:", ("Classificação", "Regressão"))
    if problema == "Classificação":
        modelo = st.selectbox("Selecione o modelo:", ("Logistic Regression", "Random Forest", "XGBoost"))
    if problema == "Regressão":
        modelo = st.selectbox("Selecione o modelo:", ("Linear Regression", "Random Forest", "XGBoost"))
        if modelo == "Linear Regression":
            st.warning("AVISO: Este modelo não é adequado para o problema de classificação!")
            # selectbox para selecionar o alvo
            alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
            # botão para treinar o modelo de regressão linear
            st.button("Treinar modelo de Regressão linear")
            if st.button:
                # treinando o modelo
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error
                X = dataframe.drop(alvo, axis=1)
                y = dataframe[alvo]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.subheader("**Avaliação do Modelo**")
                st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
                st.write("R2:", model.score(X_test, y_test))
                st.success("Modelo treinado com sucesso!")
            # botão para salvar o modelo
            st.button("Salvar modelo")
            if st.button:
                import pickle
                pickle.dump(model, open("model.pkl", "wb"))
                st.success("Modelo salvo com sucesso!")
                st.balloons()
    if modelo == "Logistic Regression":
        st.warning("Este modelo não é adequado para o problema de regressão.")
        # selectbox para selecionar o alvo
        alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
        # botão para treinar o modelo de regressão logística
        st.button("Treinar modelo de Regressão logística")
        if st.button:
            # treinando o modelo
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            X = dataframe.drop(alvo, axis=1)
            y = dataframe[alvo]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.subheader("**Avaliação do Modelo**")
            st.write("Acurácia:", accuracy_score(y_test, y_pred)*100,"%")
            st.success("Modelo treinado com sucesso!")
        # botão para salvar o modelo
        st.button("Salvar modelo")
        if st.button:
            import pickle
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("Modelo salvo com sucesso!")
            st.balloons()
    elif modelo == "Random Forest":
        # selectbox para selecionar o alvo
        alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
        # botão para treinar o modelo de random forest
        st.button("Treinar modelo de Random Forest")
        if st.button:
            # treinando o modelo
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            X = dataframe.drop(alvo, axis=1)
            y = dataframe[alvo]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.subheader("**Avaliação do Modelo**")
            st.write("Acurácia:", accuracy_score(y_test, y_pred)*100,"%")
            st.success("Modelo treinado com sucesso!")
        # botão para salvar o modelo
        st.button("Salvar modelo")
        if st.button:
            import pickle
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("Modelo salvo com sucesso!")
            st.balloons()
    elif modelo == "XGBoost":
        #selectbox para selecionar o alvo
        alvo = st.selectbox("Selecione o alvo: ", dataframe.columns)
        # botão para treinar o modelo de xgboost
        st.button("Treinar modelo de XGBoost")
        if st.button:
            # treinando o modelo
            from xgboost import XGBClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            X = dataframe.drop(alvo, axis=1)
            y = dataframe[alvo]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.subheader("**Avaliação do Modelo**")
            st.write("Acurácia:", accuracy_score(y_test, y_pred)*100,"%")
            st.success("Modelo treinado com sucesso!")
        # botão para salvar o modelo
        st.button("Salvar modelo")
        if st.button:
            import pickle
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("Modelo salvo com sucesso!")
            st.balloons()


if choice == "Previsão":
    st.header("Previsão do estado dos alunos")
    st.subheader("Insira os dados do aluno em cada campo respeitando as indicações e clique em 'Prever'")
    curso = st.selectbox("Curso", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
    st.info("1 - Tec. de Biocombustíveis 2 - Animação e Design 3 - Serv. Social (Noturno) 4 - Agronomia "
            "5 - Design de Comunicação 6 - Medicina Veterinária 7 - Engenharia da Computação 8 - Equinicultura "
            "9 - Gestão 10 - Serviço Social 11 - Turismo 12 - Enfermagem 13 - Higiene Oral 14 - Publicidade "
            "15 - Jornalismo e Comunicação 16 - Letras 17 - Administração de Empresas")
    turno = st.selectbox("Turno", (0, 1))
    st.info("0 - Noturno 1 - Diurno")
    qualificacao_previa = st.selectbox("Qualificação prévia", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
    st.info("1 - Educação secundária 2 - Ensino superior - bacharelado 3 - Ensino superior - licenciatura "
            "4 - Ensino superior - mestrado 5 - Ensino superior - doutorado 6 - Frequência de ensino superior "
            "7 - 12º ano de escolaridade - não concluído 8 - 11º ano de escolaridade - não concluído "
            "9 - Outros - 11º ano de escolaridade 10 - 10º ano de escolaridade 11 - 10º ano de escolaridade - não concluído "
            "12 - 3º ciclo do ensino básico (9º/10º/11º ano) ou equivalente "
            "13 - 2º ciclo do ensino básico (6º/7º/8º ano) ou equivalente 14 - Curso de especialização tecnológica "
            "15 - Ensino superior - licenciatura (1º ciclo) 16 - Curso técnico superior profissional "
            "17 - Ensino superior - mestrado (2º ciclo)")
    nacionalidade = st.selectbox("Nacionalidade", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
    # significado dos valore de nacionalidade
    st.info("1 - Português 2 - Alemão 3 - Espanhol 4 - Italiano 5 - Holandês 6 - Inglês 7 - Lituano "
            "8 - Angolano 9 - Cabo-verdiano 10 - Guineense 11 - Moçambicano 12 - Santomense 13 - Turco 14 - Brasileiro "
            "15 - Romeno 16 - Moldávia (República) 17 - Mexicano 18 - Ucraniano 19 - Russo 20 - Cubano 21 - Colombiano")
    necessidades_especiais = st.selectbox("Necessidades especiais", (0,1))
    st.info("0 - Não 1 - Sim")
    mensalidade_em_dia = st.selectbox("Mensalidade em dia", (0, 1))
    st.info("0 - Não 1 - Sim")
    sexo = st.selectbox("Gênero", (0, 1))
    st.info("0 - Feminino 1 - Masculino")
    bolsista = st.selectbox("Bolsista", (0, 1))
    st.info("0 - Não 1 - Sim")
    aprovacoes = st.number_input("Aprovações", min_value=0, max_value=100, value=0, step=1)
    aproveitamentos = st.number_input("Aproveitamentos", min_value=0, max_value=100, value=0, step=1)
    matriculas = st.number_input("Turmas matriculadas", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    media = st.number_input("Média anual", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    indice_desemprego = st.number_input("Índice de desemprego %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    indice_inflacao = st.number_input("Índice de inflação %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    PIB = st.number_input("PIB (Trilhões $)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

 

    if st.button("Prever"):
        import pickle
        model = pickle.load(open("model.pkl", "rb"))
        prediction = model.predict([[curso, 
                                   turno, 
                                   qualificacao_previa, 
                                   nacionalidade, 
                                   necessidades_especiais,
                                   mensalidade_em_dia, 
                                   sexo, 
                                   bolsista, 
                                   aprovacoes, 
                                   aproveitamentos, 
                                   matriculas,
                                   media,
                                   indice_desemprego, 
                                   indice_inflacao, 
                                   PIB]])
        if prediction == 0:
            st.subheader("O aluno tem **grandes** chances de se evasão")
            #probabilidade de evadir
            prob = model.predict_proba([[curso, turno, qualificacao_previa, nacionalidade, necessidades_especiais,
                                     mensalidade_em_dia, sexo, bolsista, aprovacoes, aproveitamentos, matriculas,
                                     media, indice_desemprego, indice_inflacao, PIB]])
            # % de chance de evadir
            st.write("Probabilidade de evasão: ", prob[0][0]*100, "%")

        else:
            st.subheader("O aluno tem grandes chances de concluir o curso")
            #probabilidade de se formar
            prob = model.predict_proba([[curso, turno, qualificacao_previa, nacionalidade, necessidades_especiais,
                                     mensalidade_em_dia, sexo, bolsista, aprovacoes, aproveitamentos, matriculas,
                                     media, indice_desemprego, indice_inflacao, PIB]])
            # % de chance de se formar
            st.write("Probabilidade de sucesso: ", prob[0][1]*100, "%")

if choice == "Previsão de Conjunto de Dados":
    st.header("Previsão de Conjunto de Dados em CSV")
    st.subheader("Faça o upload do arquivo CSV")
    st.info("O arquivo deve conter as colunas: curso, turno, qualificacao_previa, "
            "nacionalidade, necessidades_especiais, mensalidade_em_dia, sexo, bolsista, "
            "aprovacoes, aproveitamentos, matriculas, media, indice_desemprego, "
            "indice_inflacao, PIB e a ordem das colunas deve ser a mesma do treino.")
    data_conj = st.file_uploader("Upload CSV", type=["csv"])
    if data_conj is not None:
        dfp = pd.read_csv(data_conj, index_col=0)
        st.dataframe(dfp.head())
        st.subheader("Selecione as colunas alvo")
        colunas_selecionadas = st.multiselect("Colunas", dfp.columns)
        st.info("Selecione as colunas para prever a evasão dos alunos")
        if st.button("Prever"):
            import pickle
            model = pickle.load(open("model.pkl", "rb"))
            prediction = model.predict(dfp[colunas_selecionadas])
            prob = model.predict_proba(dfp[colunas_selecionadas])
            dfp["Previsão"] = prediction
            dfp["Probabilidade de Sucesso (%)"] = prob[:,1] * 100
            st.dataframe(dfp)
            #botão para download do arquivo csv com as previsões para a pasta Downloads
            if st.button("Download CSV (Previsão)"):
                dfp.to_csv("Downloads/Previsão.csv", index=False)
                st.success("O arquivo foi baixado com sucesso!")
    else:
        st.warning("Por favor, faça o upload do arquivo CSV para realização das previsões!")
        
st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
