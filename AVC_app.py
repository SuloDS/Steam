import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
st.title(':blue[Acidente Vascular Cerebral (AVC)]')
texto = """Um AVC (Acidente Vascular Cerebral) ocorre quando há uma interrupção do fluxo sanguíneo para 
uma parte do cérebro, levando a uma lesão no tecido cerebral e a possíveis complicações neurológicas.
Existem dois tipos principais de AVC: o isquêmico, que ocorre quando há uma obstrução no fluxo sanguíneo
para o cérebro, e o hemorrágico, que ocorre quando há ruptura de um vaso sanguíneo no cérebro. O AVC é 
uma emergência médica que requer atenção imediata, pois pode causar danos cerebrais permanentes e 
até mesmo levar à morte. Os sintomas do AVC incluem fraqueza ou paralisia em um lado do corpo, dificuldade
em falar ou entender a fala, visão turva ou dupla, tontura e dor de cabeça intensa. O tratamento precoce e 
adequado do AVC pode ajudar a reduzir o risco de complicações e melhorar o prognóstico do paciente."""

st.markdown(f'<p style="text-align: justify">{texto}</p>', unsafe_allow_html=True)
st.image('header5.png')


texto = "Este aplicativo prevê se voce tera um AVC ou não --"

st.markdown(f'<p style="font-size: 18px;">{texto}</p>', unsafe_allow_html=True)



# Collects user input features into dataframe

def user_input_features():
    idade = st.sidebar.number_input('Insira a sua idade (idade.00): ')
    sexo = st.sidebar.selectbox('Insira o seu sexo (0=Mulher, 1=Homen): ', (0, 1))
    hipertesao = st.sidebar.selectbox('Você é hipertenso (0=não, 1=sim) : ', (0, 1))
    doenca_cardiaca = st.sidebar.selectbox('Você tem histórico ou possui alguma doença cardíaca  (0=não, 1=sim): ',(0, 1))
    ja_casado = st.sidebar.selectbox('Voce é casado (0=não, 1=sim):', (0, 1))
    tipo_trabalho = st.sidebar.selectbox('Categoria de trabalho (0 - Nunca trabalhou, 1 - Menor de Idade, 2 - Govt_job, 3 - Autônomo, 4 - Privado):',(0, 1, 2, 3, 4))
    tipo_residencia = st.sidebar.selectbox('Tipo da sua habitação (0 - Rural,  1 - Urbano):', (0, 1))
    imc = st.sidebar.number_input('IMC (IMC = peso (kg) / altura^2) : ')
    fumante = st.sidebar.selectbox('Voce é fumante  (0 - não,  1 - sim): ', (0, 1))
    glicose_nivel = st.sidebar.selectbox('Nivel de glicose no sangue (1 - baixo, 2 - normal, 3 - alto): ', (1, 2, 3))


    data = {'sexo':sexo, 'idade':idade, 'hipertesao':hipertesao, 'doenca_cardiaca':doenca_cardiaca, 'ja_casado':ja_casado, 'tipo_trabalho':tipo_trabalho, 'tipo_residencia':tipo_residencia, 'imc':imc,
       'fumante':fumante, 'glicose_nivel':glicose_nivel}

    df = pd.DataFrame(data,index=[0])

    df['idade'] = df['idade'].astype(int)
    df['sexo'] = df['sexo'].astype(int)
    df['hipertesao'] = df['hipertesao'].astype(int)
    df['doenca_cardiaca'] = df['doenca_cardiaca'].astype(int)
    df['ja_casado'] = df['ja_casado'].astype(int)
    df['tipo_trabalho'] = df['tipo_trabalho'].astype(int)
    df['tipo_residencia'] = df['tipo_residencia'].astype(int)
    df['imc'] = df['imc'].astype(float)
    df['fumante'] = df['fumante'].astype(int)
    df['glicose_nivel'] = df['glicose_nivel'].astype(int)


    return df

input_df = user_input_features()

#st.write(input_df)

def predict(data):
    clf = joblib.load("KNNModel.sav")
    return clf.predict(data)


# Apply model to make predictions

if st.button("Clique aqui para realizar a previsão"):

    #st.table(input_df)
    result = predict(input_df)

    if (result[0]== 0):
        st.markdown('<div style="text-align: justify;">' +
                    '<h3>Caro Paciente <span style="color: green;">' +
                    'as chances de teres um AVC são mininas, continue mantendo um estilo de vida saudável para ajudar a prevenir um AVC.</span>' +
                    '<span style="font-size: 26px; margin-left: 10px;">😎</span>' +
                    '<span style="font-size: 26px; margin-left: 10px;">💖</span></h3></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="display: flex; justify-content: center;">' +
                    '<div style="text-align: justify;">' +
                    '<h3>Caro Paciente <span style="color: red;">' +
                    'é importante que você saiba que existe um grande risco de teres um AVC. É crucial que você tome medidas preventivas para reduzir o risco de um AVC.</span>' +
                    '<span style="font-size: 26px; margin-left: 10px;">😟</span></h3></div></div>',
                    unsafe_allow_html=True)
#https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.write("""

# 
# 
### 🤠 Sobre mim: 
#### 


- 🔭 Atualmente estou aprendendo Data Analytics, Machine Learning, Deep Learning e Data Science.
- 👯 Estou procurando colaborar em Data Science.
- 💬 Pergunte-me sobre Análise de Dados, Engenharia e ML.
- 📫 Como chegar até mim Email: <claudiosulo52@gmail.com> ClaudioSulo 
- 📫 Fluxo de trabalho no GitHub: <a href="https://github.com/SuloDS/Data-Science-Projects.git">



#### Mais detalhes sobre mim:  https://www.kaggle.com/claudiosulo
### ✨ Conhecimento técnico: 

![](https://img.shields.io/badge/python-3670A0?style=for-the-badge&amp;logo=python&amp;logoColor=ffdd54)
![](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&amp;logo=numpy&amp;logoColor=white)
![](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&amp;logo=pandas&amp;logoColor=white)
![](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&amp;logo=plotly&amp;logoColor=white)
 ![](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&amp;logo=PyTorch&amp;logoColor=white)
![](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&amp;logo=scikit-learn&amp;logoColor=white) 
![](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&amp;logo=scipy&amp;logoColor=%white)
![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&amp;logo=TensorFlow&amp;logoColor=white)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F9de19bc8674de7e909cfdc555ab8199b%2Fpower%20bi.JPG?generation=1674674584825248&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F2984bf7961a04d79aa992de7e25fa036%2Ftableau.JPG?generation=1674674585096135&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9733051%2F224ba3f0a7a6dd52c6c5d57b4c6768bc%2Fmysql.JPG?generation=1674674585250106&alt=media)
![](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white)
![]()
![]()
![]()

![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmdriponmiah%2Fkaggle-badge&count_bg=%23DDAA17&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)


""")                                                                                  
