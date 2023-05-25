import streamlit as st  # pip install streamlit
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
import joblib
from streamlit_card import card



st.markdown("<h2 style='text-align: center; color: #339665;'>Sistema de diagnóstico de Diabetes</h2>", unsafe_allow_html=True)
st.image('ft2.jpg')

texto = """ Caros amigos e amigas, gostaria de apresentar a vocês um sistema de diagnóstico de diabetes, uma ferramenta inovadora que pode ajudar a 
identificar a possibilidade de ter essa condição médica. O diabetes é uma das doenças crônicas com risco de vida que mais 
cresce e já afetou 422 milhões de pessoas em todo o mundo de acordo com o relatório da Organização Mundial da Saúde (OMS), em 2018. Devido à presença
 de uma fase assintomática relativamente longa, a deteção precoce do diabetes é sempre desejado para um resultado
  clinicamente significativo. Cerca de 50% de todas as pessoas que sofrem de diabetes não são diagnosticadas devido
   à sua fase assintomática de longo prazo. O tratamento precoce e  adequado do diabetes pode ajudar a reduzir o risco de complicações e melhorar o
    prognóstico do paciente."""

texto1 = """Este Algoritimo(sistema) foi treinado com dados coletados através de questionários diretos  e resultados de diagnósticos 
dos pacientes do Sylhet Diabetes Hospital em Sylhet, Bangladesh. Com o objectivo de ajudar as pessoas
a identificar os sinais de diabetes na fase incial, Espero que seja util para você."""



st.markdown(f'<p style="text-align: justify">{texto}</p>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align: justify">{texto1}</p>', unsafe_allow_html=True)


col1, col2 = st.columns(2)
# Collects user input features into dataframe
def user_input_features():

    with col1:
        idade = st.number_input('Insira a sua idade: ', min_value=0, max_value=150)
        genero = st.selectbox('Qual e o seu genero? ', ('Selecione uma opção', 'Masculino', 'Femenino'))
        cura_atrasada = st.selectbox('Você teve uma cura tardia notada quando ferido? ', ('Selecione uma opção', 'Sim', 'Nao'))
        poliúria = st.selectbox('Você esta tendo um volume excessivo de urina? ', ('Selecione uma opção', 'Sim', 'Nao'))
        polidipsia = st.selectbox('Você esta tendo uma sede excessiva? ', ('Selecione uma opção', 'Sim', 'Nao'))
        Perda_peso_repentina = st.selectbox('Você teve um episódio de perda súbita de peso? ', ('Selecione uma opção', 'Sim', 'Nao'))
        fraqueza = st.selectbox('Você teve um episódio de sensação de fraqueza? ', ('Selecione uma opção', 'Sim', 'Nao'))
        obesidade = st.selectbox('Você pode ser considerado obeso ou não usando seu índice de massa corporal? ',('Selecione uma opção', 'Sim', 'Nao'))

    with col2:
        polifagia = st.selectbox('Você teve um episódio de fome excessiva/extrema? ', ('Selecione uma opção', 'Sim', 'Nao'))
        candidiase_genital = st.selectbox('Você teve uma infecção por fungos? ', ('Selecione uma opção', 'Sim', 'Nao'))
        desfoque_visual = st.selectbox('Você teve um episódio de visão turva? ', ('Selecione uma opção', 'Sim', 'Nao'))
        coceira = st.selectbox('Você teve um episódio de coceira excessiva? ', ('Selecione uma opção', 'Sim', 'Nao'))
        irritabilidade = st.selectbox('Você teve um episódio de irritabilidade? ', ('Selecione uma opção', 'Sim', 'Nao'))
        calvicie = st.selectbox('Você esta tendo perda de cabelo? ', ('Selecione uma opção', 'Sim', 'Nao'))
        rigidez_muscular = st.selectbox('Você teve um episódio de rigidez muscular? ', ('Selecione uma opção', 'Sim', 'Nao'))
        partial_paresis = st.selectbox('Você teve um episódio de enfraquecimento de um músculo/grupo de músculos? ',
                                       ('Selecione uma opção', 'Sim', 'Nao'))



    data = {'idade': idade, 'genero': genero, 'poliúria': poliúria, 'polidipsia': polidipsia,
            'Perda_peso_repentina': Perda_peso_repentina, 'fraqueza': fraqueza,'polifagia':polifagia, 'candidiase_genital': candidiase_genital, 'desfoque_visual': desfoque_visual,
            'coceira': coceira, 'irritabilidade': irritabilidade, 'cura_atrasada': cura_atrasada, 'partial_paresis': partial_paresis, 'rigidez_muscular': rigidez_muscular,
            'calvicie': calvicie, 'obesidade': obesidade}


    Novos_Pacientes = pd.DataFrame(data, index=[0])
    Novos_Pacientes['genero'] = np.where(
        (Novos_Pacientes['genero'] == 'Masculino'), 1,
        np.where(
        (Novos_Pacientes['genero'] == 'Femenino'), 0, '2'))
    Novos_Pacientes['poliúria'] = np.where(
        (Novos_Pacientes['poliúria'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['poliúria'] == 'Nao'), 0, '2'))
    Novos_Pacientes['polidipsia'] = np.where(
        (Novos_Pacientes['polidipsia'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['polidipsia'] == 'Nao'), 0, '2'))
    Novos_Pacientes['Perda_peso_repentina'] = np.where(
        (Novos_Pacientes['Perda_peso_repentina'] == 'Sim'), 1,
        np.where(
         (Novos_Pacientes['Perda_peso_repentina'] == 'Nao'), 0, '2'))
    Novos_Pacientes['fraqueza'] = np.where(
        (Novos_Pacientes['fraqueza'] == 'Sim'), 1,
    np.where(
        (Novos_Pacientes['fraqueza'] == 'Nao'), 0 , '2'))
    Novos_Pacientes['polifagia'] = np.where(
        (Novos_Pacientes['polifagia'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['polifagia'] == 'Nao'), 0, '2'))

    Novos_Pacientes['candidiase_genital'] = np.where(
        (Novos_Pacientes['candidiase_genital'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['candidiase_genital'] == 'Nao'), 0, '2'))
    Novos_Pacientes['desfoque_visual'] = np.where(
        (Novos_Pacientes['desfoque_visual'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['desfoque_visual'] == 'Nao'), 0, '2'))
    Novos_Pacientes['coceira'] = np.where(
        (Novos_Pacientes['coceira'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['coceira'] == 'Nao'), 0, '2'))
    Novos_Pacientes['irritabilidade'] = np.where(
        (Novos_Pacientes['irritabilidade'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['irritabilidade'] == 'Nao'), 0, '2'))
    Novos_Pacientes['cura_atrasada'] = np.where(
        (Novos_Pacientes['cura_atrasada'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['cura_atrasada'] == 'Nao'), 0, '2'))
    Novos_Pacientes['partial_paresis'] = np.where(
        (Novos_Pacientes['partial_paresis'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['partial_paresis'] == 'Nao'), 0, '2'))
    Novos_Pacientes['rigidez_muscular'] = np.where(
        (Novos_Pacientes['rigidez_muscular'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['rigidez_muscular'] == 'Nao'), 0, '2'))
    Novos_Pacientes['calvicie'] = np.where(
        (Novos_Pacientes['calvicie'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['calvicie'] == 'Nao'), 0, '2'))
    Novos_Pacientes['obesidade'] = np.where(
        (Novos_Pacientes['obesidade'] == 'Sim'), 1,
        np.where(
        (Novos_Pacientes['obesidade'] == 'Nao'), 0, '2'))

    Novos_Pacientes['genero'] = Novos_Pacientes['genero'].astype(int)
    Novos_Pacientes['poliúria'] = Novos_Pacientes['poliúria'].astype(int)
    Novos_Pacientes['polidipsia'] = Novos_Pacientes['polidipsia'].astype(int)
    Novos_Pacientes['Perda_peso_repentina'] = Novos_Pacientes['Perda_peso_repentina'].astype(int)
    Novos_Pacientes['fraqueza'] = Novos_Pacientes['fraqueza'].astype(int)
    Novos_Pacientes['polifagia'] = Novos_Pacientes['polifagia'].astype(int)
    Novos_Pacientes['candidiase_genital'] = Novos_Pacientes['candidiase_genital'].astype(int)
    Novos_Pacientes['desfoque_visual'] = Novos_Pacientes['desfoque_visual'].astype(int)
    Novos_Pacientes['coceira'] = Novos_Pacientes['coceira'].astype(int)
    Novos_Pacientes['irritabilidade'] = Novos_Pacientes['irritabilidade'].astype(int)
    Novos_Pacientes['cura_atrasada'] = Novos_Pacientes['cura_atrasada'].astype(int)
    Novos_Pacientes['partial_paresis'] = Novos_Pacientes['partial_paresis'].astype(int)
    Novos_Pacientes['rigidez_muscular'] = Novos_Pacientes['rigidez_muscular'].astype(int)
    Novos_Pacientes['calvicie'] = Novos_Pacientes['calvicie'].astype(int)
    Novos_Pacientes['obesidade'] = Novos_Pacientes['obesidade'].astype(int)


    data = {
        'idade': 47.557333,
        'genero': 0.621333,
        'poliúria': 0.506667,
        'polidipsia': 0.456000,
        'Perda_peso_repentina': 0.408000,
        'fraqueza': 0.592000,
        'polifagia': 0.445333,
        'candidiase_genital': 0.208000,
        'desfoque_visual': 0.461333,
        'coceira': 0.469333,
        'irritabilidade': 0.245333,
        'cura_atrasada': 0.437333,
        'partial_paresis': 0.442667,
        'rigidez_muscular': 0.378667,
        'calvicie': 0.344000,
        'obesidade': 0.176000
    }

    treino_mean = pd.Series(data)

    data = {
        'idade': 11.931457,
        'genero': 0.485703,
        'poliúria': 0.500623,
        'polidipsia': 0.498726,
        'Perda_peso_repentina': 0.492120,
        'fraqueza': 0.492120,
        'polifagia': 0.497667,
        'candidiase_genital': 0.406419,
        'desfoque_visual': 0.499169,
        'coceira': 0.499725,
        'irritabilidade': 0.430860,
        'cura_atrasada': 0.496720,
        'partial_paresis': 0.497366,
        'rigidez_muscular': 0.485703,
        'calvicie': 0.475676,
        'obesidade': 0.381329
    }

    treino_std = pd.Series(data)

    Novos_Pacientes = (Novos_Pacientes - treino_mean) / treino_std

    return Novos_Pacientes

input_df = user_input_features()

#st.write(input_df)


def predict(data):
    clf = joblib.load("modelo_v2.sav")
    return clf.predict(data)
col3, col4,col5 = st.columns(3)
with col4:
 if st.button("Clique para obter o resultado!"):
    #st.table(input_df)
    result = predict(input_df)

    if (result[0]== 0):
        st.markdown('<div style="text-align: justify;border: 2px solid green;">' +
                    '<h5>Caro Amigo 😎 💖<span style="color: green;">' +
                    'as chances de teres diabetes são mininas, continue mantendo um estilo de vida saudável.</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="display: flex; justify-content: center;border: 2px solid red;">' +
                    '<div style="text-align: justify;">' +
                    '<h5>Caro Amigo 😟<span style="color: red;">' +
                    'é importante que você saiba que existe um grande risco de teres diabetes. É crucial que você tome medidas preventivas!</span>',
                    unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* CSS for input labels */
    .stNumberInput input[type="number"] {
         color: #302c2c;
        background-color: #68ffc7;
        background-clip: padding-box;
        border: 1px solid #ced4da;
        border-radius: 0.25rem;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }

    /* CSS for select boxes */
    .st-ci {
        color: #302c2c;
        background-color: #68ffc7;
        background-clip: padding-box;
        border: 1px solid #ced4da;
        border-radius: 0.25rem;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    
    
    </style>
    """,
    unsafe_allow_html=True
)
hasClicked = card(
  title="SuloDS!",
  text="Para saber mas sobre mim, click aque.",
  image="https://avatars.githubusercontent.com/u/109469430?v=4",
  url="https://github.com/SuloDS"
)

