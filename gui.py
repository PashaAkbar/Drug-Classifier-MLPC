import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Drug Classification using Multi Layer Perceptron Classifier')

loaded_model = joblib.load("mlpc_model.joblib")

c  = st.container()


with c:
    col1, col2,col3,col4,col5 = c.columns(5)
    
    with col1:
        umur = st.number_input("Umur", min_value=0, max_value=100, value=5, step=1, format="%d")
        # st.write('0', number)

    with col2:
        optionSex = st.selectbox(
            "Sex",
            ("M", "F")
        )

    with col3:
        optionBP = st.selectbox(
            "BP",
            ("HIGH", "NORMAL","LOW")
        )

    with col4:
        optionCholesterol = st.selectbox(
            "Cholesterol",
            ("HIGH", "NORMAL","LOW")
        )
    with col5:
        Na_to_K = st.number_input("Na_to_K")


click = st.button("Prediksi Obat")

def pred():
    table = [umur, optionSex, optionBP, optionCholesterol, Na_to_K]

    if(table[1]=="M"):
        table[1] = 0
    else:
        table[1] = 1

    if(table[2]=="HIGH"):
        table[2] = 3
    elif(table[2]=="NORMAL"):
        table[2] = 2
    else:
        table[2] = 1

    if(table[3]=="HIGH"):
        table[3] = 3
    elif(table[3]=="NORMAL"):
        table[3] = 2
    else:
        table[3] = 1
    # st.write(table)

    untukPrediksi = np.array(table)
    input_data_reshaped =untukPrediksi.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)[0]
    st.write(prediction)


    
if(click):
    pred()


