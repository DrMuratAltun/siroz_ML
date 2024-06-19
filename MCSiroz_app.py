import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modeli yükle
loaded_model = joblib.load('best_model_pipeline.pkl')

# Kategorik ve sayısal özniteliklerin listesi
categorical_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
numeric_columns = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

# Streamlit formu
st.title("Cirrhosis Outcome Prediction")
st.write("Lütfen aşağıdaki formu doldurun ve tahmin sonuçlarını görmek için 'Gönder' butonuna tıklayın.")

with st.form(key='prediction_form'):
    # Kategorik girdiler
    categorical_inputs = {}
    for col in categorical_columns:
        categorical_inputs[col] = st.selectbox(f"{col}", options=['A', 'B', 'C'])  # Kategorik değerlerinizi buraya ekleyin

    # Sayısal girdiler
    numeric_inputs = {}
    for col in numeric_columns:
        numeric_inputs[col] = st.number_input(f"{col}", min_value=0.0, max_value=100.0, step=0.1)
    
    submit_button = st.form_submit_button(label='Gönder')

# Kullanıcı formu gönderdiğinde tahmin yap
if submit_button:
    # Girdileri veri çerçevesine dönüştür
    input_data = {**categorical_inputs, **numeric_inputs}
    input_df = pd.DataFrame([input_data])

    # Tahmin yap
    predictions_proba = loaded_model.predict_proba(input_df)[0]
    class_names = ['C', 'CL', 'D']  # Sınıf isimleri
    predictions_class = np.argmax(predictions_proba)
    
    st.write("Tahmin Olasılıkları:")
    for idx, proba in enumerate(predictions_proba):
        st.write(f"Sınıf {class_names[idx]}: {proba:.2f}")
    
    st.write(f"En yüksek olasılıklı sınıf: {class_names[predictions_class]}")
    
    # Sınıf açıklamaları
    if class_names[predictions_class] == 'C':
        st.write("Tahmin: C (censored) - Hasta N_Days'de hayattaydı.")
    elif class_names[predictions_class] == 'CL':
        st.write("Tahmin: CL - Hasta N_Days'de karaciğer nakli nedeniyle hayattaydı.")
    elif class_names[predictions_class] == 'D':
        st.write("Tahmin: D - Hasta N_Days'de vefat etti.")
