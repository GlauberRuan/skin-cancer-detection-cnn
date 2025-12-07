import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(page_title="Detecção de Câncer de Pele", page_icon="🩺")

st.title("Detecção de Câncer de Pele 🩺")
st.write("Utilizando Inteligência Artificial para auxiliar no diagnóstico.")

# --- CONFIGURAÇÃO DO MODELO ---
MODEL_FILE = 'best_model_ResNet50.keras' 

# COLOCA O TEU ID DO GOOGLE DRIVE AQUI DENTRO DAS ASPAS:
file_id = '1gaekLtSkAKR7eBh71y0Yrk0QafWKNX5_' 

# URL de download direto do Google Drive
url = f'https://drive.google.com/uc?id={file_id}'

@st.cache_resource
def load_model_from_drive():
    # Se o arquivo não existir localmente, baixa do Drive
    if not os.path.exists(MODEL_FILE):
        gdown.download(url, MODEL_FILE, quiet=False)
    
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        return model
    except Exception as e:
        return None

with st.spinner('Baixando e carregando o modelo de IA... (Isso pode demorar um tiquinho)'):
    model = load_model_from_drive()

if model is None:
    st.error("Erro ao carregar o modelo! Verifique o ID do Google Drive.")
else:
    st.success("Modelo pronto para uso!")

# --- INTERFACE DE UPLOAD ---
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem enviada', use_column_width=True)
    
    st.write("Analisando...")
    
    # Pré-processamento
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    # Redimensiona para 176x176 (Tamanho do treino)
    img_array = tf.image.resize(img_array, [176, 176])
    img_array = tf.expand_dims(img_array, 0)

    if st.button("Classificar Lesão"):
        prediction = model.predict(img_array)
        classes = ['Benigno', 'Maligno']
        
        score = tf.nn.softmax(prediction[0])
        class_index = np.argmax(score)
        confidence = 100 * np.max(score)
        result_text = classes[class_index]
        
        st.write("---")
        if result_text == 'Maligno':
            st.error(f"### Resultado: {result_text}")
        else:
            st.success(f"### Resultado: {result_text}")
            
        st.write(f"Confiança da IA: **{confidence:.2f}%**")
