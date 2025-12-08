import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Configuração da página
st.set_page_config(page_title="Detecção de Câncer de Pele (EfficientNet)", page_icon="🩺")

st.title("Detecção de Câncer de Pele 🩺")
st.write("Modelo: **EfficientNetB1** (Experimento 2)")
st.write("Utilizando Inteligência Artificial para auxiliar no diagnóstico.")

# --- CONFIGURAÇÃO DO MODELO ---
# Mude o nome aqui se o seu arquivo tiver outro nome
MODEL_FILE = 'best_efficientnet_b4.keras'

# ---------------------------------------------------------
# ⚠️ IMPORTANTE: COLOCA O ID DO TEU NOVO MODELO DO DRIVE AQUI:
file_id = '15bmb-Rqbnn8b7wiozmL3PSqnMmtT7iFz'
# ---------------------------------------------------------

@st.cache_resource
def load_model_from_drive():
    # Se o arquivo não existir localmente, baixa do Drive
    if not os.path.exists(MODEL_FILE):
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(id=file_id, output=MODEL_FILE, quiet=False)
        except Exception as e:
            st.error(f"Não consegui baixar o modelo. Verifique o ID do Google Drive. Erro: {e}")
            return None
    
    try:
        # Carrega o modelo
        model = tf.keras.models.load_model(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Erro ao ler o arquivo do modelo: {e}")
        return None

with st.spinner('Carregando o modelo EfficientNet...'):
    model = load_model_from_drive()

if model is None:
    st.warning("⚠️ O modelo ainda não foi carregado. Verifique o ID no código.")
else:
    st.success("Modelo carregado e pronto!")

# --- INTERFACE DE UPLOAD ---
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem enviada', use_column_width=True)
    
    st.write("Analisando...")
    
    # --- PRÉ-PROCESSAMENTO ---
    # Atenção: O EfficientNetB1 costuma usar 240x240, mas se tu treinou
    # com 176x176, mantém 176x176. Vou deixar 176 padrão do nosso projeto.
    TAMANHO_TREINO = (176, 176) 
    
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    # Redimensiona
    img_array = tf.image.resize(img_array, TAMANHO_TREINO)
    img_array = tf.expand_dims(img_array, 0) # Lote de 1

    # --- PREVISÃO ---
    if st.button("Classificar Lesão"):
        if model:
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
                
            st.write(f"Confiança: **{confidence:.2f}%**")
