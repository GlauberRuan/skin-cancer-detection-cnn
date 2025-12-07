import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuração da página
st.set_page_config(page_title="Detecção de Câncer de Pele", page_icon="🩺")

st.title("Detecção de Câncer de Pele 🩺")
st.write("Utilizando Inteligência Artificial para auxiliar no diagnóstico.")

# --- CARREGAR O MODELO ---
# Estou usando o ResNet50 como padrão, mas você pode mudar o nome do arquivo abaixo
MODEL_FILE = 'best_model_ResNet50.keras' 

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error(f"Erro: O arquivo '{MODEL_FILE}' não foi encontrado! Certifique-se de que ele está no mesmo repositório do GitHub.")
else:
    st.success("Modelo de IA carregado e pronto!")

# --- INTERFACE DE UPLOAD ---
uploaded_file = st.file_uploader("Escolha uma imagem de lesão de pele...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostra a imagem na tela
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem enviada', use_column_width=True)
    
    st.write("Analisando...")
    
    # --- PRÉ-PROCESSAMENTO (IGUAL AO TREINO) ---
    # Converte para array
    img_array = np.array(image)
    
    # Se a imagem tiver 4 canais (PNG transparente), converte para 3 (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Redimensiona para 176x176 (Tamanho que usamos no treino)
    img_array = tf.image.resize(img_array, [176, 176])
    
    # Expande dimensões (de (176,176,3) para (1, 176, 176, 3))
    img_array = tf.expand_dims(img_array, 0)

    # --- PREVISÃO ---
    if st.button("Classificar Lesão"):
        prediction = model.predict(img_array)
        
        # O modelo retorna probabilidades. Vamos pegar a maior.
        # Assumindo classes: 0 = Benigno, 1 = Maligno
        classes = ['Benigno', 'Maligno']
        
        # Pega a probabilidade bruta
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
