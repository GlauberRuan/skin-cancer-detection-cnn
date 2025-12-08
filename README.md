# 🩺 Detecção de Câncer de Pele com Deep Learning

Este repositório contém um estudo comparativo e experimental para a classificação automática de lesões de pele (Benigno vs. Maligno) utilizando Redes Neurais Convolucionais (CNNs).

O projeto foi dividido em duas fases: a reprodução de um artigo de referência e a implementação de uma arquitetura otimizada para superar os resultados iniciais.

## 🎯 Objetivos
* **Reprodução:** Replicar a arquitetura proposta por *Hasan et al.* utilizando o dataset ISIC Archive.
* **Otimização:** Implementar técnicas avançadas de Deep Learning (Transfer Learning, Fine-Tuning, XAI) para melhorar a acurácia e a generalização do modelo.
* **Deploy:** Disponibilizar o modelo para uso prático via aplicação Web.

---

## 🧪 Experimentos Realizados

### 🔹 Experimento 01: Reprodução (Baseline)
Focamos em testar arquiteturas clássicas de Transfer Learning para estabelecer uma linha de base.
* **Modelos Testados:** ResNet50, DenseNet201, Xception, MobileNetV2.
* **Estratégia:** Treinamento padrão com congelamento de camadas base.
* **Resultado:** O modelo sofreu com overfitting e acurácia limitada devido ao desbalanceamento do dataset.
* 📂 [Ver Notebook do Experimento 01](skin_cancer_cnn_classifier_py.ipynb)

### 🚀 Experimento 02: Abordagem Avançada (EfficientNetB7)
Para superar as limitações do primeiro experimento, implementamos uma pipeline de treinamento robusta baseada nas melhores práticas atuais de visão computacional.

* **Modelo:** **EfficientNetB7** (Arquitetura mais eficiente e poderosa que as anteriores).
* **Resultado:** Acurácia de **46.61%** (com melhor generalização).
* 📂 [Ver Notebook do Experimento 02](experimento_02_efficientnetb1.ipynb)

#### 🔧 Principais Melhorias Implementadas:
1.  **Label Smoothing (0.1):** Técnica de regularização que impede o modelo de ser "confiante demais", reduzindo o erro em casos ambíguos.
2.  **Treinamento em 2 Estágios:**
    * *Warmup:* Treinamento apenas do classificador final (Top Layers).
    * *Fine-Tuning:* Descongelamento das últimas 100 camadas com taxa de aprendizado baixa (`1e-5`) para refinar os pesos sem "esquecer" o conhecimento prévio.
3.  **Class Weights:** Cálculo automático de pesos para penalizar mais os erros na classe minoritária, combatendo o desbalanceamento.
4.  **Data Augmentation Agressivo:** Rotações de até 40º, ajustes de brilho e zoom para forçar o modelo a aprender características invariantes.

---

## 📊 Visualização e Explicabilidade (XAI)

O projeto não apenas classifica, mas explica *onde* o modelo está olhando para tomar a decisão:

* **Matriz de Confusão Visual:** Gráfico detalhado para analisar falsos positivos e negativos.
* **Grad-CAM (Gradient Class Activation Maps):** Gera mapas de calor sobre a imagem da lesão, destacando as áreas que mais influenciaram a decisão da IA (garantindo que o modelo olhe para a lesão e não para a pele ao redor).

---

## 💻 Como Executar o Projeto

### Pré-requisitos
```bash
pip install -r requirements.txt

Rodando a Aplicação Web (Streamlit)
O projeto conta com uma interface gráfica para upload e classificação de imagens em tempo real.

streamlit run app.py

🛠️ Tecnologias Utilizadas
Linguagem: Python

Deep Learning: TensorFlow / Keras

Processamento de Imagem: OpenCV, PIL

Interface Web: Streamlit

Ambiente de Treino: Google Colab (T4 GPU)

## 📄 Artigo Base
*Classification of Skin Cancer Images using Convolutional Neural Networks*

---
Desenvolvido por João Pedro, GLauber Ruan e MarcioJr - IFPE Jaboatão
