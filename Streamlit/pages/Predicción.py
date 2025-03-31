import streamlit as st
import numpy as np
from PIL import Image
import pickle
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.ndimage import gaussian_filter

# Inicializar la variable en el session_state
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False

# Cargar el modelo desde el archivo
try:
    with open("svm_digits_model.pkl", "rb") as f:
        modelo = pickle.load(f)
    scaler = modelo["scaler"]
    clf = modelo["clf"]
except FileNotFoundError:
    st.error("Archivo 'svm_digits_model.pkl' no encontrado. Asegúrate de que el modelo esté disponible.")

# Título y descripción
st.title("🎨 Predicción de Dígitos con SVM")
st.markdown("""
Este es un modelo de predicción de dígitos manuscritos usando **Support Vector Machines (SVM)**. 
Dibuja un dígito en el lienzo o sube una imagen para ver cómo el modelo predice el número.
""")
st.markdown("---")

# Estilo para los botones
button_style = """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Configuración del lienzo
st.subheader("✍️ Dibuja un dígito en el lienzo")
canvas_result = st_canvas(
    fill_color="rgb(0, 0, 0)",          # Fondo negro
    stroke_width=20,                   # Grosor del trazo
    stroke_color="rgb(255, 255, 255)", # Trazo blanco
    background_color="rgb(0, 0, 0)",   # Fondo negro
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas"
)

# Subir una imagen desde un archivo
archivo_subido = st.file_uploader("Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"])

# Función para recortar el área del dígito
def crop_digit(image_array):
    coords = np.argwhere(image_array > 50)  # Coordenadas de píxeles relevantes
    if coords.size == 0:  # Si no se encuentra nada, devolver la imagen original
        return image_array
    x0, y0 = coords.min(axis=0)  # Coordenada inicial
    x1, y1 = coords.max(axis=0) + 1  # Coordenada final
    cropped = image_array[x0:x1, y0:y1]
    return cropped

# Función para suavizar la imagen
def smooth_image(image_array):
    return gaussian_filter(image_array, sigma=1)  # Aplicar filtro Gaussiano

# Función para preprocesar la imagen (8x8 píxeles)
def preprocess_image(image):
    if image is None:
        return None
    try:
        # Convertir a escala de grises
        img = Image.fromarray(image.astype('uint8')).convert('L')
        img_array = np.array(img)

        # Recortar el área del dígito
        img_cropped = crop_digit(img_array)

        # Suavizar para capturar gradientes
        img_smoothed = smooth_image(img_cropped)

        # Cambiar tamaño a 8x8
        img_resized = Image.fromarray(img_smoothed).resize((8, 8))
        img_array = np.array(img_resized)

        # Normalizar al rango [0, 16] esperado por load_digits
        img_normalized = (img_array / 255.0) * 16.0
        img_flat = img_normalized.flatten().reshape(1, -1)

        # Mostrar la imagen preprocesada
        st.image(img_array, caption="Imagen ajustada y reducida a 8x8", width=150)

        return img_flat
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

# Función para realizar la predicción
def predict(image, clf):
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return None
    try:
        prediction = clf.predict(preprocessed_image)  # preprocessed_image tiene forma (1, 64)
        return prediction[0]
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None

# Botón para realizar la predicción
if st.button("🔍 Predecir", key="predict_button"):
    if canvas_result.image_data is not None:
        image_array = np.array(canvas_result.image_data)
        predicted_class = predict(image_array, clf)
        if predicted_class is not None:
            st.subheader("✨ Predicción")
            st.write(f"El modelo predice que el número es: **{predicted_class}**")
        else:
            st.write("No se pudo realizar la predicción. Intenta dibujar nuevamente.")
    else:
        st.write("Por favor, dibuja un dígito en el lienzo.")

# Comparación del preprocesamiento del lienzo y la imagen del dataset
if st.button("🔄 Comparar Preprocesamiento con Dataset", key="compare_button"):
    lienzo_preprocessed = None

    # Imagen desde el lienzo
    if canvas_result.image_data is not None:
        lienzo_image = np.array(canvas_result.image_data)
        lienzo_preprocessed = preprocess_image(lienzo_image)
        st.write("Imagen del lienzo procesada:")
        st.image(lienzo_image, caption="Lienzo - Original", width=150)

    # Cargar ejemplo del dataset load_digits
    digits = load_digits()
    ejemplo_digit = digits.images[1]  # Usar un ejemplo de "1" del dataset

    # Normalizar el ejemplo de load_digits al rango [0.0, 1.0] para st.image()
    ejemplo_normalizado = ejemplo_digit / 16.0  # Convertir del rango [0, 16] al rango [0.0, 1.0]
    st.image(ejemplo_normalizado, caption="Ejemplo de load_digits: '1'", width=100)

    # Mostrar los valores preprocesados del lienzo y del dataset
    if lienzo_preprocessed is not None:
        st.write("Comparación de imágenes preprocesadas:")
        st.write(f"Lienzo: {lienzo_preprocessed.flatten()}")
        st.write(f"load_digits Ejemplo (1): {digits.data[1]}")
    else:
        st.write("Dibuja un dígito en el lienzo para compararlo con el dataset.")

# Procesar imagen subida
if archivo_subido is not None:
    try:
        image = Image.open(archivo_subido)
        st.image(image, caption="Imagen subida", width=150)
        image_np = np.array(image.convert("L"))
        prediction = predict(image_np, clf)
        if prediction is not None:
            st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")
        else:
            st.write("No se pudo realizar la predicción.")
    except Exception as e:
        st.error(f"Error al procesar la imagen subida: {e}")

# Información adicional sobre la aplicación
st.markdown("""
### 🚀 ¿Cómo funciona?

Este modelo usa **Support Vector Machines (SVM)** para clasificar dígitos manuscritos. 
El modelo se ha entrenado con un dataset de dígitos y ahora puede predecir los números que dibujes en el lienzo o subas como imágenes.

#### 🧠 ¿Te interesa aprender más?
- [Documentación de SVM en Scikit-learn](https://scikit-learn.org/stable/modules/svm.html)
- [Tutorial sobre SVM en Towards Data Science](https://towardsdatascience.com/support-vector-machines-a-guide-for-beginners-940a7f6a1a0b)
""")

