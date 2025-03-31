from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import streamlit as st

# 1. Cargar el dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=10)

# 4. Optimizar el modelo SVM
param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto"]
}
clf = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
clf.fit(x_train, y_train)

# 5. Evaluar el modelo
st.write("Mejores parámetros:", clf.best_params_)
accuracy = clf.score(x_test, y_test)
st.write(f"Precisión en datos de prueba: {accuracy * 100:.2f}%")

# 6. Probar con un ejemplo específico
test_image = X_scaled[0].reshape(1, -1)
prediction = clf.predict(test_image)
st.write(f"Predicción para el primer dígito: {prediction[0]}, Valor real: {y[0]}")

# Normalizar la imagen para Streamlit (0-16 -> 0-255)
normalized_image = digits.images[0] / 16  # Normalizamos en el rango [0.0, 1.0]
st.image(normalized_image, caption="Primer dígito de load_digits", width=150)

# 7. Guardar el modelo
modelo = {"scaler": scaler, "clf": clf.best_estimator_}
with open("svm_digits_model.pkl", "wb") as f:
    pickle.dump(modelo, f)

st.success("Modelo guardado como 'svm_digits_model.pkl'.")
