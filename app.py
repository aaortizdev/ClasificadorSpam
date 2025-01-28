from flask import Flask, request, render_template
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Descargar stopwords de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inicializar Flask
app = Flask(__name__)

# Cargar y preparar el modelo
def cargar_modelo():
    try:
        # Leer los datos
        df = pd.read_csv("spam.csv", encoding="latin1")
        df = df.iloc[:, :2]
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

        # Extracción del texto
        stemer = PorterStemmer()
        stopwords_set = set(stopwords.words('english'))
        corpus = []

        for i in range(len(df)):
            text = df["text"].iloc[i].lower()
            text = text.translate(str.maketrans("", "", string.punctuation)).split()
            text = [stemer.stem(word) for word in text if word not in stopwords_set]
            text = " ".join(text)
            corpus.append(text)

        # Creación de la matriz de características
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus).toarray()
        y = df['target']

        # Entrenamiento del modelo
        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(X, y)

        return clf, vectorizer
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

# Cargar el modelo y el vectorizador
clf, vectorizer = cargar_modelo()

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def home():
    resultado = ""
    if request.method == "POST":
        # Obtener el texto del formulario
        texto = request.form.get("correo")

        # Procesar el texto
        stemer = PorterStemmer()
        stopwords_set = set(stopwords.words('english'))
        texto_procesado = texto.lower().translate(str.maketrans('', '', string.punctuation))
        texto_procesado = texto_procesado.split()
        texto_procesado = [stemer.stem(word) for word in texto_procesado if word not in stopwords_set]
        texto_procesado = ' '.join(texto_procesado)

        # Clasificar el texto
        X_texto = vectorizer.transform([texto_procesado])
        prediccion = clf.predict(X_texto)[0]

        # Mostrar el resultado
        resultado = "Spam" if prediccion == "spam" else "Ham"

    return render_template("index.html", resultado=resultado)

# Ejecutar la aplicación
if __name__ == "__main__":
    if clf is not None and vectorizer is not None:
        app.run(debug=True)
    else:
        print("No se pudo cargar el modelo. Verifica el archivo de datos y las dependencias.")