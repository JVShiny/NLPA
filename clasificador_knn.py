import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize
from unidecode import unidecode
import pandas as pd
from nltk.corpus import stopwords

# Función de normalización de texto
def normalizar(texto):
    texto = texto.lower()
    texto = unidecode(texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.split()
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    texto_normalizado = ' '.join(palabras_filtradas)
    return texto_normalizado

# Cargar los datos y normalizarlos
data = pd.read_excel('data.xlsx')
data_test = pd.read_excel('data_test.xlsx')
opiniones = data['Opinion']
opiniones_test = data_test['Opinion']

op_normalizadas = [normalizar(texto) for texto in opiniones]
test_normalizado = [normalizar(texto) for texto in opiniones_test]

# Cargar el modelo Doc2Vec entrenado
modeld2v = Doc2Vec.load('d2v_01.model')

# Extraer vectores de características para los datos de entrenamiento
x = []
y = []

for item in op_normalizadas:
    words = word_tokenize(item)
    x.append(modeld2v.infer_vector(words))
    y.append(item[1])

# Crear etiquetas para el modelo KNN
y_knn = [str(i) for i in range(len(op_normalizadas))]

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y_knn, test_size=0.2, random_state=42)

# Crear y entrenar el modelo KNN
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model_knn.predict(x_test)

# Calcular la precisión del modelo KNN
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Reducción de dimensionalidad a 2D utilizando PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# Crear un DataFrame para facilitar el manejo de los datos
df = pd.DataFrame(data={'X': x_pca[:, 0], 'Y': x_pca[:, 1], 'Category': y})

# Colores para las categorías
colors = ['red', 'green', 'blue', 'purple', 'orange']  # Puedes agregar más colores si es necesario

# Crear un gráfico de dispersión con colores según la categoría
plt.figure(figsize=(10, 8))
for category, color in zip(df['Category'].unique(), colors):
    subset = df[df['Category'] == category]
    plt.scatter(subset['X'], subset['Y'], c=color, label=category)

plt.title('Gráfico de dispersión de palabras categorizadas')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.legend(title='Categoría')
plt.grid(True)
plt.show()
