import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ThreadPoolExecutor
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def parallel_processing_threads(data, func):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, data))
    return results

def normailizar_texto(text):
    # Tokenizar el texto
    tokens = word_tokenize(text)
    tokens_lower = [word.lower() for word in tokens]
    tokens_no_punct = [word for word in tokens_lower if word.isalnum()]
    tokens_no_stopwords = [word for word in tokens_no_punct if word not in stop_words]
    tokens = " ".join(tokens_no_stopwords)
    doc = nlp(tokens)
    normalizar_texto = [word.lemma_ for word in doc if word.pos_ in tags]
    normalizar_texto = " ".join(normalizar_texto)
    return normalizar_texto

# Cargar el archivo Excel
file_path = 'corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx'
df = pd.read_excel(file_path)
df.head()

# Eliminamos valores nulos
df = df.dropna()
# Lematizador
lemmatizer = WordNetLemmatizer()
# Stopwords en español
stop_words = set(stopwords.words('spanish'))
# Creamos lista de tags a utilizar
tags = ['ADJ', 'NOUN', 'VERB']
# Cargar el modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

# Normaliza el dataframe en paraleloS
df['Opinion'] = parallel_processing_threads(df['Opinion'], normailizar_texto)
# Mostrar el dataframe
df.head()
# Guardar el dataframe en un archivo CSV
df.to_csv('corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train_Normalized.csv', index=False)

# Preparar los datos
tagged_data = [TaggedDocument(words=opinion.split(), tags=[str(i)]) for i, opinion in enumerate(df['Opinion'])]

# Construir y entrenar el modelo
model = Doc2Vec(vector_size=20, window=2, min_count=1, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
# Guardar el modelo para uso futuro
model.save("doc2vec_model_cluster")

# Cargar el dataframe normalizado
df = pd.read_csv('corpus/Rest_Mex_2022_Sentiment_Analysis_Track_Train_Normalized.csv')
# Cargar el modelo
model = Doc2Vec.load("doc2vec_model_cluster")
# Preparar los datos
vector_op = np.array([model.infer_vector(opinion.split()) for opinion in df['Opinion']])
# Dimensiones del vector
vector_op.shape

df_vector = pd.DataFrame(vector_op)
# Eliminar valores nulos
df_vector = df_vector.dropna()
# Nombrar columnas
df_vector.columns = ['V' + str(i) for i in range(20)]
# Mostrar el dataframe
df_vector.head()

# Lista para almacenar los valores de la inercia
inertia = []
# Lista para almacenar el número de clusters
k = []

# Iterar de 1 a 10 clusters
for i in range(1, 11):
    # Crear el modelo
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    # Entrenar el modelo
    model.fit(df_vector)
    # Almacenar el valor de la inercia
    inertia.append(model.inertia_)
    # Almacenar el número de clusters
    k.append(i)

# Graficar el método del codo
plt.plot(k, inertia, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.xticks(k)
plt.show()

# Prepapar los datos
X = df_vector
# Crear el modelo
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Entrenar el modelo
kmeans.fit(X)

# Almacenar las etiquetas
ymeans = kmeans.labels_

print("El índice de silueta es: ", silhouette_score(X, ymeans))
print("La puntuación de Calinski-Harabasz es: ", calinski_harabasz_score(X, ymeans))

# Crear el modelo
pca = PCA(n_components=2)
# Entrenar el modelo
pca.fit(X)

# Transformar los datos
X_pca = pca.transform(X)

# Graficar los clusters con centroides y con nombre de los clusters
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Preparar los datos
X = df_vector
# Crear el modelo
scaler = StandardScaler()
# Entrenar el modelo
scaler.fit(X)
# Transformar los datos
X_scaled = scaler.transform(X)
# Visualizamos los datos como dataframe
df_scaled = pd.DataFrame(X_scaled)
df_scaled.head()

# Preparar los datos
y = df['Attraction'].copy()

# Crear el modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
rf.fit(X_scaled, y)
# Extraer las importancias de las características
importances = rf.feature_importances_

# Graficar las importancias de las características
plt.figure(figsize=(12, 8))
plt.bar(range(len(importances)), importances)
plt.xlabel('Número de característica')
plt.ylabel('Importancia')
plt.show()

df_vector_important_features = df_vector.iloc[:, importances.argsort()[::-1][:15]]
# Mostrar el dataframe
df_vector_important_features.head()
# Prepapar los datos
X_best = df_vector_important_features

# Crear el modelo
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Entrenar el modelo
kmeans.fit(X_best)

ymeans_best = kmeans.labels_

print("El índice de silueta es: ", silhouette_score(X_best, ymeans_best))
print("La puntuación de Calinski-Harabasz es: ", calinski_harabasz_score(X_best, ymeans_best))

# Crear el modelo
pca = PCA(n_components=2)
# Entrenar el modelo
pca.fit(X_best)
# Transformar los datos
X_pca_best = pca.transform(X_best)

# Graficar los clusters con centroides
plt.figure(figsize=(12, 8))
plt.scatter(X_pca_best[:, 0], X_pca_best[:, 1], c=ymeans_best, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()