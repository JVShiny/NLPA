import re
import unidecode
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Función de normalización de texto
def normalizar(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.split()
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    texto_normalizado = ' '.join(palabras_filtradas)
    return texto_normalizado

data = pd.read_excel('data_etiquetada.xlsx')

#etiquetas = ['Categoria']
#opiniones = ['Opinion']
opiniones = data['Opinion']
etiquetas = data['Categoria']

op_normalizadas = [normalizar(texto) for texto in opiniones]

tagged_data = [TaggedDocument(words=word_tokenize(opinion.lower()), tags=[str(tag)]) for tag, opinion in enumerate(op_normalizadas)]

#model_d2v = Doc2Vec(vector_size=10, window=3, min_count=2, dm=2, epochs=50)
model_d2v = Doc2Vec(vector_size=100, window=5, min_count=1, dm=1, epochs=50)
model_d2v.build_vocab(tagged_data)

for epoch in range(50):
    model_d2v.train(tagged_data, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs)
    model_d2v.alpha -= 0.002
    model_d2v.min_alpha = model_d2v.alpha

model_d2v.save("d2v_model")

print("el modelo se guardo correcatmente... ")
print("creando el modelo knn")

x = [model_d2v.dv[str(tag)] for tag in range(len(op_normalizadas))]
#y = etiquetas
y = etiquetas.astype(str)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)

y_pred = model_knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.externals import joblib
joblib.dump(model_knn, 'modelo_knn.pkl')


    