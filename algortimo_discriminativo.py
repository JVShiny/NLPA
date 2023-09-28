import nltk
import re
from gensim import models
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize
from unidecode import unidecode
import pandas as pd
from nltk.corpus import stopwords

def normalizar(texto):
    texto = texto.lower()
    texto = unidecode(texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.split()
    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    texto_normalizado = ' '.join(palabras_filtradas)
    return texto_normalizado

data = pd.read_excel('data.xlsx')
data_test = pd.read_excel('data_test.xlsx')
opiniones = data['Opinion']
opiniones_test = data_test['Opinion']

op_normalizadas = [normalizar(texto) for texto in opiniones]
test_normalizado = [normalizar(texto) for texto in opiniones_test]

# list_tags = [str(i) for i in range(len(op_normalizadas))]

# tagged_data = [TaggedDocument(words = word_tokenize(op_normalizadas), tags=[list_tags[i]]) for i, op_normalizadas in enumerate(op_normalizadas)]

# model = Doc2Vec(vector_size=100, window=10, min_alpha = 0.00025, min_count = 1, dm=1, epochs= 50)
# model.build_vocab(tagged_data)


# for epoch in range(50):
#     model.train(tagged_data,
#                 total_examples=model.corpus_count,
#                 epochs=model.epochs)
#     model.alpha = 0.0002
#     model.min_alpha = model.alpha

# model.save("d2v_01.model")




    