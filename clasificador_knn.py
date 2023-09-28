from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import Doc2Vec
from algortimo_discriminativo import op_normalizadas, test_normalizado
from nltk import word_tokenize


modeld2v = Doc2Vec.load('d2v_01.model')

x = []
y = []

for item in op_normalizadas:
    words = word_tokenize(item[0])
    x.append(modeld2v.infer_vector(words))
    y.append(item[1])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

