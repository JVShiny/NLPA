import pandas as pd
import spacy
import unidecode
from concurrent.futures import ThreadPoolExecutor
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Función para normalizar el texto usando spaCy
def normalize_text(text):
    text = str(text).lower()
    text = unidecode.unidecode(text)
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return text

# Función para mostrar nubes de palabras
def show_wordcloud(categoria, text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Nube de palabras para la categoría {categoria}', fontsize=16)
    plt.axis('off')
    plt.show()

# Cargar el modelo en español de spaCy
nlp = spacy.load("es_core_news_sm")

# Reemplazar los valores NaN con cadenas vacías en la columna "Opinion"
df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
df['Opinion'].fillna('', inplace=True)

# Normalizar el dataframe en paralelo
with ThreadPoolExecutor() as executor:
    df['Opinion'] = list(executor.map(normalize_text, df['Opinion']))

clases = df['Attraction'].unique()

# Generar y mostrar nubes de palabras
for categoria in clases:
    text = ' '.join(df[df['Attraction'] == categoria]['Opinion'].tolist())
    show_wordcloud(categoria, text)

# Calcular las palabras más comunes para cada categoría
top50 = {}
for categoria in clases:
    palabras = [word for opinion in df[df['Attraction'] == categoria]['Opinion'] for word in opinion.split()]
    palabras_comunes = [word for word, _ in Counter(palabras).most_common(50)]
    top50[categoria] = palabras_comunes

data = {}

# Procesar palabras según su categoría
for categoria, palabras in top50.items():
    text = ' '.join(palabras)
    doc = nlp(text)
  
    adjetivos = [token.text for token in doc if token.pos_ == 'ADJ']
    verbos = [token.text for token in doc if token.pos_ == 'VERB']
    sustantivos = [token.text for token in doc if token.pos_ == 'NOUN']
    ner = [ent.text for ent in doc.ents]

    data[f'Adjetivos {categoria}'] = adjetivos
    data[f'Verbos {categoria}'] = verbos
    data[f'Sustantivos {categoria}'] = sustantivos
    data[f'NER {categoria}'] = ner

# Función para crear el vector de características
def create_vector(row, columns):
    return [row[col] for col in columns]

for categoria in clases:
    for component in ['Adjetivos', 'Verbos', 'Sustantivos', 'NER']:
        col_name = f"{component} {categoria} Count"
        palabras_to_count = data[f"{component} {categoria}"]
        df[col_name] = df['Opinion'].apply(lambda x: sum(word in x.split() for word in palabras_to_count))

new_columns = []

for categoria in clases:
    for component in ['Adjetivos', 'Verbos', 'Sustantivos', 'NER']:
        new_columns.append(f"{component} {categoria} Count")

# Entrenar el modelo
df['Feature_Vector'] = df.apply(lambda row: create_vector(row, new_columns), axis=1)
y = df['Attraction']
X = df['Feature_Vector'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb = GaussianNB()

# Entrenar el clasificador
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()