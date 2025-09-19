import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


df = pd.read_csv('SMSSpams.txt', sep='\t', header=None, names=['label', 'message'])


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Función para limpiar y preprocesar el texto de un mensaje.
    """
    
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_message'] = df['message'].apply(preprocess_text)

print("### Muestra de Datos Preprocesados ###")
print(df.head())
print("\n" + "="*50 + "\n")


X = df['cleaned_message']
y = df['label']

# Dividir el 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} mensajes")
print(f"Tamaño del conjunto de prueba: {len(X_test)} mensajes")
print("\n" + "="*50 + "\n")


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# NAIVE
model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"### Resultados de la Evaluación ###\n")
print(f"Exactitud (Accuracy) del modelo: {accuracy:.4f} (o {accuracy*100:.2f}%)\n")

print("Matriz de Confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(conf_matrix, index=['Real: Ham', 'Real: Spam'], columns=['Pred: Ham', 'Pred: Spam']))
print("\n")

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\n" + "="*50 + "\n")
print("### Prueba con nuevos mensajes ###\n")

new_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts.", # Spam
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?", # Ham
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!" # Spam
]


cleaned_new_messages = [preprocess_text(msg) for msg in new_messages]
new_messages_vec = vectorizer.transform(cleaned_new_messages)

predictions = model.predict(new_messages_vec)

for msg, pred in zip(new_messages, predictions):
    print(f'Mensaje: "{msg}"\nPredicción: {pred.upper()}\n')
