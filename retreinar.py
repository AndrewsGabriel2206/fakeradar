import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
import os
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('portuguese'))
os.makedirs('models', exist_ok=True)

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'[^a-zA-ZáéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palavras = [p for p in texto.split() if p not in stop_words]
    return ' '.join(palavras)

def extrair_texto(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        paragrafos = soup.find_all('p')
        texto = ' '.join([p.get_text() for p in paragrafos])
        return texto[:3000] if len(texto) > 200 else ""
    except:
        return ""

# ── 1. CARREGA DATASET ORIGINAL ───────────────────────────────
print("Carregando dataset original...")
dados = []

pasta_true = r'C:\Users\andre\Downloads\trabalho\tcc\data\Fake.br-Corpus-master\full_texts\true'
pasta_fake = r'C:\Users\andre\Downloads\trabalho\tcc\data\Fake.br-Corpus-master\full_texts\fake'

for arquivo in Path(pasta_true).glob('*.txt'):
    with open(arquivo, 'r', encoding='utf-8', errors='ignore') as f:
        texto = f.read().strip()
        if texto:
            dados.append({'texto_limpo': limpar_texto(texto), 'label': 0})

for arquivo in Path(pasta_fake).glob('*.txt'):
    with open(arquivo, 'r', encoding='utf-8', errors='ignore') as f:
        texto = f.read().strip()
        if texto:
            dados.append({'texto_limpo': limpar_texto(texto), 'label': 1})

print(f"Dataset original: {len(dados)} noticias")

# ── 2. COLETA NOTICIAS REAIS DO G1 ───────────────────────────
print("\nColetando noticias reais do G1...")
urls_g1 = [
    "https://g1.globo.com/saude/",
    "https://g1.globo.com/economia/",
    "https://g1.globo.com/politica/",
    "https://g1.globo.com/tecnologia/",
    "https://g1.globo.com/mundo/",
]

noticias_coletadas = 0
for url_secao in urls_g1:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url_secao, timeout=10, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'g1.globo.com' in href and '/noticia/' in href:
                links.append(href)
        links = list(set(links))[:15]

        for link in links:
            texto = extrair_texto(link)
            if texto:
                dados.append({'texto_limpo': limpar_texto(texto), 'label': 0})
                noticias_coletadas += 1
                print(f"  Coletado: {link[:60]}...")
    except Exception as e:
        print(f"Erro: {e}")

print(f"\nNoticias G1 coletadas: {noticias_coletadas}")

# ── 3. RETREINA O MODELO ──────────────────────────────────────
df = pd.DataFrame(dados).dropna()
print(f"\nTotal dataset: {len(df)}")
print(f"Verdadeiras: {sum(df.label==0)} | Falsas: {sum(df.label==1)}")

X = df['texto_limpo']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
modelo.fit(X_train_vec, y_train)

y_pred = modelo.predict(X_test_vec)
print(f"\nAcuracia: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Verdadeira', 'Fake News']))

joblib.dump(modelo, 'models/modelo_fakenews.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("\nModelo salvo!")