import pandas as pd
import re
import nltk
import os
import joblib
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('portuguese'))
os.makedirs('models', exist_ok=True)

# ── FEATURES LINGUISTICAS ─────────────────────────────────────
# Ensina o modelo a reconhecer CARACTERISTICAS de fake news
# independente do site de origem

PALAVRAS_SENSACIONALISTAS = [
    'bomba', 'urgente', 'exclusivo', 'inacreditavel', 'chocante',
    'escandalo', 'revelado', 'segredo', 'conspiração', 'mentira',
    'fraude', 'golpe', 'traição', 'absurdo', 'vergonha',
    'destruir', 'acabar', 'colapso', 'catastrofe', 'caos',
    'nunca', 'sempre', 'todos', 'ninguem', 'impossivel',
    'milagre', 'cura', 'perigo', 'ameaça', 'ataque',
    'vaza', 'vazou', 'confirmado', 'provado', 'definitivo'
]

PALAVRAS_JORNALISTICAS = [
    'afirmou', 'declarou', 'informou', 'segundo', 'conforme',
    'de acordo', 'informação', 'dados', 'pesquisa', 'estudo',
    'governo', 'ministério', 'secretaria', 'presidente',
    'aprovado', 'votação', 'projeto', 'lei', 'decreto',
    'porcentagem', 'resultado', 'relatório', 'documento'
]

def extrair_features_linguisticas(texto):
    texto_lower = texto.lower()
    palavras = texto_lower.split()
    total = max(len(palavras), 1)

    features = []

    # 1. Proporção de palavras sensacionalistas
    qtd_sens = sum(1 for p in PALAVRAS_SENSACIONALISTAS if p in texto_lower)
    features.append(qtd_sens / total)

    # 2. Proporção de palavras jornalisticas
    qtd_jorn = sum(1 for p in PALAVRAS_JORNALISTICAS if p in texto_lower)
    features.append(qtd_jorn / total)

    # 3. Proporção de letras maiusculas (GRITO)
    maiusculas = sum(1 for c in texto if c.isupper())
    features.append(maiusculas / max(len(texto), 1))

    # 4. Quantidade de pontos de exclamacao
    features.append(texto.count('!') / total)

    # 5. Quantidade de pontos de interrogacao
    features.append(texto.count('?') / total)

    # 6. Tamanho medio das palavras
    tam_medio = sum(len(p) for p in palavras) / total
    features.append(tam_medio)

    # 7. Tem aspas (citação de fontes)?
    features.append(1 if '"' in texto or "'" in texto else 0)

    # 8. Tem numeros e porcentagens (dados concretos)?
    features.append(1 if re.search(r'\d+%|\d+\.\d+|\d{4}', texto) else 0)

    # 9. Comprimento do texto
    features.append(min(len(palavras) / 500, 1))

    # 10. Ratio sensacionalista/jornalistico
    ratio = qtd_sens / max(qtd_jorn, 1)
    features.append(min(ratio, 10) / 10)

    return features

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'[^a-zA-ZáéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palavras = [p for p in texto.split() if p not in stop_words]
    return ' '.join(palavras)

# ── CARREGAR DADOS ────────────────────────────────────────────
print("Carregando dados...")
df = pd.read_csv('data/dados_processados.csv')
df = df.dropna(subset=['texto_limpo'])
df = df[df['texto_limpo'].str.len() > 50]

print(f"Total: {len(df)}")
print(f"Verdadeiras: {sum(df.label==0)} | Falsas: {sum(df.label==1)}")

X_texto = df['texto_limpo']
X_original = df['text'].fillna(df['texto_limpo'])
y = df['label']

# ── DIVIDIR ───────────────────────────────────────────────────
X_texto_train, X_texto_test, X_orig_train, X_orig_test, y_train, y_test = train_test_split(
    X_texto, X_original, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {len(X_texto_train)} | Teste: {len(X_texto_test)}")

# ── VETORIZACAO TF-IDF ────────────────────────────────────────
print("\nVetorizando...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),    # unigrams, bigrams e trigrams
    min_df=2,
    sublinear_tf=True,     # normalização logarítmica
    analyzer='word'
)

X_train_tfidf = vectorizer.fit_transform(X_texto_train)
X_test_tfidf  = vectorizer.transform(X_texto_test)

# ── FEATURES LINGUISTICAS ─────────────────────────────────────
print("Extraindo features linguisticas...")
from scipy.sparse import csr_matrix

X_train_ling = csr_matrix([extrair_features_linguisticas(t) for t in X_orig_train])
X_test_ling  = csr_matrix([extrair_features_linguisticas(t) for t in X_orig_test])

# Combina TF-IDF + features linguisticas
X_train_final = hstack([X_train_tfidf, X_train_ling])
X_test_final  = hstack([X_test_tfidf,  X_test_ling])

# ── TREINAR MODELO ────────────────────────────────────────────
print("\nTreinando modelo...")

modelo = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    solver='lbfgs'
)

modelo.fit(X_train_final, y_train)
print("Modelo treinado!")

# ── AVALIAR ───────────────────────────────────────────────────
y_pred = modelo.predict(X_test_final)
print(f"\nAcuracia: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Verdadeira', 'Fake News']))

# ── SALVAR ────────────────────────────────────────────────────
joblib.dump(modelo, 'models/modelo_fakenews.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# Salva função de features para usar no app
import pickle
with open('models/feature_extractor.pkl', 'wb') as f:
    pickle.dump(extrair_features_linguisticas, f)

print("\nModelo salvo!")