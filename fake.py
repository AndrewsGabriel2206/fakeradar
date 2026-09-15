import pandas as pd
import numpy as np
import re, nltk
from nltk.corpus import stopwords
import os

os.makedirs('data', exist_ok=True)

nltk.download('stopwords')
nltk.download('punkt')

# ── 1. CARREGAR OS DADOS ──────────────────────────────────────
caminho_true = r'C:\Users\ANDREWS\Documents\trabalho\data\True.csv'
caminho_fake = r'C:\Users\ANDREWS\Documents\trabalho\data\Fake.csv'

df_true = pd.read_csv(caminho_true)
df_fake = pd.read_csv(caminho_fake)

df_true['label'] = 0
df_fake['label'] = 1

df = pd.concat([df_true, df_fake], ignore_index=True)  # ← só uma vez!

print(f"total de noticias: {len(df)}")
print(f"Verdadeiras: {len(df_true)} | Falsas: {len(df_fake)}")
print(df.head())

# ── 2. STOPWORDS ──────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

# ── 3. LIMPEZA DO TEXTO ───────────────────────────────────────
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()  # ← espaço entre as aspas!
    palavras = texto.split()
    palavras = [p for p in palavras if p not in stop_words]
    return ' '.join(palavras)               # ← espaço entre as aspas!

# ── 4. APLICAR LIMPEZA ────────────────────────────────────────
df['texto_limpo'] = df['text'].apply(limpar_texto)

print("\nExemplo de texto original:")
print(df['text'][0][:200])          # ← [:200] com dois pontos, não [200]!
print("\nApós limpeza:")
print(df['texto_limpo'][0][:200])   # ← mostra o texto limpo também

# ── 5. SALVAR ─────────────────────────────────────────────────
df.to_csv('data/dados_processados.csv', index=False)
print("\n✅ Dados salvos em data/dados_processados.csv")