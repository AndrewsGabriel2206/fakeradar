# gerar_metricas.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              classification_report, accuracy_score)

os.makedirs('resultados', exist_ok=True)

# ── CARREGAR DADOS ────────────────────────────────────────────
print("Carregando dados...")
df = pd.read_csv('data/dados_processados.csv')
df = df.dropna(subset=['texto_limpo'])

X = df['texto_limpo']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── CARREGAR MODELO ───────────────────────────────────────────
modelo     = joblib.load('models/modelo_fakenews.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

X_test_vec = vectorizer.transform(X_test)
y_pred     = modelo.predict(X_test_vec)
y_prob     = modelo.predict_proba(X_test_vec)[:, 1]

# ── MATRIZ DE CONFUSAO ────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
np.save('resultados/confusion_matrix.npy', cm)

# ── CURVA ROC ─────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
np.save('resultados/roc_fpr.npy', fpr)
np.save('resultados/roc_tpr.npy', tpr)
np.save('resultados/roc_auc.npy', np.array([roc_auc]))

print(f"Acuracia: {accuracy_score(y_test, y_pred):.2%}")
print(f"AUC-ROC:  {roc_auc:.4f}")
print("Metricas salvas em resultados/")