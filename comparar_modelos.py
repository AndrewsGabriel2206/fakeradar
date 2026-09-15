# comparar_modelos.py - compara varios algoritmos de ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# ── 1. CARREGAR DADOS ─────────────────────────────────────────
print("Carregando dados...")
df = pd.read_csv('data/dados_processados.csv')
df = df.dropna(subset=['texto_limpo'])

X = df['texto_limpo']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

# ── 2. VETORIZAÇÃO ────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 3. MODELOS PARA COMPARAR ──────────────────────────────────
modelos = {
    'Regressao Logistica':    LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest':          RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Naive Bayes':            MultinomialNB(),
    'SVM Linear':             LinearSVC(max_iter=1000, class_weight='balanced'),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ── 4. TREINAR E AVALIAR CADA MODELO ─────────────────────────
resultados = {}

for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train_vec, y_train)
    y_pred = modelo.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    resultados[nome] = acc
    print(f"Acuracia: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=['Verdadeira', 'Fake News']))

# ── 5. GRAFICO COMPARATIVO ────────────────────────────────────
print("\nGerando grafico...")
nomes = list(resultados.keys())
acuracias = [v * 100 for v in resultados.values()]

plt.figure(figsize=(10, 6))
bars = plt.barh(nomes, acuracias, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
plt.xlabel('Acuracia (%)')
plt.title('Comparacao de Algoritmos de Machine Learning')
plt.xlim(0, 100)

for bar, acc in zip(bars, acuracias):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{acc:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('resultados/comparacao_modelos.png', dpi=150)
plt.show()
print("Grafico salvo em resultados/comparacao_modelos.png")

# ── 6. SALVA O MELHOR MODELO ─────────────────────────────────
melhor_nome = max(resultados, key=resultados.get)
melhor_acc  = resultados[melhor_nome]
print(f"\nMelhor modelo: {melhor_nome} com {melhor_acc:.2%}")

melhor_modelo = modelos[melhor_nome]
joblib.dump(melhor_modelo, 'models/modelo_fakenews.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print(f"Melhor modelo salvo!")

# ── 7. SALVA TABELA DE RESULTADOS ────────────────────────────
df_resultados = pd.DataFrame({
    'Modelo': list(resultados.keys()),
    'Acuracia': [f"{v:.2%}" for v in resultados.values()]
})
df_resultados = df_resultados.sort_values('Acuracia', ascending=False)
df_resultados.to_csv('resultados/comparacao_modelos.csv', index=False)
print("\nResultados:")
print(df_resultados.to_string(index=False))