import streamlit as st
import joblib
import re
import nltk
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from verificador import buscar_noticias, extrair_texto_pagina, analisar_com_ia
from banco import (criar_banco, salvar_noticia, buscar_noticia_por_url,
                   estatisticas, estatisticas_por_dominio,
                   historico_noticias, comparacao_bert_vs_ml)

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('portuguese'))

criar_banco()

st.set_page_config(
    page_title="FakeRadar — Detector de Fake News",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .titulo-principal {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitulo {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card-resultado {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    .card-verdadeira {
        background: linear-gradient(135deg, #0d2b1d, #1a4731);
        border: 1px solid #2ecc71;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .card-fake {
        background: linear-gradient(135deg, #2b0d0d, #471a1a);
        border: 1px solid #e74c3c;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .fonte-link {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-left: 3px solid #00d2ff;
    }
    .tag-confiavel {
        background: #1a4731;
        color: #2ecc71;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .tag-suspeito {
        background: #471a1a;
        color: #e74c3c;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

SITES_CONFIAVEIS = [
    'g1.globo.com', 'oglobo.globo.com', 'bbc.com', 'folha.uol.com.br',
    'estadao.com.br', 'cnnbrasil.com.br', 'agenciabrasil.ebc.com.br',
    'gov.br', 'tse.jus.br', 'stf.jus.br', 'poder360.com.br',
    'veja.abril.com.br', 'gazetadopovo.com.br', 'uol.com.br',
    'metropoles.com', 'r7.com', 'correiobraziliense.com.br'
]

@st.cache_resource
def carregar_modelo_ml():
    modelo     = joblib.load('models/modelo_fakenews.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    return modelo, vectorizer

@st.cache_resource
def carregar_bert():
    try:
        from transformers import pipeline
        import torch
        return pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"Erro BERT: {e}")
        return None

modelo, vectorizer = carregar_modelo_ml()

if 'bert_carregado' not in st.session_state:
    st.session_state.bert_carregado = False

def analisar_bert_fn(texto, bert_model):
    try:
        resultado = bert_model(
            texto[:512],
            candidate_labels=["noticia verdadeira", "fake news"],
            hypothesis_template="Este texto e uma {}."
        )
        label = resultado['labels'][0]
        score = resultado['scores'][0]
        return (1, score) if 'fake' in label else (0, score)
    except Exception as e:
        print(f"Erro analisar_bert_fn: {e}")
        return None, None

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'[^a-zA-ZáéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palavras = [p for p in texto.split() if p not in stop_words]
    return ' '.join(palavras)

def extrair_texto_do_link(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragrafos = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragrafos])[:5000]
    except:
        return ""

def detectar_periodo(texto):
    anos = re.findall(r'202[4-6]', texto)
    if anos:
        return f"Noticia de {max(anos)}"
    meses = ['janeiro','fevereiro','marco','abril','maio','junho',
             'julho','agosto','setembro','outubro','novembro','dezembro']
    for mes in meses:
        if mes in texto.lower():
            return f"Menciona {mes.capitalize()}"
    return "Periodo nao identificado"

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 FakeRadar")
    st.markdown("*Detector de Fake News com IA*")
    st.markdown("---")
    pagina = st.radio(
        "Navegar",
        ["🔍 Verificar Noticia",
         "📊 Comparacao de Modelos",
         "📈 Estatisticas",
         "📋 Historico",
         "ℹ️ Sobre o Projeto"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📊 Resumo Geral")
    stats = estatisticas()
    if stats and stats['total'] > 0:
        total = stats['total']
        verd  = stats['verdadeiras'] or 0
        fake  = stats['fake_news'] or 0
        st.metric("Total analisadas", total)
        st.metric("Verdadeiras", f"{verd} ({verd/max(total,1):.0%})")
        st.metric("Fake News",   f"{fake} ({fake/max(total,1):.0%})")
    else:
        st.info("Nenhuma noticia analisada ainda.")
    st.markdown("---")
    st.markdown("### 🤖 Modelos Ativos")
    st.success("Gradient Boosting (99.43%)")
    st.success("Busca Web (DuckDuckGo)")
    st.info("Dataset: 42.000+ noticias")
    st.info("Foco: Eleicoes 2026")

# ══════════════════════════════════════════════════════════════
# PAGINA 1 — VERIFICAR NOTICIA
# ══════════════════════════════════════════════════════════════
if pagina == "🔍 Verificar Noticia":

    st.markdown('<div class="titulo-principal">🎯 FakeRadar</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Sistema de Deteccao de Fake News com IA — Eleicoes 2026</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        usar_bert = st.toggle("🧠 Ativar BERT", value=False, help="Ative ANTES de clicar em Verificar")
    with col2:
        periodo_busca = st.selectbox("📅 Periodo das noticias buscadas",
            ["Ultimas 24h", "Ultima semana", "Ultimo mes", "Qualquer periodo"], index=2)
    with col3:
        limiar_fake = st.slider("Limiar de deteccao (%)", 50, 90, 70)

    col1, col2 = st.columns([3, 1])
    with col1:
        link = st.text_input("🔗 Link da noticia", placeholder="https://qualquer-site-de-noticias.com/...")
    with col2:
        filtro_fonte = st.text_input("🔍 Filtrar fonte", placeholder="ex: globo, sbt...")

    noticia = st.text_area(
        "📝 Ou cole o texto completo da noticia",
        placeholder="Cole aqui pelo menos 3 paragrafos completos da noticia...",
        height=150
    )

    with st.expander("📌 Noticias de Demonstracao"):
        demo = st.selectbox("Escolha um exemplo:", [
            "Selecione...",
            "Exemplo Verdadeiro — Camara aprova PEC das igrejas",
            "Exemplo Fake News — Lula cancela eleicoes de 2026",
            "Exemplo Verdadeiro — TSE divulga regras para eleicoes 2026"
        ])
        if demo != "Selecione...":
            if "Camara" in demo:
                noticia = """A Camara dos Deputados aprovou nesta quinta-feira, em primeiro turno,
uma proposta de emenda a Constituicao que amplia a imunidade tributaria de igrejas.
O placar foi de 385 votos favoraveis e 93 contrarios. O texto sera votado
em segundo turno e seguira para o Senado."""
            elif "Lula" in demo:
                noticia = """URGENTE!!! Lula vai CANCELAR as eleicoes de 2026 e se tornar
DITADOR! Fonte exclusiva revelou que o presidente assinou decreto secreto
suspendendo o processo eleitoral. Compartilhe antes que censurem!
O Brasil esta em PERIGO! ACORDE BRASIL!!!"""
            elif "TSE" in demo:
                noticia = """O Tribunal Superior Eleitoral divulgou nesta semana as regras
para as eleicoes de 2026. Segundo o TSE, o periodo de campanha eleitoral
tera inicio em agosto de 2026. A resolucao foi aprovada por unanimidade pelos
ministros do tribunal."""
            st.success("Texto carregado! Clique em Verificar.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        verificar = st.button("🔎 VERIFICAR AGORA", use_container_width=True, type="primary")

    if verificar:
        if link.strip() != "":
            noticia_no_banco = buscar_noticia_por_url(link)
            if noticia_no_banco:
                st.info("📂 Noticia ja analisada anteriormente!")
                if noticia_no_banco['label'] == 1:
                    st.markdown('<div class="card-fake"><h2>🚨 FAKE NEWS</h2></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="card-verdadeira"><h2>✅ VERDADEIRA</h2></div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Confianca", f"{noticia_no_banco['confianca']:.0%}")
                col2.metric("Modelo", noticia_no_banco.get('modelo_usado', 'N/A'))
                col3.metric("Data", str(noticia_no_banco.get('data_coleta', ''))[:10])
                st.stop()

            with st.spinner("🌐 Extraindo texto do link..."):
                texto_extraido = extrair_texto_do_link(link)
                if texto_extraido:
                    noticia = texto_extraido
                    st.success("Texto extraido com sucesso!")
                    with st.expander("📄 Ver texto extraido"):
                        st.write(noticia[:800] + "...")
                else:
                    st.error("Nao consegui ler o link. Cole o texto manualmente.")
                    st.stop()

        if link.strip() == "" and len(noticia.split()) < 30:
            st.warning("Texto muito curto! Cole pelo menos 3 paragrafos ou use o link.")
            st.stop()

        if noticia.strip() != "" or link.strip() != "":
            st.markdown("---")
            periodo = detectar_periodo(noticia)

            with st.spinner("🤖 Analisando com Gradient Boosting..."):
                texto_limpo = limpar_texto(noticia)
                vetor       = vectorizer.transform([texto_limpo])
                prob        = modelo.predict_proba(vetor)[0]
                resultado_ml = 1 if prob[1] > (limiar_fake/100) else 0

            resultado_bert, confianca_bert = None, None
            if usar_bert:
                with st.spinner("🧠 Analisando com BERT (pode demorar ~30s)..."):
                    bert = carregar_bert()
                    if bert is not None:
                        resultado_bert, confianca_bert = analisar_bert_fn(noticia, bert)
                        st.session_state.bert_carregado = True

            with st.spinner("🌐 Buscando fontes na web..."):
                urls = buscar_noticias(noticia)
                if filtro_fonte:
                    urls = [u for u in urls if filtro_fonte.lower() in u.lower()]
                fontes = []
                for url in urls:
                    texto_fonte = extrair_texto_pagina(url)
                    if texto_fonte:
                        fontes.append(f"{url}\n{texto_fonte[:200]}")

            with st.spinner("🧠 Gerando veredicto final..."):
                analise = analisar_com_ia(noticia, urls, prob[1] * 100)

            st.markdown("---")

            if resultado_ml == 1:
                st.markdown('<div class="card-fake"><h1 style="color:#e74c3c">🚨 FAKE NEWS DETECTADA</h1></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card-verdadeira"><h1 style="color:#2ecc71">✅ NOTICIA VERDADEIRA</h1></div>', unsafe_allow_html=True)

            st.markdown("### 🔬 Comparacao dos Modelos na Analise")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🤖 Gradient Boosting (ML)")
                c1, c2 = st.columns(2)
                c1.metric("Prob. Verdadeira", f"{prob[0]:.0%}")
                c2.metric("Prob. Fake News",  f"{prob[1]:.0%}")
                st.metric("Limiar utilizado", f"{limiar_fake}%")
                if resultado_ml == 1:
                    st.error("Resultado: FAKE NEWS")
                else:
                    st.success("Resultado: VERDADEIRA")

            with col2:
                st.markdown("#### 🧠 BERT (XLM-RoBERTa)")
                if not usar_bert:
                    st.info("BERT desativado. Ative o toggle acima para comparacao.")
                elif resultado_bert is None:
                    st.warning("BERT nao retornou resultado. Verifique o terminal.")
                else:
                    st.metric("Confianca BERT", f"{confianca_bert:.0%}")
                    if resultado_bert == 1:
                        st.error("Resultado: FAKE NEWS")
                    else:
                        st.success("Resultado: VERDADEIRA")
                    if resultado_ml == resultado_bert:
                        st.info("Modelos concordam no resultado")
                    else:
                        st.warning("Modelos divergem — verifique o veredicto")

            st.markdown("### 📊 Metricas da Analise")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Prob. Verdadeira", f"{prob[0]:.0%}")
            col2.metric("Prob. Fake News",  f"{prob[1]:.0%}")
            col3.metric("Links Encontrados", len(urls))
            col4.metric("Fontes Confiaveis", len([u for u in urls if any(s in u for s in SITES_CONFIAVEIS)]))
            col5.metric("Periodo", periodo)

            st.markdown("#### Nivel de Risco")
            st.progress(float(prob[1]))
            if prob[1] < 0.3:
                st.success("🟢 Risco Baixo — Alta probabilidade de ser verdadeira")
            elif prob[1] < limiar_fake/100:
                st.warning("🟡 Risco Medio — Verifique as fontes abaixo")
            else:
                st.error("🔴 Risco Alto — Grande probabilidade de fake news")

            if urls:
                st.markdown("### 🌐 Fontes Encontradas na Web")
                st.caption(f"Periodo configurado: {periodo_busca}")
                for url in urls[:6]:
                    confiavel = any(s in url for s in SITES_CONFIAVEIS)
                    icone = "✅" if confiavel else "⚠️"
                    tag = '<span class="tag-confiavel">Confiavel</span>' if confiavel else '<span class="tag-suspeito">Verificar</span>'
                    st.markdown(
                        f'<div class="fonte-link">{icone} <a href="{url}" target="_blank">{url[:75]}...</a> {tag}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Nenhuma fonte encontrada na web.")

            st.markdown("### 🧠 Veredicto Final")
            st.markdown(f'<div class="card-resultado">{analise}</div>', unsafe_allow_html=True)

            fonte = link if link.strip() != "" else "texto_manual"
            salvar_noticia(
                url=fonte, texto=noticia, texto_limpo=texto_limpo,
                label=resultado_ml, confianca=prob[1], fonte=fonte,
                modelo_usado="Gradient Boosting + BERT" if usar_bert else "Gradient Boosting",
                bert_resultado=resultado_bert, bert_confianca=confianca_bert,
                fontes_encontradas=str(urls[:5]), periodo_noticia=periodo
            )

# ══════════════════════════════════════════════════════════════
# PAGINA 2 — COMPARACAO DE MODELOS
# ══════════════════════════════════════════════════════════════
elif pagina == "📊 Comparacao de Modelos":

    st.markdown("## 📊 Comparacao de Algoritmos de Machine Learning")
    st.markdown("Resultados com **42.000+ noticias** em portugues brasileiro — foco em Eleicoes 2026.")

    dados_modelos = {
        'Modelo': ['Gradient Boosting', 'Random Forest', 'SVM Linear', 'Regressao Logistica', 'Naive Bayes'],
        'Acuracia (%)': [99.43, 99.41, 99.40, 98.92, 95.87],
        'Precisao (%)': [99.4, 99.4, 99.4, 98.9, 96.0],
        'Recall (%)':   [99.4, 99.4, 99.4, 98.9, 95.8],
        'F1-Score (%)': [99.4, 99.4, 99.4, 98.9, 95.9],
        'Velocidade':   ['Media', 'Lenta', 'Rapida', 'Rapida', 'Muito Rapida'],
        'Interpretavel':['Nao', 'Nao', 'Parcial', 'Sim', 'Sim'],
    }
    df_modelos = pd.DataFrame(dados_modelos)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cores = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    bars = axes[0].barh(df_modelos['Modelo'], df_modelos['Acuracia (%)'], color=cores)
    axes[0].set_xlabel('Acuracia (%)', color='white')
    axes[0].set_title('Acuracia por Modelo', color='white', fontsize=13)
    axes[0].set_xlim(90, 100)
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#1a1a2e')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#333')
    for bar, acc in zip(bars, df_modelos['Acuracia (%)']):
        axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', color='white', fontweight='bold', fontsize=9)
    bars2 = axes[1].barh(df_modelos['Modelo'], df_modelos['F1-Score (%)'], color=cores)
    axes[1].set_xlabel('F1-Score (%)', color='white')
    axes[1].set_title('F1-Score por Modelo', color='white', fontsize=13)
    axes[1].set_xlim(90, 100)
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#1a1a2e')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#333')
    for bar, f1 in zip(bars2, df_modelos['F1-Score (%)']):
        axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{f1:.2f}%', va='center', color='white', fontweight='bold', fontsize=9)
    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 📋 Tabela Completa")
    st.dataframe(df_modelos, use_container_width=True, hide_index=True)

    # ── MATRIZ DE CONFUSAO E CURVA ROC ────────────────────────
    st.markdown("### 📊 Metricas Avancadas do Modelo Principal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matriz de Confusao")
        if os.path.exists('resultados/confusion_matrix.npy'):
            cm = np.load('resultados/confusion_matrix.npy')
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
            ax_cm.set_title('Matriz de Confusao', color='white')
            ax_cm.set_xlabel('Predito', color='white')
            ax_cm.set_ylabel('Real', color='white')
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(['Verdadeira', 'Fake News'], color='white')
            ax_cm.set_yticklabels(['Verdadeira', 'Fake News'], color='white')
            ax_cm.tick_params(colors='white')
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm[i, j]),
                               ha='center', va='center',
                               color='white' if cm[i,j] > cm.max()/2 else 'black',
                               fontsize=16, fontweight='bold')
            fig_cm.patch.set_facecolor('#1a1a2e')
            ax_cm.set_facecolor('#1a1a2e')
            plt.tight_layout()
            st.pyplot(fig_cm)
            st.markdown("""<div class="card-resultado">
            <p><b>Verdadeiro Positivo (VP):</b> Fake news detectadas corretamente</p>
            <p><b>Verdadeiro Negativo (VN):</b> Noticias verdadeiras detectadas corretamente</p>
            <p><b>Falso Positivo (FP):</b> Noticias verdadeiras classificadas como fake</p>
            <p><b>Falso Negativo (FN):</b> Fake news nao detectadas</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Rode: python gerar_metricas.py para gerar a matriz.")

    with col2:
        st.markdown("#### Curva ROC")
        if os.path.exists('resultados/roc_fpr.npy'):
            fpr     = np.load('resultados/roc_fpr.npy')
            tpr     = np.load('resultados/roc_tpr.npy')
            roc_auc = np.load('resultados/roc_auc.npy')[0]
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            ax_roc.plot(fpr, tpr, color='#2ecc71', lw=2,
                        label=f'AUC = {roc_auc:.4f}')
            ax_roc.plot([0, 1], [0, 1], color='#e74c3c', lw=1,
                        linestyle='--', label='Aleatorio (AUC = 0.5)')
            ax_roc.fill_between(fpr, tpr, alpha=0.1, color='#2ecc71')
            ax_roc.set_xlabel('Taxa de Falso Positivo', color='white')
            ax_roc.set_ylabel('Taxa de Verdadeiro Positivo', color='white')
            ax_roc.set_title('Curva ROC', color='white')
            ax_roc.tick_params(colors='white')
            ax_roc.set_facecolor('#1a1a2e')
            fig_roc.patch.set_facecolor('#1a1a2e')
            ax_roc.legend(facecolor='#1a1a2e', labelcolor='white')
            for spine in ax_roc.spines.values():
                spine.set_edgecolor('#333')
            plt.tight_layout()
            st.pyplot(fig_roc)
            st.markdown(f"""<div class="card-resultado">
            <p><b>AUC-ROC: {roc_auc:.4f}</b></p>
            <p>Quanto mais proximo de 1.0, melhor o modelo.</p>
            <p>AUC = 1.0 modelo perfeito</p>
            <p>AUC = 0.5 equivale a chute aleatorio</p>
            <p>Nosso modelo: <b style="color:#2ecc71">{roc_auc:.4f}</b> excelente!</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Rode: python gerar_metricas.py para gerar a curva ROC.")

    st.markdown("### 📚 Sobre Cada Modelo")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gradient Boosting", "Random Forest",
        "SVM Linear", "Regressao Logistica", "Naive Bayes"
    ])
    with tab1:
        st.markdown("""<div class="card-verdadeira">
        <h3>Gradient Boosting — Modelo Escolhido (99.43%)</h3>
        <p><b>Como funciona:</b> Combina varias arvores de decisao em sequencia, onde cada nova arvore corrige os erros da anterior.</p>
        <p><b>Por que foi escolhido:</b> Maior acuracia, robusto contra overfitting.</p>
        <p><b>Desvantagem:</b> Mais lento para treinar e dificil de interpretar.</p>
        </div>""", unsafe_allow_html=True)
    with tab2:
        st.markdown("""<div class="card-resultado">
        <h3>Random Forest (99.41%)</h3>
        <p><b>Como funciona:</b> Cria centenas de arvores de decisao aleatorias e vota na classificacao final.</p>
        <p><b>Vantagem:</b> Muito robusto e resistente a overfitting.</p>
        </div>""", unsafe_allow_html=True)
    with tab3:
        st.markdown("""<div class="card-resultado">
        <h3>SVM Linear (99.40%)</h3>
        <p><b>Como funciona:</b> Encontra o hiperplano que melhor separa fake news de noticias verdadeiras.</p>
        <p><b>Vantagem:</b> Excelente para texto, eficiente e rapido.</p>
        </div>""", unsafe_allow_html=True)
    with tab4:
        st.markdown("""<div class="card-resultado">
        <h3>Regressao Logistica (98.92%)</h3>
        <p><b>Como funciona:</b> Modelo estatistico que calcula probabilidade de fake com base nos pesos de cada palavra.</p>
        <p><b>Vantagem:</b> O mais interpretavel dos modelos.</p>
        </div>""", unsafe_allow_html=True)
    with tab5:
        st.markdown("""<div class="card-resultado">
        <h3>Naive Bayes (95.87%)</h3>
        <p><b>Como funciona:</b> Usa o Teorema de Bayes para calcular probabilidade de fake.</p>
        <p><b>Vantagem:</b> Extremamente rapido e simples.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📦 Base de Dados")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Noticias", "42.211")
    col2.metric("Verdadeiras", "21.218 (50.3%)")
    col3.metric("Fake News", "20.993 (49.7%)")

    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie([21218, 20993],
                labels=['Verdadeiras\n(21.218)', 'Fake News\n(20.993)'],
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
                textprops={'color': 'white'})
        ax2.set_facecolor('#1a1a2e')
        fig2.patch.set_facecolor('#1a1a2e')
        st.pyplot(fig2)
    with col2:
        st.markdown("""<div class="card-resultado">
        <h4>Fontes do Dataset</h4>
        <ul>
        <li><b>Fake.br-Corpus</b> 7.200 noticias academicas em PT-BR</li>
        <li><b>G1 Globo</b> 75+ noticias coletadas automaticamente</li>
        <li><b>Eleicoes 2026</b> Noticias das secoes eleitorais</li>
        <li><b>CNN Brasil / Estadao</b> Noticias politicas adicionais</li>
        </ul>
        <h4>Pre-processamento</h4>
        <ul>
        <li>Remocao de stopwords em portugues</li>
        <li>Limpeza de links e caracteres especiais</li>
        <li>TF-IDF com 20.000 features</li>
        <li>N-grams (1, 2 e 3 palavras)</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🧠 Comparacao: ML vs BERT")
    comp = comparacao_bert_vs_ml()
    if comp and comp['total'] > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total comparados", comp['total'])
        col2.metric("Concordam", f"{comp['concordam']} ({comp['concordam']/comp['total']:.0%})")
        col3.metric("Divergem",  f"{comp['divergem']} ({comp['divergem']/comp['total']:.0%})")
    else:
        st.info("Ative o BERT na verificacao para ver a comparacao ML vs BERT.")

# ══════════════════════════════════════════════════════════════
# PAGINA 3 — ESTATISTICAS
# ══════════════════════════════════════════════════════════════
elif pagina == "📈 Estatisticas":

    st.markdown("## 📈 Estatisticas do Sistema")

    col1, col2, col3 = st.columns(3)
    with col1:
        filtro_label_stat = st.selectbox("Filtrar resultado", ["Todos", "Verdadeiras", "Fake News"])
    with col2:
        filtro_dominio_stat = st.text_input("Filtrar dominio", placeholder="ex: globo")
    with col3:
        dias = st.selectbox("Periodo", ["Ultimos 7 dias", "Ultimos 30 dias", "Tudo"])

    data_inicio = None
    if dias == "Ultimos 7 dias":
        data_inicio = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    elif dias == "Ultimos 30 dias":
        data_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    stats = estatisticas()
    if stats and stats['total'] > 0:
        total = stats['total']
        verd  = stats['verdadeiras'] or 0
        fake  = stats['fake_news'] or 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analisadas", total)
        col2.metric("Verdadeiras", verd)
        col3.metric("Fake News", fake)
        col4.metric("Taxa de Fake", f"{fake/max(total,1):.0%}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribuicao Geral")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            if verd + fake > 0:
                ax3.pie([verd, fake],
                        labels=['Verdadeiras', 'Fake News'],
                        colors=['#2ecc71', '#e74c3c'],
                        autopct='%1.1f%%',
                        textprops={'color': 'white'})
            ax3.set_facecolor('#1a1a2e')
            fig3.patch.set_facecolor('#1a1a2e')
            st.pyplot(fig3)

        with col2:
            st.markdown("#### % Verdadeira vs Fake por Dominio")
            stats_dom = estatisticas_por_dominio()
            if stats_dom:
                df_dom = pd.DataFrame(stats_dom)
                df_dom['pct_fake'] = (df_dom['fake_news'] / df_dom['total'] * 100).round(1)
                df_dom['pct_verd'] = (df_dom['verdadeiras'] / df_dom['total'] * 100).round(1)
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                dominios = df_dom['dominio'].str[:20]
                ax4.barh(dominios, df_dom['pct_verd'], color='#2ecc71', label='Verdadeiras %')
                ax4.barh(dominios, df_dom['pct_fake'], left=df_dom['pct_verd'],
                         color='#e74c3c', label='Fake %')
                ax4.set_xlabel('%', color='white')
                ax4.set_title('Por Dominio', color='white')
                ax4.tick_params(colors='white')
                ax4.set_facecolor('#1a1a2e')
                fig4.patch.set_facecolor('#1a1a2e')
                ax4.legend(facecolor='#1a1a2e', labelcolor='white')
                for spine in ax4.spines.values():
                    spine.set_edgecolor('#333')
                plt.tight_layout()
                st.pyplot(fig4)

        st.markdown("### 🌐 Detalhamento por Dominio")
        if stats_dom:
            df_show = pd.DataFrame(stats_dom)
            df_show['% Fake']       = (df_show['fake_news']   / df_show['total'] * 100).round(1).astype(str) + '%'
            df_show['% Verdadeira'] = (df_show['verdadeiras'] / df_show['total'] * 100).round(1).astype(str) + '%'
            df_show.columns = ['Dominio', 'Total', 'Verdadeiras', 'Fake News', '% Fake', '% Verdadeira']
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("Analise noticias de diferentes sites para ver estatisticas por dominio.")
    else:
        st.info("Nenhuma noticia analisada ainda. Use a pagina de verificacao!")

# ══════════════════════════════════════════════════════════════
# PAGINA 4 — HISTORICO
# ══════════════════════════════════════════════════════════════
elif pagina == "📋 Historico":

    st.markdown("## 📋 Historico de Noticias Analisadas")

    col1, col2, col3 = st.columns(3)
    with col1:
        filtro_h_dominio = st.text_input("🔍 Filtrar por dominio", placeholder="ex: globo, sbt")
    with col2:
        filtro_h_label = st.selectbox("Filtrar resultado", ["Todos", "Verdadeiras", "Fake News"])
    with col3:
        filtro_h_limite = st.slider("Quantidade", 5, 50, 20)

    label_h = None
    if filtro_h_label == "Verdadeiras": label_h = 0
    elif filtro_h_label == "Fake News": label_h = 1

    historico = historico_noticias(
        limite=filtro_h_limite,
        filtro_dominio=filtro_h_dominio if filtro_h_dominio else None,
        filtro_label=label_h
    )

    if historico:
        for item in historico:
            label_icon = "🚨 FAKE NEWS" if item['label'] == 1 else "✅ VERDADEIRA"
            cor = "#e74c3c" if item['label'] == 1 else "#2ecc71"
            url_curta = str(item.get('url', ''))[:70]
            data = str(item.get('data_coleta', ''))[:16]
            confianca = item.get('confianca', 0) or 0
            modelo_u = item.get('modelo_usado', 'N/A')
            periodo = item.get('periodo_noticia', 'N/A')
            dominio = item.get('dominio', 'N/A')
            st.markdown(f"""
            <div class="card-resultado">
            <b style="color:{cor}">{label_icon}</b> &nbsp;
            <span style="color:#aaa; font-size:0.85rem">🌐 {dominio} | 📅 {data}</span><br>
            <a href="{item.get('url','')}" target="_blank" style="color:#00d2ff">{url_curta}...</a><br>
            <span style="color:#888; font-size:0.8rem">
            Confianca: {confianca:.0%} | Modelo: {modelo_u} | Periodo: {periodo}
            </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhuma noticia encontrada com os filtros aplicados.")

# ══════════════════════════════════════════════════════════════
# PAGINA 5 — SOBRE O PROJETO
# ══════════════════════════════════════════════════════════════
elif pagina == "ℹ️ Sobre o Projeto":

    st.markdown("## ℹ️ Sobre o FakeRadar")

    st.markdown("""<div class="card-resultado">
    <h3>O que e o FakeRadar?</h3>
    <p>Sistema de deteccao de fake news desenvolvido como TCC, utilizando IA e Machine Learning
    para identificar noticias falsas em portugues brasileiro, com foco em noticias politicas das Eleicoes 2026.</p>
    <p>Combina tres abordagens: <b>Machine Learning</b> (TF-IDF + Gradient Boosting),
    <b>analise semantica</b> (BERT/XLM-RoBERTa) e <b>verificacao em tempo real</b> via busca web.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="card-resultado">
        <h3>Tecnologias</h3>
        <ul>
        <li>Python 3.13</li>
        <li>Scikit-learn (ML)</li>
        <li>Gradient Boosting (99.43%)</li>
        <li>TF-IDF Vectorizer (20k features)</li>
        <li>XLM-RoBERTa (BERT multilingual)</li>
        <li>Streamlit (Interface Web)</li>
        <li>SQLite (Banco de Dados)</li>
        <li>DuckDuckGo Search API</li>
        <li>BeautifulSoup (Web Scraping)</li>
        </ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card-resultado">
        <h3>Resultados</h3>
        <ul>
        <li>42.000+ noticias no dataset</li>
        <li>99.43% de acuracia</li>
        <li>5 algoritmos comparados</li>
        <li>Foco em Eleicoes 2026</li>
        <li>Verificacao em tempo real</li>
        <li>17 fontes confiaveis cadastradas</li>
        <li>Banco de dados com historico</li>
        <li>Estatisticas por dominio</li>
        <li>Comparacao ML vs BERT</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="card-resultado">
    <h3>Como o Sistema Funciona</h3>
    <ol>
    <li><b>Entrada:</b> Usuario cola link ou texto da noticia</li>
    <li><b>Extracao:</b> Sistema extrai e limpa o texto automaticamente</li>
    <li><b>ML:</b> Gradient Boosting analisa padroes linguisticos via TF-IDF</li>
    <li><b>BERT:</b> XLM-RoBERTa faz analise semantica profunda (opcional)</li>
    <li><b>Web:</b> DuckDuckGo busca fontes confiaveis para confirmar</li>
    <li><b>Veredicto:</b> Resultado combinado com explicacao detalhada</li>
    <li><b>Historico:</b> Resultado salvo no SQLite para analise futura</li>
    </ol>
    </div>""", unsafe_allow_html=True)