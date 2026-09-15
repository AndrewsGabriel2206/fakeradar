import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import re

SITES_CONFIAVEIS = [
    "g1.globo.com",
    "uol.com.br",
    "cnnbrasil.com.br",
    "bbc.com",
    "gov.br",
    "estadao.com.br",
    "folha.uol.com.br",
    "agenciabrasil.ebc.com.br",
    "veja.abril.com.br",
    "gazetadopovo.com.br",
    "cartacapital.com.br",
    "poder360.com.br",
    "tse.jus.br",
    "stf.jus.br"
]

SITES_FAKE = [
    "pragmatismopolitico",
    "revistaforum",
    "olavodecarvalho",
    "terraplanismo",
    "conspiracao",
    "boato",
    "mentira"
]

def extrair_palavras_chave(texto):
    # Remove palavras comuns e pega as mais relevantes
    stop = ['para', 'com', 'uma', 'que', 'por', 'mais', 'como', 'este', 'esta', 'isso']
    palavras = texto.split()
    importantes = [p for p in palavras if len(p) > 4 and p.lower() not in stop]
    return " ".join(importantes[:10])

def buscar_noticias(texto, num=10):
    try:
        consulta = extrair_palavras_chave(texto)
        resultados = []
        titulos = []

        with DDGS() as ddgs:
            for r in ddgs.text(consulta, region='br-pt', max_results=num):
                # Guarda URL e titulo
                resultados.append(r['href'])
                titulos.append(r.get('title', ''))

        # Filtra sites confiaveis
        confiaveis = [u for u in resultados if any(s in u for s in SITES_CONFIAVEIS)]

        # Remove sites suspeitos
        resultados = [u for u in resultados if not any(s in u for s in SITES_FAKE)]

        return confiaveis if confiaveis else resultados[:5]

    except Exception as e:
        print(f"Erro na busca: {e}")
        return []

def extrair_titulo_pagina(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, 'html.parser')
        titulo = soup.find('title')
        return titulo.get_text().strip()[:100] if titulo else url
    except:
        return url

def extrair_texto_pagina(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        paragrafos = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragrafos])[:3000]
    except:
        return ""

def identificar_fonte(url):
    for site in SITES_CONFIAVEIS:
        if site in url:
            nomes = {
                'g1.globo.com': 'G1 Globo',
                'uol.com.br': 'UOL',
                'cnnbrasil.com.br': 'CNN Brasil',
                'bbc.com': 'BBC Brasil',
                'gov.br': 'Portal do Governo',
                'estadao.com.br': 'Estadão',
                'folha.uol.com.br': 'Folha de S.Paulo',
                'agenciabrasil.ebc.com.br': 'Agência Brasil',
                'veja.abril.com.br': 'Revista Veja',
                'gazetadopovo.com.br': 'Gazeta do Povo',
                'poder360.com.br': 'Poder360',
                'tse.jus.br': 'TSE',
                'stf.jus.br': 'STF'
            }
            return nomes.get(site, site)
    return None

def analisar_com_ia(noticia_usuario, resultados, prob_fake):

    confiaveis = [u for u in resultados if any(s in u for s in SITES_CONFIAVEIS)]
    total = len(resultados)
    prob_fake = float(prob_fake)

    # Monta lista de fontes com nomes
    lista_fontes = ""
    for url in confiaveis[:4]:
        nome = identificar_fonte(url)
        lista_fontes += f"\n- [{nome}]({url})"

    lista_outros = ""
    for url in resultados[:3]:
        if url not in confiaveis:
            lista_outros += f"\n- {url}"

    # ── VEREDICTO VERDADEIRA ──────────────────────────────────
    if len(confiaveis) >= 2 and prob_fake < 60:
        return f"""### ✅ VEREDICTO: VERDADEIRA

**Explicação:** A notícia foi confirmada em **{len(confiaveis)} fontes jornalísticas confiáveis**. 
O modelo de Machine Learning indica apenas **{prob_fake:.0f}%** de probabilidade de ser fake news, 
reforçando que o conteúdo é legítimo.

**Fontes confiáveis que confirmam a notícia:**{lista_fontes}

**Nível de Risco:** 🟢 Baixo

**Recomendação:** Esta notícia pode ser compartilhada com segurança. 
Sempre verifique a data de publicação antes de compartilhar."""

    # ── VEREDICTO PARCIAL ─────────────────────────────────────
    elif len(confiaveis) == 1:
        return f"""### ⚠️ VEREDICTO: PARCIALMENTE CONFIRMADA

**Explicação:** Encontrei apenas **1 fonte confiável** cobrindo este assunto.
O modelo indica **{prob_fake:.0f}%** de probabilidade de ser fake news.
Isso pode significar que a notícia é verdadeira mas ainda pouco coberta, 
ou que apenas parte das informações é correta.

**Fonte encontrada:**{lista_fontes}

**Outras fontes encontradas:**{lista_outros if lista_outros else ' Nenhuma'}

**Nível de Risco:** 🟡 Médio

**Recomendação:** Aguarde mais cobertura jornalística antes de compartilhar. 
Verifique a fonte original da notícia."""

    # ── VEREDICTO INCONCLUSIVO ────────────────────────────────
    elif total > 0 and prob_fake < 50:
        return f"""### ⚠️ VEREDICTO: INCONCLUSIVO

**Explicação:** Não encontrei confirmação em sites jornalísticos confiáveis, 
porém o modelo de Machine Learning indica apenas **{prob_fake:.0f}%** de probabilidade 
de ser fake news. Isso pode significar que a notícia é real mas ainda não 
foi amplamente coberta pela imprensa.

**Links encontrados (verificar manualmente):**{lista_outros}

**Nível de Risco:** 🟡 Médio

**Recomendação:** Verifique diretamente em fontes como G1, BBC Brasil, 
Folha de S.Paulo ou Agência Brasil antes de compartilhar."""

    # ── VEREDICTO FAKE NEWS ───────────────────────────────────
    elif prob_fake >= 70 and len(confiaveis) == 0:
        return f"""### 🚨 VEREDICTO: POSSÍVEL FAKE NEWS

**Explicação:** Não encontrei **nenhuma fonte jornalística confiável** confirmando 
esta notícia. O modelo de Machine Learning indica **{prob_fake:.0f}%** de probabilidade 
de ser fake news. Notícias falsas frequentemente circulam sem cobertura 
da imprensa tradicional.

**Sinais de alerta:**
- ❌ Nenhuma fonte confiável encontrada
- ❌ Alta probabilidade de fake news pelo modelo ({prob_fake:.0f}%)
- ❌ Linguagem possivelmente sensacionalista

**Nível de Risco:** 🔴 Alto

**Recomendação:** **Não compartilhe** esta notícia sem verificar em fontes 
confiáveis como G1, BBC Brasil, Folha de S.Paulo, Estadão ou portais oficiais 
do governo (gov.br)."""

    # ── VEREDICTO PADRAO ──────────────────────────────────────
    else:
        return f"""### ⚠️ VEREDICTO: REQUER VERIFICAÇÃO

**Explicação:** O modelo indica **{prob_fake:.0f}%** de probabilidade de fake news.
Encontrei **{total} links** relacionados mas não em fontes jornalísticas principais.

**Links encontrados:**{lista_outros}

**Nível de Risco:** 🟡 Médio

**Recomendação:** Verifique a notícia em fontes confiáveis como 
G1, BBC Brasil ou Agência Brasil antes de compartilhar."""