import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = 'data/fakenews.db'
os.makedirs('data', exist_ok=True)

def conectar():
    return sqlite3.connect(DB_PATH)

def criar_banco():
    conn = conectar()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS noticias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            texto TEXT,
            texto_limpo TEXT,
            label INTEGER,
            confianca REAL,
            fonte TEXT,
            dominio TEXT,
            data_coleta TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modelo_usado TEXT,
            bert_resultado INTEGER,
            bert_confianca REAL,
            fontes_encontradas TEXT,
            periodo_noticia TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verificacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            noticia_id INTEGER REFERENCES noticias(id),
            resultado_ml INTEGER,
            confianca_ml REAL,
            veredicto_ia TEXT,
            fontes_web TEXT,
            data_verificacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def extrair_dominio(url):
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except:
        return 'desconhecido'

def salvar_noticia(url, texto, texto_limpo, label, confianca, fonte="",
                   modelo_usado="Gradient Boosting", bert_resultado=None,
                   bert_confianca=None, fontes_encontradas="", periodo_noticia=""):
    try:
        conn = conectar()
        cur = conn.cursor()
        dominio = extrair_dominio(url) if url.startswith('http') else fonte
        cur.execute("""
            INSERT OR REPLACE INTO noticias
            (url, texto, texto_limpo, label, confianca, fonte, dominio,
             modelo_usado, bert_resultado, bert_confianca, fontes_encontradas, periodo_noticia)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (url, texto, texto_limpo, label, confianca, fonte, dominio,
              modelo_usado, bert_resultado, bert_confianca, fontes_encontradas, periodo_noticia))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Erro ao salvar: {e}")
        return False

def buscar_noticia_por_url(url):
    try:
        conn = conectar()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM noticias WHERE url = ?", (url,))
        resultado = cur.fetchone()
        conn.close()
        return dict(resultado) if resultado else None
    except:
        return None

def salvar_verificacao(noticia_id, resultado_ml, confianca_ml, veredicto_ia, fontes_web):
    try:
        conn = conectar()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verificacoes
            (noticia_id, resultado_ml, confianca_ml, veredicto_ia, fontes_web)
            VALUES (?, ?, ?, ?, ?)
        """, (noticia_id, resultado_ml, confianca_ml, veredicto_ia, str(fontes_web)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao salvar verificacao: {e}")

def estatisticas():
    try:
        conn = conectar()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) as total,
                SUM(CASE WHEN label=0 THEN 1 ELSE 0 END) as verdadeiras,
                SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) as fake_news
            FROM noticias
        """)
        row = cur.fetchone()
        conn.close()
        return {'total': row[0], 'verdadeiras': row[1], 'fake_news': row[2]}
    except:
        return None

def estatisticas_por_dominio():
    try:
        conn = conectar()
        cur = conn.cursor()
        cur.execute("""
            SELECT dominio,
                COUNT(*) as total,
                SUM(CASE WHEN label=0 THEN 1 ELSE 0 END) as verdadeiras,
                SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) as fake_news
            FROM noticias
            WHERE dominio IS NOT NULL AND dominio != '' AND dominio != 'desconhecido'
            GROUP BY dominio
            ORDER BY total DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        conn.close()
        return [{'dominio': r[0], 'total': r[1], 'verdadeiras': r[2], 'fake_news': r[3]} for r in rows]
    except:
        return []

def historico_noticias(limite=20, filtro_dominio=None, filtro_label=None,
                        filtro_data_inicio=None, filtro_data_fim=None):
    try:
        conn = conectar()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = "SELECT * FROM noticias WHERE 1=1"
        params = []
        if filtro_dominio:
            query += " AND dominio LIKE ?"
            params.append(f"%{filtro_dominio}%")
        if filtro_label is not None:
            query += " AND label = ?"
            params.append(filtro_label)
        if filtro_data_inicio:
            query += " AND data_coleta >= ?"
            params.append(filtro_data_inicio)
        if filtro_data_fim:
            query += " AND data_coleta <= ?"
            params.append(filtro_data_fim)
        query += " ORDER BY data_coleta DESC LIMIT ?"
        params.append(limite)
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except:
        return []

def comparacao_bert_vs_ml():
    try:
        conn = conectar()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) as total,
                SUM(CASE WHEN label=bert_resultado THEN 1 ELSE 0 END) as concordam,
                SUM(CASE WHEN label!=bert_resultado AND bert_resultado IS NOT NULL THEN 1 ELSE 0 END) as divergem
            FROM noticias WHERE bert_resultado IS NOT NULL
        """)
        row = cur.fetchone()
        conn.close()
        if row and row[0] > 0:
            return {'total': row[0], 'concordam': row[1], 'divergem': row[2]}
        return None
    except:
        return None

if __name__ == '__main__':
    criar_banco()
    print("Banco de dados pronto!")