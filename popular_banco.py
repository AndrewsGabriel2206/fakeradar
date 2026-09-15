# popular_banco.py - adiciona noticias de exemplo no banco
from banco import criar_banco, salvar_noticia

criar_banco()

noticias = [
    # G1 - Verdadeiras
    ("https://g1.globo.com/politica/noticia/2026/01/camara-aprova-pec.html",
     "A Camara aprovou a PEC com 385 votos favoraveis", "camara aprovou pec votos favoraveis", 0, 0.12, "g1.globo.com"),
    ("https://g1.globo.com/economia/noticia/2026/02/inflacao-cai.html",
     "Inflacao recua em fevereiro segundo IBGE", "inflacao recua fevereiro ibge", 0, 0.08, "g1.globo.com"),
    ("https://g1.globo.com/politica/eleicoes/2026/03/tse-divulga-regras.html",
     "TSE divulga regras para eleicoes 2026", "tse divulga regras eleicoes", 0, 0.15, "g1.globo.com"),

    # CNN Brasil - Verdadeiras
    ("https://cnnbrasil.com.br/politica/lula-assina-decreto.html",
     "Lula assina decreto de medidas economicas", "lula assina decreto medidas economicas", 0, 0.18, "cnnbrasil.com.br"),
    ("https://cnnbrasil.com.br/economia/pib-cresce.html",
     "PIB brasileiro cresce 2.3% no primeiro trimestre", "pib brasileiro cresce trimestre", 0, 0.11, "cnnbrasil.com.br"),

    # Fake News
    ("https://site-fake1.com/lula-cancela-eleicoes",
     "URGENTE Lula vai CANCELAR eleicoes 2026 e virar ditador compartilhe", "urgente lula cancelar eleicoes ditador", 1, 0.95, "site-fake1.com"),
    ("https://site-fake2.com/vacina-mata",
     "BOMBA vacina causa morte em massa governo esconde a verdade", "bomba vacina causa morte governo esconde", 1, 0.92, "site-fake2.com"),
    ("https://site-fake3.com/fraude-eleicoes",
     "EXCLUSIVO fraude nas eleicoes comprovada urnas adulteradas", "exclusivo fraude eleicoes comprovada urnas adulteradas", 1, 0.89, "site-fake3.com"),
    ("https://site-fake4.com/bolsonaro-volta",
     "Bolsonaro vai VOLTAR ao poder em golpe militar confirmado", "bolsonaro volta poder golpe militar confirmado", 1, 0.91, "site-fake4.com"),

    # Estadao - Verdadeiras
    ("https://estadao.com.br/politica/congresso-vota.html",
     "Congresso vota reforma tributaria nesta semana", "congresso vota reforma tributaria semana", 0, 0.22, "estadao.com.br"),
    ("https://estadao.com.br/economia/dolar-cai.html",
     "Dolar recua frente ao real com dados de inflacao", "dolar recua real dados inflacao", 0, 0.19, "estadao.com.br"),

    # BBC - Verdadeiras
    ("https://bbc.com/portuguese/brasil-2026-eleicoes",
     "BBC analisa cenario politico brasileiro para eleicoes 2026", "bbc analisa cenario politico eleicoes", 0, 0.14, "bbc.com"),

    # Mais Fake News
    ("https://whatsapp-fake.com/chip-vacina",
     "VERDADE chip na vacina ativa 5G e controla populacao nao tome", "verdade chip vacina ativa controla populacao", 1, 0.97, "whatsapp-fake.com"),
    ("https://telegram-fake.com/stf-preso",
     "BOMBA ministros do STF serao presos amanha exercito na rua", "bomba ministros stf presos amanha exercito", 1, 0.94, "telegram-fake.com"),
]

for url, texto, texto_limpo, label, confianca, fonte in noticias:
    salvar_noticia(
        url=url, texto=texto, texto_limpo=texto_limpo,
        label=label, confianca=confianca, fonte=fonte,
        modelo_usado="Gradient Boosting",
        periodo_noticia="Noticia de 2026"
    )
    print(f"Salvo: {'FAKE' if label==1 else 'VERDADEIRA'} - {fonte}")

print("\nBanco populado com sucesso!")
print(f"Total: {len(noticias)} noticias")