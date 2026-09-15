from transformers import pipeline
import torch

print("Carregando modelo de analise de sentimento em portugues...")

# Modelo de zero-shot classification - funciona sem treino especifico
classificador = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=0 if torch.cuda.is_available() else -1
)

def analisar_bert(texto):
    try:
        texto_curto = texto[:512]
        resultado = classificador(
            texto_curto,
            candidate_labels=["noticia verdadeira", "fake news", "desinformacao"],
            hypothesis_template="Este texto e uma {}."
        )
        
        label = resultado['labels'][0]
        score = resultado['scores'][0]
        
        if 'fake' in label or 'desinformacao' in label:
            return f"FAKE NEWS ({score:.0%} de confianca)"
        else:
            return f"VERDADEIRA ({score:.0%} de confianca)"
    except Exception as e:
        return f"Erro: {e}"

# Testes
textos = [
    "Camara aprova PEC que amplia imunidade tributaria de igrejas com 385 votos",
    "Lula vai transformar o Brasil em Venezuela amanha e acabar com a democracia",
    "STF aceita denuncia contra ministros do STJ"
]

for texto in textos:
    print(f"\nTexto: {texto[:60]}...")
    print(f"Resultado: {analisar_bert(texto)}")