import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re

# === CONFIGURAÇÕES ===
API_KEY = "SUA_CHAVE_OPENR_OUTER"
OPENROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
SIMILARIDADE_LIMIAR = 2.0  # quanto menor, mais parecido. Se for maior que isso, ignoramos

# === 1. Carrega a base de conhecimento ===
with open("base.txt", "r", encoding="utf-8") as f:
    base = f.read()

# Divide a base em blocos (um parágrafo por linha, por exemplo)
blocos = [b.strip() for b in base.split("\n") if b.strip()]

# === 2. Gera vetores (embeddings) para cada bloco ===
modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")
vetores = modelo_embedding.encode(blocos)

# === 3. Cria o índice vetorial com FAISS ===
dim = vetores.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 = distância euclidiana
index.add(np.array(vetores))

# === 4. Função para buscar o contexto relevante com base na pergunta ===
def buscar_contexto(pergunta, k=2, limiar_similaridade=SIMILARIDADE_LIMIAR):
    vetor_pergunta = modelo_embedding.encode([pergunta])
    distancias, indices = index.search(np.array(vetor_pergunta), k)

    # Checa a distância (quanto menor, mais próximo). Se for maior que o limiar, ignora
    if min(distancias[0]) > limiar_similaridade:
        return None  # nada relevante encontrado

    # Junta os blocos mais relevantes em um único contexto
    return "\n".join([blocos[i] for i in indices[0]])

# === 5. Função para gerar resposta com OpenRouter ===
def gerar_resposta(pergunta, contexto):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://openrouter.ai",  # obrigatório para OpenRouter
        "Content-Type": "application/json"
    }

    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um assistente que responde perguntas **apenas com base no contexto fornecido abaixo**.\n"
                "Se a pergunta estiver fora do contexto, diga: 'Não encontrei informações suficientes para responder com base na base de conhecimento.'\n\n"
                f"Contexto:\n{contexto}"
            )
        },
        {"role": "user", "content": pergunta}
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": mensagens
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Erro {response.status_code}:\n{response.text}"

# === 6. Função para validar a entrada do usuário ===
def pergunta_valida(pergunta):
    # Remove espaços extras e checa se tem pelo menos 3 letras
    if len(pergunta.strip()) < 3:
        return False
    # Se for só letras repetidas ou caracteres sem sentido
    if re.match(r"^[a-zA-Z]{1,2}$", pergunta):
        return False
    if re.search(r"[^\w\s\?]", pergunta):  # muitos símbolos estranhos
        return False
    return True

# === 7. Loop principal do jogo ===
print("📚 Assistente RAG (Responde com base na base.txt)\nDigite 'sair' para encerrar.\n")

while True:
    pergunta = input("👤 Sua pergunta: ").strip()
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("✅ Encerrando. Até mais!")
        break

    if not pergunta_valida(pergunta):
        print("⚠️ Pergunta inválida. Tente escrever uma pergunta mais clara.\n")
        continue

    contexto = buscar_contexto(pergunta)
    if contexto is None:
        print("🤖 Resposta:\nDesculpe, não encontrei informações relevantes na base para essa pergunta.\n")
        continue

    resposta = gerar_resposta(pergunta, contexto)
    print(f"\n🤖 Resposta:\n{resposta}\n")
