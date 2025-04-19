# 🧠 Projeto RAG com FAISS + OpenRouter

Este projeto implementa um sistema de **Pergunta e Resposta baseado em RAG (Retrieval-Augmented Generation)** utilizando:

- **FAISS** para busca semântica eficiente
- **SentenceTransformers** para gerar embeddings
- **OpenRouter** para conectar com um modelo LLM (como o Mistral) sem treinar nada
- **Python** com lógica clara e extensível

---

## 📚 O que é RAG?

**Retrieval-Augmented Generation** é uma técnica onde:

1. **Você fornece uma base de conhecimento (ex: `base.txt`)**
2. O sistema busca **trechos mais relevantes** com base na pergunta do usuário
3. Um modelo de linguagem (LLM) **gera a resposta com base nesse contexto**

✅ Isso garante que a IA **só responda com base no conteúdo que você forneceu**  
❌ Sem necessidade de treinar modelos  
❌ Sem "alucinações" fora da sua base

## 📚 Similaridade dos vetores

Quando transformamos textos em vetores (números), textos com significados parecidos ficam "próximos" matematicamente. O sistema:
- Converte sua pergunta em vetor
- Calcula a "distância" entre esse vetor e os vetores dos parágrafos
- Seleciona os parágrafos com menor distância (mais relevantes)

## 📚 Limiar de similaridade:

Este é um filtro de qualidade:
- Se a distância for maior que 2.0, significa que nenhum parágrafo está suficientemente relacionado à pergunta
- O sistema só usa parágrafos com distância menor que 2.0
- Se nada for encontrado abaixo desse limiar, o sistema responde que não tem informações suficientes

Essencialmente, o limiar evita que o sistema use informações irrelevantes para responder sua pergunta.

## 📚 Embeddings e Sentence Transformer

**Embeddings** são representações numéricas de texto. No código, cada parágrafo é convertido em um vetor de números que captura seu significado semântico, permitindo comparações matemáticas de similaridade entre textos.

**SentenceTransformer "all-MiniLM-L6-v2"** é um modelo específico usado para criar esses embeddings:

- É uma versão compacta ("Mini") do modelo LM (Language Model)
- Possui 6 camadas (L6)
- Gera vetores de 384 dimensões
- É otimizado para equilibrar eficiência e qualidade
- Funciona bem para buscas semânticas em português e outros idiomas
- É mais rápido e leve que modelos maiores, mantendo boa precisão

Este modelo foi projetado para transformar frases e parágrafos em representações vetoriais que preservam relações semânticas, tornando possível encontrar conteúdo relacionado mesmo quando as palavras exatas são diferentes.

---

## ✅ O que este projeto faz

- Lê um arquivo `base.txt` com qualquer conteúdo textual
- Separa em blocos e gera vetores (embeddings)
- Indexa com FAISS para busca semântica rápida
- Recebe perguntas do usuário via terminal
- Busca os trechos mais relevantes da base
- Envia a pergunta + contexto para o modelo da OpenRouter
- Exibe a resposta gerada com base **apenas na sua base**

---

## 🛠️ Tecnologias utilizadas

| Componente                | Descrição                                            |
|---------------------------|------------------------------------------------------|
| FAISS                     | Busca de vetores eficiente                           |
| `sentence-transformers`   | Embedding textual com `MiniLM-L6-v2` (pré-treinado)  |
| OpenRouter (via API)      | Conexão com LLMs como `mistral-small-3.1`            |
| Python 3.10+              | Código principal                                     |

---

## 🧪 Exemplo de uso

Instale as dependencias utilizando o VENV para projetos em Python 

``` 
python3 -m venv venv
.\venv\Scripts\activate
(venv) PS> pip install faiss-cpu sentence-transformers requests

```

```bash
$ python rag_faiss.py

📚 Assistente RAG (Responde com base na base.txt)
Digite 'sair' para encerrar.

👤 Sua pergunta: o que é EC2?
🤖 Resposta:
O Amazon EC2 permite criar e gerenciar servidores virtuais na nuvem.
```

---

## 📂 Estrutura do projeto

```
rag-basic/
├── base.txt          # Sua base de conhecimento
├── rag_faiss.py      # Código principal do assistente
```

---

## 🔐 Sobre a API Key

Este projeto usa o modelo `mistralai/mistral-small-3.1-24b-instruct:free` via OpenRouter.  
Para funcionar, você precisa de uma chave válida:

1. Crie uma conta gratuita em [https://openrouter.ai](https://openrouter.ai)
2. Gere sua API Key e substitua no código:

```python
API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> Obs: nunca comitar ou compartilhar chaves publicamente

---

## 🧠 O que você NÃO está fazendo

- ❌ Você **não está treinando** um modelo
- ❌ Você **não precisa de GPU local**
- ✅ Você está **aproveitando modelos já prontos**, mas **controlando o conhecimento com RAG**

---

## 📈 Próximos passos (opcional)

- Transformar o assistente em **jogo de perguntas e respostas com pontuação**
- Rodar tudo local com **LLMs como Mistral, LLaMA ou Phi usando Ollama**
- Criar versão web com **Streamlit** ou **React + FastAPI**
- Evoluir para **Fine-tuning** com seus próprios dados

---

## 🧩 Licença

Este projeto é livre para uso educacional e pode ser adaptado para treinamentos, jogos, bots internos e muito mais.

