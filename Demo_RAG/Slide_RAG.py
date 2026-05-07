import ollama
import numpy as np

# -----------------------------------
# CONFIG
# -----------------------------------

LLM_MODEL = "tinyllama:1.1b"

# -----------------------------------
# KNOWLEDGE BASE (WITH LABELS)
# -----------------------------------

documents = [

    ("Montoya", """
    Juan Pablo Montoya is a Colombian racing driver born in Bogota, Colombia, in 1975.
    He is considered one of the most versatile drivers of his generation, competing in Formula 1, NASCAR, IndyCar and endurance racing.

    Montoya Career timeline:
    - Competed in Formula 3000 in the late 1990s.
    - Won the CART Championship in 1999 as a rookie.
    - Won the Indianapolis 500 in 2000 with Chip Ganassi Racing.
    - Competed in Formula 1 from 2001 to 2006.
    - Drove for Williams from 2001 to 2004.
    - Drove for McLaren from 2005 to 2006.
    - Won 7 Formula 1 Grand Prix races.
    - Moved to NASCAR after leaving Formula 1.
    - Competed in NASCAR from 2007 to 2013 with Chip Ganassi Racing.
    - Returned to IndyCar and won the Indianapolis 500 again in 2015.
    - Also competed in endurance racing series such as IMSA.

    Montoya Major achievements:
    - CART Champion (1999)
    - Indianapolis 500 winner (2000, 2015)
    - 7 Formula 1 wins

    Montoya is one of the few drivers in history to succeed across multiple racing disciplines.
    """),

    ("Checo Perez", """
    Sergio "Checo" Perez is a Mexican Formula 1 driver born in Guadalajara, Mexico, in 1990.
    He is known for his tire management skills and consistency in races.

    Checo Career timeline:
    - Debuted in Formula 1 in 2011 with Sauber.
    - Drove for Sauber from 2011 to 2012, achieving several podium finishes.
    - Joined McLaren in 2013.
    - Joined Force India in 2014.
    - Continued with the same organization through Racing Point until 2020.
    - Won the Sakhir Grand Prix in 2020 (his first Formula 1 victory).
    - Joined Red Bull Racing in 2021.
    - Won multiple races with Red Bull between 2021 and 2024.
    - Finished runner-up in the Formula 1 World Championship in 2023.

    Checo Major achievements:
    - Multiple Formula 1 wins
    - Formula 1 World Championship runner-up (2023)

    He is considered the most successful Mexican Formula 1 driver of his era.
    """),

    ("Chicharito", """
    Javier "Chicharito" Hernandez is a Mexican football striker born in Guadalajara, Mexico, in 1988.
    He is known for his positioning, finishing ability and goal-scoring instinct.

    Chicharito Career timeline:
    - Began professional career with Chivas in Mexico.
    - Played for Chivas from 2006 to 2010.
    - Joined Manchester United in 2010.
    - Played for Manchester United from 2010 to 2015.
    - Reached the UEFA Champions League final in 2011.
    - Loan spell at Real Madrid during the 2014 to 2015 season.
    - Joined Bayer Leverkusen in 2015.
    - Played for Bayer Leverkusen from 2015 to 2017, with strong scoring seasons.
    - Joined West Ham United in 2017.
    - Played for West Ham from 2017 to 2019.
    - Played for Sevilla from 2019 to 2020.
    - Joined LA Galaxy in 2020.
    - Later returned to Chivas.

    Chicharito International career:
    - Represented Mexico in multiple FIFA World Cups.
    - One of the all-time top scorers for Mexico.

    Chicharito Major achievements:
    - Premier League titles with Manchester United
    - International success with Mexico
    """),

    ("Valderrama", """
    Carlos "El Pibe" Valderrama is a Colombian football legend born in Santa Marta, Colombia, in 1961.
    He is known for his vision, passing ability and leadership.

    Valderrama Career timeline:
    - Began professional career in Colombia.
    - Played for Union Magdalena early in his career.
    - Played for Millonarios and Deportivo Cali.
    - Played in France for Montpellier from 1988 to 1991.
    - Returned to Colombia after his European spell.
    - Later played in Major League Soccer in the United States.
    - Played for Tampa Bay Mutiny, Miami Fusion and Colorado Rapids during the 1990s.

    Valderrama International career:
    - Represented Colombia from 1985 to 1998.
    - Played in the FIFA World Cups of 1990, 1994 and 1998.
    - Captain and playmaker of Colombia's golden generation.

    Valderrama Major achievements:
    - One of the greatest Colombian footballers ever
    - Icon of South American football
    """)
]

# -----------------------------------
# EMBEDDINGS (LOCAL)
# -----------------------------------

def get_embedding(text):
    response = ollama.embeddings(
        model="bge-m3",
        prompt=text
    )
    return response["embedding"]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------------
# INDEX DOCUMENTS
# -----------------------------------

print("Generating embeddings for documents...")

document_embeddings = [
    get_embedding(doc_text) for _, doc_text in documents
]

print("Knowledge base ready.\n")


# -----------------------------------
# RETRIEVAL
# -----------------------------------

def retrieve_context(query, top_k=2, threshold=0.5):

    query_embedding = get_embedding(query)

    scores = []

    for i, doc_embedding in enumerate(document_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        doc_name, doc_text = documents[i]
        scores.append((score, doc_name, doc_text))

    # Sort by similarity
    scores.sort(reverse=True, key=lambda x: x[0])

    # PRINT SCORES
    print("\n[Similarity Scores]")
    for score, name, _ in scores:
        print(f"{name}: {score:.4f}")

    # Filter by threshold
    filtered = [(s, n, d) for s, n, d in scores if s >= threshold]

    if not filtered:
        print("\n No document passed threshold")
        return None

    selected = filtered[:top_k]

    # PRINT SELECTED
    print(" Selected documents:")
    for score, name, _ in selected:
        print(f"- {name} ({score:.4f})")

    return "\n".join([doc for _, _, doc in selected])


# -----------------------------------
# WITHOUT RAG
# -----------------------------------

def ask_without_rag(question):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": question}]
    )
    return response["message"]["content"]


# -----------------------------------
# WITH RAG
# -----------------------------------

def ask_rag(question):

    context = retrieve_context(question)

    if context is None:
        return "I don't know"

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# -----------------------------------
# MAIN LOOP
# -----------------------------------

while True:
    question = input("Ask a question (or type exit): ")

    if question.lower() == "exit":
        break

    print("\n--- WITHOUT RAG ---")
    print(ask_without_rag(question))

    print("\n--- WITH RAG ---")
    print(ask_rag(question))

    print("-" * 60)
