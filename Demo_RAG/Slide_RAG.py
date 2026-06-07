"""
RAG (Retrieval-Augmented Generation) Demo
This script demonstrates how to use Ollama (local LLM) with RAG to answer questions
based on a knowledge base. It compares answers with and without RAG.

Key Concepts:
- Embeddings: Convert text to numbers for similarity comparison
- RAG: Retrieve relevant documents before generating answers
- Cosine Similarity: Measure how similar two text pieces are
"""

import ollama
import numpy as np

# ==========================================
# CONFIG - Model Selection
# ==========================================
# This is the LLM model we'll use for generating answers
# tinyllama:1.1b is a lightweight, fast model good for local testing
LLM_MODEL = "tinyllama:1.1b"


# ==========================================
# KNOWLEDGE BASE - Sample Documents
# ==========================================
# Create a list of documents (text) about various sports figures
# Each document is a tuple: (label, content)
# The label helps us identify which document was retrieved
# The content is the actual text that will be used to answer questions

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


# ==========================================
# EMBEDDINGS - Convert Text to Vectors
# ==========================================
# Embeddings are numerical representations of text
# Similar texts have similar embeddings
# We use the "bge-m3" model to create embeddings

def get_embedding(text):
    """
    Convert a text string into an embedding (vector of numbers).
    
    Args:
        text (str): The text to convert
        
    Returns:
        list: A list of numbers representing the text
        
    How it works:
    1. Send the text to Ollama with the "bge-m3" embedding model
    2. The model analyzes the text and returns a numerical representation
    3. Return just the embedding vector from the response
    """
    response = ollama.embeddings(
        model="bge-m3",
        prompt=text
    )
    return response["embedding"]


def cosine_similarity(a, b):
    """
    Calculate how similar two embeddings are (0 = not similar, 1 = identical).
    
    Args:
        a (list): First embedding vector
        b (list): Second embedding vector
        
    Returns:
        float: Similarity score between 0 and 1
        
    How it works:
    1. Convert both embeddings to numpy arrays for math operations
    2. Use the cosine similarity formula: dot_product / (norm_a * norm_b)
    3. Return the similarity score
    
    Example:
    - cosine_similarity([1,0], [1,0]) = 1.0 (identical)
    - cosine_similarity([1,0], [0,1]) = 0.0 (perpendicular)
    """
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==========================================
# INDEX DOCUMENTS - Create Embeddings for Knowledge Base
# ==========================================
# This step happens once at startup
# We convert all our knowledge base documents into embeddings
# This allows us to quickly find relevant documents later

print("Generating embeddings for documents...")

# Create embeddings for each document in our knowledge base
# We use a list comprehension to generate embeddings for all documents
# Note: We only use the document text (second element), not the label (first element)
document_embeddings = [
    get_embedding(doc_text) for _, doc_text in documents
]

print("Knowledge base ready.\n")


# ==========================================
# RETRIEVAL - Find Relevant Documents
# ==========================================
# This function searches the knowledge base for documents similar to the query

def retrieve_context(query, top_k=2, threshold=0.5):
    """
    Find the most relevant documents for a given query.
    
    Args:
        query (str): The user's question or search text
        top_k (int): Maximum number of documents to return (default: 2)
        threshold (float): Minimum similarity score (0-1) to include a document (default: 0.5)
        
    Returns:
        str: Concatenated text of relevant documents, or None if no documents pass threshold
        
    How it works:
    1. Convert the query into an embedding
    2. Compare the query embedding to each document embedding
    3. Calculate similarity scores for all documents
    4. Filter documents by threshold
    5. Select the top_k most similar documents
    6. Return the text of selected documents
    """
    
    # Step 1: Convert the user's query into an embedding
    query_embedding = get_embedding(query)

    # Step 2: Calculate similarity between query and each document
    scores = []

    for i, doc_embedding in enumerate(document_embeddings):
        # Calculate how similar this document is to the query
        score = cosine_similarity(query_embedding, doc_embedding)
        
        # Get the document label and text
        doc_name, doc_text = documents[i]
        
        # Store the score, name, and text as a tuple
        scores.append((score, doc_name, doc_text))

    # Step 3: Sort documents by similarity (highest first)
    scores.sort(reverse=True, key=lambda x: x[0])

    # Step 4: Print all similarity scores for transparency
    print("\n[Similarity Scores]")
    for score, name, _ in scores:
        print(f"{name}: {score:.4f}")

    # Step 5: Filter documents - keep only those above the threshold
    # Example: if threshold=0.5, only documents with score >= 0.5 are kept
    filtered = [(s, n, d) for s, n, d in scores if s >= threshold]

    # Step 6: Check if any documents passed the threshold
    if not filtered:
        print("\n No document passed threshold")
        return None

    # Step 7: Select the top_k best documents
    # If we have 3 documents with scores [0.8, 0.7, 0.6] and top_k=2,
    # we select only the first 2 documents
    selected = filtered[:top_k]

    # Step 8: Print which documents were selected
    print(" Selected documents:")
    for score, name, _ in selected:
        print(f"- {name} ({score:.4f})")

    # Step 9: Combine the selected documents into one text block
    # This will be used as context for the LLM
    return "\n".join([doc for _, _, doc in selected])


# ==========================================
# WITHOUT RAG - Direct LLM Response
# ==========================================
# This function asks the LLM without providing any context
# The LLM uses only its training data to answer

def ask_without_rag(question):
    """
    Ask the LLM a question without any additional context.
    
    Args:
        question (str): The user's question
        
    Returns:
        str: The LLM's answer based only on its training data
        
    How it works:
    1. Send the question directly to the LLM
    2. No context or knowledge base is provided
    3. The LLM generates an answer from its training knowledge only
    
    Limitations:
    - May provide incorrect information
    - May not know about recent events or specialized knowledge
    - No source attribution
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": question}]
    )
    return response["message"]["content"]


# ==========================================
# WITH RAG - Context-Augmented LLM Response
# ==========================================
# This function asks the LLM after retrieving relevant context
# The LLM can only use the context to answer (grounded answer)

def ask_rag(question):
    """
    Ask the LLM a question using RAG (Retrieval-Augmented Generation).
    
    Args:
        question (str): The user's question
        
    Returns:
        str: The LLM's answer based on relevant context from the knowledge base
        
    How it works:
    1. Retrieve relevant documents from the knowledge base
    2. If no relevant documents found, return "I don't know"
    3. Combine the context with explicit instructions for the LLM
    4. Send the prompt (with instructions + context + question) to the LLM
    5. The LLM answers based only on the provided context
    
    Advantages over non-RAG:
    - Answers are grounded in the knowledge base
    - More accurate and reliable information
    - Can cite sources
    - Works well for specialized domains
    """
    
    # Step 1: Retrieve relevant documents
    context = retrieve_context(question)

    # Step 2: Check if any relevant documents were found
    if context is None:
        return "I don't know"

    # Step 3: Create a prompt that instructs the LLM to use only the provided context
    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    # Step 4: Send the prompt to the LLM
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    # Step 5: Return the LLM's response
    return response["message"]["content"]


# ==========================================
# MAIN LOOP - Interactive Question-Answering
# ==========================================
# This creates an interactive loop where the user can ask questions
# We show both RAG and non-RAG answers for comparison

while True:
    # Get a question from the user
    question = input("Ask a question (or type exit): ")

    # Check if user wants to exit
    if question.lower() == "exit":
        break

    # Ask without RAG and show the response
    print("\n--- WITHOUT RAG ---")
    print(ask_without_rag(question))

    # Ask with RAG and show the response
    print("\n--- WITH RAG ---")
    print(ask_rag(question))

    # Print a separator line for clarity
    print("-" * 60)
