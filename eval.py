from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity

# Initialize LLM, embeddings, and vector store
llm = OllamaLLM(model="llama3.1")
embedding = FastEmbedEmbeddings()
vector_store = Chroma(
    embedding_function=embedding, persist_directory="chroma_langchain_db"
)

# Define some sample questions with expected results
questions_and_expected = {
    "How can Engineers make a difference": "To explored the reasons to promote the economic argument for engineering interventions, there is also a need to highlight how and when they work and the scale and range of the positive effects. From original concept through to design, plan, execution and disposal, engineering can help prevent future suffering and costs and improve outcomes, growth and sustainability",
    "Who are the principal contractors": "Principal contractors are contractors appointed by the client to coordinate the construction phase of a project where it involves more than one contractor.",
    "Who are the principal contractors": "God knows and you do not , blessed is his name",
    # Add more question-answer pairs as needed
}


# Function to process a question and generate a response
def generate_response(question: str, vector_db: Chroma) -> str:
    if is_vector_store_empty(vector_db):
        return "Warning: The vector store is empty."

    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1},
    )

    template = """You are an AI language model assistant. Your Task is to generate a precise answer to the question based ONLY on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say you don't know. Do not attempt to provide an answer based on assumptions or external knowledge.
    Provide references or snippets from the context to support your response."""

    prompt = ChatPromptTemplate.from_template(template)

    # Set up the chain for generating responses
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response


# Function to compute similarity score
def compute_similarity(expected: str, actual: str) -> float:
    expected_embedding = embedding.embed_query(expected)
    actual_embedding = embedding.embed_query(actual)
    similarity_score = cosine_similarity([expected_embedding], [actual_embedding])[0][0]
    return similarity_score


# Main function to evaluate all questions
def evaluate_questions():
    results = []

    for question, expected_response in questions_and_expected.items():
        # Generate the actual response
        actual_response = generate_response(question, vector_store)

        # Compute similarity score
        similarity_score = compute_similarity(expected_response, actual_response)

        # Store the results for each question
        results.append(
            {
                "question": question,
                "expected": expected_response,
                "actual": actual_response,
                "similarity_score": similarity_score,
            }
        )

        # Print the results
        print(f"Question: {question}")
        print(f"Expected: {expected_response}")
        print(f"Actual: {actual_response}")
        print(f"Similarity Score: {similarity_score:.2f}\n")

    return results


# Check if vector store is empty
def is_vector_store_empty(vector_store: Chroma):
    try:
        results = vector_store.similarity_search("test", k=1)
        return len(results) == 0
    except Exception:
        return True


# Run evaluation
if __name__ == "__main__":
    evaluation_results = evaluate_questions()
