import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.memory.db_client import MemoryDB
from app.intelligence.ollama_client import OllamaClient

def main():
    print("Initializing Sentinel Memory Hub...")
    db = MemoryDB(persist_directory="chroma_db")
    llm = OllamaClient()

    print("\n==============================================")
    print(" AI Sentinel Lite - Security Chat Active")
    print("==============================================")
    print("Ask me anything about what I've seen.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You >> ")
            if query.lower() in ['exit', 'quit']:
                break
                
            if not query.strip():
                continue

            print("Sentinel >> Searching memories...")
            relevant_docs = db.query_events(query, n_results=5)
            
            if not relevant_docs:
                print("Sentinel >> I have no memory records yet. Let the detector run for a bit in the other terminal.\n")
                continue

            # Format the context
            context = "\n".join(f"- {doc}" for doc in relevant_docs)
            
            prompt = f"""You are an AI security assistant named Sentinel.
The user is asking you a question about the video feed you are monitoring.

User Question: "{query}"

Here are the most relevant temporal events from your memory log:
{context}

Answer the user's question clearly and concisely based ONLY on the provided memory log. Do not make up events.
If the memory log doesn't contain enough information to answer the question, specifically say "I have not seen that in my recent memory log."
"""
            print("Sentinel >> Thinking...")
            response = llm.ask(prompt)
            print(f"\nSentinel >> {response}\n")
            
        except KeyboardInterrupt:
            break

    print("Closing memory hub.")

if __name__ == "__main__":
    main()
