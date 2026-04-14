import chromadb
from datetime import datetime
import os

class MemoryDB:
    def __init__(self, persist_directory="chroma_db"):
        print(f"Initializing MemoryDB at {persist_directory}...")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="sentinel_memory")

    def save_event(self, summary: str):
        """Save a new AI summary to the vector database with a timestamp."""
        # Don't save empty states or errors
        if not summary or summary.startswith("LLM error") or "Waiting" in summary or summary.startswith("Thinking"):
            return

        timestamp = datetime.now()
        doc_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        metadata = {"timestamp": timestamp.isoformat()}
        
        # Embed the time into the text so the LLM understands when it happened
        time_str = timestamp.strftime("%H:%M:%S")
        document = f"At {time_str}, {summary}"
        
        try:
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"Database error writing event: {e}")

    def query_events(self, query: str, n_results=7):
        """Retrieve the most relevant past events based on a string query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            # results['documents'] is a list of lists of strings
            if not results['documents'] or not results['documents'][0]:
                return []
            return results['documents'][0]
        except Exception as e:
            print(f"Database error querying event: {e}")
            return []
