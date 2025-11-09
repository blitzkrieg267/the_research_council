import os
from pathlib import Path
from typing import List, Dict, Any
import tempfile
from embeddings_agent import DocumentEmbeddingsAgent
from langchain.schema import Document

class DocumentIngestionAgent:
    """
    Document Ingestion Agent - now uses Azure OpenAI embeddings.
    Supplements document processing with advanced embeddings.
    """
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory

        # Use the new Azure embeddings agent
        self.embeddings_agent = DocumentEmbeddingsAgent(persist_directory=persist_directory)

    def process_and_store(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and store it in the vector database using Azure embeddings.

        Args:
            file_path: Path to the document file

        Returns:
            Processing results
        """
        return self.embeddings_agent.process_document_file(file_path)

    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents in the vector store.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of similar documents
        """
        return self.embeddings_agent.search_documents(query, k=k)

    def get_all_documents(self) -> List[str]:
        """
        Get list of all processed document filenames.

        Returns:
            List of filenames
        """
        return self.embeddings_agent.get_database_stats().get('filenames', [])

    def clear_database(self):
        """
        Clear all documents from the vector store.
        """
        self.embeddings_agent.clear_all_documents()

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings database.

        Returns:
            Database statistics
        """
        return self.embeddings_agent.get_database_stats()

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m the_council___research_tool.document_ingestion_agent <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    agent = DocumentIngestionAgent()

    try:
        result = agent.process_and_store(file_path)
        print("Document processed successfully!")
        print(f"Filename: {result['filename']}")
        print(f"Type: {result['file_type']}")
        print(f"Chunks created: {result['total_chunks']}")
        print(f"Source documents: {result['total_documents']}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
