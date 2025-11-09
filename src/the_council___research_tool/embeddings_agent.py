import os
from typing import List, Dict, Any
from pathlib import Path
import tempfile
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma
from openai import AzureOpenAI
import numpy as np

class AzureEmbeddingsAgent(Embeddings):
    """
    Azure OpenAI Embeddings Agent for document processing and vector storage.
    Uses Azure OpenAI text-embedding-ada-002 for creating embeddings.
    """

    def __init__(
        self,
        azure_endpoint: str = "https://nachalo-2324-resource.cognitiveservices.azure.com",
        api_key: str = "os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")",
        api_version: str = "2023-05-15",
        deployment_name: str = "text-embedding-ada-002",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the Azure Embeddings Agent.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name for embeddings
            persist_directory: Directory to persist ChromaDB
        """
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        # Initialize ChromaDB vector store
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using Azure OpenAI.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embeddings (list of floats)
        """
        embeddings = []

        # Process in batches to avoid token limits
        batch_size = 10  # Adjust based on text length

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.deployment_name
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error embedding batch {i//batch_size}: {e}")
                # Return zero embeddings for failed batches
                embeddings.extend([[0.0] * 1536] * len(batch_texts))  # ada-002 has 1536 dimensions

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using Azure OpenAI.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.deployment_name
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 1536  # ada-002 has 1536 dimensions

    def process_and_store_documents(self, documents: List[Document], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process documents and store them in the vector database.

        Args:
            documents: List of LangChain Document objects
            metadata: Additional metadata to add to all documents

        Returns:
            Processing results
        """
        # Add metadata to documents
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # Store in vector database
        try:
            self.vectorstore.add_documents(documents)

            return {
                'status': 'success',
                'documents_processed': len(documents),
                'total_documents_in_db': len(self.vectorstore.get()['ids']) if hasattr(self.vectorstore, 'get') else 0
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'documents_processed': 0
            }

    def search_similar(self, query: str, k: int = 5, filter: Dict = None) -> List[Document]:
        """
        Search for similar documents in the vector store.

        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of similar documents
        """
        try:
            return self.vectorstore.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def get_all_documents(self) -> List[str]:
        """
        Get list of all processed document filenames.

        Returns:
            List of filenames
        """
        try:
            collection = self.vectorstore._collection
            metadata = collection.get(include=['metadatas'])
            filenames = set()
            for meta in metadata['metadatas']:
                if meta and 'filename' in meta:
                    filenames.add(meta['filename'])
            return list(filenames)
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []

    def clear_database(self):
        """
        Clear all documents from the vector store.
        """
        import shutil
        try:
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            self.persist_directory.mkdir(exist_ok=True)

            # Reinitialize vector store
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self
            )
        except Exception as e:
            print(f"Error clearing database: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Collection statistics
        """
        try:
            collection = self.vectorstore._collection
            data = collection.get(include=['metadatas'])

            total_docs = len(data['ids']) if 'ids' in data else 0
            filenames = set()
            file_types = set()

            for meta in data.get('metadatas', []):
                if meta:
                    if 'filename' in meta:
                        filenames.add(meta['filename'])
                    if 'file_type' in meta:
                        file_types.add(meta['file_type'])

            return {
                'total_documents': total_docs,
                'unique_files': len(filenames),
                'file_types': list(file_types),
                'filenames': list(filenames)
            }

        except Exception as e:
            return {
                'error': str(e),
                'total_documents': 0,
                'unique_files': 0,
                'file_types': [],
                'filenames': []
            }


class DocumentEmbeddingsAgent:
    """
    High-level agent that manages document processing and embeddings.
    Supplements the document ingestion agent with Azure OpenAI embeddings.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings_agent = AzureEmbeddingsAgent(persist_directory=persist_directory)

    def process_document_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file: load, chunk, embed, and store.

        Args:
            file_path: Path to the document file

        Returns:
            Processing results
        """
        from langchain.document_loaders import (
            PyPDFLoader,
            Docx2txtLoader,
            UnstructuredExcelLoader,
            TextLoader
        )
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load document based on type
        file_extension = file_path_obj.suffix.lower()

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(str(file_path))
            elif file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
                loader = TextLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()

        except Exception as e:
            # Try alternative loading for PDFs
            if file_extension == '.pdf':
                try:
                    # Fallback PDF loading
                    from langchain.document_loaders import PyMuPDFLoader
                    loader = PyMuPDFLoader(str(file_path))
                    documents = loader.load()
                except:
                    raise Exception(f"Failed to load PDF {file_path}: {str(e)}")
            else:
                raise Exception(f"Error loading document {file_path}: {str(e)}")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)

        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source': str(file_path_obj),
                'filename': file_path_obj.name,
                'file_type': file_path_obj.suffix,
                'chunk_id': f"{file_path_obj.name}_{i}",
                'chunk_index': i,
                'total_chunks': len(chunks)
            })

        # Process and store with embeddings
        result = self.embeddings_agent.process_and_store_documents(
            chunks,
            metadata={
                'processed_at': str(Path.cwd()),
                'embedding_model': 'text-embedding-ada-002'
            }
        )

        # Add additional info
        result.update({
            'filename': file_path_obj.name,
            'file_type': file_path_obj.suffix,
            'chunks_created': len(chunks),
            'source_documents': len(documents)
        })

        return result

    def search_documents(self, query: str, k: int = 5, filter: Dict = None) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter

        Returns:
            List of similar documents
        """
        return self.embeddings_agent.search_similar(query, k=k, filter=filter)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings database.

        Returns:
            Database statistics
        """
        return self.embeddings_agent.get_collection_info()

    def clear_all_documents(self):
        """
        Clear all documents from the database.
        """
        self.embeddings_agent.clear_database()


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m the_council___research_tool.embeddings_agent <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    agent = DocumentEmbeddingsAgent()

    try:
        result = agent.process_document_file(file_path)
        print("✅ Document processed and embedded successfully!")
        print(f"📄 Filename: {result['filename']}")
        print(f"📁 Type: {result['file_type']}")
        print(f"📊 Chunks created: {result['chunks_created']}")
        print(f"📚 Source documents: {result['source_documents']}")
        print(f"💾 Status: {result['status']}")

        # Show database stats
        stats = agent.get_database_stats()
        print(f"\n📈 Database Stats:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Unique files: {stats['unique_files']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
