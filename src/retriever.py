import duckdb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



class DuckDBRetriever:
    def __init__(self, db_path, k=4):
        """
        Initialize the DuckDBRetriever.

        Args:
            db_path: Path to the DuckDB database file.
            k: Number of relevant documents to retrieve (top-k).
        """
        self.db_path = db_path
        self.k = k

    def _get_embeddings(self):
        """
        Load the embeddings and document data from DuckDB.

        Returns:
            document_data: A list of dictionaries containing 'id', 'content', 'metadata', and 'embedding'.
        """
        con = duckdb.connect(self.db_path)
        result = con.execute("SELECT id, content, metadata, embedding FROM documents").fetchall()
        document_data = []
        for row in result:
            doc_id, content, metadata, embedding_blob = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            document_data.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            })
        con.close()
        return document_data

    def _calculate_similarity(self, query_embedding, document_embeddings):
        """
        Calculate cosine similarity between the query embedding and all document embeddings.

        Args:
            query_embedding: A NumPy array representing the query's embedding.
            document_embeddings: A list of NumPy arrays representing document embeddings.

        Returns:
            similarity_scores: A list of similarity scores between the query and each document.
        """
        query_embedding = np.array([query_embedding])
        document_embeddings = np.array(document_embeddings)
        similarity_scores = cosine_similarity(query_embedding, document_embeddings).flatten()
        return similarity_scores

    def retrieve(self, query_embedding):
        """
        Retrieve the top-k most similar documents based on the query embedding.

        Args:
            query_embedding: The embedding for the query.

        Returns:
            top_k_documents: A list of the top-k most similar documents.
        """
        # Step 1: Load document data and embeddings
        documents = self._get_embeddings()

        # Step 2: Extract document embeddings
        document_embeddings = [doc["embedding"] for doc in documents]

        # Step 3: Calculate similarity between query embedding and document embeddings
        similarity_scores = self._calculate_similarity(query_embedding, document_embeddings)

        # Step 4: Get the top-k most similar documents
        top_k_indices = np.argsort(similarity_scores)[-self.k:][::-1]  # Get top-k indices (sorted by similarity)

        top_k_documents = [documents[i] for i in top_k_indices]

        return top_k_documents
