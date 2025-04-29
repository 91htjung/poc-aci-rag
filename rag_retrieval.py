import faiss
import numpy as np
import math
import re
import openai
import os

class DocumentIndex:
    """
    A document index for retrieval using FAISS, with methods for building the index,
    querying with temporal weighting, and saving/loading the index and metadata.
    """

    def __init__(self):
        self.index = None          # FAISS index
        self.metadata = []         # List of metadata dicts corresponding to indexed documents
        self.embeddings = None     # Numpy array of document embeddings
        self.dim = None            # Dimension of embeddings
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def build_index(self, docs, metadata_list=None):
        """
        Build a FAISS index from a list of document texts. Embeds the texts using OpenAI,
        stores metadata for each document, and creates a FAISS index for retrieval.
        """
        if metadata_list is None:
            # Initialize empty metadata if not provided
            metadata_list = [{} for _ in docs]
        # Embed documents in batches using OpenAI embeddings
        embeddings = []
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            #response = openai.embeddings.create(model="text-embedding-ada-002", input=batch_docs)
            response = self.client.embeddings.create(model="text-embedding-ada-002", input=batch_docs)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        # Convert to numpy array and normalize vectors for cosine similarity
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = np.divide(embedding_matrix, norms, out=np.zeros_like(embedding_matrix), where=(norms != 0))
        self.embeddings = embedding_matrix
        self.dim = embedding_matrix.shape[1]
        # Create FAISS index (Inner Product for cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embedding_matrix)
        # Store metadata (ensure each metadata contains at least text and year if available)
        self.metadata = metadata_list

    def query(self, query_text, top_k=5):
        """
        Query the index with a question. Returns top_k relevant document chunks with temporal weighting.
        Temporal weighting: if a year is mentioned in the query, results from that year are boosted;
        if no year mentioned, recent documents are given higher weight.
        """
        if self.index is None or self.dim is None:
            raise ValueError("Index not built. Build or load an index before querying.")
        # Embed the query
        response = self.client.embeddings.create(model="text-embedding-ada-002", input=[query_text])
        #response = openai.Embedding.create(model="text-embedding-ada-002", input=[query_text])
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm != 0:
            query_vec = query_vec / norm
        # Search the FAISS index for initial candidates
        total_docs = len(self.metadata)
        initial_k = min(total_docs, top_k * 5)  # retrieve more candidates for re-ranking
        D, I = self.index.search(np.array([query_vec], dtype=np.float32), initial_k)
        candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            candidates.append((float(score), int(idx)))
        # Determine temporal bias from query (find year mentioned, if any)
        year_matches = re.findall(r'\d{4}', query_text)
        years_in_q = [int(y) for y in year_matches if 1900 <= int(y) <= 2100]
        year_bias = None
        if len(years_in_q) == 1:
            year_bias = years_in_q[0]
        elif len(years_in_q) > 1:
            year_bias = None  # multiple years mentioned, skip explicit bias
        if year_bias is None:
            # If no specific year in query, use the latest year in the indexed documents for bias
            doc_years = [m.get("year") for m in self.metadata if m.get("year")]
            if doc_years:
                year_bias = max(doc_years)
        # Apply temporal weighting to candidate scores
        weighted_candidates = []
        for score, idx in candidates:
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            doc_year = meta.get("year")
            if year_bias and doc_year:
                # Exponential decay weight based on year difference
                diff = abs(doc_year - year_bias)
                weight = math.exp(-diff / 5.0)  # decay factor (adjust 5.0 for faster/slower decay as needed)
                new_score = score * weight
            else:
                new_score = score
            weighted_candidates.append((new_score, idx))
        # Sort by weighted score and pick top_k results
        weighted_candidates.sort(key=lambda x: x[0], reverse=True)
        top_results = weighted_candidates[:top_k]
        # Prepare output list of results with text and metadata
        results = []
        for new_score, idx in top_results:
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            result = {
                "score": new_score,
                "text": meta.get("text", ""),
                "metadata": meta
            }
            results.append(result)
        return results

    def save(self, dir_path):
        """
        Save the FAISS index, embeddings, and metadata to the specified directory.
        """
        import os, pickle
        os.makedirs(dir_path, exist_ok=True)
        # Save FAISS index to file
        faiss.write_index(self.index, os.path.join(dir_path, "faiss_index.bin"))
        # Save metadata list to pickle
        with open(os.path.join(dir_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        # Save embeddings matrix to .npy for reuse
        if self.embeddings is not None:
            np.save(os.path.join(dir_path, "embeddings.npy"), self.embeddings)

    def load(self, dir_path):
        """
        Load the FAISS index, embeddings, and metadata from the specified directory.
        """
        import os, pickle
        # Load FAISS index
        index_path = os.path.join(dir_path, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index file not found in directory.")
        self.index = faiss.read_index(index_path)
        # Load metadata
        meta_path = os.path.join(dir_path, "metadata.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata file not found in directory.")
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        # Load embeddings if available (for visualization or re-use)
        emb_path = os.path.join(dir_path, "embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)
            self.dim = self.embeddings.shape[1] if self.embeddings is not None else None
        else:
            # If embeddings not saved, we can reconstruct if needed (not implemented here)
            self.embeddings = None
            try:
                self.dim = self.index.d
            except AttributeError:
                self.dim = None