import numpy as np
import faiss
import torch
from transformers import pipeline
from functools import lru_cache
from sentence_transformers import SentenceTransformer

class DocumentIndexingPipeline:
    def __init__(self, embedding_model_name='BAAI/bge-m3', dimension=1024, max_history_turns=3):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self.doc_store = {}
        self._next_id = 0
        self.conversation_history = []
        self.max_history_turns = max_history_turns

        model_id = "unsloth/Llama-3.2-1B-Instruct"
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            print("Warning: LLM expansion pipeline failed to initialize. Expansions won't use LLM:", e)
            self.pipe = None

    @lru_cache(maxsize=50)
    def _cached_llm_expansion(self, processed_query):
        if not self.pipe:
            return ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert search query rewriter. Your task is to transform the user's input "
                    "into a single, optimized search query expansion; Do not repeat the user input.\n\n"
                    "Rules:\n"
                    "1. Use relevant synonyms and semantic variations.\n"
                    "2. Respond ONLY with the rewritten query."
                )
            },
            {
                "role": "user",
                "content": f"Input: {processed_query}"
            },
        ]
        outputs = self.pipe(messages)
        output = outputs[0]["generated_text"][-1]["content"].strip()
        if '"' in output:
            output = output.split('"')[1]
        return output

    def add_to_history(self, role, text):
        self.conversation_history.append({"role": role, "text": text})
        max_messages = self.max_history_turns * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def clear_history(self):
        self.conversation_history = []

    def add_documents(self, chunks_df):
        texts = chunks_df['chunk_text'].tolist()
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True).astype('float32')

        ids = np.arange(self._next_id, self._next_id + len(texts), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        for i, idx in enumerate(ids):
            row = chunks_df.iloc[i]
            self.doc_store[int(idx)] = {
                "chunk_id": row['chunk_id'],
                "text": row['chunk_text'],
                "metadata": row.get('metadata', {})
            }

        self._next_id += len(texts)
        print(f"Successfully indexed {len(texts)} documents. Total in index: {self.index.ntotal}")

    def _query_preprocessing(self, query):
        query = query.lower()
        if not self.conversation_history:
            return query
        history_str = " | ".join(
            [f"{msg['role'].lower()}: {msg['text'].lower()}" for msg in self.conversation_history]
        )
        return f"context: {history_str} | current query: {query}"

    def _query_expansion(self, query):
        processed_query = self._query_preprocessing(query)
        expanded_term = self._cached_llm_expansion(processed_query)
        if expanded_term:
            final_query = f"{query}\n{expanded_term}"
        else:
            final_query = query
        return final_query

    def search(self, query, k=1):
        expanded_query = self._query_expansion(query)
        print(f"Searching for: {expanded_query}")

        query_vector = self.embedding_model.encode([expanded_query], normalize_embeddings=True).astype('float32')
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                doc = self.doc_store[int(idx)]
                results.append({
                    "score": float(distances[0][i]),
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                })
        return results
