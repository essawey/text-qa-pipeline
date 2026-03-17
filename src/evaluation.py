import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class PipelineEvaluator:
    def __init__(self):
        self.metrics_log = []
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.bleu_smoother = SmoothingFunction().method1

    def evaluate_retrieval(self, expected_source_id, retrieved_chunks, k=1):
        retrieved_ids = [chunk['metadata']['source_row_index'] for chunk in retrieved_chunks[:k]]

        is_relevant_retrieved = int(expected_source_id in retrieved_ids)

        precision_at_k = is_relevant_retrieved / k
        recall_at_k = is_relevant_retrieved / 1.0

        return precision_at_k, recall_at_k

    def evaluate_generation(self, reference_answer, generated_answer):
        if not generated_answer or not reference_answer:
            return 0.0, 0.0, 0.0

        ref_tokens = str(reference_answer).split()
        gen_tokens = str(generated_answer).split()

        bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.bleu_smoother)

        rouge_scores = self.rouge_scorer.score(str(reference_answer), str(generated_answer))
        rouge1 = rouge_scores['rouge1'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure

        return bleu_score, rouge1, rougeL

    def track_query(self, query_id, query_text, expected_source_id, reference_answer, search_fn, generate_fn):
        t0 = time.time()
        search_results = search_fn(query_text)
        retrieval_time = time.time() - t0

        precision, recall = self.evaluate_retrieval(expected_source_id, search_results, k=1)

        t1 = time.time()
        gen_result = generate_fn(query_text, search_results)
        generation_time = time.time() - t1

        generated_text = gen_result.get("answer", "")
        confidence = gen_result.get("confidence", 0.0)

        bleu, rouge1, rougeL = self.evaluate_generation(reference_answer, generated_text)

        total_time = retrieval_time + generation_time

        self.metrics_log.append({
            "query_id": query_id,
            "retrieval_time_sec": retrieval_time,
            "generation_time_sec": generation_time,
            "total_time_sec": total_time,
            "precision@1": precision,
            "recall@1": recall,
            "bleu_score": bleu,
            "rouge1_f1": rouge1,
            "rougeL_f1": rougeL,
            "confidence": confidence
        })

    def generate_report(self):
        if not self.metrics_log:
            print("No metrics to report.")
            return pd.DataFrame()

        df = pd.DataFrame(self.metrics_log)

        total_queries = len(df)
        total_time = df['total_time_sec'].sum()
        throughput = total_queries / total_time if total_time > 0 else 0

        print("--- RAG Pipeline Performance Report ---")
        print(f"Total Queries Processed: {total_queries}")
        print(f"System Throughput: {throughput:.2f} queries/sec")
        print("\n--- Average Latency ---")
        print(f"Retrieval latency: {df['retrieval_time_sec'].mean():.4f}s")
        print(f"Generation latency: {df['generation_time_sec'].mean():.4f}s")
        print(f"Total latency per query: {df['total_time_sec'].mean():.4f}s")
        print("\n--- Retrieval Accuracy ---")
        print(f"Precision@1: {df['precision@1'].mean():.4f}")
        print(f"Recall@1: {df['recall@1'].mean():.4f}")
        print("\n--- Generation Quality ---")
        print(f"BLEU Score: {df['bleu_score'].mean():.4f}")
        print(f"ROUGE-1 F1: {df['rouge1_f1'].mean():.4f}")
        print(f"ROUGE-L F1: {df['rougeL_f1'].mean():.4f}")
        print(f"Average Confidence: {df['confidence'].mean():.4f}")
        print("---------------------------------------")

        return df

    def plot_benchmarks(self, df):
        if df.empty: return
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(df['rougeL_f1'], bins=10, alpha=0.7, color='green', edgecolor='black', label='ROUGE-L')
        ax.hist(df['bleu_score'], bins=10, alpha=0.7, color='blue', edgecolor='black', label='BLEU')
        ax.set_title('Distribution of Generation Scores')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Score')
        ax.legend()
        plt.tight_layout()
        plt.show()
