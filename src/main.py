import pandas as pd
import os
from data_processing import process_and_chunk_data
from indexing import DocumentIndexingPipeline
from generation import generate_answer
from evaluation import PipelineEvaluator

def main():

    base_csv_path = os.path.join(os.path.dirname(__file__), "..", "Natural-Questions-Base.csv")

    df_base = pd.read_csv(base_csv_path, nrows=100).dropna()
    chunks_df, df_base_processed = process_and_chunk_data(df_base)
    rag_pipeline = DocumentIndexingPipeline()
    rag_pipeline.add_documents(chunks_df)

    query = "Who was the first person in space?"
    print(f"\nEvaluating single query: '{query}'")
    search_results = rag_pipeline.search(query, k=1)
    
    for result in search_results:
        print(f"Distance Score: {result['score']:.2f}")
        print(f"Domain: {result['metadata']['question_domain']}")
        print(f"Text snippet:\n{result['text']}")

    print("Evaluation")
    evaluator = PipelineEvaluator()
    test_df = df_base_processed.sample(2, random_state=2026)


    for index, row in test_df.iterrows():
        query_text = row['question']
        expected_source = index
        reference_answer = row['long_answers_clean']

        evaluator.track_query(
            query_id=index,
            query_text=query_text,
            expected_source_id=expected_source,
            reference_answer=reference_answer,
            search_fn=lambda q: rag_pipeline.search(q, k=1),
            generate_fn=lambda q, ctx: generate_answer(q, ctx)
        )

    metrics_df = evaluator.generate_report()
    evaluator.plot_benchmarks(metrics_df)

if __name__ == "__main__":
    main()
