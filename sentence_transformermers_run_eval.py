import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm.auto import tqdm
import argparse
from huggingface_hub import login


class YourCustomDEModel:
    def __init__(self, model_name="intfloat/e5-base-v2", **kwargs):
        self.model = SentenceTransformer(model_name,trust_remote_code=True, model_kwargs ={"torch_dtype":torch.float16})

    def encode_text(self, texts: List[str], batch_size: int = 12, show_progress_bar: bool = True, **kwargs) -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar, **kwargs)
        
        if embeddings is None:
            logging.error("Embeddings are None.")
        else:
            logging.info(f"Encoded {len(embeddings)} embeddings.")
        
        return np.array(embeddings.cpu())

    def encode_queries(self, queries: List[str], batch_size: int = 12, show_progress_bar: bool = True, **kwargs) -> np.ndarray:
        all_queries = ["query: "+ query for query in queries]
        return self.encode_text(all_queries, batch_size, show_progress_bar, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 12, show_progress_bar: bool = True, **kwargs) -> np.ndarray:
        all_texts = ["passage: "+ doc['text'] for doc in corpus]
        return self.encode_text(all_texts, batch_size, show_progress_bar, **kwargs)

# Load the model
def main():
    parser = argparse.ArgumentParser(description="Run evaluation with a specified model and tasks.")
    parser.add_argument('--model_name', type=str, default="intfloat/e5-base-v2", help='Name of the model to use.')
    parser.add_argument('--tasks', type=str, nargs='+', default=["codetrans-dl"], help='List of tasks to evaluate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--hf_token', type=str, help='Hugging Face token for login.')

    args = parser.parse_args()
    if args.hf_token:
        login(token=args.hf_token)

    # Load the model
    model = YourCustomDEModel(model_name=args.model_name)

    # Get tasks
    tasks = coir.get_tasks(tasks=args.tasks)

    # Initialize evaluation
    evaluation = COIR(tasks=tasks, batch_size=args.batch_size)

    # Run evaluation
    results = evaluation.run(model, output_folder=f"results/{args.model_name}")
    print(results)

if __name__ == "__main__":
    main()