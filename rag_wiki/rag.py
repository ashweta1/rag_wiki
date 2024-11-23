import pandas as pd
import torch
import faiss
from rag_wiki.util import best_device
from rag_wiki.util import batched_iterable
from datasets import load_dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer

def load_dataset_num(path, num_examples=None, debug=False):
    print(f"Loading dataset from {path}")
    dataset = load_dataset(path, split="full", streaming=True, trust_remote_code=True)
    if num_examples is not None:
        dataset = dataset.take(num_examples)
    debug and print(dataset)
    return dataset

def load_wikiqa(file_path):
    """
    Load, clean, and filter the WikiQA dataset to extract only answers with true labels.
    Args:
        file_path (str): Path to the WikiQA dataset file (typically TSV format).
    Returns:
        pd.DataFrame: A DataFrame containing questions and their corresponding correct answers.
    """
    # Load the dataset
    df = pd.read_csv(file_path, sep='\t')

    # Inspect columns (use appropriate column names based on your dataset version)
    print("Columns in dataset:", df.columns)

    # Ensure the dataset contains required columns
    required_columns = ['Question', 'Sentence', 'Label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter only answers with true labels (Label == 1)
    true_answers = df[df['Label'] == 1]
    # Drop duplicates to keep unique (Question, Answer) pairs
    true_answers = true_answers[['Question', 'Sentence']].drop_duplicates()
    # Reset index for a clean DataFrame
    true_answers.reset_index(drop=True, inplace=True)

    return true_answers

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get the embedding model.
    :param model_name:
    :return: model, tokenizer
    """
    model = AutoModel.from_pretrained(model_name).to(best_device())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def initialize_model_and_index(debug=False):
    # print what device we have.
    debug and print("device = ", best_device())

    # Get the model to generate embeddings
    debug and print("Getting the model and tokenizer for embeddings...")
    model, tokenizer = get_embedding_model()
    # Get the embeddings size
    embedding_size = model.config.hidden_size
    debug and print("Embedding size = ", embedding_size)
    debug and print("Getting the model and tokenizer for embeddings... done")

    # Create FAISS index
    debug and print("Creating FAISS index for storing and searching the embeddings...")
    index = faiss.IndexFlatL2(embedding_size)
    if best_device() == "cuda":
        print("Moving the index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    debug and print("Creating FAISS index... done")

    return model, tokenizer, index


def encode(texts, model, tokenizer, debug=False):
    tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tokens = {key: value.to(best_device()) for key, value in tokens.items()}  # Move input tensors to GPU

    with torch.no_grad():
        # Do a forward pass and use [CLS] token embeddings
        model_output = model(**tokens)
        embeddings = model_output.last_hidden_state[:, 0, :]

    debug and print("torch embeddings tensor shape: ", embeddings.shape)
    embeddings = embeddings.cpu().numpy()
    debug and print("numpy embeddings shape: ", embeddings.shape)

    return embeddings


def preprocess_kiltwiki(dataset, batch_size=100, debug=False):
    model, tokenizer, index = initialize_model_and_index(debug)

    # Encode and store embeddings for texts in batches
    texts = []
    for batch_idx, batch in enumerate(batched_iterable(dataset, batch_size)):
        print(f"Batch {batch_idx + 1}")

        # batch_texts = [str(example['text']) for example in islice(dataset, batch_size)]
        batch_texts = [str(example['text']) for example in batch]
        debug and print(f"Got {len(batch_texts)} texts")
        debug and print("Getting batch texts... done.")

        # Encode embeddings using the model and tokenizer
        debug and print("Encoding the dataset embeddings...")
        embeddings = encode(batch_texts, model, tokenizer, debug)
        debug and print("Encoding the dataset embeddings... done.")
        debug and print("#Dataset embeddings = ", embeddings.shape, "d = ", embeddings.shape[1])

        # Add the embeddings to an index
        debug and print(f"Adding embeddings to the index (dimension {index.d})...")
        index.add(embeddings)
        debug and print("Adding embeddings to the index... done.")

        texts.extend(batch_texts)

    debug and print("Length of texts = ", len(texts))  # expect total size.
    return index, texts


def preprocess_wikiqa(dataset, batch_size=100, debug=False):
    model, tokenizer, index = initialize_model_and_index(debug)

    # Encode and store embeddings for texts in batches
    texts = []
    for batch_idx, batch in enumerate(batched_iterable(dataset, batch_size)):
        print(f"Batch {batch_idx + 1}")

        batch_texts = [str(example['question'] + " " + example["answer"])
                            for example in batch if example['label'] == 1]

        if len(batch_texts) > 0:
            # Encode batch_texts using the model and tokenizer
            embeddings = encode(batch_texts, model, tokenizer, debug)
            # Add the embeddings to the index
            index.add(embeddings)
            # Add batch_texts to texts
            texts.extend(batch_texts)

    debug and print("Length of texts = ", len(texts))  # expect total size.
    return index, texts


def preprocess_wikiqa_pdframe(pdframe, batch_size=100, debug=False):
    model, tokenizer, index = initialize_model_and_index(debug)

    # Split DataFrame into n batches
    batches = np.array_split(pdframe, len(pdframe) // batch_size)

    texts = []
    for batch_idx, batch in enumerate(batches):
        print(f"Batch {batch_idx + 1}")

        # Convert pdframe columns question and sentence into texts.
        batch_texts = (batch['Question'] + " " + batch['Sentence']).tolist()

        # Generate embeddings
        embeddings = encode(batch_texts, model, tokenizer, debug)

        # Add embeddings to the index
        index.add(embeddings)

        # Add batch_texts to texts
        texts.extend(batch_texts)

    return index, texts


def retrieve(prompts, index, texts, top_k=5, debug=False):
    """
    Retrieve relevant texts to this prompt.
    :param prompts: prompts or query for which RAG should retrieve relevant texts
    :param index: FAISS index of the dataset
    :param texts: texts in the dataset
    :param model: model for encoding the prompt
    :param tokenizer: tokenizer for encoding the prompt
    :param top_k: number of texts to retrieve
    :return: texts relevant to the prompts
    """
    # Encode the prompts using the model and tokenizer to get (num_prompts, embedding_size) array
    debug and print("Encoding prompt embeddings...")
    model, tokenizer = get_embedding_model()
    prompt_embeddings = encode(prompts, model, tokenizer, debug=debug)
    debug and print("Encoding prompt embeddings... done.")
    debug and print("Length of prompt_embeddings = ", len(prompt_embeddings))

    # Search the index and returns (num_prompts, top_k) arrays of distances and indices
    debug and print("Searching for texts in the index...")
    debug and print("Index: ", index)
    debug and print("Index dimension: ", index.d)
    # debug and print("Query vectors: ", prompt_embeddings)
    debug and print("Top k ", top_k)
    _, indices = index.search(prompt_embeddings, top_k)
    debug and print("Search for texts in the index... done.")
    debug and print("All prompt indices: ", indices)

    # Get actual retrieved texts from indices
    debug and print("Getting actual text from indices...")
    retrieved_texts = [[texts[idx] for idx in prompt_indices] for prompt_indices in indices]
    debug and print("Getting actual text from indices... done.")
    debug and print("Retrieved texts size: ", len(retrieved_texts))

    return retrieved_texts
