import torch
import faiss
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

def load_wiki_dataset(num_examples=None, debug=False):
    path = "kilt_wikipedia"
    print(f"Loading dataset from {path}")
    dataset = load_dataset(path, split="full", streaming=True, trust_remote_code=True)
    if num_examples is not None:
        dataset = dataset.take(num_examples)
    if debug:
        print(dataset)
        for example in dataset:
            print(example)
            break

    return dataset

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get the embedding model.
    :param model_name:
    :return: model, tokenizer
    """
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def encode(texts, model, tokenizer, debug=False):
    tokens = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        # Do a forward pass and use [CLS] token embeddings
        model_output = model(**tokens)
        embeddings = model_output.last_hidden_state[:, 0, :].numpy()
        if debug:
            print("Embeddings shape: ", embeddings.shape)
    return embeddings

def preprocess(dataset, debug=False):
    """
    Preprocess the dataset.
    :param dataset:
    :return: FAISS index
    """
    # Get the list of texts from the dataset
    texts = [f"{example['text']}" for example in dataset]
    if debug:
        print("List of texts size: ", len(texts))

    # Encode embeddings using the model and tokenizer
    if debug:
        print("Encoding the embeddings...")
    model, tokenizer = get_embedding_model()
    embeddings = np.vstack([encode(texts, model, tokenizer, debug)])
    if debug:
        print("Embeddings shape: ", embeddings.shape)

    # Add the embeddings to an index
    if debug:
        print("Adding embeddings to the index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    if debug:
        print("Index is added with embeddings")

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
    model, tokenizer = get_embedding_model()
    prompt_embeddings = encode(prompts, model, tokenizer, debug)

    # Search the index and returns (num_prompts, top_k) arrays of distances and indices
    distances, indices = index.search(prompt_embeddings, top_k)
    if debug:
        print("Distances shape: ", distances.shape)
        print("Indices shape: ", indices.shape)

    retrieved_texts = [[texts[idx] for idx in prompt_indices] for prompt_indices in indices]
    if debug:
        print("Retrieved texts: ", retrieved_texts)
    return retrieved_texts
