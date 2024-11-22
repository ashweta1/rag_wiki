import torch
import faiss
from rag_wiki.util import best_device
from rag_wiki.util import is_gpu
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

def load_wiki_dataset(num_examples=None, debug=False):
    path = "kilt_wikipedia"
    print(f"Loading dataset from {path}")
    dataset = load_dataset(path, split="full", streaming=True, trust_remote_code=True)
    if num_examples is not None:
        dataset = dataset.take(num_examples)
    debug and print(dataset)
    if debug:
        for x in dataset:
            print(x)
            break

    return dataset


def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get the embedding model.
    :param model_name:
    :return: model, tokenizer
    """
    model = AutoModel.from_pretrained(model_name).to(best_device())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def encode(texts, model, tokenizer, batch_size=1000, debug=False):
    # create a list of torch tensor embeddings
    embeddings = []
    for i in range(0, len(texts), batch_size):
        debug and print(f"Processing batch {i//batch_size} from {i}:{min(i + batch_size, len(texts))}")
        batch_texts = texts[i:i + batch_size]  # get a batch of texts
        tokens = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        tokens = {key: value.to(best_device()) for key, value in tokens.items()}  # Move input tensors to GPU

        with torch.no_grad():
            # Do a forward pass and use [CLS] token embeddings
            model_output = model(**tokens)
            batch_embeddings = model_output.last_hidden_state[:, 0, :]
            debug and print("Batch embeddings shape: ", batch_embeddings.shape)
            embeddings.append(batch_embeddings)

    embeddings = torch.vstack(embeddings).cpu().numpy()
    debug and print("# embeddings: ", embeddings.shape)

    return embeddings


def preprocess(dataset, batch_size=1000, debug=False):
    """
    Preprocess the dataset.
    :param dataset:
    :return: FAISS index
    """
    # print what device we have.
    debug and print("device = ", best_device())

    # Get the list of texts from the dataset
    debug and print("Listing text values in the dataset...")
    texts = [str(example['text']) for example in dataset]
    debug and print("Listing text values in the dataset... done.")
    debug and print("List of texts size = ", len(texts))

    # Encode embeddings using the model and tokenizer
    debug and print("Encoding the dataset embeddings...")
    model, tokenizer = get_embedding_model()
    embeddings = encode(texts, model, tokenizer, batch_size, debug)
    debug and print("Encoding the dataset embeddings... done.")
    debug and print("#Dataset mbeddings = ", embeddings.shape, "d = ", embeddings.shape[1])

    # Add the embeddings to an index
    debug and print("Adding embeddings to the index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    if best_device() == "cuda":
        debug and print("Creating GPU faiss index")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    debug and print("Index dimension: ", index.d)
    debug and   print("Adding embeddings to the index... done.")

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
    debug and print(prompt_embeddings)

    # Search the index and returns (num_prompts, top_k) arrays of distances and indices
    debug and print("Searching for texts in the index...")
    debug and print("Index: ", index)
    debug and print("Index dimension: ", index.d)
    debug and print("Query vectors: ", prompt_embeddings)
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
