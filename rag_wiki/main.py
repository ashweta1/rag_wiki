"""
Main module for the RAG project.

"""
from rag_wiki import rag

if __name__ == '__main__':
    NUM_EXAMPLES = 100
    BATCH_SIZE = 100
    TOP_K_TEXTS = 1

    # Load dataset
    print("Loading dataset...")
    dataset = rag.load_wiki_dataset(num_examples=NUM_EXAMPLES, debug=True)
    print("Loading dataset...done")
    print("")

    # Preprocess the dataset
    print("Preprocessing dataset...")
    index, texts = rag.preprocess(dataset, batch_size=BATCH_SIZE, debug=True)
    print("Preprocessing dataset...done")
    print("")

    # Retrieve relevant texts
    prompts = ["What is the capital of India?",
               "Who is the president of the United States?",
               "What is the population of China?",
               "The captial of France is ",
               "Where is the Eiffel Tower located?"]

    print("Retrieving relevant texts...")
    retrieved_texts = rag.retrieve(prompts, index, texts, top_k=TOP_K_TEXTS, debug=True)
    print("Retrieving relevant texts...done")
    print("")

    contexts = [f"{' '.join(texts)} {prompt}" for prompt, texts in zip(prompts, retrieved_texts)]
    print("Prompts with contexts: ", contexts)
