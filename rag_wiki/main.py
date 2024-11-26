"""
Main module for the RAG project.

"""
import datasets

from rag_wiki import rag

if __name__ == '__main__':
    TOP_K_TEXTS = 1
    # DATA_SOURCE = "kiltwiki"
    DATA_SOURCE = "wikiqa"

      # Load and preprocess the dataset
    if DATA_SOURCE == "wikiqa":
        model, tokenizer, index = rag.initialize_model_and_index(debug=False)

        print("Loading wikiqa dataset...")
        # pdframe = rag.load_wikiqa("../data/WikiQACorpus/WikiQA-train.tsv")
        dataset = datasets.load_dataset("wiki_qa", split="train")
        print("Loading dataset...done")
        print(dataset)
        for example in dataset:
            print(example['label'])
            break

        print("")

        print("Preprocessing dataset...")
        # index, texts = rag.preprocess_wikiqa(dataset, batch_size=500, debug=True)
        texts = []
        rag.add_dataset_to_index(dataset, "wikiqa",
                                 model, tokenizer,
                                 index, texts,
                                 batch_size=500,
                                 debug=False)
        print("Preprocessing dataset...done")

    else:
        print("Loading dataset...")
        dataset = rag.load_dataset_num("kilt_wikipedia", num_examples=100, debug=True)
        print("Loading dataset...done")

        print("")

        print("Preprocessing dataset...")
        index, texts = rag.preprocess_kiltwiki(dataset, batch_size=100, debug=True)
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
