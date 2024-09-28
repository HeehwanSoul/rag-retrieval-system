## Heehwan Soul, 885941
## Mehmet Görkem Basar, 921637

from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from sentence_transformers import SentenceTransformer

# Because of the graphic cards in labor
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Assuming you've already set up the Qdrant client
from qdrant_client import QdrantClient
client = QdrantClient("http://dsm.bht-berlin.de:6333/dashboard", port=6333) # http://dsm.bht-berlin.de:6333/dashboard

device = "cuda"  # the device to load the model onto

# Load the language model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Load the sentence transformer model for encoding
sentence_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b', device=device)  ###########

class NeuralSearcher:
    def __init__(self, collection_name, model):
        self.collection_name = collection_name
        self.model = model
        self.qdrant_client = client

    def search(self, text: str):
        vector = self.model.encode(text).tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5,
        )
        payloads = [hit.payload for hit in search_result]
        texts = [hit.payload['text'] for hit in search_result]
        return payloads, texts

neural_searcher = NeuralSearcher(collection_name="Basar_Soul_long_model_semantic_method", model=sentence_model)  ###########


def generate_response_with_rag(query):
    seconds_start = time.time()

    # Retrieval step
    _, retrieved_texts = neural_searcher.search(query)
    
    # Combine retrieved texts into context
    context = " ".join(retrieved_texts)
    
    # Create prompt with context and query
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generation step
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512 
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    seconds_end = time.time()
    duration = seconds_end - seconds_start

    return response, duration

# Main loop for user input
print("Enter your questions. Type 'quit' to exit.")
while True:
    user_input = input("Question: ")
    if user_input.lower() == 'quit':
        break
    
    answer, duration = generate_response_with_rag(user_input)
    
    print(f"Answer: {answer}")
    print(f"Duration: {duration:.2f} seconds")
    print("\n" + "="*50 + "\n")

# Check if the model is on GPU
device = next(model.parameters()).device

print(f"Model is on device: {device}")