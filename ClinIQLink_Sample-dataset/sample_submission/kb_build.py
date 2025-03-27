# from datasets import load_dataset
# import pickle
# import faiss
# import json
# import torch
# from transformers import AutoTokenizer, AutoModel


# # 1. Load MedRAG Textbooks dataset
# textbooks = load_dataset("MedRAG/textbooks")

# # 2. Extract and clean the content
# medical_facts = []
# for item in textbooks['train']:
#     try:
#         data = json.loads(item['content'])
#         medical_facts.append(data.get('content', '') + " " + data.get('contents', ''))
#     except json.JSONDecodeError:
#         medical_facts.append(item['content'])

# # 3. Load BioMistral tokenizer and model
# biomistral_tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
# biomistral_model = AutoModel.from_pretrained("BioMistral/BioMistral-7B").to("cuda")

# # 4. Tokenize and generate embeddings with BioMistral
# def get_biomistral_embeddings(texts, tokenizer, model, batch_size=8):
#     """
#     Generate embeddings using BioMistral.
#     """
#     model.eval()
#     all_embeddings = []

#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]

#             # Tokenize batch
#             tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

#             # Generate embeddings
#             outputs = model(**tokens)
            
#             # Use the [CLS] token embedding as the sentence representation
#             embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
#             all_embeddings.extend(embeddings)

#     return all_embeddings

# # Generate embeddings for the medical facts
# embeddings = get_biomistral_embeddings(medical_facts, biomistral_tokenizer, biomistral_model)

# # 5. Build FAISS index
# index = faiss.IndexFlatL2(embeddings[0].shape[0])
# index.add(embeddings)

# # 6. Save the index and knowledge base
# with open('rag_biomistral_embeddings.pkl', 'wb') as f:
#     pickle.dump({
#         'knowledge': medical_facts,
#         'index': faiss.serialize_index(index)
#     }, f)

# print("Combined medical facts with BioMistral embeddings and FAISS index saved successfully.")
from datasets import load_dataset
import pickle
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# 1. Load MedRAG Textbooks dataset
textbooks = load_dataset("MedRAG/textbooks")

# 2. Extract and clean the content
medical_facts = []
for item in textbooks['train']:
    try:
        data = json.loads(item['content'])
        medical_facts.append(data.get('content', '') + " " + data.get('contents', ''))
    except json.JSONDecodeError:
        medical_facts.append(item['content'])

# 3. Load BioMistral tokenizer
biomistral_tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")

# 4. Tokenize and embed with Sentence Transformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')   # You can change this to BioMistral embedding later
medical_facts_tokenized = [
    biomistral_tokenizer.decode(biomistral_tokenizer.encode(text, truncation=True, max_length=512))
    for text in medical_facts
]

# 5. Create sentence embeddings
embeddings = encoder.encode(medical_facts_tokenized)

# 6. Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 7. Save the index and knowledge base
with open('rag_biomistral.pkl', 'wb') as f:
    pickle.dump({
        'knowledge': medical_facts,
        'index': faiss.serialize_index(index)
    }, f)

print("Combined medical facts with BioMistral tokenizer and FAISS index saved successfully.")
