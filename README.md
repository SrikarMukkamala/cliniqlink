# Steps to run locally:
1. git clone https://github.com/SrikarMukkamala/cliniqlink.git <br> 
2. cd cliniqlink <br>
3. pip install -r ClinIQLink_Sample-dataset/sample_submission/requirements.txt <br>
4. gdown 1yNeL5XFqS_tPS8QNPAYtohuR1oSd-HLq -O ClinIQLink_Sample-dataset/sample_submission/rag_model.pkl
5. python3 ClinIQLink_Sample-dataset/sample_submission/submit.py <br>

PS: Running for the first might take some time to download the model. <br>

# Update the RAG's Knowledge base:
The current Knowledge base is solely built on [textbooks data](https://huggingface.co/datasets/MedRAG/textbooks). <br>
You can update the RAG KB by using the kb_builder.py file which creates faiss index. <br>
The output will be a pickle (.pkl) file which you should paste in submission_template folder. <br>

# Output:
You can find the output of that run in enhanced_medical_evaluation_results.json file. <br>

# Results:
# Model Performance Comparison

| Category               | BioMistral7B F1 Score | Gemini 2.0 Flash F1 Score | Difference |
|------------------------|--------------------|------------------------|------------|
| **True/False**         | 0.800              | 0.600                  | +0.200     |
| **Multiple Choice**    | 0.800              | 1.000                  | -0.200     |
| **List**              | 0.870              | 0.678                  | +0.192     |
| **Short Answer**      | 0.550              | 0.538                  | +0.120     |
| **Multi-hop**         | 0.500              | 0.587                  | -0.087     |
| **Short Inverse**     | 0.640              | 0.708                  | -0.068     |
| **Multi-hop Inverse** | 0.420              | 0.626                  | -0.206     |
| **Overall Score**     | 0.655              | 0.677                  | -0.022     |

# Updates Done For Phase III:
1. Adjust BioMistral7B to work for Multi-hop Inverse Questions.
2. Update RAG's Knowledge base with [PubMed data](https://huggingface.co/datasets/MedRAG/pubmed) and [Wikipedia data](https://huggingface.co/datasets/MedRAG/wikipedia).
3. Enhanced the KB efficiency using Knowledge Distillation.
4. Use BM25 + Faiss for re-ranking during information retrieval.
5. Add Hallucination Detection Framework.
6. Include External Fact Checking.
7. Experiment with other models.

# Note: 
The model is around 14.5 GB in size, so kindly ensure that <br>
the system you are running on has atleast 16 GB RAM or you can also try <br>
quantizing the model but that may result in a slightly reducing the performace. <br>
You can find the quantized model [here](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF).

# References:
1. [ClinIQLink Sample Dataset](https://github.com/Brandonio-c/ClinIQLink_Sample-dataset)
2. Labrak, Y., Bazoge, A., Morin, E., Gourraud, P.-A., Rouvier, M., & Dufour, R. (2024). BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. arXiv. Retrieved from https://arxiv.org/abs/2402.10373
3. Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). Benchmarking Retrieval-Augmented Generation for Medicine. *arXiv preprint arXiv:2402.13178*. Retrieved from [https://arxiv.org/abs/2402.13178](https://arxiv.org/abs/2402.13178)
4. [Quantized BioMistral Model](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF)


