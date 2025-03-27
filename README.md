# Steps to follow:
1. git clone https://github.com/SrikarMukkamala/cliniqlink.git <br> 
2. cd cliniqlink <br>
3. pip install -r ClinIQLink_Sample-dataset/sample_submission/requirements.txt <br>
4. python3 ClinIQLink_Sample-dataset/sample_submission/submit.py <br>

# Update the RAG's Knowledge base:
You can update the RAG KB by using the kb_builder.py file which creates faiss index. <br>
The output will be a pickle (.pkl) file which you should paste in submission_template folder. <br>

# Output:
You can find the output of that run in enhanced_medical_evaluation_results.json file. <br>

# Results:
# Model Performance Comparison

| Category               | BioMistral7B Score | Gemini 2.0 Flash Score | Difference |
|------------------------|--------------------|------------------------|------------|
| **True/False**         | 0.800              | 0.600                  | +0.200     |
| **Multiple Choice**    | 0.800              | 1.000                  | -0.200     |
| **List**              | 0.825              | 0.678                  | +0.147     |
| **Short Answer**      | 0.415              | 0.538                  | -0.123     |
| **Multi-hop**         | 0.565              | 0.587                  | -0.022     |
| **Short Inverse**     | 0.633              | 0.708                  | -0.075     |
| **Multi-hop Inverse** | 0.000              | 0.626                  | -0.626     |
| **Overall Score**     | 0.577              | 0.677                  | -0.100     |

## Detailed Breakdown

### True/False Questions
- **BioMistral7B**: 80% accuracy
- **Gemini 2.0 Flash**: 60% accuracy
- **Best Performer**: BioMistral7B (+20%)

### Multiple Choice Questions
- **BioMistral7B**: 80% accuracy
- **Gemini 2.0 Flash**: 100% accuracy
- **Best Performer**: Gemini 2.0 Flash (+20%)

### List Questions
- **BioMistral7B**: 82.5% accuracy
- **Gemini 2.0 Flash**: 67.8% accuracy
- **Best Performer**: BioMistral7B (+14.7%)

### Short Answer Questions
- **BioMistral7B**: 41.5% F1 score
- **Gemini 2.0 Flash**: 53.8% F1 score
- **Best Performer**: Gemini 2.0 Flash (+12.3%)

### Multi-hop Questions
- **BioMistral7B**: 56.5% F1 score
- **Gemini 2.0 Flash**: 58.7% F1 score
- **Best Performer**: Gemini 2.0 Flash (+2.2%)

### Short Inverse Questions
- **BioMistral7B**: 63.3% F1 score
- **Gemini 2.0 Flash**: 70.8% F1 score
- **Best Performer**: Gemini 2.0 Flash (+7.5%)

### Multi-hop Inverse Questions
- **BioMistral7B**: 0% F1 score
- **Gemini 2.0 Flash**: 62.6% F1 score
- **Best Performer**: Gemini 2.0 Flash (+62.6%)

## Overall Performance
- **BioMistral7B**: 57.7% overall score
- **Gemini 2.0 Flash**: 67.7% overall score
- **Best Overall Performer**: Gemini 2.0 Flash (+10.0%)

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


