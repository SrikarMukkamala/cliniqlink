# Steps to follow:
1. git clone https://github.com/SrikarMukkamala/cliniqlink.git <br> 
2. cd cliniqlink <br>
3. pip install -r ClinIQLink_Sample-dataset/sample_submission/requirements.txt <br>
4. python3 ClinIQLink_Sample-dataset/sample_submission/submit.py <br>

# Update the RAG's Knowledge base:
You can update the RAG KB by using the kb_builder.py file. <br>
The output will be a pickle (.pkl) file which you should paste in submission_template folder. <br>

# Results:
You can find the results of that run in enhanced_medical_evaluation_results.json file. <br>

# Note: 
The model is around 14.5 GB in size, so kindly ensure that <br>
the system you are running on has atleast 16 GB RAM or you can also try <br>
quantizing the model but that may result in a slightly reducing the performace. <br>
You can find the quantized model [here](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF).

# References:
1. [ClinIQLink Sample Dataset](https://github.com/Brandonio-c/ClinIQLink_Sample-dataset)
2. Labrak, Y., Bazoge, A., Morin, E., Gourraud, P.-A., Rouvier, M., & Dufour, R. (2024). BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. arXiv. Retrieved from https://arxiv.org/abs/2402.10373
3. https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF


