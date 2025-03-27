# import json
# import os
# import re
# import time
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import faiss
# import pickle
# from typing import List, Dict, Union
# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama
# import nltk
# nltk.download('punkt_tab')

# # Initialize NLTK
# nltk.download('punkt')
# nltk.download('wordnet')

# class MedicalRAG:
#     """Enhanced Medical RAG system with better context handling"""
#     def __init__(self, knowledge_base_path: str = "ClinIQLink_Sample-dataset/sample_submission/rag_biomistral.pkl"):
#         self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
#         self.knowledge = []
#         self.index = None
        
#         if os.path.exists(knowledge_base_path):
#             self.load_knowledge_base(knowledge_base_path)
#         else:
#             print("Initializing with enhanced medical knowledge...")
#             self.initialize_enhanced_knowledge()
    
#     def load_knowledge_base(self, path: str):
#         """Load FAISS index and knowledge base"""
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
#             self.knowledge = data['knowledge']
#             self.index = faiss.deserialize_index(data['index'])
    
#     def initialize_enhanced_knowledge(self):
#         """Enhanced medical knowledge base"""
#         self.knowledge = [
#             "Hypertension: Blood pressure >140/90 mmHg. Treated with ACE inhibitors, diuretics, or calcium channel blockers.",
#             "Diabetes: HbA1c >6.5%. Metformin is first-line therapy. Monitor for complications like retinopathy and neuropathy.",
#             "COVID-19: Caused by SARS-CoV-2. Symptoms include fever, cough, fatigue. Prevent with vaccination and masks.",
#             "Antibiotics: Penicillins for strep throat, fluoroquinolones for UTIs, vancomycin for MRSA.",
#             "EKG interpretation: Check rate, rhythm, axis, intervals, segments, and waves for abnormalities.",
#             "Common lab values: Na 135-145, K 3.5-5.0, WBC 4-11, Hgb 12-16 (women) 13-17 (men), Platelets 150-400."
#         ]
#         embeddings = self.encoder.encode(self.knowledge)
#         self.index = faiss.IndexFlatL2(embeddings.shape[1])
#         self.index.add(embeddings)
    
#     def retrieve(self, query: str, k: int = 3) -> List[str]:
#         """Enhanced retrieval with query expansion"""
#         expanded_query = f"medical {query} diagnosis treatment guidelines"
#         query_embed = self.encoder.encode([expanded_query])
#         _, indices = self.index.search(query_embed, k)
#         return [self.knowledge[i] for i in indices[0]]

# class BioMistralEvaluator:
#     def __init__(self):
#         self.base_dir = os.path.dirname(os.path.abspath(__file__))
#         self.qa_dir = os.path.join(self.base_dir, "..", "sample_QA_pairs")
#         self.template_dir = os.path.join(self.base_dir, "submission_template")
        
#         # Initialize components
#         self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.rag = MedicalRAG()
#         # self.model, self.tokenizer = self._load_biomistral()
#         self.pipeline = self._create_pipeline()



#     # def _load_biomistral(self):
#     #     print("Loading OpenBioLLM-7B...")
#     #     try:
#     #         tokenizer = AutoTokenizer.from_pretrained("openbio/OpenBioLLM-7B")
#     #         model = AutoModelForCausalLM.from_pretrained(
#     #             "openbio/OpenBioLLM-7B",
#     #             device_map="auto",
#     #             torch_dtype=torch.float16
#     #         )
#     #         return model, tokenizer
#     #     except Exception as e:
#     #         print(f"Error: {e}")
#     #         raise

#     model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
#     model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"

#     model_path = hf_hub_download(model_name,
#                                 filename=model_file,
#                                 local_dir='./')
#     model = Llama(model_path=model_path,
#                 n_gpu_layers=-1)
        

#     # def _load_biomistral(self):
#     #     """Optimized model loading with error handling"""
#     #     print("Loading BioMistral-7B with optimized settings...")
#     #     try:
#     #         tokenizer = AutoTokenizer.from_pretrained(
#     #             "BioMistral/BioMistral-7B",
#     #             padding_side="left",
#     #             truncation_side="left"
#     #         )
#     #         tokenizer.pad_token = tokenizer.eos_token
            
#     #         model = AutoModelForCausalLM.from_pretrained(
#     #             "BioMistral/BioMistral-7B",
#     #             torch_dtype=torch.float16,
#     #             device_map="auto",
#     #             trust_remote_code=True,
#     #             low_cpu_mem_usage=True,
#     #             offload_folder="./offload"
#     #         )
#     #         print("Model loaded successfully with optimized settings")
#     #         return model, tokenizer
#     #     except Exception as e:
#     #         print(f"Error loading model: {e}")
#     #         raise

#     # def _create_pipeline(self):
#     #     """Optimized generation pipeline"""
#     #     print("Creating optimized generation pipeline...")
#     #     try:
#     #         pipe = pipeline(
#     #             "text-generation",
#     #             model=self.model,
#     #             tokenizer=self.tokenizer,
#     #             torch_dtype=torch.float16,
#     #             device_map="auto",
#     #             max_new_tokens=300,  # Reduced for more concise answers
#     #             do_sample = True,
#     #             temperature=0.7,     # Balanced between creativity and accuracy
#     #             top_p=0.9,
#     #             # top_k=50,            # Added for better diversity
#     #             repetition_penalty=1.1,
#     #             pad_token_id=self.tokenizer.eos_token_id
#     #         )
#     #         print("Pipeline created with optimized settings")
#     #         return pipe
#     #     except Exception as e:
#     #         print(f"Error creating pipeline: {e}")
#     #         raise


#     def load_json(self, filepath):
#         """
#         Load JSON data from the specified file.
#         """
#         try:
#             with open(filepath, "r") as f:
#                 return json.load(f)
#         except Exception as e:
#             print(f"Error loading JSON from {filepath}: {e}", flush=True)
#             return None


#     def load_template(self, filename):
#         """
#         Load the content of the template file.
#         """
#         filepath = os.path.join(self.template_dir, filename)
#         try:
#             with open(filepath, "r") as f:
#                 return f.read()
#         except Exception as e:
#             print(f"Error loading template {filename} from {filepath}: {e}", flush=True)
#             return None


#     def generate_prompt(self, template, qa, qa_type):
#         """
#         Generates a prompt for the given QA pair using the specified template.

#         Args:
#             template (str): The prompt template.
#             qa (dict): A dictionary containing question and options (if applicable).
#             qa_type (str): The type of question (e.g., "true_false", "multiple_choice", "list", etc.).

#         Returns:
#             str: A formatted prompt.
#         """
#         try:
#             # Extract common fields
#             # question = qa.get("question", "Unknown Question")
#             question_only = qa.get("question", "Unknown Question")
#             retrieved_docs = self.rag.retrieve(query=question_only)
#             rag_context = "\n\nRelevant Context:\n" + "\n---\n".join(retrieved_docs)
#             question = question_only+rag_context
#             question = f"""<s>[INST] <<SYS>>
#             You are a highly knowledgeable medical expert. Answer concisely (max 100 words).
#             <</SYS>>

#             Question: {question} [/INST]"""
#             answer = qa.get("answer", "")
#             options = qa.get("options", {})
#             reasoning = qa.get("reasoning", "")
#             false_answer = qa.get("false_answer", "")
    

#             if qa_type == "true_false":
#                 return template.format(question=question)

#             elif qa_type == "multiple_choice":
#                 # Ensure the placeholders match your MC template
#                 return template.format(
#                     question=question,
#                     options_A=options.get("A", "Option A missing"),
#                     options_B=options.get("B", "Option B missing"),
#                     options_C=options.get("C", "Option C missing"),
#                     options_D=options.get("D", "Option D missing")
#                 )

#             elif qa_type == "list":
#                 # Convert list to a joined string for {options_joined}
#                 options_joined = "\n".join(options) if isinstance(options, list) else str(options)
#                 return template.format(
#                     question=question,
#                     options_joined=options_joined
#                 )

#             elif qa_type == "multi_hop":
#                 return template.format(question=question)

#             elif qa_type == "multi_hop_inverse":
#                 return template.format(
#                     question=question,
#                     answer=answer,
#                     reasoning=reasoning
#                 )

#             elif qa_type == "short":
#                 return template.format(question=question)

#             elif qa_type == "short_inverse":
#                 return template.format(
#                     question=question,
#                     false_answer=false_answer
#                 )

#             else:
#                 print(f"Warning: Unknown QA type '{qa_type}'", flush=True)
#                 return "Invalid QA type."

#         except Exception as e:
#             print(f"Error generating prompt: {e}", flush=True)
#             return "Error generating prompt."


#     def evaluate_true_false(self, expected, prediction):
#         """
#         Evaluate True/False questions: returns 1 if answers match, else 0.
#         """
#         try:
#             return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
#         except Exception as e:
#             print(f"Error evaluating True/False question: {e}", flush=True)
#             return 0.0

#     def evaluate_multiple_choice(self, expected, prediction):
#         """
#         Evaluate Multiple Choice questions: returns 1 if the selected option matches the expected answer.
#         """
#         try:
#             return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
#         except Exception as e:
#             print(f"Error evaluating Multiple Choice question: {e}", flush=True)
#             return 0.0

#     def evaluate_list(self, expected, prediction):
#         """
#         Evaluate List questions using the F1 score.
#         'expected' should be a list of strings and 'prediction' can be a comma-separated string or list.
#         """
#         try:
#             # Convert prediction to a list if it's a string
#             if isinstance(prediction, str):
#                 pred_list = [item.strip().lower() for item in prediction.split(",")]
#             else:
#                 pred_list = [item.strip().lower() for item in prediction]
#             exp_list = [item.strip().lower() for item in expected]
#             _, _, f1 = self.compute_f1_score(exp_list, pred_list)
#             return f1
#         except Exception as e:
#             print(f"Error evaluating List question: {e}", flush=True)
#             return 0.0


#     def compute_word_level_similarity(self, expected_text, prediction_text):
#         """
#         Compute a word-level similarity score using token embeddings.
#         For each word in expected_text, find the maximum cosine similarity with any word in prediction_text,
#         and vice versa, then compute the harmonic mean of the averaged precision and recall.
#         Returns a float score between 0 and 1.
#         """
#         try:
#             expected_words = expected_text.split()
#             prediction_words = prediction_text.split()
#             if not expected_words or not prediction_words:
#                 return 0.0
#             expected_embeds = self.st_model.encode(expected_words, convert_to_tensor=True).cpu().numpy()
#             prediction_embeds = self.st_model.encode(prediction_words, convert_to_tensor=True).cpu().numpy()
            
#             sims_expected = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
#             sims_prediction = [np.max(cosine_similarity([embed], expected_embeds)) for embed in prediction_embeds]
            
#             recall = np.mean(sims_expected)
#             precision = np.mean(sims_prediction)
#             if (precision + recall) == 0:
#                 return 0.0
#             return 2 * precision * recall / (precision + recall)
#         except Exception as e:
#             print(f"Error computing word-level similarity: {e}", flush=True)
#             return 0.0


#     def compute_sentence_level_similarity(self, expected_text, prediction_text):
#         """
#         Compute sentence-level similarity by splitting texts into sentences,
#         encoding them, and averaging the maximum cosine similarity for each expected sentence.
#         Returns a float score between 0 and 1.
#         """
#         try:
#             expected_sentences = nltk.sent_tokenize(expected_text)
#             prediction_sentences = nltk.sent_tokenize(prediction_text)
#             if not expected_sentences or not prediction_sentences:
#                 return 0.0
#             expected_embeds = self.st_model.encode(expected_sentences, convert_to_tensor=True).cpu().numpy()
#             prediction_embeds = self.st_model.encode(prediction_sentences, convert_to_tensor=True).cpu().numpy()
#             sims = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
#             return np.mean(sims)
#         except Exception as e:
#             print(f"Error computing sentence-level similarity: {e}", flush=True)
#             return 0.0


#     def compute_paragraph_level_similarity(self, expected_text, prediction_text):
#         """
#         Compute paragraph-level similarity using embeddings for the full texts.
#         Returns a similarity score between 0 and 1.
#         """
#         try:
#             expected_embed = self.st_model.encode(expected_text, convert_to_tensor=True).cpu().numpy()
#             prediction_embed = self.st_model.encode(prediction_text, convert_to_tensor=True).cpu().numpy()
#             sim = cosine_similarity([expected_embed], [prediction_embed])[0][0]
#             return sim
#         except Exception as e:
#             print(f"Error computing paragraph-level similarity: {e}", flush=True)
#             return 0.0


#     def evaluate_open_ended(self, expected, prediction):
#         """
#         Evaluate open-ended questions by first checking for an exact match.
#         If the response exactly matches the expected answer, return 1.0.
#         Otherwise, compute a weighted semantic similarity using:
#             - Word-level similarity (weight 0.3)
#             - Sentence-level similarity (weight 0.3)
#             - Paragraph-level similarity (weight 0.4)
#         Full points are given if the final semantic score is >= 0.9,
#         0 points if below 0.4, and linear interpolation is used between.
#         """
#         try:
#             if expected.strip().lower() == prediction.strip().lower():
#                 return 1.0

#             word_sim = self.compute_word_level_similarity(expected, prediction)
#             sentence_sim = self.compute_sentence_level_similarity(expected, prediction)
#             paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)

#             # Weights that sum to 1
#             w_word = 0.3
#             w_sentence = 0.3
#             w_paragraph = 0.4
#             semantic_score = w_word * word_sim + w_sentence * sentence_sim + w_paragraph * paragraph_sim

#             if semantic_score >= 0.9:
#                 return 1.0
#             elif semantic_score < 0.4:
#                 return 0.0
#             else:
#                 return (semantic_score - 0.4) / 0.5
#         except Exception as e:
#             print(f"Error evaluating open-ended question: {e}", flush=True)
#             return 0.0

        
#     # def evaluate_open_ended_metrics(self, expected, prediction):
#     #   """Robust metric calculation with proper tokenization and type handling"""
#     #   try:
#     #       # Convert inputs to strings if they're lists
#     #       expected_str = ' '.join(expected) if isinstance(expected, list) else str(expected)
#     #       prediction_str = ' '.join(prediction) if isinstance(prediction, list) else str(prediction)
          
#     #       # Tokenize for BLEU
#     #       expected_tokens = nltk.word_tokenize(expected_str.lower())
#     #       pred_tokens = nltk.word_tokenize(prediction_str.lower())
          
#     #       # Calculate BLEU with smoothing
#     #       smoothing = SmoothingFunction().method1
#     #       bleu = sentence_bleu([expected_tokens], pred_tokens, smoothing_function=smoothing)
          
#     #       # Calculate METEOR
#     #       meteor = meteor_score([expected_str], prediction_str)
          
#     #       # Calculate ROUGE
#     #       scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     #       rouge_scores = scorer.score(expected_str, prediction_str)
#     #       rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
          
#     #       return {
#     #           "bleu": float(bleu),
#     #           "meteor": float(meteor),
#     #           "rouge": float(rouge_avg)
#     #       }
#     #   except Exception as e:
#     #       print(f"Metric calculation error: {e}")
#     #       return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}

#     def evaluate_open_ended_metrics(self, expected, prediction):
#         try:
#             # Convert to strings if they're lists
#             expected_str = ' '.join(expected) if isinstance(expected, list) else str(expected)
#             prediction_str = ' '.join(prediction) if isinstance(prediction, list) else str(prediction)
            
#             # Tokenize for BLEU
#             expected_tokens = [word_tokenize(expected_str.lower())]  # Note: Wrapped in a list
#             pred_tokens = word_tokenize(prediction_str.lower())
            
#             # Calculate BLEU with smoothing
#             smoothing = SmoothingFunction().method1
#             bleu = sentence_bleu(expected_tokens, pred_tokens, smoothing_function=smoothing)
            
#             # Calculate METEOR
#             meteor = meteor_score([expected_str], prediction_str)
            
#             # Calculate ROUGE
#             scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#             rouge_scores = scorer.score(expected_str, prediction_str)
#             rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            
#             return {"bleu": float(bleu), "meteor": float(meteor), "rouge": float(rouge_avg)}
#         except Exception as e:
#             print(f"Metric error: {e}")
#             return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}


#     def compute_f1_score(self, true_list, pred_list):
#         """
#         Compute precision, recall, and F1 score for list-type answers.
#         """
#         try:
#             true_set = set(item.strip().lower() for item in true_list)
#             pred_set = set(item.strip().lower() for item in pred_list)
#             tp = len(true_set & pred_set)
#             fp = len(pred_set - true_set)
#             fn = len(true_set - pred_set)
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
#             return precision, recall, f1
#         except Exception as e:
#             print(f"Error computing F1 score: {e}", flush=True)
#             return 0.0, 0.0, 0.0


#     # def generate_response(self, prompt: str) -> str:
#     #   """Enhanced response generation with strict formatting"""
#     #   try:
#     #       # Generate with more constrained parameters
#     #       outputs = self.pipeline(
#     #           prompt,
#     #           max_new_tokens=300,
#     #           do_sample = True,
#     #           temperature=0.7,  # Minimize randomness
#     #           top_p=1.0,
#     #           num_return_sequences=1,
#     #           eos_token_id=self.tokenizer.eos_token_id,
#     #           pad_token_id=self.tokenizer.eos_token_id
#     #       )
          
#     #       # Extract and clean response
#     #       response = outputs[0]['generated_text']
#     #       answer = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response

          
#     #       # Remove common prefixes and clean up
#     #       for prefix in ["Answer:", "Final Answer:", "Response:", "The correct answer is"]:
#     #           if answer.startswith(prefix):
#     #               answer = answer[len(prefix):].strip()
          
#     #       # Remove any numbering or bullet points
#     #       answer = re.sub(r'^[\d\.\)\-•]\s*', '', answer)
          
#     #       return answer if answer else "[No answer generated]"
#     #   except Exception as e:
#     #       print(f"Generation error: {e}")
#     #       return "[Error generating response]"


#     def generate_response(self, prompt: str) -> str:
#         try:
            
#             response = model(prompt, max_tokens=300)['choices'][0]['text']
#             return self._clean_response(response.text)
        
#         except Exception as e:
#             print(f"Response Generation Error: {e}")
#             return "[Response Generation Error]"


        
#     def _clean_response(self, response: str) -> str:
#         response = response.strip()
#         # Remove unwanted prefixes
#         for prefix in ["Answer:", "Response:", "**"]:
#             if response.startswith(prefix):
#                 response = response[len(prefix):].strip()
#         return response if response else "[No answer]"




#     # Enhanced evaluation methods
#     # def evaluate_open_ended_metrics(self, expected, prediction):
#     #     """Robust metric calculation with error handling"""
#     #     try:
#     #         # Tokenize for BLEU
#     #         expected_tokens = nltk.word_tokenize(expected.lower())
#     #         pred_tokens = nltk.word_tokenize(prediction.lower())
            
#     #         # Calculate BLEU with smoothing
#     #         smoothing = SmoothingFunction().method1
#     #         bleu = sentence_bleu([expected_tokens], pred_tokens, smoothing_function=smoothing)
            
#     #         # Calculate METEOR
#     #         meteor = meteor_score([expected], prediction)
            
#     #         # Calculate ROUGE
#     #         scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     #         rouge_scores = scorer.score(expected, prediction)
#     #         rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            
#     #         return {
#     #             "bleu": float(bleu),
#     #             "meteor": float(meteor),
#     #             "rouge": float(rouge_avg)
#     #         }
#     #     except Exception as e:
#     #         print(f"Metric calculation error: {e}")
#     #         return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}

#     def evaluate_true_false_questions(self):
#         """Enhanced TF evaluation with better answer parsing"""
#         try:
#             tf_path = os.path.join(self.qa_dir, "TF.json")
#             tf_data = self.load_json(tf_path)
#             if not tf_data:
#                 return {"average": 0.0, "scores": {}}
                
#             template = self.load_template("tf_template.prompt")
#             results = {}
#             scores = []
            
#             for qa in tf_data:
#                 try:
#                     prompt = self.generate_prompt(template, qa, "true_false")
#                     response = self.generate_response(prompt)
#                     response = response.replace(prompt,"").strip()
                    
#                     # Enhanced answer parsing
#                     response_lower = response.lower().strip()
#                     expected = qa.get("answer", "").lower().strip()
                    
#                     # Determine prediction
#                     if any(x in response_lower for x in ["true", "yes", "correct", "affirmative"]) and response_lower != "correct_answer: false":
#                         predicted = "true"
#                     elif any(x in response_lower for x in ["false", "no", "incorrect", "negative"]):
#                         predicted = "false"
#                     else:
#                         # Fallback to first word if unclear
#                         predicted = response_lower.split()[0] if response_lower else "unknown"
                    
#                     score = 1.0 if expected == predicted else 0.0
                    
#                     results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
#                         "question": qa.get("question", ""),
#                         "expected": expected,
#                         "predicted": predicted,
#                         "score": score,
#                         "response": response
#                     }
#                     scores.append(score)
                    
#                 except Exception as e:
#                     print(f"Error processing TF QA: {e}")
#                     scores.append(0.0)
            
#             avg = sum(scores) / len(scores) if scores else 0.0
#             print(f"True/False Average Score: {avg:.2f}")
#             return {"average": avg, "scores": results}
            
#         except Exception as e:
#             print(f"Error evaluating TF questions: {e}")
#             return {"average": 0.0, "scores": {}}

#     # [Include other evaluation methods with similar enhancements]


#     def evaluate_multiple_choice_questions(self):
#       """Enhanced MC evaluation with strict answer extraction"""
#       try:
#           mc_path = os.path.join(self.qa_dir, "MC.json")
#           mc_data = self.load_json(mc_path)
#           if not mc_data:
#               return {"average": 0.0, "scores": {}}
              
#           template = self.load_template("MC_template.prompt")
#           results = {}
#           scores = []
          
#           for qa in mc_data:
#               try:
#                   # Generate prompt with explicit answer format instructions
#                   enhanced_prompt = (
#                       f"{template}\n\n"
#                       "Answer ONLY with the single uppercase letter of the correct option (A, B, C, or D). "
#                       "Do not include any explanation or additional text."
#                   )
#                   prompt = self.generate_prompt(enhanced_prompt, qa, "multiple_choice")
#                   response = self.generate_response(prompt)
#                   response = response.replace(prompt,"").strip()
                  
#                   # Extract just the option letter
#                   predicted = ""
#                   for char in response.upper():
#                       if char in ['A', 'B', 'C', 'D']:
#                           predicted = char
#                           break
                  
#                   expected = qa.get("correct_answer", "").strip().upper()[0]
                  
#                   score = 1.0 if expected == predicted else 0.0
                  
#                   results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
#                       "question": qa.get("question", ""),
#                       "expected": expected,
#                       "predicted": predicted,
#                       "score": score,
#                       "response": response
#                   }
#                   scores.append(score)
                  
#               except Exception as e:
#                   print(f"Error processing MC QA: {e}")
#                   scores.append(0.0)
          
#           avg = sum(scores) / len(scores) if scores else 0.0
#           print(f"Multiple Choice Average Score: {avg:.2f}")
#           return {"average": avg, "scores": results}
          
#       except Exception as e:
#           print(f"Error evaluating MC questions: {e}")
#           return {"average": 0.0, "scores": {}}


#     def evaluate_list_questions(self):
#       """Improved list question evaluation with better parsing"""
#       try:
#           list_path = os.path.join(self.qa_dir, "list.json")
#           list_data = self.load_json(list_path)
#           if not list_data:
#               return {"average": 0.0, "scores": {}}
              
#           template = self.load_template("list_template.prompt")
#           results = {}
#           scores = []
          
#           for qa in list_data:
#               try:
#                   # Generate prompt with clear list formatting instructions
#                   enhanced_prompt = (
#                       f"{template}\n\n"
#                       "Provide ONLY a comma-separated list of items. "
#                       "Do not include any introductory text or explanations."
#                   )
#                   prompt = self.generate_prompt(enhanced_prompt, qa, "list")
#                   response = self.generate_response(prompt)
#                   response = response.replace(prompt,"").strip()
                  
#                   # Clean and parse the response
#                   response_clean = response.split(":")[-1].strip()  # Remove any "Answer:" prefix
#                   predicted = [item.strip() for item in response_clean.split(",") if item.strip()]
                  
#                   # Handle expected answers
#                   expected = qa.get("answer", [])
#                   if isinstance(expected, str):
#                       expected = [item.strip() for item in expected.split(",")]
                  
#                   # Calculate F1 score
#                   _, _, f1 = self.compute_f1_score(expected, predicted)
                  
#                   results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
#                       "question": qa.get("question", ""),
#                       "expected": expected,
#                       "predicted": predicted,
#                       "score": f1,
#                       "response": response
#                   }
#                   scores.append(f1)
                  
#               except Exception as e:
#                   print(f"Error processing List QA: {e}")
#                   scores.append(0.0)
          
#           avg = sum(scores) / len(scores) if scores else 0.0
#           print(f"List Question Average F1 Score: {avg:.2f}")
#           return {"average": avg, "scores": results}
          
#       except Exception as e:
#           print(f"Error evaluating List questions: {e}")
#           return {"average": 0.0, "scores": {}}


#     # def evaluate_short_questions(self):
#     #     """
#     #     Evaluate all Short Answer questions using semantic similarity metrics.
#     #     Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
#     #     """
#     #     try:
#     #         short_path = os.path.join(self.qa_dir, "short.json")
#     #         short_data = self.load_json(short_path)
#     #         if short_data is None:
#     #             print("No Short Answer data loaded.", flush=True)
#     #             return {"average": 0.0, "scores": {}}
#     #         template = self.load_template("short_template.prompt")
#     #         results = {}
#     #         scores = []
#     #         for qa in short_data:
#     #             try:
#     #                 prompt = self.generate_prompt(template, qa, "short")
#     #                 response = self.generate_response(prompt)
#     #                 response = response.replace(prompt,"").strip()
#     #                 expected = qa.get("answer", "")
#     #                 f1_score = self.evaluate_open_ended(expected, response)
#     #                 metrics = self.evaluate_open_ended_metrics(expected, response)
#     #                 para_id = qa.get("source", {}).get("paragraph_id", "unknown")
#     #                 results[para_id] = {
#     #                     "question": qa.get("question", ""),
#     #                     "expected": expected,
#     #                     "predicted": response,
#     #                     "f1_score": f1_score,
#     #                     "metrics": metrics
#     #                 }
#     #                 scores.append(f1_score)
#     #             except Exception as inner_e:
#     #                 print(f"Error processing Short Answer QA: {inner_e}", flush=True)
#     #         avg = sum(scores) / len(scores) if scores else 0.0
#     #         print(f"Average Short Answer F1 Score: {avg:.2f}", flush=True)
#     #         return {"average": avg, "scores": results}
#     #     except Exception as e:
#     #         print(f"Error evaluating Short Answer questions: {e}", flush=True)
#     #         return {"average": 0.0, "scores": {}}


#     def evaluate_short_questions(self):
#         """
#         Evaluate all Short Answer questions using semantic similarity metrics.
#         Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
#         """
#         try:
#             short_path = os.path.join(self.qa_dir, "short.json")
#             short_data = self.load_json(short_path)
#             if short_data is None:
#                 print("No Short Answer data loaded.", flush=True)
#                 return {"average": 0.0, "scores": {}}
#             template = self.load_template("short_template.prompt")
#             results = {}
#             scores = []
#             for qa in short_data:
#                 try:
#                     prompt = self.generate_prompt(template, qa, "short")
#                     response = self.generate_response(prompt)
#                     response = response.replace(prompt,"").strip()
#                     expected = qa.get("answer", "")
#                     f1_score = self.evaluate_open_ended(expected, response)
#                     metrics = self.evaluate_open_ended_metrics(expected, response)
#                     para_id = qa.get("source", {}).get("paragraph_id", "unknown")
#                     results[para_id] = {
#                         "question": qa.get("question", ""),
#                         "expected": expected,
#                         "predicted": response,
#                         "f1_score": f1_score,
#                         "metrics": metrics
#                     }
#                     scores.append(f1_score)
#                 except Exception as inner_e:
#                     print(f"Error processing Short Answer QA: {inner_e}", flush=True)
#             avg = sum(scores) / len(scores) if scores else 0.0
#             print(f"Average Short Answer F1 Score: {avg:.2f}", flush=True)
#             return {"average": avg, "scores": results}
#         except Exception as e:
#             print(f"Error evaluating Short Answer questions: {e}", flush=True)
#             return {"average": 0.0, "scores": {}}
    
#     def evaluate_short_inverse_questions(self):
#         """
#         Evaluate Short Inverse questions by comparing the LLM's response to the provided incorrect explanation.
#         Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
#         """
#         try:
#             short_inverse_path = os.path.join(self.qa_dir, "short_inverse.json")
#             short_inverse_data = self.load_json(short_inverse_path)
#             if short_inverse_data is None:
#                 print("No Short Inverse data loaded.", flush=True)
#                 return {"average": 0.0, "scores": {}}
#             template = self.load_template("short_inverse_template.prompt")
#             results = {}
#             scores = []
#             for qa in short_inverse_data:
#                 try:
#                     prompt = self.generate_prompt(template, qa, "short_inverse")
#                     response = self.generate_response(prompt)
#                     response = response.replace(prompt,"").strip()
#                     print("Short Inverse Response:", response, flush=True)
#                     # Use the provided incorrect explanation as the expected text.
#                     expected = qa.get("incorrect_explanation", "")
#                     f1_score = self.evaluate_open_ended(expected, response)
#                     metrics = self.evaluate_open_ended_metrics(expected, response)
#                     para_id = qa.get("source", {}).get("paragraph_id", "unknown")
#                     results[para_id] = {
#                         "question": qa.get("question", ""),
#                         "expected": expected,
#                         "predicted": response,
#                         "f1_score": f1_score,
#                         "metrics": metrics
#                     }
#                     scores.append(f1_score)
#                 except Exception as inner_e:
#                     print(f"Error processing Short Inverse QA: {inner_e}", flush=True)
#             avg = sum(scores) / len(scores) if scores else 0.0
#             print(f"Average Short Inverse F1 Score: {avg:.2f}", flush=True)
#             return {"average": avg, "scores": results}
#         except Exception as e:
#             print(f"Error evaluating Short Inverse questions: {e}", flush=True)
#             return {"average": 0.0, "scores": {}}


#     def evaluate_multi_hop_questions(self):
#         """
#         Evaluate all Multi-hop questions using semantic similarity metrics.
#         Returns a dictionary containing the average F1 score and a mapping (by paragraph_id)
#         of individual QA scores.
#         """
#         try:
#             mh_path = os.path.join(self.qa_dir, "multi_hop.json")
#             mh_data = self.load_json(mh_path)
#             if mh_data is None:
#                 print("No Multi-hop data loaded.", flush=True)
#                 return {"average": 0.0, "scores": {}}
#             template = self.load_template("multi_hop_template.prompt")
#             results = {}
#             scores = []
#             for qa in mh_data:
#                 try:
#                     prompt = self.generate_prompt(template, qa, "multi_hop")
#                     response = self.generate_response(prompt)
#                     response = response.replace(prompt,"").strip()
#                     expected = qa.get("answer", "")
#                     f1_score = self.evaluate_open_ended(expected, response)
#                     metrics = self.evaluate_open_ended_metrics(expected, response)
#                     para_id = qa.get("source", {}).get("paragraph_id", "unknown")
#                     results[para_id] = {
#                         "question": qa.get("question", ""),
#                         "expected": expected,
#                         "predicted": response,
#                         "f1_score": f1_score,
#                         "metrics": metrics
#                     }
#                     scores.append(f1_score)
#                 except Exception as inner_e:
#                     print(f"Error processing Multi-hop QA: {inner_e}", flush=True)
#             avg = sum(scores) / len(scores) if scores else 0.0
#             print(f"Average Multi-hop F1 Score: {avg:.2f}", flush=True)
#             return {"average": avg, "scores": results}
#         except Exception as e:
#             print(f"Error evaluating Multi-hop questions: {e}", flush=True)
#             return {"average": 0.0, "scores": {}}


#     def evaluate_multi_hop_inverse_questions(self):
#       """Properly handle list-type expected answers for multi-hop inverse"""
#       try:
#           mh_inverse_path = os.path.join(self.qa_dir, "multi_hop_inverse.json")
#           mh_inverse_data = self.load_json(mh_inverse_path)
#           if not mh_inverse_data:
#               return {"average": 0.0, "scores": {}}
              
#           template = self.load_template("multi_hop_inverse_template.prompt")
#           results = {}
#           scores = []
          
#           for qa in mh_inverse_data:
#               try:
#                   prompt = self.generate_prompt(template, qa, "multi_hop_inverse")
#                   response = self.generate_response(prompt)
#                   response = response.replace(prompt,"").strip()
                  
#                   # Handle list-type expected answers
#                   expected = qa.get("incorrect_reasoning_step", [])
#                   if isinstance(expected, list):
#                       expected_str = ' '.join(expected)
#                   else:
#                       expected_str = str(expected)
                  
#                   # Handle prediction
#                   if isinstance(response, list):
#                       prediction_str = ' '.join(response)
#                   else:
#                       prediction_str = str(response)
                  
#                   f1_score = self.evaluate_open_ended(expected_str, prediction_str)
                  
#                   results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
#                       "question": qa.get("question", ""),
#                       "expected": expected,
#                       "predicted": response,
#                       "f1_score": f1_score,
#                       "metrics": self.evaluate_open_ended_metrics(expected_str, prediction_str)
#                   }
#                   scores.append(f1_score)
                  
#               except Exception as e:
#                   print(f"Error processing Multi-hop Inverse QA: {e}")
#                   scores.append(0.0)
          
#           avg = sum(scores) / len(scores) if scores else 0.0
#           print(f"Average Multi-hop Inverse F1 Score: {avg:.2f}")
#           return {"average": avg, "scores": results}
          
#       except Exception as e:
#           print(f"Error evaluating Multi-hop Inverse questions: {e}")
#           return {"average": 0.0, "scores": {}}


#     def run_all_evaluations(self):
#         """Enhanced evaluation pipeline with progress tracking"""
#         try:
#             print("\nStarting comprehensive evaluation...")
#             results = {
#                 "true_false": self.evaluate_true_false_questions(),
#                 "multiple_choice": self.evaluate_multiple_choice_questions(),
#                 "list": self.evaluate_list_questions(),
#                 "short": self.evaluate_short_questions(),
#                 "multi_hop": self.evaluate_multi_hop_questions(),
#                 "short_inverse": self.evaluate_short_inverse_questions(),
#                 "multi_hop_inverse": self.evaluate_multi_hop_inverse_questions()
#             }
            
#             # Calculate overall score excluding None values
#             valid_scores = [v['average'] for v in results.values() if isinstance(v, dict) and 'average' in v]
#             overall_score = np.mean(valid_scores) if valid_scores else 0.0
#             results["overall_score"] = float(overall_score)
            
#             # Enhanced results saving
#             output_path = os.path.join(self.base_dir, "enhanced_medical_evaluation_results.json")
#             with open(output_path, 'w') as f:
#                 json.dump(results, f, indent=2, ensure_ascii=False)
                
#             print(f"\nEvaluation Complete. Overall Score: {overall_score:.2f}")
#             print(f"Detailed results saved to {output_path}")
            
#             return results
            
#         except Exception as e:
#             print(f"Evaluation failed: {e}")
#             raise

# if __name__ == "__main__":
#     print("Initializing Enhanced BioMistral Medical Evaluator...")
#     evaluator = BioMistralEvaluator()
#     results = evaluator.run_all_evaluations()


import json
import os
import re
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from peft import PeftModel
from transformers import AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import pickle
from typing import List, Dict, Union
import nltk
nltk.download('punkt_tab')

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')

class MedicalRAG:
    """Enhanced Medical RAG system with better context handling"""
    def __init__(self, knowledge_base_path: str = "ClinIQLink_Sample-dataset/sample_submission/rag_biomistral.pkl"):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge = []
        self.index = None
        
        if os.path.exists(knowledge_base_path):
            self.load_knowledge_base(knowledge_base_path)
        else:
            print("Initializing with enhanced medical knowledge...")
            self.initialize_enhanced_knowledge()
    
    def load_knowledge_base(self, path: str):
        """Load FAISS index and knowledge base"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.knowledge = data['knowledge']
            self.index = faiss.deserialize_index(data['index'])
    
    def initialize_enhanced_knowledge(self):
        """Enhanced medical knowledge base"""
        self.knowledge = [
            "Hypertension: Blood pressure >140/90 mmHg. Treated with ACE inhibitors, diuretics, or calcium channel blockers.",
            "Diabetes: HbA1c >6.5%. Metformin is first-line therapy. Monitor for complications like retinopathy and neuropathy.",
            "COVID-19: Caused by SARS-CoV-2. Symptoms include fever, cough, fatigue. Prevent with vaccination and masks.",
            "Antibiotics: Penicillins for strep throat, fluoroquinolones for UTIs, vancomycin for MRSA.",
            "EKG interpretation: Check rate, rhythm, axis, intervals, segments, and waves for abnormalities.",
            "Common lab values: Na 135-145, K 3.5-5.0, WBC 4-11, Hgb 12-16 (women) 13-17 (men), Platelets 150-400."
        ]
        embeddings = self.encoder.encode(self.knowledge)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Enhanced retrieval with query expansion"""
        expanded_query = f"medical {query} diagnosis treatment guidelines"
        query_embed = self.encoder.encode([expanded_query])
        _, indices = self.index.search(query_embed, k)
        return [self.knowledge[i] for i in indices[0]]

class BioMistralEvaluator:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.qa_dir = os.path.join(self.base_dir, "..", "sample_QA_pairs")
        self.template_dir = os.path.join(self.base_dir, "submission_template")
        
        # Initialize components
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rag = MedicalRAG()
        self.model, self.tokenizer = self._load_biomistral()
        # self.pipeline = self._create_pipeline()




    # def _load_biomistral(self):
    #     print("Loading OpenBioLLM...")
    #     try:
    #         from transformers import AutoTokenizer
            
    #         # Load tokenizer
    #         tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            
    #         # Load model
    #         base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    #         model = PeftModel.from_pretrained(base_model, "HiTZ/Mistral-7B-MedExpQA-EN")
            
    #         return model, tokenizer
            
    #     except Exception as e:
    #         print(f"Error loading model: {e}")
    #         raise


    def _load_biomistral(self):
        print("Loading OpenBioLLM...")
        try:
            from transformers import AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("aaditya/OpenBioLLM-Llama3-8B")
            
            # Load model
            model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
            model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
            model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')
            
            model = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_gpu_layers=-1,
                verbose=False
            )
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        

    # def _load_biomistral(self):
    #     """Optimized model loading with error handling"""
    #     print("Loading BioMistral-7B with optimized settings...")
    #     try:
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             "BioMistral/BioMistral-7B",
    #             padding_side="left",
    #             truncation_side="left"
    #         )
    #         tokenizer.pad_token = tokenizer.eos_token
            
    #         model = AutoModelForCausalLM.from_pretrained(
    #             "BioMistral/BioMistral-7B",
    #             torch_dtype=torch.float16,
    #             device_map="auto",
    #             trust_remote_code=True,
    #             low_cpu_mem_usage=True,
    #             offload_folder="./offload"
    #         )
    #         print("Model loaded successfully with optimized settings")
    #         return model, tokenizer
    #     except Exception as e:
    #         print(f"Error loading model: {e}")
    #         raise

    # def _create_pipeline(self):
    #     """Optimized generation pipeline"""
    #     print("Creating optimized generation pipeline...")
    #     try:
    #         pipe = pipeline(
    #             "text-generation",
    #             model=self.model,
    #             tokenizer=self.tokenizer,
    #             torch_dtype=torch.float16,
    #             device_map="auto",
    #             max_new_tokens=300,  # Reduced for more concise answers
    #             do_sample = True,
    #             temperature=0.7,     # Balanced between creativity and accuracy
    #             top_p=0.9,
    #             # top_k=50,            # Added for better diversity
    #             repetition_penalty=1.1,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
    #         print("Pipeline created with optimized settings")
    #         return pipe
    #     except Exception as e:
    #         print(f"Error creating pipeline: {e}")
    #         raise


    def load_json(self, filepath):
        """
        Load JSON data from the specified file.
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}", flush=True)
            return None


    def load_template(self, filename):
        """
        Load the content of the template file.
        """
        filepath = os.path.join(self.template_dir, filename)
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading template {filename} from {filepath}: {e}", flush=True)
            return None


    def generate_prompt(self, template, qa, qa_type):
        """
        Generates a prompt for the given QA pair using the specified template.

        Args:
            template (str): The prompt template.
            qa (dict): A dictionary containing question and options (if applicable).
            qa_type (str): The type of question (e.g., "true_false", "multiple_choice", "list", etc.).

        Returns:
            str: A formatted prompt.
        """
        try:
            # Extract common fields
            # question = qa.get("question", "Unknown Question")
            question_only = qa.get("question", "Unknown Question")
            retrieved_docs = self.rag.retrieve(query=question_only)
            rag_context = "\n\nRelevant Context:\n" + "\n---\n".join(retrieved_docs)
            question = question_only+rag_context
            question = f"""<s>[INST] <<SYS>>
            You are a highly knowledgeable medical expert. Answer concisely (max 100 words).
            <</SYS>>

            Question: {question} [/INST]"""
            answer = qa.get("answer", "")
            options = qa.get("options", {})
            reasoning = qa.get("reasoning", "")
            false_answer = qa.get("false_answer", "")
    

            if qa_type == "true_false":
                return template.format(question=question)

            elif qa_type == "multiple_choice":
                # Ensure the placeholders match your MC template
                return template.format(
                    question=question,
                    options_A=options.get("A", "Option A missing"),
                    options_B=options.get("B", "Option B missing"),
                    options_C=options.get("C", "Option C missing"),
                    options_D=options.get("D", "Option D missing")
                )

            elif qa_type == "list":
                # Convert list to a joined string for {options_joined}
                options_joined = "\n".join(options) if isinstance(options, list) else str(options)
                return template.format(
                    question=question,
                    options_joined=options_joined
                )

            elif qa_type == "multi_hop":
                return template.format(question=question)

            elif qa_type == "multi_hop_inverse":
                return template.format(
                    question=question,
                    answer=answer,
                    reasoning=reasoning
                )

            elif qa_type == "short":
                return template.format(question=question)

            elif qa_type == "short_inverse":
                return template.format(
                    question=question,
                    false_answer=false_answer
                )

            else:
                print(f"Warning: Unknown QA type '{qa_type}'", flush=True)
                return "Invalid QA type."

        except Exception as e:
            print(f"Error generating prompt: {e}", flush=True)
            return "Error generating prompt."


    def evaluate_true_false(self, expected, prediction):
        """
        Evaluate True/False questions: returns 1 if answers match, else 0.
        """
        try:
            return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            print(f"Error evaluating True/False question: {e}", flush=True)
            return 0.0

    def evaluate_multiple_choice(self, expected, prediction):
        """
        Evaluate Multiple Choice questions: returns 1 if the selected option matches the expected answer.
        """
        try:
            return 1.0 if expected.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            print(f"Error evaluating Multiple Choice question: {e}", flush=True)
            return 0.0

    def evaluate_list(self, expected, prediction):
        """
        Evaluate List questions using the F1 score.
        'expected' should be a list of strings and 'prediction' can be a comma-separated string or list.
        """
        try:
            # Convert prediction to a list if it's a string
            if isinstance(prediction, str):
                pred_list = [item.strip().lower() for item in prediction.split(",")]
            else:
                pred_list = [item.strip().lower() for item in prediction]
            exp_list = [item.strip().lower() for item in expected]
            _, _, f1 = self.compute_f1_score(exp_list, pred_list)
            return f1
        except Exception as e:
            print(f"Error evaluating List question: {e}", flush=True)
            return 0.0


    def compute_word_level_similarity(self, expected_text, prediction_text):
        """
        Compute a word-level similarity score using token embeddings.
        For each word in expected_text, find the maximum cosine similarity with any word in prediction_text,
        and vice versa, then compute the harmonic mean of the averaged precision and recall.
        Returns a float score between 0 and 1.
        """
        try:
            expected_words = expected_text.split()
            prediction_words = prediction_text.split()
            if not expected_words or not prediction_words:
                return 0.0
            expected_embeds = self.st_model.encode(expected_words, convert_to_tensor=True).cpu().numpy()
            prediction_embeds = self.st_model.encode(prediction_words, convert_to_tensor=True).cpu().numpy()
            
            sims_expected = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
            sims_prediction = [np.max(cosine_similarity([embed], expected_embeds)) for embed in prediction_embeds]
            
            recall = np.mean(sims_expected)
            precision = np.mean(sims_prediction)
            if (precision + recall) == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        except Exception as e:
            print(f"Error computing word-level similarity: {e}", flush=True)
            return 0.0


    def compute_sentence_level_similarity(self, expected_text, prediction_text):
        """
        Compute sentence-level similarity by splitting texts into sentences,
        encoding them, and averaging the maximum cosine similarity for each expected sentence.
        Returns a float score between 0 and 1.
        """
        try:
            expected_sentences = nltk.sent_tokenize(expected_text)
            prediction_sentences = nltk.sent_tokenize(prediction_text)
            if not expected_sentences or not prediction_sentences:
                return 0.0
            expected_embeds = self.st_model.encode(expected_sentences, convert_to_tensor=True).cpu().numpy()
            prediction_embeds = self.st_model.encode(prediction_sentences, convert_to_tensor=True).cpu().numpy()
            sims = [np.max(cosine_similarity([embed], prediction_embeds)) for embed in expected_embeds]
            return np.mean(sims)
        except Exception as e:
            print(f"Error computing sentence-level similarity: {e}", flush=True)
            return 0.0


    def compute_paragraph_level_similarity(self, expected_text, prediction_text):
        """
        Compute paragraph-level similarity using embeddings for the full texts.
        Returns a similarity score between 0 and 1.
        """
        try:
            expected_embed = self.st_model.encode(expected_text, convert_to_tensor=True).cpu().numpy()
            prediction_embed = self.st_model.encode(prediction_text, convert_to_tensor=True).cpu().numpy()
            sim = cosine_similarity([expected_embed], [prediction_embed])[0][0]
            return sim
        except Exception as e:
            print(f"Error computing paragraph-level similarity: {e}", flush=True)
            return 0.0


    def evaluate_open_ended(self, expected, prediction):
        """
        Evaluate open-ended questions by first checking for an exact match.
        If the response exactly matches the expected answer, return 1.0.
        Otherwise, compute a weighted semantic similarity using:
            - Word-level similarity (weight 0.3)
            - Sentence-level similarity (weight 0.3)
            - Paragraph-level similarity (weight 0.4)
        Full points are given if the final semantic score is >= 0.9,
        0 points if below 0.4, and linear interpolation is used between.
        """
        try:
            if expected.strip().lower() == prediction.strip().lower():
                return 1.0

            word_sim = self.compute_word_level_similarity(expected, prediction)
            sentence_sim = self.compute_sentence_level_similarity(expected, prediction)
            paragraph_sim = self.compute_paragraph_level_similarity(expected, prediction)

            # Weights that sum to 1
            w_word = 0.3
            w_sentence = 0.3
            w_paragraph = 0.4
            semantic_score = w_word * word_sim + w_sentence * sentence_sim + w_paragraph * paragraph_sim

            if semantic_score >= 0.9:
                return 1.0
            elif semantic_score < 0.4:
                return 0.0
            else:
                return (semantic_score - 0.4) / 0.5
        except Exception as e:
            print(f"Error evaluating open-ended question: {e}", flush=True)
            return 0.0

        
    # def evaluate_open_ended_metrics(self, expected, prediction):
    #   """Robust metric calculation with proper tokenization and type handling"""
    #   try:
    #       # Convert inputs to strings if they're lists
    #       expected_str = ' '.join(expected) if isinstance(expected, list) else str(expected)
    #       prediction_str = ' '.join(prediction) if isinstance(prediction, list) else str(prediction)
          
    #       # Tokenize for BLEU
    #       expected_tokens = nltk.word_tokenize(expected_str.lower())
    #       pred_tokens = nltk.word_tokenize(prediction_str.lower())
          
    #       # Calculate BLEU with smoothing
    #       smoothing = SmoothingFunction().method1
    #       bleu = sentence_bleu([expected_tokens], pred_tokens, smoothing_function=smoothing)
          
    #       # Calculate METEOR
    #       meteor = meteor_score([expected_str], prediction_str)
          
    #       # Calculate ROUGE
    #       scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    #       rouge_scores = scorer.score(expected_str, prediction_str)
    #       rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
          
    #       return {
    #           "bleu": float(bleu),
    #           "meteor": float(meteor),
    #           "rouge": float(rouge_avg)
    #       }
    #   except Exception as e:
    #       print(f"Metric calculation error: {e}")
    #       return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}


    def evaluate_open_ended_metrics(self, expected, prediction):
        try:
            # Convert to strings if they're lists
            expected_str = ' '.join(expected) if isinstance(expected, list) else str(expected)
            prediction_str = ' '.join(prediction) if isinstance(prediction, list) else str(prediction)
            
            # Tokenize for BLEU
            expected_tokens = [word_tokenize(expected_str.lower())]  # Note: Wrapped in a list
            if isinstance(word_tokenize(prediction_str.lower()), list):
                pred_tokens = word_tokenize(prediction_str.lower())
            else:
                pred_tokens = [word_tokenize(prediction_str.lower())] 
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu(expected_tokens, pred_tokens, smoothing_function=smoothing)
            
            # Calculate METEOR
            if isinstance(word_tokenize(prediction_str.lower()), list):
                meteor = meteor_score([expected_str], prediction_str)
            else:
                meteor = meteor_score([expected_str], [prediction_str])
            
            # Calculate ROUGE
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(expected_str, prediction_str)
            rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            
            return {"bleu": float(bleu), "meteor": float(meteor), "rouge": float(rouge_avg)}
        except Exception as e:
            print(f"Metric error: {e}")
            return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}


    def compute_f1_score(self, true_list, pred_list):
        """
        Compute precision, recall, and F1 score for list-type answers.
        """
        try:
            true_set = set(item.strip().lower() for item in true_list)
            pred_set = set(item.strip().lower() for item in pred_list)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return precision, recall, f1
        except Exception as e:
            print(f"Error computing F1 score: {e}", flush=True)
            return 0.0, 0.0, 0.0


    # def generate_response(self, prompt: str) -> str:
    #   """Enhanced response generation with strict formatting"""
    #   try:
    #       # Generate with more constrained parameters
    #       outputs = self.pipeline(
    #           prompt,
    #           max_new_tokens=300,
    #           do_sample = True,
    #           temperature=0.7,  # Minimize randomness
    #           top_p=1.0,
    #           num_return_sequences=1,
    #           eos_token_id=self.tokenizer.eos_token_id,
    #           pad_token_id=self.tokenizer.eos_token_id
    #       )
          
    #       # Extract and clean response
    #       response = outputs[0]['generated_text']
    #       answer = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response

          
    #       # Remove common prefixes and clean up
    #       for prefix in ["Answer:", "Final Answer:", "Response:", "The correct answer is"]:
    #           if answer.startswith(prefix):
    #               answer = answer[len(prefix):].strip()
          
    #       # Remove any numbering or bullet points
    #       answer = re.sub(r'^[\d\.\)\-•]\s*', '', answer)
          
    #       return answer if answer else "[No answer generated]"
    #   except Exception as e:
    #       print(f"Generation error: {e}")
    #       return "[Error generating response]"


    def generate_response(self, prompt: str) -> str:
        try:
            # Get token count using the model's tokenize method
            tokens = self.model.tokenize(prompt.encode('utf-8'))
            if len(tokens) > 2000:  # Leave room for response
                # Truncate the prompt if too long
                truncated = self.model.detokenize(tokens[:2000])
                prompt = truncated.decode('utf-8', errors='ignore')
            
            response = self.model(
                prompt, 
                max_tokens=300,
                temperature=0.7,
                top_p=0.9
            )
            
            if isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['text'].strip()
            return str(response).strip()
            
        except Exception as e:
            print(f"Response Generation Error: {e}")
            return "[Response Generation Error]"



            
        def _clean_response(self, response: str) -> str:
            response = response.strip()
            # Remove unwanted prefixes
            for prefix in ["Answer:", "Response:", "**"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            return response if response else "[No answer]"




    # Enhanced evaluation methods
    # def evaluate_open_ended_metrics(self, expected, prediction):
    #     """Robust metric calculation with error handling"""
    #     try:
    #         # Tokenize for BLEU
    #         expected_tokens = nltk.word_tokenize(expected.lower())
    #         pred_tokens = nltk.word_tokenize(prediction.lower())
            
    #         # Calculate BLEU with smoothing
    #         smoothing = SmoothingFunction().method1
    #         bleu = sentence_bleu([expected_tokens], pred_tokens, smoothing_function=smoothing)
            
    #         # Calculate METEOR
    #         meteor = meteor_score([expected], prediction)
            
    #         # Calculate ROUGE
    #         scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    #         rouge_scores = scorer.score(expected, prediction)
    #         rouge_avg = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0
            
    #         return {
    #             "bleu": float(bleu),
    #             "meteor": float(meteor),
    #             "rouge": float(rouge_avg)
    #         }
    #     except Exception as e:
    #         print(f"Metric calculation error: {e}")
    #         return {"bleu": 0.0, "meteor": 0.0, "rouge": 0.0}

    def evaluate_true_false_questions(self):
        """Enhanced TF evaluation with better answer parsing"""
        try:
            tf_path = os.path.join(self.qa_dir, "TF.json")
            tf_data = self.load_json(tf_path)
            if not tf_data:
                return {"average": 0.0, "scores": {}}
                
            template = self.load_template("tf_template.prompt")
            results = {}
            scores = []
            
            for qa in tf_data:
                try:
                    prompt = self.generate_prompt(template, qa, "true_false")
                    response = self.generate_response(prompt)
                    response = response.replace(prompt,"").strip()
                    
                    # Enhanced answer parsing
                    response_lower = response.lower().strip()
                    expected = qa.get("answer", "").lower().strip()
                    
                    # Determine prediction
                    if any(x in response_lower for x in ["true", "yes", "correct", "affirmative"]) and response_lower != "correct_answer: false":
                        predicted = "true"
                    elif any(x in response_lower for x in ["false", "no", "incorrect", "negative"]):
                        predicted = "false"
                    else:
                        # Fallback to first word if unclear
                        predicted = response_lower.split()[0] if response_lower else "unknown"
                    
                    score = 1.0 if expected == predicted else 0.0
                    
                    results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": predicted,
                        "score": score,
                        "response": response
                    }
                    scores.append(score)
                    
                except Exception as e:
                    print(f"Error processing TF QA: {e}")
                    scores.append(0.0)
            
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"True/False Average Score: {avg:.2f}")
            return {"average": avg, "scores": results}
            
        except Exception as e:
            print(f"Error evaluating TF questions: {e}")
            return {"average": 0.0, "scores": {}}

    # [Include other evaluation methods with similar enhancements]


    def evaluate_multiple_choice_questions(self):
      """Enhanced MC evaluation with strict answer extraction"""
      try:
          mc_path = os.path.join(self.qa_dir, "MC.json")
          mc_data = self.load_json(mc_path)
          if not mc_data:
              return {"average": 0.0, "scores": {}}
              
          template = self.load_template("MC_template.prompt")
          results = {}
          scores = []
          
          for qa in mc_data:
              try:
                  # Generate prompt with explicit answer format instructions
                  enhanced_prompt = (
                      f"{template}\n\n"
                      "Answer ONLY with the single uppercase letter of the correct option (A, B, C, or D). "
                      "Do not include any explanation or additional text."
                  )
                  prompt = self.generate_prompt(enhanced_prompt, qa, "multiple_choice")
                  response = self.generate_response(prompt)
                  response = response.replace(prompt,"").strip()
                  
                  # Extract just the option letter
                  predicted = ""
                  for char in response.upper():
                      if char in ['A', 'B', 'C', 'D']:
                          predicted = char
                          break
                  
                  expected = qa.get("correct_answer", "").strip().upper()[0]
                  
                  score = 1.0 if expected == predicted else 0.0
                  
                  results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
                      "question": qa.get("question", ""),
                      "expected": expected,
                      "predicted": predicted,
                      "score": score,
                      "response": response
                  }
                  scores.append(score)
                  
              except Exception as e:
                  print(f"Error processing MC QA: {e}")
                  scores.append(0.0)
          
          avg = sum(scores) / len(scores) if scores else 0.0
          print(f"Multiple Choice Average Score: {avg:.2f}")
          return {"average": avg, "scores": results}
          
      except Exception as e:
          print(f"Error evaluating MC questions: {e}")
          return {"average": 0.0, "scores": {}}


    def evaluate_list_questions(self):
      """Improved list question evaluation with better parsing"""
      try:
          list_path = os.path.join(self.qa_dir, "list.json")
          list_data = self.load_json(list_path)
          if not list_data:
              return {"average": 0.0, "scores": {}}
              
          template = self.load_template("list_template.prompt")
          results = {}
          scores = []
          
          for qa in list_data:
              try:
                  # Generate prompt with clear list formatting instructions
                  enhanced_prompt = (
                      f"{template}\n\n"
                      "Provide ONLY a comma-separated list of items. "
                      "Do not include any introductory text or explanations."
                  )
                  prompt = self.generate_prompt(enhanced_prompt, qa, "list")
                  response = self.generate_response(prompt)
                  response = response.replace(prompt,"").strip()
                  
                  # Clean and parse the response
                  response_clean = response.split(":")[-1].strip()  # Remove any "Answer:" prefix
                  predicted = [item.strip() for item in response_clean.split(",") if item.strip()]
                  
                  # Handle expected answers
                  expected = qa.get("answer", [])
                  if isinstance(expected, str):
                      expected = [item.strip() for item in expected.split(",")]
                  
                  # Calculate F1 score
                  _, _, f1 = self.compute_f1_score(expected, predicted)
                  
                  results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
                      "question": qa.get("question", ""),
                      "expected": expected,
                      "predicted": predicted,
                      "score": f1,
                      "response": response
                  }
                  scores.append(f1)
                  
              except Exception as e:
                  print(f"Error processing List QA: {e}")
                  scores.append(0.0)
          
          avg = sum(scores) / len(scores) if scores else 0.0
          print(f"List Question Average F1 Score: {avg:.2f}")
          return {"average": avg, "scores": results}
          
      except Exception as e:
          print(f"Error evaluating List questions: {e}")
          return {"average": 0.0, "scores": {}}


    # def evaluate_short_questions(self):
    #     """
    #     Evaluate all Short Answer questions using semantic similarity metrics.
    #     Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
    #     """
    #     try:
    #         short_path = os.path.join(self.qa_dir, "short.json")
    #         short_data = self.load_json(short_path)
    #         if short_data is None:
    #             print("No Short Answer data loaded.", flush=True)
    #             return {"average": 0.0, "scores": {}}
    #         template = self.load_template("short_template.prompt")
    #         results = {}
    #         scores = []
    #         for qa in short_data:
    #             try:
    #                 prompt = self.generate_prompt(template, qa, "short")
    #                 response = self.generate_response(prompt)
    #                 response = response.replace(prompt,"").strip()
    #                 expected = qa.get("answer", "")
    #                 f1_score = self.evaluate_open_ended(expected, response)
    #                 metrics = self.evaluate_open_ended_metrics(expected, response)
    #                 para_id = qa.get("source", {}).get("paragraph_id", "unknown")
    #                 results[para_id] = {
    #                     "question": qa.get("question", ""),
    #                     "expected": expected,
    #                     "predicted": response,
    #                     "f1_score": f1_score,
    #                     "metrics": metrics
    #                 }
    #                 scores.append(f1_score)
    #             except Exception as inner_e:
    #                 print(f"Error processing Short Answer QA: {inner_e}", flush=True)
    #         avg = sum(scores) / len(scores) if scores else 0.0
    #         print(f"Average Short Answer F1 Score: {avg:.2f}", flush=True)
    #         return {"average": avg, "scores": results}
    #     except Exception as e:
    #         print(f"Error evaluating Short Answer questions: {e}", flush=True)
    #         return {"average": 0.0, "scores": {}}


    def evaluate_short_questions(self):
        """
        Evaluate all Short Answer questions using semantic similarity metrics.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            short_path = os.path.join(self.qa_dir, "short.json")
            short_data = self.load_json(short_path)
            if short_data is None:
                print("No Short Answer data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("short_template.prompt")
            results = {}
            scores = []
            for qa in short_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short")
                    response = self.generate_response(prompt)
                    response = response.replace(prompt,"").strip()
                    expected = qa.get("answer", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Short Answer QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Answer F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Short Answer questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}
    
    def evaluate_short_inverse_questions(self):
        """
        Evaluate Short Inverse questions by comparing the LLM's response to the provided incorrect explanation.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id) of individual QA scores.
        """
        try:
            short_inverse_path = os.path.join(self.qa_dir, "short_inverse.json")
            short_inverse_data = self.load_json(short_inverse_path)
            if short_inverse_data is None:
                print("No Short Inverse data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("short_inverse_template.prompt")
            results = {}
            scores = []
            for qa in short_inverse_data:
                try:
                    prompt = self.generate_prompt(template, qa, "short_inverse")
                    response = self.generate_response(prompt)
                    response = response.replace(prompt,"").strip()
                    print("Short Inverse Response:", response, flush=True)
                    # Use the provided incorrect explanation as the expected text.
                    expected = qa.get("incorrect_explanation", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Short Inverse QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Short Inverse F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Short Inverse questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multi_hop_questions(self):
        """
        Evaluate all Multi-hop questions using semantic similarity metrics.
        Returns a dictionary containing the average F1 score and a mapping (by paragraph_id)
        of individual QA scores.
        """
        try:
            mh_path = os.path.join(self.qa_dir, "multi_hop.json")
            mh_data = self.load_json(mh_path)
            if mh_data is None:
                print("No Multi-hop data loaded.", flush=True)
                return {"average": 0.0, "scores": {}}
            template = self.load_template("multi_hop_template.prompt")
            results = {}
            scores = []
            for qa in mh_data:
                try:
                    prompt = self.generate_prompt(template, qa, "multi_hop")
                    response = self.generate_response(prompt)
                    response = response.replace(prompt,"").strip()
                    expected = qa.get("answer", "")
                    f1_score = self.evaluate_open_ended(expected, response)
                    metrics = self.evaluate_open_ended_metrics(expected, response)
                    para_id = qa.get("source", {}).get("paragraph_id", "unknown")
                    results[para_id] = {
                        "question": qa.get("question", ""),
                        "expected": expected,
                        "predicted": response,
                        "f1_score": f1_score,
                        "metrics": metrics
                    }
                    scores.append(f1_score)
                except Exception as inner_e:
                    print(f"Error processing Multi-hop QA: {inner_e}", flush=True)
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"Average Multi-hop F1 Score: {avg:.2f}", flush=True)
            return {"average": avg, "scores": results}
        except Exception as e:
            print(f"Error evaluating Multi-hop questions: {e}", flush=True)
            return {"average": 0.0, "scores": {}}


    def evaluate_multi_hop_inverse_questions(self):
      """Properly handle list-type expected answers for multi-hop inverse"""
      try:
          mh_inverse_path = os.path.join(self.qa_dir, "multi_hop_inverse.json")
          mh_inverse_data = self.load_json(mh_inverse_path)
          if not mh_inverse_data:
              return {"average": 0.0, "scores": {}}
              
          template = self.load_template("multi_hop_inverse_template.prompt")
          results = {}
          scores = []
          
          for qa in mh_inverse_data:
              try:
                  prompt = self.generate_prompt(template, qa, "multi_hop_inverse")
                  response = self.generate_response(prompt)
                  response = response.replace(prompt,"").strip()
                  
                  # Handle list-type expected answers
                  expected = qa.get("incorrect_reasoning_step", [])
                  if isinstance(expected, list):
                      expected_str = ' '.join(expected)
                  else:
                      expected_str = str(expected)
                  
                  # Handle prediction
                  if isinstance(response, list):
                      prediction_str = ' '.join(response)
                  else:
                      prediction_str = str(response)
                  
                  f1_score = self.evaluate_open_ended(expected_str, prediction_str)
                  
                  results[qa.get("source", {}).get("paragraph_id", "unknown")] = {
                      "question": qa.get("question", ""),
                      "expected": expected,
                      "predicted": response,
                      "f1_score": f1_score,
                      "metrics": self.evaluate_open_ended_metrics(expected_str, prediction_str)
                  }
                  scores.append(f1_score)
                  
              except Exception as e:
                  print(f"Error processing Multi-hop Inverse QA: {e}")
                  scores.append(0.0)
          
          avg = sum(scores) / len(scores) if scores else 0.0
          print(f"Average Multi-hop Inverse F1 Score: {avg:.2f}")
          return {"average": avg, "scores": results}
          
      except Exception as e:
          print(f"Error evaluating Multi-hop Inverse questions: {e}")
          return {"average": 0.0, "scores": {}}


    def run_all_evaluations(self):
        """Enhanced evaluation pipeline with progress tracking"""
        try:
            print("\nStarting comprehensive evaluation...")
            results = {
                "true_false": self.evaluate_true_false_questions(),
                "multiple_choice": self.evaluate_multiple_choice_questions(),
                "list": self.evaluate_list_questions(),
                "short": self.evaluate_short_questions(),
                "multi_hop": self.evaluate_multi_hop_questions(),
                "short_inverse": self.evaluate_short_inverse_questions(),
                "multi_hop_inverse": self.evaluate_multi_hop_inverse_questions()
            }
            
            # Calculate overall score excluding None values
            valid_scores = [v['average'] for v in results.values() if isinstance(v, dict) and 'average' in v]
            overall_score = np.mean(valid_scores) if valid_scores else 0.0
            results["overall_score"] = float(overall_score)
            
            # Enhanced results saving
            output_path = os.path.join(self.base_dir, "enhanced_medical_evaluation_results.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            print(f"\nEvaluation Complete. Overall Score: {overall_score:.2f}")
            print(f"Detailed results saved to {output_path}")
            
            return results
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise

if __name__ == "__main__":
    print("Initializing Enhanced BioMistral Medical Evaluator...")
    evaluator = BioMistralEvaluator()
    results = evaluator.run_all_evaluations()