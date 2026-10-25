# IndicRAG-QA 
## Evidence-Grounded Cross-Lingual Question Answering for Indic and Code-Mixed Languages

In this project I have made a domain specific RAG chatbot that will answer user queries in Hindi(Indic Language), English and Hinglish(mixed).  
Domain of the project: Government Schemes for Education  
Project Status: In Progress  

## Objectives
- Build a domain-specific RAG-based question-answering system
- Support Hindi, English and Hinglish queries
- Handle code mixed and Cross Lingual queries
- Retrieve relevant evidence from government education scheme documents
- Reduce unsupported or hallucinated answers by grounding responses in retrieved evidence
- Evaluate and compare different information-retrieval approaches

## System Architecture
<img width="1920" height="1536" alt="IndicRAG QA Architecture" src="https://github.com/user-attachments/assets/f66edeb5-aed9-4874-9f21-238236ce4141" />
The system first retrieves potentially relevant passages from the document collection. A cross-encoder is then used to verify the relevance of the retrieved passages before the context is passed to the answer-generation model.

## Retrieval Approaches
1. TF-IDF retrieval  
  A traditional sparse retrieval approach based on Term Frequency-Inverse Document Frequency (TF-IDF).  
  It is useful for identifying passages containing important lexical terms from the query.

2. Dense vector retrieval
  Dense retrieval represents queries and passages as vector embeddings and retrieves semantically similar passages.  
  The project uses multilingual sentence embeddings to support Hindi, English, and code-mixed queries.

3. Hybrid retrieval
  Hybrid retrieval combines the strengths of sparse and dense retrieval methods to improve retrieval performance across different query types.
  Hybrid Score = 0.9 x TF-IDF Score + 0.1 x Dense Score
  The weighting is based on the current retrieval evaluation, where TF-IDF performs better than dense retrieval on the project dataset.

## Answerability Checking
After retrieval, the system performs an additional relevance verification step using a cross-encoder.
The verifier evaluates the relationship between question and retrieved passage.
The purpose of this stage is to determine whether the retrieved evidence is sufficiently relevant before allowing the system to generate an answer.
If sufficient supporting evidence is not detected, the system will not generate an answer.
This is intended to reduce unsupported answers and hallucinations.

## Dataset Used
The project includes a manually constructed multilingual question-answering dataset covering government education schemes.  
- Dataset Contains:
- 325 questions
- 225 answerable questions
- 100 unanswerable questions
- English, Hindi, and Hinglish queries
- Monolingual, code-mixed, and cross-lingual queries
- Multiple difficulty levels
- 417 retrieved passages

## Retrieval
Retrieval performance is evaluated using:  
- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)
Current Retrieval Results:
<img width="672" height="185" alt="Screenshot 2026-10-25 115600" src="https://github.com/user-attachments/assets/29fa08ef-1406-4054-ac6d-16e66adf980e" />

## Technologies Used
- Python
- FAISS: Vector similarity search
- Sentence Transformers: Multilingual text embeddings
- Scikit-learn: TF-IDF and evaluation
- NumPy: Numerical computation
- JSON: Dataset and result storage
- PyPDF: Document processing
- LLM: Answer generation
- Streamlit: UI

## Screenshots
UI
<img width="1900" height="986" alt="Screenshot 2026-10-25 112610" src="https://github.com/user-attachments/assets/2318e736-5b57-4b45-8ca6-bc3122d1d1ad" />  

Answerable question with retrieved context
- English
  <img width="1906" height="1007" alt="Screenshot 2026-10-25 113332" src="https://github.com/user-attachments/assets/fa802314-295c-4004-a5bc-d7f0d0a1ac88" />
  
- Hindi
<img width="1897" height="872" alt="Screenshot 2026-10-25 124407" src="https://github.com/user-attachments/assets/f174d98a-3348-4cd0-849f-919447999ff0" />

- Hinglish
<img width="1907" height="880" alt="Screenshot 2026-10-25 123411" src="https://github.com/user-attachments/assets/8ce57fd8-fdfa-4009-9674-39a2dd28e1a3" />

Unanswerable question
<img width="1917" height="892" alt="Screenshot 2026-10-25 113621" src="https://github.com/user-attachments/assets/7a4e47b7-7fbb-4081-93d2-e45ae048f920" />

## Current Limitation
- Dense retrieval currently performs worse than TF-IDF on the evaluation dataset.
- The answer-generation stage requires further evaluation for multilingual answer quality.
- CPU-based inference with the generative model can be slow.
- Answerability thresholds require further tuning to improve the distinction between answerable and unanswerable questions.

## Future Improvements
- Improve dense retrieval performance for Hindi and Hinglish queries.
- Tune hybrid retrieval weights using validation data.
- Improve answerability threshold selection.
- Reduce time taken to answer.
- Add more government schemes to dataset collection.
- Deploy the final stable version of the application.
