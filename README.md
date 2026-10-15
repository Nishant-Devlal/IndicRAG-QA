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

## Retrieval Approaches
1. TF-IDF retrieval  
  A traditional sparse retrieval approach based on Term Frequency-Inverse Document Frequency (TF-IDF).  
  It is useful for identifying passages containing important lexical terms from the query.

2. Dense vector retrieval
  Dense retrieval represents queries and passages as vector embeddings and retrieves semantically similar passages.  
  The project uses multilingual sentence embeddings to support Hindi, English, and code-mixed queries.

3. Hybrid retrieval
  Hybrid retrieval combines the strengths of sparse and dense retrieval methods to improve retrieval performance across different query types.

## Dataset Used
The project includes a manually constructed multilingual question-answering dataset covering government education schemes.  
- Dataset Contains:
- 325 questions
- 225 answerable questions
- 100 unanswerable questions
- English, Hindi, and Hinglish queries
- Monolingual, code-mixed, and cross-lingual queries
- Multiple difficulty levels

## Retrieval
Retrieval performance is evaluated using:  
- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)

## Technologies Used
- Python
- FAISS: Vector similarity search
- Sentence Transformers: Multilingual text embeddings
- Scikit-learn: TF-IDF and evaluation
- NumPy: Numerical computation
- JSON: Dataset and result storage
- PyPDF: Document processing
- LLM: Answer generation
