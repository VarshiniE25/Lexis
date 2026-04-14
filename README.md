# 🚀 Lexis — AI-Powered Contract Intelligence System

## 🧠 Overview
Lexis is an AI-powered contract intelligence system that leverages **LLMs and Retrieval-Augmented Generation (RAG)** to analyze legal documents and extract structured insights such as clauses, risks, and key entities.

It transforms unstructured contracts into actionable intelligence, enabling faster and more reliable decision-making.

---

## 🎯 Problem Statement
Manual contract analysis is:
- Time-consuming and inefficient  
- Prone to human error  
- Difficult to scale across large document sets  

Lexis addresses this by providing an **automated, scalable, and context-aware contract analysis pipeline**.

---

## ⚙️ System Architecture (RAG Pipeline)

1. **Document Ingestion**
   - Upload and parse contract documents  

2. **Text Chunking**
   - Split documents into semantically meaningful chunks  

3. **Embedding Generation**
   - Convert text into vector representations  

4. **Vector Storage**
   - Store embeddings in a vector database for retrieval  

5. **Retrieval**
   - Fetch relevant chunks based on query/context  

6. **LLM Reasoning**
   - Use LLM to generate structured outputs:
     - Clause extraction  
     - Risk identification  
     - Entity recognition  

---

## 💡 Key Features
- 📄 Clause Extraction (legal structure understanding)  
- ⚠️ Risk Detection (context-aware analysis)  
- 🔍 Entity & Obligation Extraction  
- 🤖 LLM-powered semantic reasoning  
- ⚡ Scalable document processing pipeline  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **LLM / NLP:** chatgpt nano 5  
- **Frameworks:** Streamlit  
- **Vector DB:**  FAISS 
- **Tools:** Git, VS Code  

---


---

## 🚀 How to Run

```bash
git clone https://github.com/VarshiniE25/Lexis.git
cd Lexis

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python app.py
