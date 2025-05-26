# 💬 Financial Chatbot for Press Release Analysis

This project presents a **Financial Conversational AI** chatbot designed to answer questions based on corporate press release report, specifically tailored for financial analysis and investor insights. The chatbot uses state-of-the-art NLP techniques to interpret text and table data extracted from company reports.

---

## 📌 Project Overview

- **Objective**: To build a financial chatbot capable of understanding and answering questions from quarterly press release report (Q3) of **Loblaw Companies Limited**.
- **Approach**: Convert report content to **SQuAD format**, train NLP models for **question answering (QA)**, and evaluate performance using standard metrics.

---

## 🧠 Features

- Parses and processes financial press release data (text and tabular).
- Fine-tuned on **BERT**, **RoBERTa**, **DistilBERT**, and **BiLSTM** architectures.
- Supports context-aware **question answering** for financial metrics, performance summaries, and insights.
- Evaluated using **BLEU**, **METEOR**, **ROUGE**, and **BERTScore**.

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Libraries**:  
  - Transformers (Hugging Face)  
  - PyTorch  
  - TensorFlow 
  - NLTK, SpaCy  
  - Scikit-learn  
  - Pandas, NumPy, Matplotlib, Seaborn  
- **NLP Techniques**: Tokenization, Contextual Embeddings, Attention Mechanisms  
- **Evaluation**: BLEU, METEOR, ROUGE, BERTScore


## ▶️ How to Run the Files

Follow the steps below to replicate the project:

- **Data Preparation**
Run data_prep_viz.ipynb to clean and structure the Q3 press release data.

- **Convert to SQuAD Format**
Execute squad_format.ipynb to transform the processed text and tables into SQuAD-style JSON format for model training.

- **View the SQuAD JSON Output**
Check the squad_format.json — this will be the training dataset for the QA models.

- **Model Training**
Use model_training.py to train your models (BERT, RoBERTa, DistilBERT, BiLSTM). You can modify model selection within the notebook.

- **Run the Chatbot App**
Launch the chatbot using the stramlit_app.py
