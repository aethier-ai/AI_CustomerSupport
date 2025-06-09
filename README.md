# 📬 Support Ticket Triage Assistant (PoC)

A demo app that analyzes unstructured support tickets and automatically:

✅ Predicts urgency (1–5)  
✅ Suggests next best action  
✅ Retrieves 3–5 similar historical tickets

## Built With:
- Streamlit for UI
- Sentence Transformers for text embeddings
- Scikit-learn for ML models
- FAISS for fast similarity search

## Setup Instructions:

1. Clone the repo:
   git clone https://github.com/aethier-ai/AI_CustomerSupport.git
   cd AI_CustomerSupport

2. Install requirements:
   pip install -r requirements.txt

3. Add or edit your data in data/historical_tickets.csv

4. Train models and build index:
   python models/train_models.py  
   python retrieval/build_index.py

5. Run the app:
   streamlit run app.py

## Folder Structure

AI_CustomerSupport/
├── app.py
├── data/
├── models/
├── retrieval/
├── utils/
├── requirements.txt
└── README.md
