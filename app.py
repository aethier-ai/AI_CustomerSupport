import streamlit as st
import numpy as np
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load models
urgency_model = pickle.load(open("models/urgency_model.pkl", "rb"))
action_model = pickle.load(open("models/action_model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# Load FAISS index & historical data
index = faiss.read_index("retrieval/faiss_index.bin")
historical_data = pd.read_csv("data/historical_tickets.csv")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📬 Support Ticket Triage Assistant")
ticket_input = st.text_area("Enter support ticket text")

if st.button("Analyze"):
    input_vec = embedder.encode([ticket_input])

    # Urgency prediction
    urgency = urgency_model.predict(input_vec)[0]
    st.subheader(f"Urgency Score: {round(urgency, 2)} / 5")

    # Action prediction
    action_encoded = action_model.predict(input_vec)[0]
    action = label_encoder.inverse_transform([action_encoded])[0]
    st.subheader(f"Suggested Action: {action}")

    # Similar tickets
    D, I = index.search(np.array(input_vec), 5)
    st.markdown("### 🔎 Similar Tickets:")
    for idx in I[0]:
        st.markdown(f"- {historical_data.iloc[idx]['ticket_text']}")
