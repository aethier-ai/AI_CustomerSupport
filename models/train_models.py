import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/historical_tickets.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(df["ticket_text"].tolist())

# Train Urgency Regressor
urgency_model = LinearRegression()
urgency_model.fit(X, df["urgency"])
pickle.dump(urgency_model, open("models/urgency_model.pkl", "wb"))

# Train Action Classifier
le = LabelEncoder()
y_action = le.fit_transform(df["suggested_action"])
action_model = RandomForestClassifier()
action_model.fit(X, y_action)

pickle.dump(action_model, open("models/action_model.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))
