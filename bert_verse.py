import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import torch
torch.classes.__path__ = []  # Neutralizes the path inspection
from transformers import AutoTokenizer, AutoModel
import gdown
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables problematic inspection

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit Page Configuration
st.set_page_config(page_title="Bible Verse Recommender", layout="wide")
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    book_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi', 40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter',
        62: '1 John', 63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }

    # Map book numbers to names
    data['Book Name'] = data['b'].map(book_names)

    # Process text: Remove stopwords
    stop_words = ['all', 'to', 'any', "he'd", "we've", 'this', 'have', 'whom', "isn't", "wasn't", 'own', 'now', 'do', "mightn't", 'but', 'yourselves', 'under', "i've", 'his', 'is', "haven't", 'over', 'doesn', "he's", 'her', 'your', "you've", 'each', 'the', "she'll", 'did', "you'll", 'until', "wouldn't", 'during', 'some', 'he', 'than', "didn't", 'then', 'with', 'had', "it's", 'and', "should've", 'few', "it'll", 'there', 'which', 'why', "we're", 'should', 'other', "i'll", 'an', 'been', 'herself', "needn't", 'above', "hasn't", 'both', 'will', 'only', "we'll", 'before', 'here', "we'd", 'again', 'what', "you'd", "shouldn't", 'has', 'me', "i'd", 'were', "aren't", 'so', "she's", "hadn't", 'she', 'o', 'from', 'on', 'ours', "they've", 'very', "don't", 'down', 'further', 'it', 'by', 'once', 'if', 'doing', 'are', 'no', 'i', 'through', 'yours', 'about', "she'd", 'most', 'how', "mustn't", 'as', 'myself', 'being', 'their', 'was', 'between', 'or', 'into', 'when', 'them', "they're", 'him', "couldn't", 'shouldn', 'who', 'my', "doesn't", 'where', 'at', 'off', 'yourself', 'for', 'its', "won't", 'such', "he'll", 'hers', 'be', 'after', 'not', 'same', 'these', 'that', 'below', "shan't", "they'll", 'nor', 'they', 'having', 'too', 'himself', 'those', 'out', "i'm", 'itself', 'just', 'while', 'does', "that'll", 'theirs', "they'd", 'in', 'can', 'of', 'am', 'because', "it'd", 'more', 'you', "weren't", 'we', 'themselves', 'ourselves', 'a', "you're", 'up', 'our', 'against']

    data['corpus'] = data['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )

    return data, book_names

# Load data
data, book_names = load_data()
##################################################################################################################

# **Embeddings & FAISS Index Loading**
@st.cache_resource
def load_embeddings_and_index():
    """Download and load embeddings and FAISS index."""
    
    # Define file paths
    emb_file = "bible_embeddings.npy"
    index_file = "bible_faiss.index"
    
    # Download embeddings if not present
    if not os.path.exists(emb_file):
        gdown.download("https://drive.google.com/uc?id=1-z5RDrWKn13t65PmsWb4FhOGyRcJbOpB", emb_file, quiet=False)
    
    # Download FAISS index if not present
    if not os.path.exists(index_file):
        gdown.download("https://drive.google.com/uc?id=1I7sqgWmMjFcjqDVic73IMPXK8tehcX-A", index_file, quiet=False)

    # Load files
    embeddings = np.load(emb_file, allow_pickle=True)
    index = faiss.read_index(index_file)

    return embeddings, index

embeddings, index = load_embeddings_and_index()

##################################################################################################################
# Load Sentence-BERT model
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
tokenizer, model = load_model()

# Function to get embedding for a new query
def get_embedding(text):
    """Generate embedding for the given text using Sentence-BERT."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().astype(np.float32)

# Find similar verses
def find_similar_verses(query, top_n=5):
    """Find similar Bible verses based on FAISS search."""
    query_embedding = get_embedding(query).reshape(1, -1)
    query_embedding = np.array(query_embedding, dtype=np.float32)  # Ensure NumPy array
    query_embedding = query_embedding.reshape(1, -1)  # Reshape for FAISS search
    
    distances, indices = index.search(query_embedding, top_n)

    if indices is None or len(indices[0]) == 0:
        st.error("No similar verses found.")
        return pd.DataFrame(columns=["Book Name", "Chapter", "Verse", "Text", "Similarity"])

    # Extract results
    results = data.iloc[indices[0]][["Book Name", "c", "v", "t"]].copy()
    
    # Ensure distances is not empty and matches the indices size
    if len(distances) > 0 and len(distances[0]) == len(results):
        results["Similarity"] = 1 - distances[0]  # Convert distance to similarity
    else:
        results["Similarity"] = np.nan  # Fallback if distances are empty
    
    return results

##################################################################################################################

# **Streamlit UI**

st.title("📖 Bible Verse Similarity Finder")
query = st.text_input("Enter a phrase or verse:", "Love your neighbor as yourself")
top_n = st.slider("Number of similar verses:", min_value=1, max_value=50, value=10, step=5)

if st.button("Find Similar Verses"):
    results = find_similar_verses(query, top_n)
    st.write("### 🔍 Similar Verses:")
    for i, row in results.iterrows():
        st.write(f"**Book:** {row['Book Name']} | **Chapter:** {row['c']} | **Verse:** {row['v']}")
        st.write(f"**Text:** {row['t']} (Similarity: {row['Similarity']:.2f})")
    


