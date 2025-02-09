import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def load_models():
    """Load SBERT and BERTopic models."""
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    bertopic_model = BERTopic(min_topic_size=5, calculate_probabilities=True, verbose=True)
    return sbert_model, bertopic_model

sbert_model, bertopic_model = load_models()

def fetch_main_content(url):
    """Fetch and extract the main content from a given URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
        return " ".join(paragraphs)
    except requests.RequestException:
        return ""

def filter_topics(topic_words):
    """Remove stopwords and low-weighted topic words."""
    return {word for word, weight in topic_words if word.lower() not in ENGLISH_STOP_WORDS and weight > 0.05}

def compute_similarity(target_embedding, corpus_embeddings):
    """Compute cosine similarity between target and corpus embeddings."""
    similarities = util.cos_sim(target_embedding, corpus_embeddings)
    return similarities.numpy().tolist()[0]

def compute_conceptual_originality(target_text, corpus_texts):
    """Compute topic-based originality using BERTopic."""
    if len(corpus_texts) < 2:
        st.warning("Not enough documents for BERTopic, defaulting to full originality.")
        return 1.0
    
    try:
        all_texts = [target_text] + corpus_texts
        topics, _ = bertopic_model.fit_transform(all_texts)
        target_topic_words = filter_topics(bertopic_model.get_topic(topics[0]))
        corpus_topic_words = [filter_topics(bertopic_model.get_topic(t)) for t in topics[1:] if t != -1]
        
        jaccard_similarities = [
            len(target_topic_words & corpus_topic) / max(len(target_topic_words | corpus_topic), 1)
            for corpus_topic in corpus_topic_words
        ]
        conceptual_originality = 1 - max(jaccard_similarities) if jaccard_similarities else 1.0
    
        # Debugging output
        st.subheader("Topic Analysis")
        st.write("Target Page Topics (Filtered):", target_topic_words)
        for i, corpus_topic in enumerate(corpus_topic_words):
            st.write(f"Corpus Page {i+1} Topics (Filtered):", corpus_topic)
            st.write(f"Jaccard Similarity with Target: {1 - conceptual_originality:.4f}")
    
        return conceptual_originality
    except Exception as e:
        st.error(f"BERTopic failed: {e}. Defaulting to full originality.")
        return 1.0

def compute_hybrid_originality(target_text, corpus_texts, alpha=0.5):
    """Compute hybrid originality score by combining SBERT semantic similarity and BERTopic conceptual originality."""
    if not corpus_texts:
        return 1.0  # Default originality score if no corpus is provided
    
    # Compute semantic originality
    corpus_combined = " ".join(corpus_texts)
    corpus_embeddings = sbert_model.encode([corpus_combined] + corpus_texts, convert_to_tensor=True)
    target_embedding = sbert_model.encode(target_text, convert_to_tensor=True)
    similarities = compute_similarity(target_embedding, corpus_embeddings)
    avg_similarity = np.mean(similarities)
    semantic_originality = 1 - avg_similarity  # Higher means more original
    
    # Compute conceptual originality
    conceptual_originality = compute_conceptual_originality(target_text, corpus_texts)
    
    # Combine both scores
    hybrid_originality = alpha * semantic_originality + (1 - alpha) * conceptual_originality
    
    # Debugging output
    st.subheader("Originality Breakdown")
    st.write(f"Semantic Originality (SBERT): {semantic_originality:.4f}")
    st.write(f"Conceptual Originality (BERTopic): {conceptual_originality:.4f}")
    st.write(f"Final Hybrid Originality Score: {hybrid_originality:.4f}")
    
    return hybrid_originality

# Streamlit UI
st.title("Web Page Originality Score")
target_url = st.text_input("Enter target page URL")
corpus_urls = st.text_area("Enter corpus URLs (one per line)").split("\n")
alpha = st.slider("Adjust weight between SBERT (semantic) and BERTopic (conceptual) originality", 0.0, 1.0, 0.5)

if st.button("Calculate Originality"):
    with st.spinner("Fetching content and computing originality..."):
        target_text = fetch_main_content(target_url)
        corpus_texts = [fetch_main_content(url) for url in corpus_urls if url.strip()]
        originality_score = compute_hybrid_originality(target_text, corpus_texts, alpha)
        st.success(f"Hybrid Originality Score: {originality_score:.4f}")
