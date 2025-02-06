import streamlit as st
import requests
import trafilatura
import csv
from sentence_transformers import SentenceTransformer, util
from io import StringIO
import pandas as pd
import openai
import os

# Use SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sbert_model = load_model()

# OpenAI embedding function
def get_openai_embedding(text, api_key):
    openai.api_key = api_key
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return None

def fetch_main_content(url):
    """Fetches and extracts main content using requests and Trafilatura."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        with st.spinner(f'Fetching content from URL: {url}'):
            response = requests.get(url, headers=headers, timeout=10)
            st.info(f"Response for {url}: {response.status_code} {response.reason}")
            response.raise_for_status()
            text = trafilatura.extract(response.text)
            if text:
                st.success(f"Content fetched and extracted successfully from {url}")
            return text if text else ""
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return ""

def compute_originality(target_text, corpus_texts, corpus_urls, use_sbert=True, openai_api_key=None):
    """Computes originality score using either SBERT or OpenAI embeddings."""
    if not corpus_texts:
        return 1.0, [], 0, "", 0  # If no corpus, assume fully original
    
    # Create aggregated "pool of knowledge"
    corpus_aggregate = " ".join(corpus_texts)
    
    with st.spinner('Computing embeddings and similarities...'):
        if use_sbert:
            # SBERT embedding logic
            target_embedding = sbert_model.encode(target_text, convert_to_tensor=True)
            corpus_embeddings = sbert_model.encode(corpus_texts, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(target_embedding, corpus_embeddings).squeeze().tolist()
        else:
            # OpenAI embedding logic
            target_embedding = get_openai_embedding(target_text, openai_api_key)
            if target_embedding is None:
                st.error("Failed to get OpenAI embedding for target text")
                return None, None, None, None, None
                
            corpus_embeddings = []
            for text in corpus_texts:
                emb = get_openai_embedding(text, openai_api_key)
                if emb is None:
                    st.error("Failed to get OpenAI embedding for corpus text")
                    return None, None, None, None, None
                corpus_embeddings.append(emb)
            
            # Compute cosine similarities using numpy
            import numpy as np
            target_embedding = np.array(target_embedding)
            corpus_embeddings = np.array(corpus_embeddings)
            similarities = [
                np.dot(target_embedding, corp_emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(corp_emb))
                for corp_emb in corpus_embeddings
            ]
        
        # Calculate statistics
        max_similarity = max(similarities) if similarities else 0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        most_similar_index = similarities.index(max_similarity) if similarities else -1
        most_similar_url = corpus_urls[most_similar_index] if most_similar_index != -1 else ""
        
        originality_score = 1 - max_similarity
        
        return originality_score, similarities, avg_similarity, most_similar_url, len(target_text.split())

def main():
    st.title("Content Originality Checker")
    st.write("Compare a target URL's content against a corpus of URLs to check originality")

    # Embedding method selection
    use_sbert = not st.checkbox("Use OpenAI Embeddings", False)
    
    # Show API key input if OpenAI is selected
    openai_api_key = None
    if not use_sbert:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use their embedding model"
        )
        if not openai_api_key:
            st.warning("Please enter an OpenAI API key to use OpenAI embeddings")

    # Rest of the input UI
    input_method = st.radio(
        "Choose input method for comparison URLs:",
        ["Text Input", "File Upload"]
    )

    corpus_urls = []
    if input_method == "Text Input":
        urls_input = st.text_area(
            "Enter URLs (one per line):",
            height=150,
            help="Enter the URLs you want to compare against, one URL per line"
        )
        if urls_input:
            corpus_urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file with URLs (one per line)",
            type=['txt']
        )
        if uploaded_file:
            corpus_urls = [url.strip() for url in uploaded_file.getvalue().decode().split('\n') if url.strip()]

    target_url = st.text_input("Enter the target URL to check:", help="This is the URL you want to check for originality")

    if st.button("Check Originality") and target_url and corpus_urls:
        # Validate OpenAI API key if selected
        if not use_sbert and not openai_api_key:
            st.error("Please provide an OpenAI API key to use OpenAI embeddings")
            return

        with st.spinner('Fetching content from URLs...'):
            corpus_texts = [fetch_main_content(url) for url in corpus_urls]
            valid_corpus = [(url, text) for url, text in zip(corpus_urls, corpus_texts) if text]
            
            if not valid_corpus:
                st.error("No valid content could be fetched from the corpus URLs.")
                return
                
            corpus_urls, corpus_texts = zip(*valid_corpus)
            
            target_text = fetch_main_content(target_url)
            if not target_text:
                st.error("Failed to fetch target content.")
                return

        # Compute originality with selected embedding method
        results = compute_originality(
            target_text,
            corpus_texts,
            corpus_urls,
            use_sbert=use_sbert,
            openai_api_key=openai_api_key
        )
        
        if results[0] is None:  # Check if computation failed
            return
            
        originality_score, similarities, avg_similarity, most_similar_url, word_count = results

        # Display results
        st.header("Results")
        st.write(f"Using {'SBERT' if use_sbert else 'OpenAI'} embeddings")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Originality Score", f"{originality_score:.2%}")
        with col2:
            st.metric("Average Similarity", f"{avg_similarity:.2%}")
        with col3:
            st.metric("Word Count", word_count)

        # Most similar URL
        if most_similar_url:
            st.subheader("Most Similar Content")
            st.write(f"URL: {most_similar_url}")

        # Detailed results table
        st.subheader("Detailed Results")
        results_df = pd.DataFrame({
            "URL": [target_url] + list(corpus_urls),
            "Type": ["Target"] + ["Corpus"] * len(corpus_urls),
            "Originality Score": [originality_score] + [1 - sim for sim in similarities],
            "Similarity to Target": [1.0] + list(similarities),
            "Word Count": [word_count] + [len(text.split()) for text in corpus_texts]
        })
        
        st.dataframe(results_df)

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="originality_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
