import streamlit as st
import pandas as pd
import openai
import os
import torch

from utils import parse_pdf, chunk_text, extract_year
from rag_retrieval import DocumentIndex
from topic_modeling import train_topic_model, save_topic_model, load_topic_model
from visualization import plot_topic_clusters, plot_documents_clusters, plot_topic_bar, plot_topics_over_time

torch.classes.__path__ = []

#openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY


# Streamlit App Configuration
st.set_page_config(page_title="pdf Document Explorer")

st.title("ACI Policy Handbook - Enhanced RAG System")
st.write("Upload one or more PDF files, process them into a knowledge base, and explore topics or ask questions.")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Buttons for processing or loading sessions
col1, col2 = st.columns(2)
with col1:
    process_clicked = st.button("Process Documents")
with col2:
    load_clicked = st.button("Load Latest Session")

# Initialize or retrieve persisted session state
if "doc_index" not in st.session_state:
    st.session_state.doc_index = None
    st.session_state.topic_model = None
    st.session_state.docs = None
    st.session_state.years = None
    st.session_state.topics = None
    st.session_state.topics_over_time = None

doc_index = st.session_state.doc_index
topic_model = st.session_state.topic_model
docs = st.session_state.docs
years = st.session_state.years
topics = st.session_state.topics
topics_over_time_df = st.session_state.topics_over_time

# Process new documents
if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before processing.")
    else:
        st.write("Processing documents... (this may take a minute)")
        # Parse all uploaded PDFs and prepare documents and metadata
        docs = []      # full text of each chunk
        metadata = []  # metadata for each chunk
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            text = parse_pdf(uploaded_file)
            if not text or text.strip() == "":
                st.warning(f"No text extracted from {file_name}. It may be an image-based PDF.")
                continue
            year = extract_year(file_name) or None
            # Split text into chunks and create metadata for each
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                metadata.append({
                    "source": file_name,
                    "year": year,
                    "text": chunk
                })
        # Build document index with embeddings
        doc_index = DocumentIndex()
        doc_index.client = openai.OpenAI(api_key=OPENAI_API_KEY)
#        try:
        doc_index.build_index(docs, metadata_list=metadata)
        #except Exception as e:
        #    st.error(f"Error during embedding/indexing: {e}")
        #    st.stop()
        # Perform topic modeling on the document chunks
        #try:
            # Use the same embeddings for topic modeling if available to avoid re-computation
        topic_model, topics = train_topic_model(docs, api_key=OPENAI_API_KEY,embeddings=doc_index.embeddings)
        #except Exception as e:
        #    st.error(f"Error during topic modeling: {e}")
        #    st.stop()

        # Compute topics over time data
        years = [m.get("year") for m in metadata]
        #try:
        topics_over_time_df = topic_model.topics_over_time(docs, years, global_tuning=True, evolution_tuning=True)
        #except Exception as e:
            # If topics_over_time fails (e.g., no year data), handle gracefully
            #topics_over_time_df = pd.DataFrame()
        # Store results in session state for reuse
        st.session_state.doc_index = doc_index
        st.session_state.topic_model = topic_model
        st.session_state.docs = docs
        st.session_state.years = years
        st.session_state.topics = topics
        st.session_state.topics_over_time = topics_over_time_df
        st.success("Documents processed successfully.")

# Load existing session from disk
if load_clicked:
    st.write("Loading saved session...")
    doc_index = DocumentIndex()
    doc_index.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        doc_index.load("session_data")
    except Exception as e:
        st.error("No saved session found. Please process and save a session first.")
        st.stop()
    try:
        topic_model = load_topic_model("session_data/topic_model.bin")
    except Exception as e:
        st.error("Topic model file not found in saved session.")
        st.stop()
    # Retrieve stored docs and years from metadata
    docs = [m.get("text", "") for m in doc_index.metadata]
    years = [m.get("year") for m in doc_index.metadata]
    # Get topic assignments for each document (using the loaded model)
    try:
        topics, _ = topic_model.transform(docs, embeddings=doc_index.embeddings)
    except Exception:
        # If transform is not available, assume topics were stored in metadata (if saved after processing)
        topics = [m.get("topic", -1) for m in doc_index.metadata]
    # Compute topics over time
    #try:
    topics_over_time_df = topic_model.topics_over_time(docs, years, global_tuning=True, evolution_tuning=True)
    #except Exception:
    #    topics_over_time_df = pd.DataFrame()
    # Update session state
    st.session_state.doc_index = doc_index
    st.session_state.topic_model = topic_model
    st.session_state.docs = docs
    st.session_state.years = years
    st.session_state.topics = topics
    st.session_state.topics_over_time = topics_over_time_df
    st.success("Session loaded successfully.")

# If we have an active document index and topic model, enable exploration UI
if doc_index and topic_model:
    st.subheader("Topic Exploration")
    # Plot topic clusters (2D visualization of document embeddings)
    try:
        cluster_fig = plot_topic_clusters(topic_model)
        st.plotly_chart(cluster_fig, use_container_width=True)
    except Exception as e:
        st.write("Error in cluster visualization:", e)

    # Plot document clusters (2D visualization of document embeddings)
    try:
        document_fig = plot_documents_clusters(topic_model, docs, doc_index.embeddings)
        st.plotly_chart(document_fig, use_container_width=True)
    except Exception as e:
        st.write("Error in cluster visualization:", e)

    # Plot topic barchart
    #try:
    #    bar_fig = plot_topic_bar(topic_model)
    #    st.plotly_chart(bar_fig, use_container_width=True)
    #except Exception as e:
    #    st.write("Error in cluster visualization:", e)

    # Plot topic evolution over time (line chart)
    if not topics_over_time_df.empty:
        # Allow user to select number of top topics to display
        top_n = st.slider("Top N Topics to Display in Trends", 1, 20, 5)
        timeline_fig = plot_topics_over_time(topics_over_time_df, topic_model, top_n_topics=top_n)
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.write("No temporal data available for topic trends.")
    # Save session button (to save processed index and model)
    if st.button("Save Session"):
        try:
            doc_index.save("session_data")
            save_topic_model(topic_model, "session_data/topic_model.bin")
            st.success("Session saved to disk.")
        except Exception as e:
            st.error(f"Failed to save session: {e}")

    st.subheader("Ask Questions")
    query = st.text_input("Enter a question about the annual reviews:")
    if query:
        # Perform retrieval-Augmented Q&A using OpenAI
        try:
            results = doc_index.query(query, top_k=3)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            results = []
        # Build context from top results
        context_sections = []
        for res in results:
            year = res["metadata"].get("year")
            source = res["metadata"].get("source", "Document")
            text_snippet = res["text"]
            context_sections.append(f"**{source} ({year})**: {text_snippet}")
        context_content = "\n\n".join(context_sections)
        # Prepare OpenAI ChatGPT prompt with context
        system_msg = {"role": "system", "content": "You are an expert assistant answering questions using IATA Annual Review documents. Only use the provided context to answer."}
        user_msg = {"role": "user", "content": f"Context:\n{context_content}\n\nQuestion: {query}\nAnswer:"}
        answer = ""
        try:

            response = openai.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[system_msg, user_msg],
                temperature=0.2,  # low randomness to keep answers factual
                max_tokens=500  # adjust as needed to allow sufficiently long answers
            )
            answer = response.choices[0].message.content

            #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[system_msg, user_msg])
            #answer = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            answer = f"(Failed to get answer: {e})"
        st.markdown("**Answer:** " + answer)
        # Optionally, display the context that was used for transparency
        with st.expander("Retrieved context"):
            for res in results:
                year = res["metadata"].get("year")
                source = res["metadata"].get("source", "Document")
                snippet = res["text"][:200].strip()
                st.write(f"**{source} ({year})** snippet: {snippet}...")
