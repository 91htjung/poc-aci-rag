import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic


def plot_topic_clusters(topic_model):
    fig = topic_model.visualize_topics()
    return fig

def plot_documents_clusters(topic_model, docs, embeddings):
    fig = topic_model.visualize_document_datamap(docs, embeddings=embeddings, width=1800, height=900)
    return fig

def plot_topic_bar(topic_model):
     fig = topic_model.visualize_barchart()
     return fig
 
def plot_topics_over_time(topics_over_time_df, topic_model, top_n_topics=10):
    fig = topic_model.visualize_topics_over_time(topics_over_time_df, top_n_topics=top_n_topics)
    return fig
