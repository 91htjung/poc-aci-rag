from bertopic import BERTopic
import openai
from bertopic.representation import OpenAI
from bertopic.representation import KeyBERTInspired
import os


def train_topic_model(docs, api_key, embeddings=None):
    """
    Train a BERTopic model on the given documents.
    Optionally uses pre-computed document embeddings for faster training.
    Returns the trained BERTopic model and the list of topic assignments for each document.
    """
    client = openai.OpenAI(api_key=api_key)

    representation_model = OpenAI(client, model="gpt-4.1-nano-2025-04-14", chat=True)
    topic_model = BERTopic(representation_model=representation_model, nr_topics=20)

    if embeddings is not None:
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    else:
        topics, _ = topic_model.fit_transform(docs)
    return topic_model, topics

def save_topic_model(topic_model, path):
    """
    Save the BERTopic model to the given file path.
    The model can be reloaded later for reuse.
    """
    topic_model.save(path)

def load_topic_model(path):
    """
    Load a BERTopic model from the given file path.
    """
    topic_model = BERTopic.load(path)
    return topic_model
