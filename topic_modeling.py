from bertopic import BERTopic
import openai
from bertopic.representation import OpenAI
from bertopic.representation import KeyBERTInspired


def train_topic_model(docs, embeddings=None):
    """
    Train a BERTopic model on the given documents.
    Optionally uses pre-computed document embeddings for faster training.
    Returns the trained BERTopic model and the list of topic assignments for each document.
    """
    client = openai.OpenAI(
        api_key='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')

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