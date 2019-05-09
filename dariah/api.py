"""
dariah.api
~~~~~~~~~~

This module implements the high-level API.
"""

from pathlib import Path

import cophi

from dariah.core import LDA, Vis


def topics(directory, stopwords, num_topics, num_iterations, **kwargs):
    """Train a topic model.

    Parameters:
        directory (str): Path to corpus directory.
        stopwords (int, list): Either a threshold for most frequent words,
            or a list of stopwords.
        num_topics (int): Number of topics.
        num_iterations (int): Number of iterations.
        alpha (float): TODO. Defaults to 0.1.
        eta (float): TODO. Defaults to 0.01.
        random_state (int): TODO. Defaults to None.
        mallet (str): TODO. Defaults to None.

    Returns:
        A topic model and its visualizations.
    """
    # Construct a corpus object:
    corpus = cophi.corpus(directory, metadata=False)
    # Get stopwords and hapax legomena:
    if isinstance(stopwords, int):
        stopwords = corpus.mfw(stopwords)
    stopwords = stopwords + corpus.hapax
    # Remove them from corpus:
    dtm = corpus.drop(corpus.dtm, stopwords)
    # Construct LDA object:
    model = LDA(num_topics=num_topics, num_iterations=num_iterations, **kwargs)
    # Train model:
    model.fit(dtm)
    # Construct visualization object:
    vis = Vis(model)
    return model, vis
