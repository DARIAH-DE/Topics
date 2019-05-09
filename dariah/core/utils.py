"""
dariah.topics.utils
~~~~~~~~~~~~~~~~~~~

This module implements helper functions for topic modeling.
"""

from typing import Generator, List
from pathlib import Path

import cophi


def read_mallet_topics(path: Path, num_words: int) -> Generator[List[str], None, None]:
    """Read a MALLET topics file.

    Args:
        path: Filepath to the topics file.
        num_words: Number of words for a topic.

    Yields:
        A list of tokens, i.e. a topic.
    """
    with path.open("r", encoding="utf-8") as file:
        for row in file:
            sequence = row.split("\t")[2]
            yield list(cophi.text.utils.find_tokens(sequence))[:200]
