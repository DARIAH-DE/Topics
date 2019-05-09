"""
dariah.topics.modeling
~~~~~~~~~~~~~~~~~~~~~~

This module implements low-level LDA modeling functions.
"""

from pathlib import Path
import tempfile
import os
import logging
import multiprocessing
import shutil
from typing import Optional, Union

import cophi
import lda
import numpy as np
import pandas as pd

from dariah.mallet import MALLET
from dariah.core import utils


logging.getLogger("lda").setLevel(logging.WARNING)


class LDA:
    """Latent Dirichlet allocation.

    Args:
        num_topics: The number of topics.
        num_iterations: The number of iterations.
        alpha:
        eta:
        random_state:
        mallet:
    """

    def __init__(
        self,
        num_topics: int,
        num_iterations: int = 1000,
        alpha: float = 0.1,
        eta: float = 0.01,
        random_state: int = None,
        mallet: Optional[Union[str, Path]] = None,
    ) -> None:
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.eta = eta
        self.random_state = random_state
        self.mallet = mallet
        if mallet:
            if not Path(self.mallet).exists():
                # Check if MALLET is in environment variable:
                if not os.environ.get(self.mallet):
                    raise OSError(
                        "MALLET executable was not found. "
                        "'{}' does not exist".format(self.mallet)
                    )
                self.mallet = os.environ.get(self.mallet)
            if not Path(self.mallet).is_file():
                raise OSError(
                    "'{}' is not a file. "
                    "Point to the 'mallet/bin/mallet' file.".format(self.mallet)
                )
        else:
            self._model = lda.LDA(
                n_topics=self.num_topics,
                n_iter=self.num_iterations,
                alpha=self.alpha,
                eta=self.eta,
                random_state=self.random_state,
            )

    def fit(self, dtm: pd.DataFrame) -> None:
        """Fit the model.

        Args:
            dtm: The document-term matrix.
        """
        self._vocabulary = list(dtm.columns)
        self._documents = list(dtm.index)
        dtm = dtm.fillna(0).astype(int)
        if self.mallet:
            self._mallet_lda(dtm)
        else:
            self._riddell_lda(dtm.values)

    @property
    def topics(self):
        """Topics with 200 top words.
        """
        if self.mallet:
            return self._mallet_topics
        else:
            return self._riddell_topics

    @property
    def topic_word(self):
        """Topic-word distributions.
        """
        if self.mallet:
            return self._mallet_topic_word
        else:
            return self._riddell_topic_word

    @property
    def topic_document(self):
        """Topic-document distributions.
        """
        if self.mallet:
            return self._mallet_topic_document
        else:
            return self._riddell_topic_document

    @property
    def topic_similarities(self):
        """Topic similarity matrix.
        """
        data = self.topic_document.T.copy()
        return self._similarities(data)

    @property
    def document_similarities(self):
        """Document similarity matrix.
        """
        data = self.topic_document.copy()
        return self._similarities(data)

    @staticmethod
    def _similarities(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cosine simliarity matrix.

        Args:
            data: A matrix to calculate similarities for.

        Returns:
            A similarity matrix.
        """
        descriptors = data.columns
        d = data.T @ data
        norm = (data * data).sum(0) ** 0.5
        similarities = d / norm / norm.T
        return pd.DataFrame(similarities, index=descriptors, columns=descriptors)

    def _riddell_lda(self, dtm: pd.DataFrame) -> None:
        """Fit the Riddell LDA model.

        Args:
            dtm: The document-term matrix.
        """
        self._model.fit(dtm)

    @property
    def _riddell_topics(self):
        """Topics of the Riddell LDA model.
        """
        maximum = len(self._vocabulary)
        num_words = 200 if maximum > 200 else maximum
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = [f"word{n}" for n in range(num_words)]
        topics = [
            np.array(self._vocabulary)[np.argsort(dist)][: -num_words - 1 : -1]
            for dist in self._model.topic_word_
        ]
        return pd.DataFrame(topics, index=index, columns=columns)

    @property
    def _riddell_topic_word(self):
        """Topic-word distributions for Riddell LDA model.
        """
        index = [f"topic{n}" for n in range(self.num_topics)]
        return pd.DataFrame(
            self._model.topic_word_, index=index, columns=self._vocabulary
        )

    @property
    def _riddell_topic_document(self):
        """Topic-document distributions for Riddell LDA model.
        """
        index = [f"topic{n}" for n in range(self.num_topics)]
        return pd.DataFrame(
            self._model.doc_topic_, index=self._documents, columns=index
        ).T

    def _mallet_lda(self, dtm: pd.DataFrame) -> None:
        """Fit the MALLET LDA model.

        Args:
            dtm: The documen-term matrix.
        """
        # Get number of CPUs for threaded processing:
        cpu = multiprocessing.cpu_count() - 1

        # Get temporary directory to dump corpus files:
        self._tempdir = Path(tempfile.gettempdir(), "dariah-topics")
        if self._tempdir.exists():
            shutil.rmtree(str(self._tempdir))
        self._tempdir.mkdir()

        # Export document-term matrix to plaintext files:
        corpus_sequence = Path(self._tempdir, "corpus.sequence")
        cophi.text.utils.export(dtm, corpus_sequence, "plaintext")

        # Construct MALLET object:
        mallet = MALLET(self.mallet)

        # Create a MALLET corpus file:
        corpus_mallet = Path(self._tempdir, "corpus.mallet")
        mallet.import_file(
            input=str(corpus_sequence), output=str(corpus_mallet), keep_sequence=True
        )

        # Construct paths to MALLET output files:
        self._topic_document_file = Path(self._tempdir, "topic-document.txt")
        self._topic_word_file = Path(self._tempdir, "topic-word.txt")
        self._topics_file = Path(self._tempdir, "topics.txt")

        # Train topics:
        mallet.train_topics(
            input=str(corpus_mallet),
            num_topics=self.num_topics,
            num_iterations=self.num_iterations,
            output_doc_topics=self._topic_document_file,
            output_topic_keys=self._topics_file,
            topic_word_weights_file=self._topic_word_file,
            alpha=self.alpha,
            beta=self.eta,
            num_top_words=200,
            num_threads=cpu,
            random_seed=self.random_state,
        )

    @property
    def _mallet_topics(self):
        """Topics of MALLET LDA model.
        """
        maximum = len(self._vocabulary)
        num_words = 200 if maximum > 200 else maximum
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = [f"word{n}" for n in range(num_words)]
        topics = utils.read_mallet_topics(self._topics_file, num_words)
        return pd.DataFrame(topics, index=index, columns=columns)

    @property
    def _mallet_topic_word(self):
        """Topic-word distributions of MALLET LDA model.
        """
        index = [f"topic{n}" for n in range(self.num_topics)]
        data = pd.read_csv(self._topic_word_file, sep="\t", header=None).dropna()
        data = data.pivot(index=0, columns=1, values=2)
        data.columns.name = None
        data.index.name = None
        data.index = index
        return data

    @property
    def _mallet_topic_document(self):
        """Topic-document distributions of MALLET LDA model.
        """
        data = pd.read_csv(self._topic_document_file, sep="\t", header=None)
        columns = [f"topic{n}" for n in range(self.num_topics)]
        index = data[1]
        data = data.drop([0, 1], axis=1)
        data.columns = list(columns)
        data.index = index
        return data.T

    def __repr__(self):
        return (
            f"<Model: LDA, "
            f"{self.num_topics} topics, "
            f"{self.num_iterations} iterations, "
            f"alpha={self.alpha}, "
            f"eta={self.eta}>"
        )
