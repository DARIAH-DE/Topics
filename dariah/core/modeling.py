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

import cophi
import lda
import numpy as np
import pandas as pd

from dariah.mallet import MALLET
from dariah.core import utils


logging.getLogger("lda").setLevel(logging.WARNING)


class LDA:
    def __init__(
        self,
        num_topics,
        num_iterations=1000,
        alpha=0.1,
        eta=0.01,
        random_state=None,
        mallet=None,
    ):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.eta = eta
        self.mallet = mallet
        if self.mallet:
            if not Path(self.mallet).is_file():
                raise OSError(
                    "'{}' is not a file. "
                    "Point to the 'mallet/bin/mallet' file.".format(self.mallet)
                )
            if not Path(self.mallet).exists():
                # Check if MALLET is in environment variable:
                if not os.environ.get(self.mallet):
                    raise OSError(
                        "MALLET executable was not found. "
                        "'{}' does not exist".format(self.mallet)
                    )
        else:
            self._model = lda.LDA(
                n_topics=self.num_topics,
                n_iter=self.num_iterations,
                alpha=self.alpha,
                eta=self.eta,
            )

    def fit(self, dtm):
        """Fit the LDA model.
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
            return self._mallet_topics()
        else:
            return self._riddell_topics()

    @property
    def topic_word(self):
        """Topic-word distributions.
        """
        if self.mallet:
            return self._mallet_topic_word()
        else:
            return self._riddell_topic_word()

    @property
    def topic_document(self):
        """Topic-document distributions.
        """
        if self.mallet:
            return self._mallet_topic_document()
        else:
            return self._riddell_topic_document()

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
    def _similarities(data):
        """Cosine simliarity matrix.
        """
        descriptors = data.columns
        d = data.T @ data
        norm = (data * data).sum(0) ** 0.5
        similarities = d / norm / norm.T
        return pd.DataFrame(similarities, index=descriptors, columns=descriptors)

    def _riddell_lda(self, dtm):
        self._model.fit(dtm)

    def _riddell_topics(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = [f"word{n}" for n in range(200)]
        topics = [
            np.array(self._vocabulary)[np.argsort(dist)][: -200 - 1 : -1]
            for dist in self._model.topic_word_
        ]
        return pd.DataFrame(topics, index=index, columns=columns)

    def _riddell_topic_word(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        return pd.DataFrame(
            self._model.topic_word_, index=index, columns=self._vocabulary
        )

    def _riddell_topic_document(self) -> pd.DataFrame:
        index = [f"topic{n}" for n in range(self.num_topics)]
        return pd.DataFrame(
            self._model.doc_topic_, index=self._documents, columns=index
        ).T

    def _mallet_lda(self, dtm):
        # Get number of CPUs for threaded processing:
        cpu = multiprocessing.cpu_count() - 1

        # Get temporary directory to dump corpus files:
        self._tempdir = Path(tempfile.gettempdir(), "dariah-topics")
        if not self._tempdir.exists():
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
        )

    def _mallet_topics(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = [f"word{n}" for n in range(200)]
        topics = utils.read_topics_file(self._topics_file)
        return pd.DataFrame(topics, index=index, columns=columns)

    def _mallet_topic_word(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        data = pd.read_csv(self._topic_word_file, sep="\t", header=None).dropna()
        data = data.pivot(index=0, columns=1, values=2)
        data.columns.name = None
        data.index.name = None
        data.index = index
        return data

    def _mallet_topic_document(self):
        data = pd.read_csv(self._topic_document_file, sep="\t", header=None)
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = data[1]
        data = data.drop([0, 1], axis=1)
        data.columns = list(columns)
        data.index = index
        return data

    def __repr__(self):
        return (
            f"<Model: LDA, "
            f"{self.num_topics} topics, "
            f"{self.num_iterations} iterations, "
            f"alpha={self.alpha}, "
            f"eta={self.eta}>"
        )
