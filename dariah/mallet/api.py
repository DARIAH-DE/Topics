"""
dariah.mallet.api
~~~~~~~~~~~~~~~~

This module implements the high-level API to communicate with
the CLI interface of MALLET.
"""

from dariah.mallet import core


class MALLET:
    """Machine Learning for Language Toolkit (MALLET).
    """

    def __init__(self, executable):
        self.executable = executable

    def import_dir(self, **parameters):
        """Load contents of a directory into MALLET instances.
        """
        return core.call("import-dir", self.executable, **parameters)

    def import_file(self, **parameters):
        """Load a file into MALLET instances.
        """
        return core.call("import-file", self.executable, **parameters)

    def import_svmlight(self, **parameters):
        """Load SVMLight data files into MALLET instances.
        """
        return core.call("import-svmlight", self.executable, **parameters)

    def info(self, **parameters):
        """Get information about MALLET instances.
        """
        return core.call("info", self.executable, **parameters)

    def train_classifier(self, **parameters):
        """Train a classifier from MALLET data files.
        """
        return core.call("train-classifier", self.executable, **parameters)

    def classify_dir(self, **parameters):
        """Classify the contents of a directory with a saved classifier.
        """
        return core.call("classify-dir", self.executable, **parameters)

    def classify_file(self, **parameters):
        """Classify data from a single file with a saved classifier.
        """
        return core.call("classify-file", self.executable, **parameters)

    def classify_svmlight(self, **parameters):
        """Classify data from a single file in SVMLight format.
        """
        return core.call("classify-svmlight", self.executable, **parameters)

    def train_topics(self, **parameters):
        """Train a topic model from MALLET data files.
        """
        return core.call("train-topics", self.executable, **parameters)

    def infer_topics(self, **parameters):
        """Use a trained topic model to infer topics for new documents.
        """
        return core.call("infer-topics", self.executable, **parameters)

    def evaluate_topics(self, **parameters):
        """Estimate the probability of new documents under a trained model.
        """
        return core.call("evaluate-topics", self.executable, **parameters)

    def prune(self, **parameters):
        """Remove features based on frequency or information gain.
        """
        return core.call("prune", self.executable, **parameters)

    def split(self, **parameters):
        """Divide data into testing, training, and validation portions.
        """
        return core.call("split", self.executable, **parameters)

    def bulk_load(self, **parameters):
        """For big input files, efficiently prune vocabulary and import docs.
        """
        return core.call("bulk-load", self.executable, **parameters)
