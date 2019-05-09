"""
dariah.topics.visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import seaborn as sns


class Vis:
    def __init__(self, model):
        self._model = model
        self.topic_document_ = model.topic_document.copy()
        self.topic_word_ = model.topic_word.copy()
        self.document_similarities_ = model.document_similarities.copy()
        self.topic_similarities_ = model.topic_similarities.copy()

    def topic_document(self, cmap="Blues", annot=False, fmt=".2g", cbar=True, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        topic_document_ = self.topic_document_.reindex(
            sorted(self.topic_document_.columns), axis=1
        )
        if topic_document_.shape[0] < topic_document_.shape[1]:
            topic_document_ = topic_document_.T
        sns.heatmap(
            topic_document_,
            linewidth=0.5,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            cbar=cbar,
        )
        return ax

    def topic_word(
        self, words, cmap="Blues", annot=False, fmt=".2g", cbar=True, **kwargs
    ):
        fig, ax = plt.subplots(**kwargs)
        topic_word_ = self.topic_word_.copy().loc[:, words]
        if topic_word_.shape[0] < topic_word_.shape[1]:
            topic_word_ = topic_word_.T
        sns.heatmap(
            topic_word_,
            linewidth=0.5,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            cbar=cbar,
        )

    def topic(self, name, num_words=10, color="grey", **kwargs):
        fig, ax = plt.subplots(**kwargs)
        data = self.topic_word_.loc[name, :].sort_values(ascending=False)
        data = data[:num_words].sort_values()
        data.plot.barh(ax=ax, color=color)
        return ax

    def document(self, name, color="grey", **kwargs):
        fig, ax = plt.subplots(**kwargs)
        data = self.topic_document_.loc[:, name].sort_values()
        data.plot.barh(ax=ax, color=color)
        return ax

    def document_similarities(
        self, cmap="Blues", annot=False, fmt=".2g", cbar=True, **kwargs
    ):
        fig, ax = plt.subplots(**kwargs)
        sns.heatmap(
            self.document_similarities_,
            linewidth=0.5,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            cbar=cbar,
        )
        return ax

    def document_similarities(
        self, cmap="Blues", annot=False, fmt=".2g", cbar=True, **kwargs
    ):
        fig, ax = plt.subplots(**kwargs)
        sns.heatmap(
            self.document_similarities_,
            linewidth=0.5,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            cbar=cbar,
        )
        return ax

    def topic_similarities(
        self, cmap="Blues", annot=False, fmt=".2g", cbar=True, **kwargs
    ):
        fig, ax = plt.subplots(**kwargs)
        sns.heatmap(
            self.topic_similarities_,
            linewidth=0.5,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            cbar=cbar,
        )
        return ax

    def __repr__(self):
        return (
            f"<Visualization: LDA, "
            f"{self._model.num_topics} topics, "
            f"{self._model.num_iterations} iterations, "
            f"alpha={self._model.alpha}, "
            f"eta={self._model.eta}>"
        )
