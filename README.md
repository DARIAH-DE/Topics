# Topics – Easy Topic Modeling in Python
[Topics](https://dariah-de.github.io/Topics/) is a Python library for topic modeling. Furthermore, this repository provides a convenient, modular workflow that can be entirely controlled from within a well documented [Jupyter](http://jupyter.org/) notebook. Users not yet familiar with programming in Python can test basic topic modeling in a standalone [GUI demonstrator](https://dariah-de.github.io/TopicsExplorer/), which **does not require a Python interpreter or any extra installations**.

At the moment, this library supports three LDA implementations:
* [lda](http://pythonhosted.org/lda/index.html), which is lightweight and provides basic LDA.
* [MALLET](http://mallet.cs.umass.edu/), which is known to be very robust.
* [Gensim](https://radimrehurek.com/gensim/), which is attractive because of its multi-core support.

## Resources
* [Topics website](https://dariah-de.github.io/Topics/)
* [Topics API documentation](https://dariah-de.github.io/Topics/docs/gen/modules.html)
* [Topics paper](https://dh2017.adho.org/abstracts/411/411.pdf)
* **[Standalone Demonstrator releases](https://github.com/DARIAH-DE/TopicsExplorer/releases/latest)**
* [An introduction to topic modeling using lda](notebooks/IntroducingLda.ipynb)
* [An introduction to topic modeling using MALLET](notebooks/IntroducingMallet.ipynb)
* [An introduction to topic modeling using Gensim](notebooks/IntroducingGensim.ipynb)
* [A demonstration of all available visualizations](notebooks/Visualization.ipynb)

## Installation
To install the latest stable version of the library `dariah_topics`:

```
$ pip install dariah_topics
```

To install the latest development version:

```
$ pip install --upgrade git+https://github.com/DARIAH-DE/Topics.git@testing
```

## Example code
In only 15 lines of code from plain text files to a visualization of the topic model output.

```python
>>> from cophi_toolbox import preprocessing
>>> from dariah_topics import postprocessing, visualization
>>> import lda
>>> import glob
>>> import os
>>> pathlist = glob.glob("notebooks/data/british-fiction-corpus/*.txt")
>>> labels = [os.path.basename(path) for path in pathlist]
>>> corpus = preprocessing.read_files(pathlist)
>>> tokens = [preprocessing.tokenize(document) for document in corpus]
>>> matrix = preprocessing.create_document_term_matrix(tokens, labels)
>>> stopwords = preprocessing.list_mfw(matrix)
>>> clean_matrix = preprocessing.remove_features(stopwords, matrix)
>>> vocabulary = clean_matrix.columns
>>> clean_matrix_array = clean_matrix.values.astype(int)
>>> model = lda.LDA(n_topics=10, n_iter=1000)
>>> model.fit(clean_matrix_array)
>>> topics = postprocessing.show_topics(model, vocabulary)
>>> document_topics = postprocessing.show_document_topics(topics, model, labels)
>>> PlotDocumentTopics = visualization.PlotDocumentTopics(document_topics)
>>> PlotDocumentTopics.static_heatmap().show()
```
<p align="center">
  <img src="docs/images/heatmap.png"/>
</p>

## Working with Jupyter Notebooks
If you wish to work through the tutorials, you can clone the repository using [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):

```
$ git clone https://github.com/DARIAH-DE/Topics.git
```

or download the [ZIP-archive](https://github.com/DARIAH-DE/Topics/archive/master.zip) (don't forget to unzip it) and install `dariah_topics` from its source code:

```
$ pip install -r requirements.txt
```

As a server-client application, Jupyter allows you to edit and run Python code interactively from within so-called notebooks via a web browser.

To install Jupyter:

```
$ pip install jupyter
```

> Python distributions like [Anaconda](https://anaconda.org/anaconda/python) come with Jupyter by default.

You can run Jupyter via:

```
$ jupyter notebook
```

## Working with MALLET
[MALLET](http://mallet.cs.umass.edu) is a Java-based package for statistical *natural language processing*. The MALLET Topic Model package includes an extremely fast and highly scalable implementation of Gibbs sampling and tools for inferring topics for new documents given trained models.

To call MALLET from within the Python environment, `dariah_topics` provides a convenient wrapper.

You can download MALLET [here](http://mallet.cs.umass.edu/download.php). For more detailed instructions, have a look at [this](http://programminghistorian.org/lessons/topic-modeling-and-mallet).

## Troubleshooting
If you are confronted with any issues regarding installation or usability, please use [GitHub issues](https://github.com/DARIAH-DE/Topics/issues).

**This library requires Python 3.4 or higher.**

### Windows-specific Issues
* You will have to install `future‑0.16.0‑py3‑none‑any.whl` from [this resource](http://www.lfd.uci.edu/~gohlke/pythonlibs/). Download the appropriate file and run `pip install future‑0.16.0‑py3‑none‑any.whl`.
* In case of the error `Microsoft Visual C++ 10.0 is required`, check if you are using Python 3.6 or higher with `python -V`. If you do, you have to install Microsoft Windows SDK from [this resource](https://developer.microsoft.com/de-de/windows/downloads/windows-10-sdk). If you do not, upgrade to Python 3.6 or higher and try installing the library again.

### UNIX-specific Issues
* In case of `PermissionError: [Errno 13] Permission denied`, try `pip install --user` or `python setup.py install --user`, respectively.
* Due to several visualization dependencies, you might have to install the distribution packages `libfreetype6-dev` and `libpng-dev` (e.g. using `sudo apt-get install`).
* Make sure to install Python 3.6 correctly and adjust the selection of the Python interpreter in your editor accordingly. See also the [Python documentation](https://docs.python.org/3/using/mac.html).


## About DARIAH-DE
[DARIAH-DE](https://de.dariah.eu) supports research in the humanities and cultural sciences with digital methods and procedures. The research infrastructure of DARIAH-DE consists of four pillars: teaching, research, research data and technical components. As a partner in [DARIAH-EU](http://dariah.eu/), DARIAH-DE helps to bundle and network state-of-the-art activities of the digital humanities. Scientists use DARIAH, for example, to make research data available across Europe. The exchange of knowledge and expertise is thus promoted across disciplines and the possibility of discovering new scientific discourses is encouraged.

This application has been developed with support from the DARIAH-DE initiative, the German branch of DARIAH-EU, the European Digital Research Infrastructure for the Arts and Humanities consortium. Funding has been provided by the German Federal Ministry for Research and Education (BMBF) under the identifier 01UG1610J.

![DARIAH-DE](docs/images/dariah-de_logo.png)
![BMBF](docs/images/bmbf_logo.png)
