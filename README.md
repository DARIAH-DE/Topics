# A library for topic modeling and visualization
`dariah` is an easy-to-use Python library for topic modeling and visualization. Getting started is _really easy_. All you have to do is import the library – you can train a model straightaway from raw textfiles.

It provides two implementations of [latent Dirichlet allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf):
- The lightweight, Cython-based package [lda](https://pypi.org/project/lda/)
- The more robust, Java-based package [MALLET](http://mallet.cs.umass.edu/topics.php)

> Topic modeling algorithms are statistical methods that analyze the words of the original texts to discover the themes that run through them, how those themes are connected to each other, and how they change over time. ([David M. Blei](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf))


## Installation
```
$ pip install dariah
```


## Example
```python
>>> import dariah
>>> dariah.topics(directory="british-fiction-corpus",
...               stopwords=100,
...               num_topics=10,
...               num_iterations=1000)
```


## Developing
[Poetry](https://poetry.eustace.io/) automatically creates a virtual environment, builds and uploads the project to [PyPI](https://pypi.org/). Install dependencies with:
```
$ poetry install
```

run tests:
```
$ poetry run pytest
```

format code:
```
$ poetry run black dariah
```

build the project:
```
$ poetry build
```

and upload it to [PyPI](https://pypi.org/):
```
$ poetry upload
```

> Save your credentials with `poetry config http-basic.pypi username password`.


## About DARIAH-DE
[DARIAH-DE](https://de.dariah.eu) supports research in the humanities and cultural sciences with digital methods and procedures. The research infrastructure of DARIAH-DE consists of four pillars: teaching, research, research data and technical components. As a partner in [DARIAH-EU](http://dariah.eu/), DARIAH-DE helps to bundle and network state-of-the-art activities of the digital humanities. Scientists use DARIAH, for example, to make research data available across Europe. The exchange of knowledge and expertise is thus promoted across disciplines and the possibility of discovering new scientific discourses is encouraged.

This software library has been developed with support from the DARIAH-DE initiative, the German branch of DARIAH-EU, the European Digital Research Infrastructure for the Arts and Humanities consortium. Funding has been provided by the German Federal Ministry for Research and Education (BMBF) under the identifier 01UG1610J.

![DARIAH-DE](docs/images/dariah-de_logo.png)
![BMBF](docs/images/bmbf_logo.png)
