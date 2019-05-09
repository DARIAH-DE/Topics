A library for topic modeling and visualization
==============================================

DARIAH Topics is an easy-to-use Python library for topic modeling and visualization. Getting started is `really easy`. All you have to do is import the library – you can train a model straightaway from raw textfiles.

It supports two implementations of `latent Dirichlet allocation <http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf>`_:

- The lightweight, Cython-based package `lda <https://pypi.org/project/lda/>`_
- The more robust, Java-based package `MALLET <http://mallet.cs.umass.edu/topics.php>`_


Installation
------------

::

    $ pip install dariah


Example
-------

>>> import dariah
>>> dariah.topics(directory="british-fiction-corpus",
...               stopwords=100,
...               num_topics=10,
...               num_iterations=1000)


Developing
----------

`Poetry <https://poetry.eustace.io/>`_ automatically creates a virtual environment, builds and publishes the project to `PyPI <https://pypi.org/>`_. Install dependencies with:

::

    $ poetry install

run tests:

::

    $ poetry run pytest


format code:

::

    $ poetry run black dariah


build the project:

::

    $ poetry build


and publish it on `PyPI <https://pypi.org/>`_:

::

    $ poetry publish


About DARIAH-DE
---------------

`DARIAH-DE <https://de.dariah.eu>`_ supports research in the humanities and cultural sciences with digital methods and procedures. The research infrastructure of DARIAH-DE consists of four pillars: teaching, research, research data and technical components. As a partner in `DARIAH-EU <http://dariah.eu/>`_, DARIAH-DE helps to bundle and network state-of-the-art activities of the digital humanities. Scientists use DARIAH, for example, to make research data available across Europe. The exchange of knowledge and expertise is thus promoted across disciplines and the possibility of discovering new scientific discourses is encouraged.

This software library has been developed with support from the DARIAH-DE initiative, the German branch of DARIAH-EU, the European Digital Research Infrastructure for the Arts and Humanities consortium. Funding has been provided by the German Federal Ministry for Research and Education (BMBF) under the identifier 01UG1610J.

.. image:: https://raw.githubusercontent.com/DARIAH-DE/Topics/master/docs/images/dariah-de_logo.png
.. image:: https://raw.githubusercontent.com/DARIAH-DE/Topics/master/docs/images/bmbf_logo.png
