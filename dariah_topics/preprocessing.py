"""
Preprocessing Text Data, Creating Matrices and Cleaning Corpora
***************************************************************

Functions of this module are for **preprocessing purpose**. You can read text \
files, `tokenize <https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)>`_ \
and segment documents (if a document is chunked into smaller segments, each segment \
counts as one document), create and read `document-term matrices <https://en.wikipedia.org/wiki/Document-term_matrix>`_, \
determine and remove features. Recurrent variable names are based on the following \
conventions:

    * ``corpus`` means an iterable containing at least one ``document``.
    * ``document`` means one single string containing all characters of a text \
    file, including whitespaces, punctuations, numbers, etc.
    * ``dkpro_document`` means a pandas DataFrame containing tokens and additional \
    information, e.g. *part-of-speech tags* or *lemmas*, produced by `DARIAH-DKPro-Wrapper <https://github.com/DARIAH-DE/DARIAH-DKPro-Wrapper>`_.
    * ``tokenized_corpus`` means an iterable containing at least one ``tokenized_document`` \
    or ``dkpro_document``.
    * ``tokenized_document`` means an iterable containing tokens of a ``document``.
    * ``document_labels`` means an iterable containing names of each ``document`` \
    and must have as much elements as ``corpus`` or ``tokenized_corpus`` does.
    * ``document_term_matrix`` means either a pandas DataFrame with rows corresponding to \
    ``document_labels`` and columns to types (distinct tokens in the corpus), whose \
    values are token frequencies, or a pandas DataFrame with a MultiIndex \
    and only one column corresponding to word frequencies. The first column of the \
    MultiIndex corresponds to a document ID (based on ``document_labels``) and the \
    second column to a type ID. The first variant is designed for small and the \
    second for large corpora.
    * ``token2id`` means a dictionary containing a token as key and an unique identifier \
    as key, e.g. ``{'first_document': 0, 'second_document': 1}``.

Contents
********
    * :func:`add_token2id` adds a token to a ``document_ids`` or ``type_ids`` dictionary \
    and assigns an unique identifier.
    * :func:`create_document_term_matrix()` creates a document-term matrix, for either \
    small or large corpora.
    * :func:`filter_pos_tags()` filters a ``dkpro_document`` by specific \
    *part-of-speech tags* and returns either tokens or, if available, lemmas.
    * :func:`find_hapax_legomena()` determines *hapax legomena* based on frequencies \
    of a ``document_term_matrix``.
    * :func:`list_mfw()` determines *most frequent words* based on frequencies \
    of a ``document_term_matrix``.
    * :func:`read_document_term_matrix()` reads a document-term matrix from a CSV file.
    * :func:`read_files()` reads one or multiple files based on a pathlist.
    * :func:`read_matrix_market_file()` reads a `Matrix Market <http://math.nist.gov/MatrixMarket/formats.html#MMformat>`_ \
    file for `Gensim <https://radimrehurek.com/gensim/>`_.
    * :func:`read_model()` reads a LDA model.
    * :func:`read_token2id()` reads a ``document_ids`` or ``type_ids`` dictionary \
    from a CSV file.
    * :func:`remove_features()` removes features from a ``document_term_matrix``.
    * :func:`segment()` is a wrapper for :func:`segment_fuzzy()` and segments a \
    ``tokenized_document`` into segments of a certain number of tokens, respecting existing chunks.
    * :func:`segment_fuzzy()` segments a ``tokenized_document``, tolerating existing \
    chunks (like paragraphs).
    * :func:`split_paragraphs()` splits a ``document`` or ``dkpro_document`` by paragraphs.
    * :func:`tokenize()` tokenizes a ``document`` based on a Unicode regular expression.
"""


from collections import Counter, defaultdict
import csv
from itertools import chain
from gensim.corpora import MmCorpus
import os
from lxml import etree
import numpy as np
import pandas as pd
import pickle
import regex
import logging

log = logging.getLogger('dariah_topics')


def add_token2id(token, token2id):
    """Adds token to token2id dictionary.

    With this function you can append a ``token`` to an existing ``token2id`` \
    dictionary. If ``token2id`` has *x* elements with *n* identifiers, ``token`` \
    will be element *x + 1* with identifier *n + 1*. 

    Args:
        token (str): Token.
        token2id (dict): A dictionary with tokens as keys and identifiers as values.

    Returns:
        An extended token2id dictionary.
    
    Raises:
        ValueError, if ``token`` has alread an ID in ``token2id``.
        
    Example:
        >>> token = 'example'
        >>> token2id = {'text': 0}
        >>> len(add_token2id(token, token2id)) == 2
        True
    """
    if token in token2id.keys():
        raise KeyError("{} has already an ID in token2id. Access its value with token2id[token].".format(token))
    token2id[token] = len(token2id) + 1
    return token2id


def create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=False):
    """Creates a document-term matrix.

    With this function you can create a document-term-matrix where rows \
    correspond to documents in the collection and columns correspond to terms. \
    Use the function :func:`read_files()` to read and :func:`tokenize()` \
    to tokenize your text files.

    Args:
        tokenized_corpus (list): Tokenized corpus as an iterable containing one
            or more iterables containing tokens.
        document_labels (list): Name or label of each text file.
        large_corpus (bool, optional): Set to True, if ``tokenized_corpus`` is
            very large. Defaults to False.

    Returns:
        Document-term matrix as pandas DataFrame.

    Example:
        >>> tokenized_corpus = [['this', 'is', 'document', 'one'], ['this', 'is', 'document', 'two']]
        >>> document_labels = ['document_one', 'document_two']
        >>> create_document_term_matrix(tokenized_corpus, document_labels) #doctest: +NORMALIZE_WHITESPACE
                      this   is  document  two  one
        document_one   1.0  1.0       1.0  0.0  1.0
        document_two   1.0  1.0       1.0  1.0  0.0
        >>> document_term_matrix, document_ids, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, True)
        >>> isinstance(document_term_matrix, pd.DataFrame) and isinstance(document_ids, dict) and isinstance(type_ids, dict)
        True
    """
    if large_corpus:
        return _create_large_corpus_model(tokenized_corpus, document_labels)
    else:
        return _create_small_corpus_model(tokenized_corpus, document_labels)


def filter_pos_tags(dkpro_document, pos_tags=['ADJ', 'V', 'NN'], lemma=True):
    """Gets tokens or lemmas respectively of selected POS-tags from pandas DataFrame.

    With this function you can filter `DARIAH-DKPro-Wrapper <https://github.com/DARIAH-DE/DARIAH-DKPro-Wrapper>`_ \
    output. Commit a list of POS-tags to get specific tokens (if ``lemma`` False) \
    or lemmas (if ``lemma`` True). 
    Use the function :func:`read_files()` to read CSV files.

    Args:
        dkpro_document (pandas.DataFrame): DARIAH-DKPro-Wrapper output.
        pos_tags (list, optional): List of desired POS-tags. Defaults
            to ``['ADJ', 'V', 'NN']``.
        lemma (bool, optional): If True, lemmas will be selected, otherwise tokens.
            Defaults to True.

    Yields:
        A pandas DataFrame containing tokens or lemmas.

    Example:
        >>> dkpro_document = pd.DataFrame({'CPOS': ['ART', 'V', 'ART', 'NN'],
        ...                                'Token': ['this', 'was', 'a', 'document'],
        ...                                'Lemma': ['this', 'is', 'a', 'document']})
        >>> list(filter_pos_tags(dkpro_document)) #doctest: +NORMALIZE_WHITESPACE
        [1          is
        3    document
        Name: Lemma, dtype: object]
    """
    tokenized_document = dkpro_document[dkpro_document['CPOS'].isin(pos_tags)]
    if lemma:
        log.info("Selecting {} lemmas ...".format(pos_tags))
        yield tokenized_document['Lemma']
    else:
        log.info("Selecting {} tokens ...".format(pos_tags))
        yield tokenized_document['Token']


def find_hapax_legomena(document_term_matrix, type_ids=None):
    """Creates a list with hapax legommena.

    With this function you can determine *hapax legomena* for each document. \
    Use the function :func:`create_document_term_matrix()` to create a \
    document-term matrix.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.
            If ``document_term_matrix`` is designed for large corpora, you have
            to commit ``type_ids``, too.

    Returns:
        Hapax legomena in a list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['hapax', 'stopword', 'stopword']]
        >>> document_term_matrix = create_document_term_matrix(tokenized_corpus, document_labels)
        >>> find_hapax_legomena(document_term_matrix)
        ['hapax']
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> find_hapax_legomena(document_term_matrix, type_ids)
        ['hapax']
    """
    log.info("Determining hapax legomena ...")
    if isinstance(document_term_matrix.index, pd.MultiIndex):
        log.debug("Large corpus model ...")
        return _hapax_legomena_large_corpus_model(document_term_matrix, type_ids)
    else:
        log.debug("Small corpus model ...")
        return document_term_matrix.loc[:, document_term_matrix.max() == 1].columns.tolist()


def list_mfw(document_term_matrix, most_frequent_tokens=100, type_ids=None):
    """Creates a list with stopword based on most frequent tokens.

    With this function you can determine *most frequent tokens*, also known as \
    *stopwords*. First, you have to translate your corpus into a document-term \
    matrix.
    Use the function :func:`create_document_term_matrix()` to create a \
    document-term matrix.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        most_frequent_tokens (int, optional): Treshold for most frequent tokens.
        type_ids (dict): If ``document_term_matrix`` is designed for large corpora,
            you have to commit ``type_ids``, too.

    Returns:
        Most frequent tokens in a list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['hapax', 'stopword', 'stopword']]
        >>> document_term_matrix = create_document_term_matrix(tokenized_corpus, document_labels)
        >>> list_mfw(document_term_matrix, 1)
        ['stopword']
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> list_mfw(document_term_matrix, 1, type_ids)
        ['stopword']
    """
    log.info("Determining stopwords ...")
    if isinstance(document_term_matrix.index, pd.MultiIndex):
        log.debug("Large corpus model ...")
        return _list_mfw_large_corpus_model(document_term_matrix, type_ids, most_frequent_tokens)
    else:
        log.debug("Small corpus model ...")
        return document_term_matrix.iloc[:, :most_frequent_tokens].columns.tolist()


def read_document_term_matrix(filepath):
    """Reads a document-term matrix from CSV file.

    With this function you can read a CSV file containing a document-term \
    matrix.
    Use the function :func:`create_document_term_matrix()` to create a document-term \
    matrix.

    Args:
        filepath (str): Path to CSV file.

    Returns:
        A document-term matrix as pandas DataFrame.
    
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
        ...     tmpfile.write(b',this,is,an,example,text\\ndocument,1,0,1,0,1') and True
        ...     tmpfile.flush()
        ...     read_document_term_matrix(tmpfile.name) #doctest: +NORMALIZE_WHITESPACE
        True
                  this  is  an  example  text
        document     1   0   1        0     1
        >>> with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
        ...     tmpfile.write(b'document_id,type_id,0\\n1,1,1') and True
        ...     tmpfile.flush()
        ...     read_document_term_matrix(tmpfile.name) #doctest: +NORMALIZE_WHITESPACE
        True
                             0
        document_id type_id   
        1           1        1
    """
    document_term_matrix = pd.read_csv(filepath)
    if 'document_id' and 'type_id' in document_term_matrix:
        return document_term_matrix.set_index(['document_id', 'type_id'])
    else:
        document_term_matrix = document_term_matrix.set_index('Unnamed: 0')
        document_term_matrix.index.name = None
        return document_term_matrix


def read_files(pathlist, file_format=None, xpath_expression='//tei:text', sep='\t', csv_columns=None):
    """Reads text files based on a pathlist.

    With this function you can read multiple file formats:
        * Plain text files (``.txt``).
        * TEI XML files (``.xml``).
        * CSV files (``.csv``), e.g. produced by `DARIAH-DKPro-Wrapper <https://github.com/DARIAH-DE/DARIAH-DKPro-Wrapper>`_. 

    The argument ``pathlist`` is an iterable of full or relative paths. In case of \
    CSV files, you have the ability to select specific columns via ``columns``. \
    If there are multiple file formats in ``pathlist``, do not specify ``file_format`` \
    and file extensions will be considered.

    Args:
        pathlist (list): One or more paths to text files.
        file_format (str, optional): Format of the files. Possible values are
            ``text``, ``xml`` and ``csv`. If None, file extensions will be considered.
            Defaults to None.
        xpath_expression (str, optional): XPath expressions to match part of the
            XML file. Defaults to ``//tei:text``.
        sep (str, optional): Separator of CSV file. Defaults to ``'\\t'``
        columns (list, optional): Column name or names for CSV files. If None, the
            whole file will be processed. Defaults to None.

    Yields:
        A ``document`` as str or, in case of a CSV file, a ``dkpro_document`` as a pandas DataFrame.

    Raises:
        ValueError, if ``file_format`` is not supported.

    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.txt') as first:
        ...     pathlist = []
        ...     first.write(b"This is the first example.") and True
        ...     first.flush()
        ...     pathlist.append(first.name)
        ...     with tempfile.NamedTemporaryFile(suffix='.txt') as second:
        ...         second.write(b"This is the second example.") and True
        ...         second.flush()
        ...         pathlist.append(second.name)
        ...         list(read_files(pathlist, 'text'))
        True
        True
        ['This is the first example.', 'This is the second example.']
    """
    log.info("Reading {} files ...".format(len(pathlist)))
    for n, file in enumerate(pathlist):
        log.debug("File #{}".format(n))
        _, extension = os.path.splitext(file)
        if file_format == 'text' or extension == '.txt':
            yield _read_txt(file)
        elif file_format == 'xml' or extension == '.xml':
            yield _read_xml(file, xpath_expression)
        elif file_format == 'csv' or extension == '.csv':
            yield _read_csv(file, sep, csv_columns)
        else:
            if file_format is None:
                log.error("Skipping {}, because the file format {} is not supported.".format(file, extension))
                pass
            else:
                raise ValueError("Unable to read {}, because the file format {} is not supported.".format(file, file_format))


def read_matrix_market_file(filepath):
    """Reads a Matrix Market file for Gensim.

    With this function you can read a Matrix Market file to process it with \
    `Gensim <https://radimrehurek.com/gensim/>`_.

    Args:
        filepath (str): Path to Matrix Market file.

    Returns:
        Matrix Market model for Gensim.
    """
    if os.path.splitext(filepath)[1] != '.mm':
        raise ValueError("The file {} is not a Matrix Market file.".format(filepath))
    return MmCorpus(filepath)


def read_model(filepath):
    """Reads a LDA model.

    With this function you can read a LDA model, if it was saved using :module:`pickle`.
    If you want to read MALLET models, you have to specify a parameter of the
    function :func:`create_mallet_model()`.

    Args:
        filepath (str): Path to LDA model, e.g. ``/home/models/model.pickle``.

    Returns:
        A LDA model.

    Example:
        >>> import lda
        >>> import gensim
        >>> import tempfile
        >>> a = lda.LDA
        >>> with tempfile.NamedTemporaryFile(suffix='.pickle') as tmpfile:
        ...     pickle.dump(a, tmpfile, protocol=pickle.HIGHEST_PROTOCOL)
        ...     tmpfile.flush()
        ...     read_model(tmpfile.name) == a
        True
        >>> a = gensim.models.LdaModel
        >>> with tempfile.NamedTemporaryFile(suffix='.pickle') as tmpfile:
        ...     pickle.dump(a, tmpfile, protocol=pickle.HIGHEST_PROTOCOL)
        ...     tmpfile.flush()
        ...     read_model(tmpfile.name) == a
        True
    """
    with open(filepath, 'rb') as model:
        return pickle.load(model)


def read_token2id(filepath):
    """Reads a token2id dictionary from CSV file.

    With this function you can read a CSV-file containing a document or type dictionary.

    Args:
        filepath (str): Path to CSV file.

    Returns:
        A dictionary.
    
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
        ...     tmpfile.write(b"0,this\\n1,is\\n2,an\\n3,example") and True
        ...     tmpfile.flush()
        ...     read_token2id(tmpfile.name)
        True
        {0: 'this', 1: 'is', 2: 'an', 3: 'example'}
    """
    dictionary = pd.read_csv(filepath, header=None)
    dictionary.index = dictionary[0]
    dictionary = dictionary[1]
    return dictionary.to_dict()
    
    
def remove_features(features, document_term_matrix=None, tokenized_corpus=None, type_ids=None):
    """Removes features based on a list of tokens.

    With this function you can clean your corpus (either a document-term matrix \
    or a ``tokenized_corpus``) from *stopwords* and *hapax legomena*.
    Use the function :func:`create_document_term_matrix()` or :func:`tokenize` to \
    create a document-term matrix or to tokenize your corpus, respectively.

    Args:
        features (list): A list of tokens.
        document_term_matrix (pandas.DataFrame, optional): A document-term matrix.
        tokenized_corpus (list, optional): An iterable of one or more ``tokenized_document``.
        type_ids (dict, optional): A dictionary with types as key and identifiers as values.

    Returns:
        A clean document-term matrix as pandas DataFrame or ``tokenized_corpus`` as list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['this', 'is', 'a', 'document']]
        >>> document_term_matrix = create_document_term_matrix(tokenized_corpus, document_labels)
        >>> features = ['this']
        >>> remove_features(features, document_term_matrix) #doctest: +NORMALIZE_WHITESPACE
                   is  document    a
        document  1.0       1.0  1.0
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> len(remove_features(features, document_term_matrix, type_ids=type_ids))
        3
        >>> list(remove_features(features, tokenized_corpus=tokenized_corpus))
        [['is', 'a', 'document']]
    """
    log.info("Removing features ...")
    if document_term_matrix is not None and tokenized_corpus is None:
        if isinstance(document_term_matrix.index, pd.MultiIndex):
            return _remove_features_from_large_corpus_model(document_term_matrix, type_ids, features)
        else:
            return _remove_features_from_small_corpus_model(document_term_matrix, features)
    elif document_term_matrix is None and tokenized_corpus is not None:
        clean_tokenized_corpus = pd.Series() # schöner machen
        for n, tokenized_document in enumerate(tokenized_corpus):
            clean_tokenized_corpus[str(n)] = _remove_features_from_tokenized_document(tokenized_document, features)
        return clean_tokenized_corpus
    else:
        raise ValueError("Commit either document-term matrix or tokenized_corpus.")


def segment(document, segment_size=1000, tolerance=0, chunker=None,
            tokenizer=None, flatten_chunks=True, materialize=True):
    """Segments a document into segments of about ``segment_size`` tokens, respecting existing chunks.

    Consider you have a document. You wish to split the document into \
    segments of about 1000 tokens, but you prefer to keep paragraphs together \
    if this does not increase or decrease the token size by more than 5%.
    This is a convenience wrapper around :func:`segment_fuzzy()`.

    Args:
        document (list): The document to process. This is an iterable of
            chunks, each of which is an iterable of tokens.
        segment_size (int): The target size of each segment, in tokens. Defaults
            to 1000.
        tolerance (float, optional): How much may the actual segment size differ from
            the segment_size? If ``0 < tolerance < 1``, this is interpreted as a
            fraction of the segment_size, otherwise it is interpreted as an
            absolute number. If ``tolerance < 0``, chunks are never split apart.
            Defaults to None.
        chunker (callable, optional): A one-argument function that cuts the document into
            chunks. If this is present, it is called on the given document.
            Defaults to None.
        tokenizer (callable, optional): A one-argument function that tokenizes each chunk.
            Defaults to None.
        flatten_chunks (bool, optional): If True, undo the effect of the chunker by
            chaining the chunks in each segment, thus each segment consists of
            tokens. This can also be a one-argument function in order to
            customize the un-chunking. Defaults to True.
        materialize (bool, optional): If True, materializes the segments. Defaults to True.

    Example:
        >>> segment([['This', 'is', 'the', 'first', 'chunk'],
        ...          ['this', 'is', 'the', 'second', 'chunk']], 2) #doctest: +NORMALIZE_WHITESPACE
        [['This', 'is'],
        ['the', 'first'],
        ['chunk', 'this'],
        ['is', 'the'],
        ['second', 'chunk']]
    """
    if chunker is not None:
        document = chunker(document)
    if tokenizer is not None:
        document = map(tokenizer, document)

    segments = segment_fuzzy(document, segment_size, tolerance)

    if flatten_chunks:
        if not callable(flatten_chunks):
            def flatten_chunks(segment):
                return list(chain.from_iterable(segment))
        segments = map(flatten_chunks, segments)
    if materialize:
        segments = list(segments)
    return segments


def segment_fuzzy(document, segment_size=5000, tolerance=0.05):
    """Segments a document, tolerating existing chunks (like paragraphs).

    Consider you have a ``document``. You wish to split the ``document`` into \
    segments of about 1000 tokens, but you prefer to keep paragraphs together \
    if this does not increase or decrease the token size by more than 5%.

    Args:
        document (list): The document to process. This is an iterable of
            chunks, each of which is an iterable of tokens.
        segment_size (int, optional): The target length of each segment in tokens.
            Defaults to 5000.
        tolerance (float, optional): How much may the actual segment size differ from
            the ``segment_size``? If ``0 < tolerance < 1``, this is interpreted as a
            fraction of the ``segment_size``, otherwise it is interpreted as an
            absolute number. If ``tolerance < 0``, chunks are never split apart.
            Defaults to 0.05.

    Yields:
        Segments. Each segment is a list of chunks, each chunk is a list of
        tokens.

    Example:
        >>> list(segment_fuzzy([['This', 'is', 'the', 'first', 'chunk'],
        ...                     ['this', 'is', 'the', 'second', 'chunk']], 2)) #doctest: +NORMALIZE_WHITESPACE
        [[['This', 'is']],
        [['the', 'first']],
        [['chunk'], ['this']],
        [['is', 'the']],
        [['second', 'chunk']]]
    """
    if tolerance > 0 and tolerance < 1:
        tolerance = round(segment_size * tolerance)

    current_segment = []
    current_size = 0
    carry = None
    doc_iter = iter(document)

    try:
        while True:
            chunk = list(carry if carry else next(doc_iter))
            carry = None
            current_segment.append(chunk)
            current_size += len(chunk)

            if current_size >= segment_size:
                too_long = current_size - segment_size
                too_short = segment_size - (current_size - len(chunk))

                if tolerance >= 0 and min(too_long, too_short) > tolerance:
                    chunk_part0 = chunk[:-too_long]
                    carry = chunk[-too_long:]
                    current_segment[-1] = chunk_part0
                elif too_long >= too_short:
                    carry = current_segment.pop()
                yield current_segment
                current_segment = []
                current_size = 0
    except StopIteration:
        pass

    if current_segment:
        yield current_segment


def split_paragraphs(document, sep=regex.compile(r'\n')):
    """Splits the given document by paragraphs.

    With this function you can split a document by paragraphs. In case of a \
    document as str, you also have the ability to select a certain regular \
    expression to split the document.
    Use the function :func:`read_files()` to read files.

    Args:
        document Union(str, pandas.DataFrame): Document text or DARIAH-DKPro-Wrapper output.
        sep (regex.Regex, optional): Separator indicating a paragraph.

    Returns:
        A list of paragraphs.

    Example:
        >>> document = "First paragraph\\nsecond paragraph."
        >>> split_paragraphs(document)
        ['First paragraph', 'second paragraph.']
        >>> dkpro_document = pd.DataFrame({'Token': ['first', 'paragraph', 'second', 'paragraph', '.'],
        ...                                'ParagraphId': [1, 1, 2, 2, 2]})
        >>> split_paragraphs(dkpro_document)[0] #doctest: +NORMALIZE_WHITESPACE
                         Token
        ParagraphId           
        1                first
        1            paragraph
    """
    if isinstance(document, str):
        if not hasattr(sep, 'match'):
            sep = regex.compile(sep)
        splitted_document = sep.split(document)
        return list(filter(str.strip, splitted_document)) #remove elements containing only whitespaces
    elif isinstance(document, pd.DataFrame):
        grouped_document = document.set_index('ParagraphId').groupby(level=0)
        return [paragraphs for _, paragraphs in grouped_document]


def tokenize(document, pattern=r'\p{L}+\p{P}?\p{L}+', lower=True):
    """Tokenizes with Unicode regular expressions.

    With this function you can tokenize a ``document`` with a regular expression. \
    You also have the ability to commit your own regular expression. The default \
    expression is ``\p{Letter}+\p{Punctuation}?\p{Letter}+``, which means one or \
    more letters, followed by one or no punctuation, followed by one or more \
    letters. So, one letter words will not match. In case you want to lower \
    all tokens, set the argument ``lower`` to True (it is by default).    
    Use the functions :func:`read_files()` to read your text files.

    Args:
        document (str): Document text.
        pattern (str, optional): Regular expression to match tokens.
        lower (boolean, optional): If True, lowers all characters. Defaults to True.

    Yields:
        All matching tokens in the ``document``.

    Example:
        >>> list(tokenize("This is 1 example text."))
        ['this', 'is', 'example', 'text']
    """
    log.debug("Tokenizing document ...")
    if lower:
        log.debug("Lowering all characters ...")
        document = document.lower()
    compiled_pattern = regex.compile(pattern)
    tokenized_document = compiled_pattern.finditer(document)
    for match in tokenized_document:
        yield match.group()


def _create_bag_of_words(document_labels, tokenized_corpus):
    """Creates a bag-of-words model.

    This private function is wrapped in :func:`_create_large_corpus_model()`. The \
    first level consists of the document label as key, and the dictionary \
    of counts as value. The second level consists of type ID as key, and the \
    count of types in document pairs as value.

    Args:
        document_labels (list): Iterable of document labels.
        tokenized_corpus (list): Tokenized corpus as an iterable
            containing one or more iterables containing tokens.

    Returns:
        A bag-of-words model as dictionary of dictionaries, document IDs and type IDs.

    Example:
        >>> document_labels = ['exampletext']
        >>> tokenized_corpus = [['this', 'is', 'an', 'example', 'text']]
        >>> bag_of_words, document_ids, type_ids = _create_bag_of_words(document_labels, tokenized_corpus)
        >>> isinstance(bag_of_words, dict) and isinstance(document_ids, dict) and isinstance(type_ids, dict)
        True
    """
    document_ids = _token2id(document_labels)
    type_ids = _token2id(tokenized_corpus)
    bag_of_words = defaultdict(dict)
    for document_label, tokenized_document in zip(document_labels, tokenized_corpus):
        bag_of_words[document_label] = Counter([type_ids[token] for token in tokenized_document])
    return {document_ids[id_]: doc for id_, doc in bag_of_words.items()}, document_ids, type_ids


def _create_large_corpus_model(tokenized_corpus, document_labels):
    """Creates a document-term matrix for large corpora.

    This private function is wrapped in :func:`create_document_term_matrix()` and \
    creates a pandas DataFrame containing document and type IDs as MultiIndex \
    and type frequencies as values representing the counts of tokens for each \
    token in each document.

    Args:
        tokenized_corpus (list): Tokenized corpus as an iterable
            containing one or more iterables containing tokens.
        document_labels (list): Iterable of document labels.

    Returns:
        A document-term matrix as pandas DataFrame, ``document_ids`` and ``type_ids``.

    Todo:
        * Make the whole function faster.

    Example:
        >>> tokenized_corpus = [['this', 'is', 'document', 'one'], ['this', 'is', 'document', 'two']]
        >>> document_labels = ['document_one', 'document_two']
        >>> document_term_matrix, document_ids, type_ids = _create_large_corpus_model(tokenized_corpus, document_labels)
        >>> isinstance(document_term_matrix, pd.DataFrame) and isinstance(document_ids, dict) and isinstance(type_ids, dict)
        True
    """
    bag_of_words, document_ids, type_ids = _create_bag_of_words(document_labels, tokenized_corpus)
    multi_index = _create_multi_index(bag_of_words)
    document_term_matrix = pd.DataFrame(np.zeros((len(multi_index), 1), dtype=int), index=multi_index)
    index_iterator = multi_index.groupby(multi_index.get_level_values('document_id'))

    for document_id in range(1, len(multi_index.levels[0]) + 1):
        for type_id in [val[1] for val in index_iterator[document_id]]:
            document_term_matrix.at[(document_id, type_id), 0] = int(bag_of_words[document_id][type_id])
    return document_term_matrix, document_ids, type_ids


def _create_multi_index(bag_of_words):
    """Creates a MultiIndex for a pandas DataFrame.

    This private function is wrapped in :func:`_create_large_corpus_model()`.

    Args:
        bag_of_words (dict): A bag-of-words model of ``{document_id: {type_id: frequency}}``.

    Returns:
        Pandas MultiIndex.

    Example:
        >>> bag_of_words = {1: {1: 2, 2: 3, 3: 4}}
        >>> _create_multi_index(bag_of_words)
        MultiIndex(levels=[[1], [1, 2, 3]],
                   labels=[[0, 0, 0], [0, 1, 2]],
                   names=['document_id', 'type_id'])
    """
    tuples = []
    for document_id in range(1, len(bag_of_words) + 1):
        if len(bag_of_words[document_id]) == 0:
            tuples.append((document_id, 0))
        for type_id in bag_of_words[document_id]:
            tuples.append((document_id, type_id))
    return pd.MultiIndex.from_tuples(tuples, names=['document_id', 'type_id'])


def _create_small_corpus_model(tokenized_corpus, document_labels):
    """Creates a document-term matrix for small corpora.

    This private function is wrapped in :func:`create_document_term_matrix()`.

    Args:
        tokenized_corpus (list): Tokenized corpus as an iterable
            containing one or more iterables containing tokens.
        document_labels (list): Name or label of each text file.


    Returns:
        Document-term matrix as pandas DataFrame.

    Example:
        >>> tokenized_corpus = [['this', 'is', 'document', 'one'], ['this', 'is', 'document', 'two']]
        >>> document_labels = ['document_one', 'document_two']
        >>> _create_small_corpus_model(tokenized_corpus, document_labels) #doctest: +NORMALIZE_WHITESPACE
                      this   is  document  two  one
        document_one   1.0  1.0       1.0  0.0  1.0
        document_two   1.0  1.0       1.0  1.0  0.0
    """
    log.info("Creating document-term matrix for small corpus ...")
    document_term_matrix = pd.DataFrame()
    for tokenized_document, document_label in zip(tokenized_corpus, document_labels):
        log.debug("Updating {} in document-term matrix ...".format(document_label))
        current_document = pd.Series(Counter(tokenized_document))
        current_document.name = document_label
        document_term_matrix = document_term_matrix.append(current_document)
    document_term_matrix = document_term_matrix.loc[:, document_term_matrix.sum().sort_values(ascending=False).index]
    return document_term_matrix.fillna(0)


def _hapax_legomena_large_corpus_model(document_term_matrix, type_ids):
    """Determines hapax legomena in large corpus model.

    This private function is wrapped in :func:`find_hapax_legomena()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.

    Returns:
        Hapax legomena in a list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['hapax', 'stopword', 'stopword']]
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> find_hapax_legomena(document_term_matrix, type_ids)
        ['hapax']
    """
    id2type = {id_: type_ for type_, id_ in type_ids.items()}
    document_term_matrix_collapsed = document_term_matrix.groupby(document_term_matrix.index.get_level_values('type_id')).sum()
    hapax_legomena = document_term_matrix_collapsed.loc[document_term_matrix_collapsed[0] == 1]
    return [id2type[token] for token in hapax_legomena.index.get_level_values('type_id')]


def _read_csv(filepath, sep, columns):
    """Reads a CSV file based on its path.
    
    This private function is wrapped in `read_files()`.
    
    Args:
        filepath (str): Path to CSV file.
        sep (str): Separator of CSV file.
        columns (list): Column names for the CSV file. If None, the whole file will be processed.

    Returns:
        A ``dkpro_document`` as pandas DataFrame with additional information,
            e.g. lemmas or POS-tags.
    
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.csv') as tmpfile:
        ...     tmpfile.write(b"Token,POS\\nThis,ART\\nis,V\\na,ART\\nCSV,NN\\nexample,NN\\n.,PUNC") and True
        ...     tmpfile.flush()
        ...     _read_csv(tmpfile.name, ',', ['Token']) #doctest: +NORMALIZE_WHITESPACE
        True
              Token
        0      This
        1        is
        2         a
        3       CSV
        4   example
        5         .
    """
    log.debug("Reading columns {} of {} ...".format(columns, filepath))
    if columns is None:
        log.warning("No column names were specified or do not match. The whole file will be processed.")
    return pd.read_csv(filepath, sep=sep, quoting=csv.QUOTE_NONE, usecols=columns)


def _read_txt(filepath):
    """Reads a plain text file based on its path.

    This private function is wrapped in `read_files()`.

    Args:
        filepath (str): Path to plain text file.
  
    Returns:
        A ``document`` as str.
    
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.txt') as tmpfile:
        ...     tmpfile.write(b"This is a plain text example.") and True
        ...     tmpfile.flush()
        ...     _read_txt(tmpfile.name)
        True
        'This is a plain text example.'
    """
    log.debug("Reading {} ...".format(filepath))
    with open(filepath, 'r', encoding='utf-8') as document:
        return document.read()


def _read_xml(filepath, xpath_expression):
    """Reads a TEI XML file based on its path.
    
    This private function is wrapped in `read_files()`.
    
    Args:
        filepath (str): Path to XML file.
        xpath_expression (str): XPath expressions to match part of the XML file.
    
    Returns:
        Either a ``document`` as str or a list of all parts of the ``document``,
            e. g. chapters of a novel.
    
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.xml') as tmpfile:
        ...     tmpfile.write(b"<text>This is a XML example.</text>") and True
        ...     tmpfile.flush()
        ...     _read_xml(tmpfile.name, '//text')
        True
        'This is a XML example.'
    """
    log.debug("Reading {} matching part or parts of {} ...".format(xpath_expression, filepath))
    ns = dict(tei='http://www.tei-c.org/ns/1.0')
    tree = etree.parse(filepath)
    document = [''.join(element.xpath('.//text()')) for element in tree.xpath(xpath_expression, namespaces=ns)]
    if len(document) == 1:
        return document[0]
    else:
        return document


def _remove_features_from_large_corpus_model(document_term_matrix, type_ids, features):
    """Removes features from large corpus model.

    This private function is wrapped in :func:`remove_features()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.
        features (list): A list of tokens.

    Returns:
        A clean document-term matrix as pandas DataFrame.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['token', 'stopword', 'stopword']]
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> len(_remove_features_from_large_corpus_model(document_term_matrix, type_ids, ['token']))
        1
    """
    features = [token for token in set(type_ids.keys()) if token in features]
    return document_term_matrix.drop([type_ids[token] for token in features], level='type_id')


def _remove_features_from_small_corpus_model(document_term_matrix, features):
    """Removes features from small corpus model.

    This private function is wrapped in :func:`remove_features()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        features (list): A list of tokens.

    Returns:
        A clean document-term matrix as pandas DataFrame.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['token', 'stopword', 'stopword']]
        >>> document_term_matrix = create_document_term_matrix(tokenized_corpus, document_labels)
        >>> _remove_features_from_small_corpus_model(document_term_matrix, ['token']) #doctest: +NORMALIZE_WHITESPACE
                  stopword
        document       2.0

    """
    features = [token for token in features if token in document_term_matrix.columns]
    return document_term_matrix.drop(features, axis=1)


def _remove_features_from_tokenized_document(tokenized_document, features):
    """Removes features from a tokenized document.

    This private function is wrapped in :func:`remove_features()`.

    Args:
        tokenized_document (list): The tokenized document to process. This is an iterable of
            tokens.
        features (list): An iterable of tokens.

    Returns:
        A clean tokenized document as list.

    Example:
        >>> tokenized_document = ['token', 'stopword', 'stopword']
        >>> features = ['stopword']
        >>> _remove_features_from_tokenized_document(tokenized_document, features)
        ['token']
    """
    tokenized_document_arr = np.array(tokenized_document)
    features_arr = np.array(features)
    indices = np.where(np.in1d(tokenized_document_arr, features_arr))
    return np.delete(tokenized_document_arr, indices).tolist()


def _list_mfw_large_corpus_model(document_term_matrix, type_ids, most_frequent_tokens):
    """Determines stopwords in large corpus model.

    This private function is wrapped in :func:`list_mfw()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.
        most_frequent_tokens (int, optional): Treshold for most frequent tokens.

    Returns:
        Most frequent tokens in a list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['hapax', 'stopword', 'stopword']]
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> list_mfw(document_term_matrix, 1, type_ids)
        ['stopword']
    """
    id2type = {id_: type_ for type_, id_ in type_ids.items()}
    document_term_matrix_collapsed = document_term_matrix.groupby(document_term_matrix.index.get_level_values('type_id')).sum()
    stopwords = document_term_matrix_collapsed[0].nlargest(most_frequent_tokens)
    return [id2type[token] for token in stopwords.index.get_level_values('type_id')]


def _token2id(tokens):
    """Creates a dictionary of tokens as keys and identifier as keys.

    This private function is wrapped in :func:`_create_largecounter()`.

    Args:
        tokens (list): Iterable of tokens.

    Returns:
        A dictionary.

    Example:
        >>> _token2id(['token'])
        {'token': 1}
        >>> _token2id([['token']])
        {'token': 1}
    """
    log.debug("Creating dictionary of tokens as keys and identifier as keys ...")
    if all(isinstance(element, list) for element in tokens):
        tokens = {token for element in tokens for token in element}
    return {token: id_ for id_, token in enumerate(set(tokens), 1)}
