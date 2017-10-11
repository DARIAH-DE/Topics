# -*- coding: utf-8 -*-

"""
Handling MALLET in Python and Generating LDA Models
===================================================

This module contains various `MALLET`_ related functions for Topic Modeling
provided by `DARIAH-DE`_.

.. _MALLET:
    http://mallet.cs.umass.edu/
.. _DARIAH-DE:
    https://de.dariah.eu
    https://github.com/DARIAH-DE
"""

__author__ = "DARIAH-DE"
__authors__ = "Steffen Pielstroem, Sina Bock, Severin Simmler"
__email__ = "pielstroem@biozentrum.uni-wuerzburg.de"

import itertools
import logging
import numpy as np
import operator
import os
import pandas as pd
import re
from platform import system
from subprocess import Popen, PIPE

log = logging.getLogger('mallet')
log.addHandler(logging.NullHandler())
logging.basicConfig(level=logging.ERROR,
                    format='%(levelname)s %(name)s: %(message)s')

def create_mallet_binary(path_to_mallet='mallet', path_to_file=False,
                         path_to_corpus=False, output_file=os.path.join('mallet_output', 'binary.mallet'),
                         encoding=None, token_regex=None, preserve_case=False,
                         remove_stopwords=True, stoplist=None, extra_stopwords=None,
                         stop_pattern_file=None, skip_header=False, skip_html=False,
                         replacement_files=None, deletion_files=None, gram_sizes=None,
                         keep_sequence=True, keep_sequence_bigrams=False,
                         binary_features=False, save_text_in_source=False,
                         print_output=False):
    """Creates a MALLET binary file.

    Description:
        bla

    Args:
        path_to_mallet (str): Path to MALLET. Defaults to 'mallet'. If MALLET is
            not properly installed, use absolute path, e.g. '/home/workspace/mallet/bin/mallet'.
        path_to_file (str): Absolute path to text file, e.g. '/home/workspace/testfile.txt'.
        path_to_corpus (str): Absolute path to corpus folder, e.g. '/home/workspace/corpus_txt'.
        output_file (str): Path to output plus filename, e.g. '/home/workspace/mallet_output/binary.mallet'.
        encoding (str): Character encoding for input file. Defaults to UTF-8.
        token_regex (str): Divides documents into tokens using a regular
            expression (supports Unicode regex). Defaults to \p{L}[\p{L}\p{P}]+\p{L}.
        preserve_case (bool): If false, converts all word features to lowercase. Defaults to False.
        remove_stopwords (bool): Ignores a standard list of very common English tokens. Defaults to True.
        stoplist (str): Absolute path to plain text stopword list. Defaults to None.
        extra_stopwords (str): Read whitespace-separated words from this file,
            and add them to either the default English stoplist or the list
            specified by `stoplist`. Defaults to None.
        stop_pattern_file (str): Read regular expressions from a file, one per
            line. Tokens matching these regexps will be removed. Defaults to None.
        skip_header (bool): If true, in each document, remove text occurring
            before a blank line. This is useful for removing email or UseNet
            headers. Defaults to False.
        skip_html (bool): If true, remove text occurring inside <...>, as in
            HTML or SGML. Defaults to False.
        replacement_files (str): Files containing string replacements, one per
            line: 'A B [tab] C' replaces A B with C, 'A B' replaces A B with A_B.
            Defaults to None.
        deletion_files (str): Files containing strings to delete after
            `replacements_files` but before tokenization (i.e. multiword stop
            terms). Defaults to False.
        gram_sizes (int): Include among the features all n-grams of sizes
            specified. For example, to get all unigrams and bigrams, use `gram_sizes=1,2`.
            This option occurs after the removal of stop words, if removed.
            Defaults to None.
        keep_sequence (bool): Preserves the document as a sequence of word features,
            rather than a vector of word feature counts. Use this option for sequence
            labeling tasks. MALLET also requires feature sequences rather than
            feature vectors. Defaults to True.
        keep_sequence_bigrams (bool): If true, final data will be a
            FeatureSequenceWithBigrams rather than a FeatureVector. Defaults to False.
        binary_features (bool): If true, features will be binary. Defaults to False.
        save_text_in_source (bool): If true, save original text of document in source.
            Defaults to False.
        print_output (bool): If true, print a representation of the processed data
            to standard output. This option is intended for debugging. Defaults to
            False.

    Returns:
        String. Absolute path to created MALLET binary file.
    """
    try:
        if re.search(r'\s', path_to_mallet):
            raise ValueError("Whitespaces are not allowed in '%s'." %path_to_mallet)

        if re.search(r'\s', output_file):
            raise ValueError("Whitespaces are not allowed in '%s'." %output_file)

        if system() == 'Windows':
            shell = True
        else:
            shell = False
            
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
            
        param = [path_to_mallet]
        if not path_to_file:
            if re.search(r'\s', path_to_corpus):
                raise ValueError(" {path_to_corpus} Whitespaces are not allowed in '%s'." %path_to_corpus)
            param.extend(['import-dir', '--input', str(path_to_corpus)])
        else:
            if re.search(r'\s', path_to_file):
                raise ValueError("Whitespaces are not allowed in '%s'." %path_to_file)
            param.extend(['import-file', '--input', str(path_to_file)])
            
        if encoding is not None:
            param.extend(['--encoding', str(encoding)])
        if token_regex is not None:
            param.extend(['--token-regex', str(token_regex)])
        if preserve_case:
            param.extend('--preserve-case')
        if remove_stopwords:
            param.extend('--remove-stopwords')
        if stoplist is not None:
            if re.search(r'\s', stoplist):
                raise ValueError("Whitespaces are not allowed in '%s'." %stoplist)
            param.extend(['--stoplist-file', str(stoplist)])
        if extra_stopwords is not None:
            if re.search(r'\s', extra_stopwords):
                raise ValueError("Whitespaces are not allowed in '%s'." %extra_stopwords)
            param.extend(['--extra-stopwords', str(extra_stopwords)])
        if stop_pattern_file is not None:
            if re.search(r'\s', stop_pattern_file):
                raise ValueError("Whitespaces are not allowed in '%s'." %stop_pattern_file)
            param.extend(['--stop-pattern-file', str(stop_pattern_file)])
        if skip_header:
            param.extend('--skip-header')
        if skip_html:
            param.extend('--skip-html')
        if replacement_files is not None:
            if re.search(r'\s', replacement_files):
                raise ValueError("Whitespaces are not allowed in '%s'." %replacement_files)
            param.extend(['--replacement-files', str(replacement_files)])
        if deletion_files is not None:
            if re.search(r'\s', deletion_files):
                raise ValueError("Whitespaces are not allowed in '%s'." %deletion_files)
            param.extend(['--deletion-files', str(deletion_files)])
        if gram_sizes is not None:
            param.extend(['--gram-sizes', str(gram_sizes)])
        if keep_sequence:
            param.extend('--keep-sequence')
        if keep_sequence_bigrams:
            param.extend('--keep-sequence-bigrams')
        if binary_features:
            param.extend('--binary-features')
        if save_text_in_source:
            param.extend('--save-text-in-source')
        if print_output:
            param.extend(['--print-output', '--output', str(output_file)])


        log.info("Running MALLET with %s ...", ' '.join(param))
        log.info("Saving MALLET binary to %s ...", output_file)
        with open('mallet.log', 'wb') as f:
            p = Popen(param, stdout=PIPE, stderr=PIPE, shell=shell)
            f.write(p.communicate()[1])
        return output_file
    except:
        log.error("Mallet model could not be created.", exc_info=True)
        raise



def create_mallet_model(path_to_mallet='mallet', path_to_binary=None, input_model=None,
                        input_state=None, folder_for_output='tutorial_supplementals/mallet_output',
                        output_model=False, output_model_interval=0, output_state=False,
                        output_state_interval=0, inferencer_file=False, evaluator_file=False,
                        output_topic_keys=True, topic_word_weights_file=True,
                        word_topic_counts_file=False, diagnostics_file=False, xml_topic_report=False,
                        xml_topic_phrase_report=False, output_topic_docs=False, num_top_docs=100,
                        output_doc_topics=True, doc_topics_threshold=0.0,
                        num_topics=10, num_top_words=10,
                        num_iterations=1000, num_threads=1, num_icm_iterations=0,
                        no_inference=False, random_seed=0, optimize_interval=0,
                        optimize_burn_in=200, use_symmetric_alpha=False, alpha=5.0,
                        beta=0.01):
    """Creates MALLET model.

    Description:
        bla

    Args:
        path_to_mallet (str): Path to MALLET. Defaults to 'mallet'. If MALLET is
            not properly installed use absolute path, e.g. '/home/workspace/mallet/bin/mallet'.
        path_to_binary (str): Path to previously created MALLET binary.
        input_model (str): Absolute path to the binary topic model created by `output_model`.
        input_state (str): Absolute path to the gzipped Gibbs sampling state created by `output_state`.
        folder_for_output (str): Folder for MALLET output.
        output_model (bool): Write a serialized MALLET topic trainer object.
            This type of output is appropriate for pausing and restarting training,
            but does not produce data that can easily be analyzed. Defaults to True.
        output_model_interval (int): The number of iterations between writing the
            model (and its Gibbs sampling state) to a binary file. You must also
            set the `output_model` parameter to use this option, whose argument
            will be the prefix of the filenames. Defaults to 0.
        output_state (bool): Write a compressed text file containing the words
            in the corpus with their topic assignments. The file format can easily
            be parsed and used by non-Java-based software. Defaults to True.
        output_state_interval (int): The number of iterations between writing the
            sampling state to a text file. You must also set the `output_state`
            to use this option, whose argument will be the prefix of the filenames.
            Defaults to 0.
        inference_file (bool): A topic inferencer applies a previously trained
            topic model to new documents. Defaults to False.
        evaluator_file (bool): A held-out likelihood evaluator for new documents.
            Defaults to False.
        output_topic_keys (bool): Write the top words for each topic and any
            Dirichlet parameters. Defaults to True.
        topic_word_weights_file (bool): Write unnormalized weights for every
            topic and word type. Defaults to True.
        word_topic_counts_file (bool): Write a sparse representation of topic-word
            assignments. By default this is null, indicating that no file will
            be written. Defaults to True.
        diagnostics_file (bool): Write measures of topic quality, in XML format.
            Defaults to True.
        xml_topic_report (bool): Write the top words for each topic and any
            Dirichlet parameters in XML format. Defaults to True.
        xml_topic_phrase_report (bool): Write the top words and phrases for each
            topic and any Dirichlet parameters in XML format. Defaults to True.
        output_topic_docs (bool): Currently not available. Write the most prominent
            documents for each topic, at the end of the iterations. Defaults to False.
        num_top_docs (int): Currently not available. Number of top documents for
            `output_topic_docs`. Defaults to False.
        output_doc_topics (bool): Write the topic proportions per document, at
            the end of the iterations. Defaults to True.
        doc_topics_threshold (float): Do not print topics with proportions less
            than this threshold value within `output_doc_topics`. Defaults to 0.0.
        num_topics (int): Number of topics. Defaults to 10.
        num_top_words (int): Number of keywords for each topic. Defaults to 10.
        num_interations (int): Number of iterations. Defaults to 1000.
        num_threads (int): Number of threads for parallel training.  Defaults to 1.
        num_icm_iterations (int): Number of iterations of iterated conditional
            modes (topic maximization).  Defaults to 0.
        no_inference (bool): Load a saved model and create a report. Equivalent
            to `num_iterations = 0`. Defaults to False.
        random_seed (int): Random seed for the Gibbs sampler. Defaults to 0.
        optimize_interval (int): Number of iterations between reestimating
            dirichlet hyperparameters. Defaults to 0.
        optimize_burn_in (int): Number of iterations to run before first
            estimating dirichlet hyperparameters. Defaults to 200.
        use_symmetric_alpha (bool): Only optimize the concentration parameter of
            the prior over document-topic distributions. This may reduce the
            number of very small, poorly estimated topics, but may disperse common
            words over several topics. Defaults to False.
        alpha (float): Sum over topics of smoothing over doc-topic distributions.
            alpha_k = [this value] / [num topics]. Defaults to 5.0.
        beta (float): Smoothing parameter for each topic-word. Defaults to 0.01.


    ToDo:
        Param 'output_topic_docs' is causing an internal error
        -> Exception in thread "main" java.lang.ClassCastException: java.net.URI cannot be cast to java.lang.String
        -> at cc.mallet.topics.ParallelTopicModel.printTopicDocuments(ParallelTopicModel.java:1773)
        -> at cc.mallet.topics.tui.TopicTrainer.main(TopicTrainer.java:281)
        Param, 'num_top_docs' is obsolete, refering to 'output_topic_docs'

    Returns:
        Nothing.
    """
    try:
        if re.search(r'\s', path_to_mallet):
            raise ValueError("Whitespaces are not allowed in '%s'." %path_to_mallet)

        if re.search(r'\s', folder_for_output):
            raise ValueError("Whitespaces are not allowed in '%s'." %folder_for_output)

        if system() == 'Windows':
            shell = True
        else:
            shell = False

        os.makedirs(folder_for_output, exist_ok=True)
        param = [path_to_mallet, 'train-topics']
        
        if input_model is None:
            param.append('--input')
        else:
            if re.search(r'\s', input_model):
                raise ValueError("Whitespaces are not allowed in '%s'." %input_model)
            param.extend(['--input-model', str(input_model)])
        if path_to_binary is not None:
            if re.search(r'\s', path_to_binary):
                raise ValueError("Whitespaces are not allowed in '%s'." %path_to_binary)
            param.append(path_to_binary)
        if input_state is not None:
            if re.search(r'\s', input_state):
                raise ValueError("Whitespaces are not allowed in '%s'." %input_state)
            param.extend(['--input-state', str(input_state)])

        log.debug("Choosing parameters ...")
        if num_topics:
            param.extend(['--num-topics', str(num_topics)])
        if num_iterations:
            param.extend(['--num-iterations', str(num_iterations)])
        if num_threads:
            param.extend(['--num-threads', str(num_threads)])
        if num_top_words:
            param.extend(['--num-top-words', str(num_top_words)])
        if num_icm_iterations:
            param.extend(['--num-icm-iterations', str(num_icm_iterations)])
        if no_inference:
            param.extend(['--no-inference', str(no_inference)])
        if random_seed:
            param.extend(['--random-seed', str(random_seed)])

        log.debug("Choosing hyperparameters ...")
        if optimize_interval is not None:
            param.extend(['--optimize-interval', str(optimize_interval)])
        if optimize_burn_in is not None:
            param.extend(['--optimize-burn-in', str(optimize_burn_in)])
        if use_symmetric_alpha is not None:
            param.extend('--use-symmetric-alpha')
        if alpha is not None:
            param.extend(['--alpha', str(alpha)])
        if beta is not None:
            param.extend(['--beta', str(beta)])

        log.debug("Choosing output parameters ...")
        if output_topic_keys:
            param.extend(['--output-topic-keys',
                         os.path.join(folder_for_output, 'topic_keys.txt')])
        if output_doc_topics:
            param.extend(['--output-doc-topics', 
                         os.path.join(folder_for_output, 'doc_topics.txt')])
            if doc_topics_threshold is not None:
                param.extend(['--doc-topics-threshold', str(doc_topics_threshold)])
        if topic_word_weights_file:
            param.extend(['--topic-word-weights-file',
                         os.path.join(folder_for_output, 'topic_word_weights.txt')])
        if word_topic_counts_file:
            param.extend(['--word-topic-counts-file',
                         os.path.join(folder_for_output, 'word_topic_counts.txt')])
        if diagnostics_file:
            param.extend(['--diagnostics-file',
                         os.path.join(folder_for_output, 'diagnostics.xml')])
        if xml_topic_report:
            param.extend(['--xml-topic-report',
                         os.path.join(folder_for_output, 'topic_report.xml')])
        if xml_topic_phrase_report:
            param.extend(['--xml-topic-phrase-report',
                         os.path.join(folder_for_output, 'topic_phrase_report.xml')])
        if output_model:
            param.extend(['--output-model', 
                         os.path.join(folder_for_output, 'mallet.model')])
            if output_model_interval is not None:
                param.extend(['--output-model-interval',
                             str(output_model_interval)])
        if output_state:
            param.extend(['--output-state',
                         os.path.join(folder_for_output, 'state.gz')])
            if output_state_interval is not None:
                param.extend(['--output-state-interval',
                             str(output_state_interval)])
        if inferencer_file:
            param.extend(['--inferencer-filename', 
                         os.path.join(folder_for_output, 'inferencer')])
        if evaluator_file:
            param.extend(['--evaluator-filename',
                         os.path.join(folder_for_output, 'evaluator')])
        # not yet working
        if output_topic_docs:
            param.extend(['--output-topic-docs',
                         os.path.join(folder_for_output, 'topic_docs.txt')])
        if num_top_docs is not None:
            param.extend(['--num-top-docs',
                         str(topic_word_weights_file)])

        with open('mallet.log', 'wb') as f:
            p = Popen(param, stdout=PIPE, stderr=PIPE, shell=shell)
            f.write(p.communicate()[1])
    except:
        log.error("Mallet model could not be created.", exc_info=True)
        raise


def _grouper(n, iterable, fillvalue=None):
    """Collects data into fixed-length chunks or blocks.

    Args:
        n (int): Length of chunks or blocks
        iterable (object): Iterable object
        fillvalue (boolean): If iterable can not be devided into evenly-sized chunks fill chunks with value.

    Returns: n-sized chunks

    """

    args=[iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def show_doc_topic_matrix(output_folder, doc_topics='doc_topics.txt', topic_keys='topic_keys.txt',
                          easy_file_format=False):
    """Shows document-topic-mapping.
    Args:
        outfolder (str): Folder for MALLET output.
        doc_topics (str): Name of MALLET's doc_topic file. Defaults to 'doc_topics.txt'.
        topic_keys (str): Name of MALLET's topic_keys file. Defaults to 'topic_keys.txt'.

    ToDo: Prettify docnames
    
    Example:    
        >>> outfolder = "tutorial_supplementals/mallet_output"
        >>> df = show_doc_topic_matrix(outfolder)
        >>> len(df.T)
        17
    """

    doc_topics=os.path.join(output_folder, doc_topics)
    assert doc_topics
    topic_keys=os.path.join(output_folder, topic_keys)
    assert topic_keys

    doctopic_triples=[]
    mallet_docnames=[]
    topics=[]

    df=pd.read_csv(topic_keys, sep='\t', header=None, encoding='utf-8')
    labels=[]
    for index, item in df.iterrows():
        label=' '.join(item[2].split()[:3])
        labels.append(label)

    with open(doc_topics, encoding='utf-8') as f:
        for line in f:
            li=line.lstrip()
            if li.startswith("#"):
                lines=f.readlines()
                for line in lines:
                    docnum, docname, *values=line.rstrip().split('\t')
                    mallet_docnames.append(docname)
                    for topic, share in _grouper(2, values):
                        triple=(docname, int(topic), float(share))
                        topics.append(int(topic))
                        doctopic_triples.append(triple)
            else:
                easy_file_format=True
                break

    if easy_file_format:
        newindex=[]
        doc_topic_matrix=pd.read_csv(
            doc_topics, sep='\t', names=labels[0:], encoding='utf-8')
        for eins, zwei in doc_topic_matrix.index:
            newindex.append(os.path.basename(zwei))
        doc_topic_matrix.index=newindex
    else:
        # sort the triples
        # triple is (docname, topicnum, share) so sort(key=operator.itemgetter(0,1))
        # sorts on (docname, topicnum) which is what we want
        doctopic_triples=sorted(
            doctopic_triples, key=operator.itemgetter(0, 1))

        # sort the document names rather than relying on MALLET's ordering
        mallet_docnames=sorted(mallet_docnames)

        # collect into a document-term matrix
        num_docs=len(mallet_docnames)

        num_topics=max(topics) + 1

        # the following works because we know that the triples are in
        # sequential order
        data=np.zeros((num_docs, num_topics))

        for triple in doctopic_triples:
            docname, topic, share=triple
            row_num=mallet_docnames.index(docname)
            data[row_num, topic]=share

        topicLabels=[]

        # creates list of topic lables consisting of the 3 most weighed topics
        df=pd.read_csv(topic_keys, sep='\t', header=None, encoding='utf-8')
        labels=[]
        for index, item in df.iterrows():

            topicLabel=' '.join(item[2].split()[:3])
            topicLabels.append(topicLabel)

        shortened_docnames=[]
        for item in mallet_docnames:
            shortened_docnames.append(os.path.basename(item))

        '''
        for topic in range(max(topics)+1):
        topicLabels.append("Topic_" + str(topic))
        '''
        doc_topic_matrix=pd.DataFrame(data=data[0:, 0:],
                                      index=shortened_docnames[0:],
                                      columns=topicLabels[0:])
    return doc_topic_matrix.T

def show_topics_keys(output_folder, topicsKeyFile="topic_keys.txt", num_topics=10, key_per_topic=10):
    """Show topic-key-mapping.

    Args:
        outfolder (str): Folder for Mallet output,
        topicsKeyFile (str): Name of Mallets' topic_key file, default "topic_keys"

    #topic-model-mallet
    Note: FBased on DARIAH-Tutorial -> https://de.dariah.eu/tatom/topic_model_mallet.html

    ToDo: Prettify index
    
    Example:    
        >>> outfolder = "tutorial_supplementals/mallet_output"
        >>> df = show_topics_keys(outfolder, num_topics=10)
        >>> len(df)
        10
    """

    path_to_topic_keys=os.path.join(output_folder, topicsKeyFile)
    assert path_to_topic_keys

    with open(path_to_topic_keys, encoding='utf-8') as input:
        topic_keys_lines=input.readlines()

    topic_keys=[]
    #topicLabels=[]


    for line in topic_keys_lines:
        _, _, words=line.split('\t')  # tab-separated
        words=words.rstrip().split(' ')  # remove the trailing '\n'
        topic_keys.append(words)

    topicKeysMatrix=pd.DataFrame(topic_keys)
    topicKeysMatrix.index=['Topic ' + str(x + 1) for x in range(num_topics)]
    topicKeysMatrix.columns=['Key ' + str(x + 1) for x in range(key_per_topic)]
    return topicKeysMatrix
