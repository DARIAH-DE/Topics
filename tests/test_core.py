import pytest
import pandas as pd

import dariah


@pytest.fixture
def dtm():
    return pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["AAA", "BBB", "CCC"],
        index=["a", "b", "c"],
    )


@pytest.fixture
def riddell_topics():
    return pd.DataFrame(
        {
            "word0": {"topic0": "AAA", "topic1": "CCC"},
            "word1": {"topic0": "CCC", "topic1": "BBB"},
            "word2": {"topic0": "BBB", "topic1": "AAA"},
        }
    )


@pytest.fixture
def riddell_topic_word():
    return pd.DataFrame(
        {
            "AAA": {"topic0": 0.9981867633726201, "topic1": 0.02967969438730532},
            "BBB": {"topic0": 0.0009066183136899366, "topic1": 0.4410813987657949},
            "CCC": {"topic0": 0.0009066183136899366, "topic1": 0.5292389068468998},
        }
    )


@pytest.fixture
def riddell_topic_document():
    return pd.DataFrame(
        {
            "a": {"topic0": 0.01612903225806452, "topic1": 0.9838709677419355},
            "b": {"topic0": 0.26973684210526316, "topic1": 0.7302631578947368},
            "c": {"topic0": 0.2933884297520661, "topic1": 0.7066115702479339},
        }
    )


@pytest.fixture
def riddell_topic_similarities():
    return pd.DataFrame(
        {
            "topic0": {"topic0": 1.0000000000000002, "topic1": 2.6409361679102195},
            "topic1": {"topic0": 0.21001814797045967, "topic1": 0.9999999999999999},
        }
    )


@pytest.fixture
def riddell_document_similarities():
    return pd.DataFrame(
        {
            "a": {
                "a": 0.9999999999999999,
                "b": 0.7465284651715263,
                "c": 0.7228895865992244,
            },
            "b": {
                "a": 1.1927144048546063,
                "b": 1.0000000000000002,
                "c": 0.9820273609083001,
            },
            "c": {
                "a": 1.1957201277450218,
                "b": 1.0166958876685324,
                "c": 0.9999999999999999,
            },
        }
    )


@pytest.fixture
def mallet_topics():
    return pd.DataFrame(
        {
            "word0": {"topic0": "aaa", "topic1": "ccc"},
            "word1": {"topic0": "bbb", "topic1": "bbb"},
            "word2": {"topic0": "ccc", "topic1": None},
        }
    )


@pytest.fixture
def mallet_topic_word():
    return pd.DataFrame(
        {
            "aaa": {"topic0": 9.01, "topic1": 0.01},
            "bbb": {"topic0": 7.01, "topic1": 8.01},
            "ccc": {"topic0": 6.01, "topic1": 12.01},
        }
    )


@pytest.fixture
def mallet_topic_document():
    return pd.DataFrame(
        {
            "a": {"topic0": 0.2058823529411765, "topic1": 0.7941176470588236},
            "b": {"topic0": 0.7127659574468086, "topic1": 0.28723404255319146},
            "c": {"topic0": 0.4783549783549784, "topic1": 0.5216450216450217},
        }
    )


@pytest.fixture
def mallet_topic_similarities():
    return pd.DataFrame(
        {
            "topic0": {"topic0": 1.0, "topic1": 0.7927620823177987},
            "topic1": {"topic0": 0.6270117939000307, "topic1": 1.0},
        }
    )


@pytest.fixture
def mallet_document_similarities():
    return pd.DataFrame(
        {
            "a": {
                "a": 0.9999999999999999,
                "b": 0.556965487064486,
                "c": 0.7618491191755972,
            },
            "b": {
                "a": 0.6347484950285212,
                "b": 0.9999999999999999,
                "c": 0.8310875275229433,
            },
            "c": {"a": 1.0235465765588327, "b": 0.97974263999169, "c": 1.0},
        }
    )


@pytest.fixture
def random_state():
    return 23


def test_read_mallet_topics(tmpdir):
    p = tmpdir.mkdir("sub").join("topics.txt")
    p.write("0\t0.05\tfoo bar\n1\t0.05\tfoo bar")
    topics = dariah.core.utils.read_mallet_topics(p, num_words=2)
    assert list(topics) == [["foo", "bar"], ["foo", "bar"]]


def test_riddell_lda(
    dtm,
    riddell_topics,
    riddell_topic_word,
    riddell_topic_document,
    riddell_topic_similarities,
    riddell_document_similarities,
    random_state,
):
    lda = dariah.core.modeling.LDA(
        num_topics=2, num_iterations=10, random_state=random_state
    )
    lda.fit(dtm)

    assert lda.topics.sum().sum() == riddell_topics.sum().sum()
    assert lda.topic_word.sum().sum() == riddell_topic_word.sum().sum()
    assert lda.topic_document.sum().sum() == riddell_topic_document.sum().sum()
    assert lda.topic_similarities.sum().sum() == riddell_topic_similarities.sum().sum()
    assert (
        lda.document_similarities.sum().sum()
        == riddell_document_similarities.sum().sum()
    )


def test_mallet_lda(
    dtm,
    mallet_topics,
    mallet_topic_word,
    mallet_topic_document,
    mallet_topic_similarities,
    mallet_document_similarities,
):
    lda = dariah.core.modeling.LDA(
        num_topics=2, num_iterations=10, random_state=23, mallet="mallet"
    )
    lda.fit(dtm)

    assert lda.topics.sum().sum() == mallet_topics.sum().sum()
    assert lda.topic_word.sum().sum() == mallet_topic_word.sum().sum()
    assert lda.topic_document.sum().sum() == mallet_topic_document.sum().sum()
    assert lda.topic_similarities.sum().sum() == mallet_topic_similarities.sum().sum()
    assert (
        lda.document_similarities.sum().sum()
        == mallet_document_similarities.sum().sum()
    )
