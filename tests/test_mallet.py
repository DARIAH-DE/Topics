from pathlib import Path
import sys
import subprocess

import pytest

sys.path.insert(0, str(Path(".").absolute()))

from dariah.mallet import utils


def test_call():
    utils.call(["whoami"]) == True


def test_call_error():
    with pytest.raises(subprocess.CalledProcessError):
        utils.call(["whoami", "-foo"])
    with pytest.raises(FileNotFoundError):
        utils.call(["foo"])


EXECUTABLE = "mallet"
INPUTFILE = Path("test", "document.txt")
OUTPUTFILE = Path("test", "document.mallet")
OUTPUTTOPIC = Path("test", "topic.txt")


def test_mallet_call():
    assert core.call(executable=EXECUTABLE,
                     command="import-file",
                     filepath=INPUTFILE,
                     output=OUTPUTFILE)
    assert OUTPUTFILE.exists()
    OUTPUTFILE.unlink()


def test_mallet_class():
    mallet = api.MALLET(EXECUTABLE)
    assert mallet.executable == EXECUTABLE
    assert mallet.import_file(filepath=INPUTFILE,
                              output=OUTPUTFILE,
                              keep_sequence=True)
    assert OUTPUTFILE.exists()
    mallet.train_topics(filepath=OUTPUTFILE,
                        output_topic_keys=OUTPUTTOPIC)
    assert OUTPUTTOPIC.exists()
    OUTPUTFILE.unlink()
    OUTPUTTOPIC.unlink()


def test_mallet_exception():
    with pytest.raises(FileNotFoundError):
        mallet = api.MALLET("does-not-exist")
        mallet.import_file(filepath=INPUTFILE,
                           output=OUTPUTFILE)
