#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subprocess import check_output, STDOUT, CalledProcessError
from pathlib import Path
import pytest
import logging
import re

project_path = Path(__file__).absolute().parent.parent

def run_notebook(notebook_name):
    try:
        check_output(["jupyter-nbconvert", "--execute",
                     "--log-level=ERROR",
                     "--ExecutePreprocessor.iopub_timeout=30",
                      "--ExecutePreprocessor.timeout=None",
                    str(Path(project_path, 'notebooks', notebook_name))],
                     stderr=STDOUT, universal_newlines=True)
    except FileNotFoundError as e:
        raise SkipTest("jupyter-nbconvert not found. Cannot run integration test. "
                   + str(e))
    except CalledProcessError as e:
        message = e.output
        cellinfo = re.search('nbconvert.preprocessors.execute.CellExecutionError: (.+)$',
                  message, re.MULTILINE | re.DOTALL)
        if cellinfo:
            message = cellinfo.group(1)
        logging.error(message)

@pytest.mark.skip
def jupyter_lda_test():
    run_notebook("IntroducingLda.ipynb")

def jupyter_gensim_test():
    run_notebook("IntroducingGensim.ipynb")

def jupyter_MALLET_test():
    run_notebook("IntroducingMallet.ipynb")

def jupyter_Visualizations_test():
    run_notebook("Visualizations.ipynb")
	
def jupyter_DKPro_test():
	path = Path('/opt/ddw-0.4.6/ddw-0.4.6.jar')
	if not path.exists():
		raise SkipTest("Could not find DKProWrapper. Integration test for DKPro Notebook is skipped.")
	else:
		run_notebook("IntroducingLda_DKProWrapper.ipynb")
