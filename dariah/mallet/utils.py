"""
dariah.utils
~~~~~~~~~~~~

This module implements general helper functions.
"""

import logging
import subprocess


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def call(args: list) -> bool:
    """Call a subprocess.

    Parameter:
        args (list): The subprocess’ arguments.

    Returns:
        True, if call was successful.
    """
    for message in _process(args):
        if len(message) < 100 and len(message) > 5:
            logger.info(message)
    return True


def _process(args: list):
    """Construct a process object.
    """
    popen = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    # Yield every line of stdout:
    for line in iter(popen.stderr.readline, ""):
        yield line.strip()
    popen.stdout.close()
    code = popen.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)
