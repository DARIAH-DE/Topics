"""
dariah.mallet.core
~~~~~~~~~~~~~~~~~~

This module implements the core functions of the MALLET sub-package.
"""

from dariah.mallet import utils


def call(command, executable, **parameters):
    """Call MALLET.

    Parameter:
        command (str): Command for MALLET.
        executable (str): Path to MALLET executable.
        **parameter: Additional parameters for MALLET.

    Returns:
        True, if call was successful.
    """
    # Basic subprocess command:
    args = [executable, command]

    # Append additional parameters:
    for parameter, value in parameters.items():
        # Support synonyms for `-input` parameter:
        if parameter in {"filepath", "directory", "path", "corpus"}:
            args.append("--input")
        elif parameter in {"random_seed", "random_state"}:
            if value:
                args.append("--random-seed")
        else:
            args.append("--{}".format(parameter.replace("_", "-")))
        if value and value != True:
            args.append(str(value))
    return utils.call(args)
