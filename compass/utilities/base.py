"""Base COMPASS utility functions"""


def title_preserving_caps(string):
    """Convert string to title case, preserving existing capitalization

    Parameters
    ----------
    string : str
        Input string potentially containing capitalized words.

    Returns
    -------
    str
        String converted to title case, preserving existing
        capitalization.
    """
    return " ".join(map(_cap, string.split(" ")))


def _cap(word):
    return "".join([word[0].upper(), word[1:]])
