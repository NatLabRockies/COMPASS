"""COMPASS validation utilities"""


def step_based_threshold(num_chunks):
    """Generate a threshold based on number of chunks

    Parameters
    ----------
    num_chunks : int
        Number of chunks being considered in the validation. This is
        used to determine how strict the validation should be, with more
        chunks generally requiring a higher fraction of chunks to pass
        for the document to be considered valid (but never no more than
        80%).

    Returns
    -------
    float
        Threshold value between 0.5 and 0.8, where higher values
        indicate a stricter requirement for the fraction of chunks that
        must pass the validation.
    """
    if num_chunks <= 2:  # ruff:ignore[magic-value-comparison]
        return min(1 / 1, 1 / 2)
    if num_chunks <= 6:  # ruff:ignore[magic-value-comparison]
        return min(2 / 3, 3 / 4, 3 / 5, 4 / 6)
    if num_chunks <= 9:  # ruff:ignore[magic-value-comparison]
        return min(5 / 7, 6 / 8, 7 / 9)
    return 0.8
