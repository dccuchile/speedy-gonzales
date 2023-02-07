import numpy as np


def modified_z_score(data: np.ndarray) -> np.ndarray:
    """
    Calculates the modified z-scores (using the median and mad as robust estimators).


    The following implementation is based on:
    https://stackoverflow.com/a/16562028
    https://stackoverflow.com/a/22357811
    And was proposed by Iglewicz and Hoaglin - How to Detect and Handle Outliers
    https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf
    """
    differences = np.abs(data - np.median(data))
    mad = np.median(differences)
    modified_z_score = 0.6745 * (data - np.median(data)) / mad
    return modified_z_score


def filter_outliers(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    The value 3.5 for the threshold is the recommended on the book.
    """
    modified_z_scores = modified_z_score(data=data)
    return data[np.abs(modified_z_scores) < threshold]
