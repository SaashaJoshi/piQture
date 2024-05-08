"""Class for Min-Max Normalization of datasets."""


# pylint: disable=too-few-public-methods
class MinMaxNormalization:
    """Normalizes input values in range [min, max]."""

    def __init__(self, normalize_min, normalize_max):
        self.min = normalize_min
        self.max = normalize_max

    def __call__(self, x):
        """Normalizes data to a range [min, max]."""
        return (x - self.min) / (self.max - self.min)
