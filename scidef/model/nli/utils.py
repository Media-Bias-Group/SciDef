def hmean(a: float, b: float):
    from scipy.stats import hmean

    return float(hmean([a, b]))


def amean(a: float, b: float):
    return float((a + b) / 2)
