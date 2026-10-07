"""COMPASS plugin utilities"""

from warnings import warn

from compass.exceptions import COMPASSPluginConfigurationError
from compass.warn import COMPASSPluginConfigurationWarning


_ORDINANCE_WEBSITE_KEYWORDS = {
    "planning",
    "plan",
    "government",
    "zoning",
    "land",
    "municipal",
    "department",
}


def normalize_website_keywords(raw):
    """Normalize website keyword tiers or legacy score mappings

    Parameters
    ----------
    raw : dict or list
        The raw website keywords, either as a dictionary of scores or a
        list of keyword tiers.

    Returns
    -------
    dict
        A normalized dictionary of website keywords with scores.
    """
    if isinstance(raw, dict):
        keywords = _augment_website_keywords(raw)
    elif isinstance(raw, list):
        keywords = _keyword_weights_from_tiers(raw)
    else:
        msg = (
            "Website keywords must be a dictionary of scores or a list of "
            "keyword tiers."
        )
        raise COMPASSPluginConfigurationError(msg)

    _warn_if_ordinance_website_keywords_are_missing(keywords)
    return keywords


def _keyword_weights_from_tiers(tiers):
    """Compute flat keyword scores from ordered keyword tiers"""
    if not tiers:
        msg = "Website keywords must contain at least one keyword tier."
        raise COMPASSPluginConfigurationError(msg)

    expanded_tiers = []
    known_keywords = set()
    for tier_ind, tier in enumerate(tiers, start=1):
        expanded = _expand_website_keyword_tier(tier, tier_ind)
        collisions = known_keywords.intersection(expanded)
        if collisions:
            msg = (
                "Website keyword tiers must not contain duplicate keywords "
                f"or URL variants: {sorted(collisions)}."
            )
            raise COMPASSPluginConfigurationError(msg)

        known_keywords.update(expanded)
        expanded_tiers.append(expanded)

    keywords = {}
    weight = 1
    for tier in reversed(expanded_tiers):
        keywords.update(dict.fromkeys(tier, weight))
        weight *= len(tier) + 1

    return keywords


def _expand_website_keyword_tier(tier, tier_ind):
    """Expand and validate one website keyword tier"""
    if isinstance(tier, str):
        tier = [tier]

    if not isinstance(tier, list) or not tier:
        msg = (
            f"Website keyword tier {tier_ind} must be a non-empty list "
            "or string."
        )
        raise COMPASSPluginConfigurationError(msg)

    expanded = {}
    for keyword in tier:
        if not isinstance(keyword, str) or not keyword.strip():
            msg = (
                f"Website keyword tier {tier_ind} must contain "
                "non-empty strings."
            )
            raise COMPASSPluginConfigurationError(msg)

        variants = _augment_website_keywords({keyword: None})
        collisions = set(expanded).intersection(variants)
        if collisions:
            msg = (
                "Website keyword tiers must not contain duplicate keywords "
                f"or URL variants: {sorted(collisions)}."
            )
            raise COMPASSPluginConfigurationError(msg)
        expanded.update(variants)

    return expanded


def _warn_if_ordinance_website_keywords_are_missing(keywords):
    """Warn when ordinance-oriented crawl keywords are absent"""
    missing = _ORDINANCE_WEBSITE_KEYWORDS - set(keywords)
    if not missing:
        return

    msg = (
        "Website keywords are missing ordinance-oriented terms: "
        f"{sorted(missing)}. These keywords help push link prioritization "
        "toward ordinance documents. Add the missing keywords to silence "
        "this warning. If they are intentionally excluded, set their scores "
        "to 0 in a flat score mapping so they have no influence."
    )
    warn(msg, COMPASSPluginConfigurationWarning)


def _augment_website_keywords(keywords):
    """Add URL-encoded variants for multi-word keywords"""
    augmented = dict(keywords)
    for keyword, score in list(augmented.items()):
        if not isinstance(keyword, str):
            continue

        if " " not in keyword:
            continue

        encoded = keyword.replace(" ", "%20")
        if encoded not in augmented:
            augmented[encoded] = score

        plus_encoded = keyword.replace(" ", "+")
        if plus_encoded not in augmented:
            augmented[plus_encoded] = score

    return augmented
