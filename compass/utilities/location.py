"""COMPASS Ordinance jurisdiction specification utilities"""

from functools import cached_property


JURISDICTION_TYPES_AS_PREFIXES = {
    "town",
    "township",
    "city",
    "borough",
    "village",
    "unorganized territory",
}


class Jurisdiction:
    """Class representing a jurisdiction"""

    def __init__(
        self,
        subdivision_type,
        state,
        county=None,
        subdivision_name=None,
        code=None,
    ):
        """

        Parameters
        ----------
        subdivision_type : str
            Type of subdivision that this jurisdiction represents.
            Typical values are "state", "county", "town", "city",
            "borough", "parish", "township", etc.
        state : str
            Name of the state containing the jurisdiction.
        county : str, optional
            Name of the county containing the jurisdiction, if
            applicable. If the jurisdiction represents a state, leave
            this input unspecified. If the jurisdiction represents a
            county or a subdivision within a county, provide the county
            name here.

            .. IMPORTANT:: Make sure this input is capitalized properly!

            By default, ``None``.
        subdivision_name : str, optional
            Name of the subdivision that the jurisdiction represents, if
            applicable. If the jurisdiction represents a state or
            county, leave this input unspecified. Otherwise, provide the
            jurisdiction name here.

            .. IMPORTANT:: Make sure this input is capitalized properly!

            By default, ``None``.
        code : int or str, optional
            Optional jurisdiction code (typically FIPS or similar).
            By default, ``None``.
        """
        self.type = subdivision_type.title()
        self.state = state.title()
        self.county = county
        self.subdivision_name = subdivision_name
        self.code = code

    @cached_property
    def full_name(self):
        """str: Full jurisdiction name"""
        name_parts = [
            self.full_subdivision_phrase,
            self.full_county_phrase,
            self.state,
        ]

        return ", ".join(filter(None, name_parts))

    @cached_property
    def full_name_the_prefixed(self):
        """str: Full jurisdiction name with `the` prefix if needed"""
        if self.type.casefold() == "state":
            return f"the state of {self.state}"

        if self.type.casefold() in JURISDICTION_TYPES_AS_PREFIXES:
            return f"the {self.full_name}"

        return self.full_name

    @cached_property
    def full_subdivision_phrase(self):
        """str: Full jurisdiction subdivision phrase, or empty str"""
        if not self.subdivision_name:
            return ""

        if self.type.casefold() in JURISDICTION_TYPES_AS_PREFIXES:
            return f"{self.type} of {self.subdivision_name}"

        return f"{self.subdivision_name} {self.type}"

    @cached_property
    def full_subdivision_phrase_the_prefixed(self):
        """str: Full jurisdiction subdivision phrase, or empty str"""
        if self.type.casefold() in JURISDICTION_TYPES_AS_PREFIXES:
            return f"the {self.full_subdivision_phrase}"

        return self.full_subdivision_phrase

    @cached_property
    def full_county_phrase(self):
        """str: Full jurisdiction county phrase, or empty str"""
        if not self.county:
            return ""

        if not self.subdivision_name:
            return f"{self.county} {self.type}"

        return f"{self.county} County"

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.full_name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.type.casefold() == other.type.casefold()
                and self.state.casefold() == other.state.casefold()
                and self.county == other.county
                and self.subdivision_name == other.subdivision_name
            )
        if isinstance(other, str):
            return self.full_name.casefold() == other.casefold()
        return False

    def __hash__(self):
        return hash(self.full_name.casefold())
