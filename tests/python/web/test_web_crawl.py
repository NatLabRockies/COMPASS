"""COMPASS web crawling tests"""

from pathlib import Path

import pytest
from crawl4ai.models import Link as TestLink

from compass.web.website_crawl import _Link


def test_link_equality():
    """Test equality of Link instances"""

    assert _Link() == _Link()
    assert _Link() == _Link(title="test")
    assert _Link() == _Link(text="test")
    assert _Link() == _Link(base_domain="test")
    assert _Link() != _Link(href="test")

    link1 = TestLink(title="test", href="http://example.com/test")
    link2 = _Link(title="Test", href="http://example.com/test")

    assert link1 == link2
    assert link2 == "http://example.com/test"

    assert link2 != "http://example.com/test2"

    assert link2 in {"http://example.com/test", "http://example.com/test2"}
    assert link2 not in {
        "http://example.com/test2",
        "http://example.com/test3",
    }

    assert link2 == "http://example.com/test"


def test_link_consistent_domain():
    """Test `Link.consistent_domain` property"""

    assert _Link().consistent_domain
    assert not _Link(base_domain="example.com").consistent_domain
    assert _Link(
        href="example.com/test", base_domain="example.com"
    ).consistent_domain


def test_link_resembles_pdf():
    """Test `Link.resembles_pdf` property"""

    assert not _Link().resembles_pdf
    assert _Link(title="example.pdf").resembles_pdf
    assert _Link(href="example.pdf").resembles_pdf
    assert not _Link(base_domain="example.pdf").resembles_pdf


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
