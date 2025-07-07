"""COMPASS web crawling tests"""

from pathlib import Path

import pytest
from crawl4ai.models import Link as TestLink

from compass.web.website_crawl import Link


def test_link_equality():
    """Test equality of Link instances"""

    assert Link() == Link()
    assert Link() == Link(title="test")
    assert Link() == Link(text="test")
    assert Link() == Link(base_domain="test")
    assert Link() != Link(href="test")

    link1 = TestLink(title="test", href="http://example.com/test")
    link2 = Link(title="Test", href="http://example.com/test")

    assert link1 == link2
    assert link2 == "http://example.com/test"

    assert link2 != "http://example.com/test2"

    assert link2 in {"http://example.com/test", "http://example.com/test2"}
    assert link2 not in {
        "http://example.com/test2",
        "http://example.com/test3",
    }

    assert "http://example.com/test" in {link2}


def test_link_consistent_domain():
    """Test `Link.consistent_domain` property"""

    assert Link().consistent_domain
    assert not Link(base_domain="example.com").consistent_domain
    assert Link(
        href="example.com/test", base_domain="example.com"
    ).consistent_domain


def test_link_resembles_pdf():
    """Test `Link.resembles_pdf` property"""

    assert not Link().resembles_pdf
    assert Link(title="example.pdf").consistent_domain
    assert Link(href="example.pdf").consistent_domain
    assert not Link(base_domain="example.pdf").consistent_domain


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
