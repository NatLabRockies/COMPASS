"""COMPASS LLM calling tests"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from compass.llm.calling import PLAIN_TEXT_INSTRUCTIONS, ChatLLMCaller
from compass.validation.location import JurisdictionWebsiteValidator


def test_chat_system_message_asks_for_plain_text():
    """Chat answers are read as Yes/No, so they must not be Markdown"""
    caller = ChatLLMCaller(SimpleNamespace(), system_message="Be brief.")

    assert caller.messages == [
        {
            "role": "system",
            "content": f"Be brief.\n\n{PLAIN_TEXT_INSTRUCTIONS}",
        }
    ]


def test_website_check_system_message_asks_for_plain_text():
    """The main-website Yes/No check also asks for plain text"""
    system_message = JurisdictionWebsiteValidator.WEB_PAGE_CHECK_SYSTEM_MESSAGE

    assert system_message.endswith(PLAIN_TEXT_INSTRUCTIONS)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
