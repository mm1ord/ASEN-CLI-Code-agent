from asen_cli.ui.slash import help_text, parse_slash_command


def test_parse_slash_command_with_args():
    parsed = parse_slash_command("/config model")

    assert parsed is not None
    assert parsed.name == "/config"
    assert parsed.args == ["model"]


def test_parse_non_slash_returns_none():
    assert parse_slash_command("read hello.py") is None


def test_help_text_lists_core_commands():
    text = help_text()

    assert "/help" in text
    assert "/tools" in text
    assert "/paste" in text
    assert "/rename" in text
    assert "!<command>" in text
