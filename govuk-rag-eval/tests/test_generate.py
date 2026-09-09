"""Answer synthesis — the deterministic offline generator."""

from src.config import from_dict
from src.generate import AnthropicGenerator, EchoGenerator, OpenAIGenerator, build_generator


def test_echo_answers_from_context():
    g = EchoGenerator(max_sentences=2)
    out = g.generate(
        "When do I register?",
        ["You must register by 5 October. Keep your UTR safe. Extra sentence."],
    )
    assert out == "You must register by 5 October. Keep your UTR safe."


def test_echo_refuses_without_context():
    g = EchoGenerator()
    out = g.generate("anything", [])
    assert "don't have enough information" in out


def test_echo_is_deterministic():
    g = EchoGenerator()
    ctx = ["A. B. C. D."]
    assert g.generate("q", ctx) == g.generate("q", ctx)


def test_build_generator_selects_provider():
    cfg = from_dict({"generation": {"provider": "echo"}})
    assert build_generator(cfg).generator_id == "echo"


def test_build_generator_openai_id_without_calling():
    cfg = from_dict({"generation": {"provider": "openai", "model": "gpt-4o-mini"}})
    # Construction must not require the SDK or a key (import is lazy).
    assert build_generator(cfg).generator_id == "openai-gpt-4o-mini"


def test_openai_blank_model_uses_default():
    cfg = from_dict({"generation": {"provider": "openai"}})  # model blank
    assert build_generator(cfg).generator_id == "openai-gpt-4o-mini"


def test_build_generator_anthropic_id_without_calling():
    cfg = from_dict({"generation": {"provider": "anthropic", "model": "claude-opus-5"}})
    # Construction must not require the anthropic SDK or a key (import is lazy).
    assert build_generator(cfg).generator_id == "anthropic-claude-opus-5"


def test_anthropic_blank_model_uses_default():
    cfg = from_dict({"generation": {"provider": "anthropic"}})  # model blank
    assert build_generator(cfg).generator_id == "anthropic-claude-sonnet-5"


def test_anthropic_falls_back_when_model_is_for_other_provider():
    # Flipping only `provider` (leaving a gpt-* model) still yields a Claude model.
    cfg = from_dict({"generation": {"provider": "anthropic", "model": "gpt-4o-mini"}})
    assert build_generator(cfg).generator_id == "anthropic-claude-sonnet-5"


def test_generators_are_interchangeable_selection():
    assert isinstance(build_generator(from_dict({"generation": {"provider": "openai"}})), OpenAIGenerator)
    assert isinstance(build_generator(from_dict({"generation": {"provider": "anthropic"}})), AnthropicGenerator)
    assert isinstance(build_generator(from_dict({"generation": {"provider": "echo"}})), EchoGenerator)


def test_build_generator_rejects_unknown():
    cfg = from_dict({"generation": {"provider": "nope"}})
    try:
        build_generator(cfg)
    except ValueError as e:
        assert "generation.provider" in str(e)
    else:
        raise AssertionError("expected ValueError")
