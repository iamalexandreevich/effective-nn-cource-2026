import wave

import pytest

from benchmark import duration, score


def test_corpus_word_error_uses_total_reference_words():
    result = score([("red green blue", "red green blue"), ("last", "")], "en")
    assert result["wer"] == pytest.approx(0.25)


def test_golos_normalization_merges_yo_and_ye():
    assert score([("Всё хорошо!", "Все хорошо")], "ru")["wer"] == 0


def test_audio_duration_comes_from_sample_count(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(16000))
    assert duration(path) == 0.5


def test_runtime_options_set_native_audio_context_and_flash_attention():
    from argparse import Namespace

    from benchmark import inference_flags

    args = Namespace(language="ru", threads=4, beam=1, audio_context=512, no_flash_attention=True)
    flags = inference_flags(args)
    assert flags[flags.index("-ac") + 1] == "512"
    assert "-nfa" in flags
