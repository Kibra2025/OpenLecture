"""Tests for backend selection and the Transformers adapter."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from typer.testing import CliRunner

from openlecture import cli, transcribe
from openlecture.models import Segment
from openlecture.transcribe import _TransformersModelAdapter, transcribe_audio


runner = CliRunner()


class FakeTensor:
    """Minimal tensor-like object supporting ``.to()`` and iteration."""

    def __init__(self, value) -> None:
        self.value = value

    def to(self, _device):
        return self

    def __iter__(self):
        return iter(self.value)


class FakeProcessor:
    """Minimal Whisper processor double for adapter tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, audio_samples, **kwargs):
        self.calls.append({"audio_samples": audio_samples, **kwargs})
        return {
            "input_features": FakeTensor([[1, 2, 3]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }

    def batch_decode(self, values, skip_special_tokens: bool = True):
        assert skip_special_tokens is True
        token_groups = []
        for value in values:
            if isinstance(value, FakeTensor):
                token_groups.append(tuple(value.value[0]))
            else:
                token_groups.append(tuple(value))

        mapping = {
            (10, 11): "First line.",
            (12,): "Second line",
            (21, 22, 23): "Full transcript without segments.",
        }
        return [mapping[token_groups[0]]]


class FakeModel:
    """Minimal Whisper model double for adapter tests."""

    def __init__(self, output) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_transcribe_audio_rejects_unknown_backend() -> None:
    """Unknown backends should fail validation before transcription starts."""
    with pytest.raises(ValueError, match="backend must be one of"):
        transcribe_audio("lecture.mp3", backend="unknown")


def test_transformers_model_adapter_uses_generate_for_long_audio(monkeypatch) -> None:
    """The adapter should use Whisper generate() and decode returned segments."""
    fake_processor = FakeProcessor()
    fake_model = FakeModel(
        {
            "segments": [
                [
                    {"start": 0.2, "end": 1.4, "tokens": [10, 11]},
                    {"start": 1.5, "end": 2.0, "tokens": [12]},
                ]
            ]
        }
    )
    adapter = _TransformersModelAdapter(fake_model, fake_processor, "dml")

    monkeypatch.setattr(
        transcribe,
        "_load_transformers_audio",
        lambda _audio_file: [0.0] * (transcribe.WHISPER_SAMPLE_RATE * 31),
    )

    segments, metadata = adapter.transcribe("lecture.wav", beam_size=3, language="it")

    assert metadata == fake_model.output
    assert segments == [
        SimpleNamespace(start=0.2, end=1.4, text="First line."),
        SimpleNamespace(start=1.5, end=2.0, text="Second line"),
    ]
    assert fake_processor.calls == [
        {
            "audio_samples": [0.0] * (transcribe.WHISPER_SAMPLE_RATE * 31),
            "sampling_rate": 16000,
            "return_tensors": "pt",
            "truncation": False,
            "padding": "longest",
            "return_attention_mask": True,
        }
    ]

    model_call = fake_model.calls[0]
    assert isinstance(model_call["input_features"], FakeTensor)
    assert isinstance(model_call["attention_mask"], FakeTensor)
    assert model_call["task"] == "transcribe"
    assert model_call["return_timestamps"] is True
    assert model_call["num_beams"] == 3
    assert model_call["language"] == "it"
    assert model_call["return_dict_in_generate"] is True


def test_transformers_model_adapter_falls_back_to_single_segment(monkeypatch) -> None:
    """The adapter should decode full sequences when no segment list is returned."""
    fake_processor = FakeProcessor()
    fake_model = FakeModel({"sequences": FakeTensor([[21, 22, 23]])})
    adapter = _TransformersModelAdapter(fake_model, fake_processor, "dml")

    monkeypatch.setattr(
        transcribe,
        "_load_transformers_audio",
        lambda _audio_file: [0.0] * (transcribe.WHISPER_SAMPLE_RATE * 10),
    )

    segments, metadata = adapter.transcribe("lecture.wav", beam_size=1)

    assert metadata == fake_model.output
    assert segments == [
        SimpleNamespace(start=0.0, end=10.0, text="Full transcript without segments.")
    ]

    model_call = fake_model.calls[0]
    assert isinstance(model_call["input_features"], FakeTensor)
    assert isinstance(model_call["attention_mask"], FakeTensor)
    assert model_call["task"] == "transcribe"
    assert model_call["return_timestamps"] is True
    assert model_call["num_beams"] == 1
    assert "return_segments" not in model_call
    assert "return_dict_in_generate" not in model_call


def test_transcribe_audio_forwards_transformers_backend_options(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """transcribe_audio should pass backend selection through to the model loader."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    model_load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeModel:
        def transcribe(self, audio_path: str, **kwargs):
            assert audio_path == str(audio_file)
            assert kwargs == {"beam_size": 3, "vad_filter": True, "language": "it"}
            return ([SimpleNamespace(start=0.0, end=1.0, text="ciao")], None)

    def fake_get_model(*args, **kwargs):
        model_load_calls.append((args, kwargs))
        return FakeModel()

    monkeypatch.setattr(transcribe, "_get_model", fake_get_model)
    monkeypatch.setattr(transcribe, "get_audio_duration_seconds", lambda _audio_path: 120.0)

    result = transcribe_audio(
        str(audio_file),
        show_progress=False,
        backend="transformers",
        beam_size=3,
        device="amd",
        compute_type="float16",
        language="it",
    )

    assert model_load_calls == [
        (
            (),
            {
                "model_size": "medium",
                "device": "amd",
                "compute_type": "float16",
                "backend": "transformers",
            },
        )
    ]
    assert result == [Segment(start=0.0, end=1.0, text="ciao")]


def test_cli_forwards_transformers_backend_option(
    workspace_tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should expose backend selection to the transcription layer."""
    audio_file = workspace_tmp_path / "lecture.mp3"
    audio_file.write_bytes(b"fake audio")
    captured_kwargs: dict[str, object] = {}

    def fake_transcribe_audio(audio_path: str, **kwargs) -> list[Segment]:
        assert audio_path == str(audio_file)
        captured_kwargs.update(kwargs)
        return [Segment(start=0.0, end=1.0, text="Only sentence.")]

    monkeypatch.setattr(cli, "transcribe_audio", fake_transcribe_audio)

    result = runner.invoke(
        cli.app,
        [
            str(audio_file),
            "--no-progress",
            "--backend",
            "transformers",
            "--device",
            "amd",
            "--compute-type",
            "float16",
        ],
    )

    assert result.exit_code == 0
    assert captured_kwargs == {
        "show_progress": False,
        "chunk_length_ms": 60000,
        "status_callback": cli.typer.echo,
        "backend": "transformers",
        "device": "amd",
        "compute_type": "float16",
    }


def test_get_transformers_model_uses_torch_dtype(monkeypatch) -> None:
    """Transformers model loading should use ``torch_dtype`` for compatibility."""
    transcribe._get_transformers_model.cache_clear()
    captured_calls: dict[str, object] = {}

    class FakeLoadedModel:
        def to(self, device) -> None:
            captured_calls["device_to"] = device

        def eval(self) -> None:
            captured_calls["eval_called"] = True

    class FakeAutoModelForSpeechSeq2Seq:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured_calls["model_id"] = model_id
            captured_calls["kwargs"] = kwargs
            return FakeLoadedModel()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str):
            captured_calls["processor_model_id"] = model_id
            return object()

    fake_transformers = SimpleNamespace(
        AutoModelForSpeechSeq2Seq=FakeAutoModelForSpeechSeq2Seq,
        AutoProcessor=FakeAutoProcessor,
    )

    monkeypatch.setattr(transcribe, "_resolve_transformers_model_id", lambda model_size: "fake-model")
    monkeypatch.setattr(transcribe, "_resolve_transformers_device", lambda device: "fake-device")
    monkeypatch.setattr(transcribe, "_resolve_transformers_dtype", lambda compute_type, device: "fake-dtype")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = transcribe._get_transformers_model(
        model_size="small",
        device="dml",
        compute_type="auto",
    )

    assert isinstance(adapter, _TransformersModelAdapter)
    assert captured_calls == {
        "model_id": "fake-model",
        "kwargs": {"torch_dtype": "fake-dtype"},
        "device_to": "fake-device",
        "eval_called": True,
        "processor_model_id": "fake-model",
    }

    transcribe._get_transformers_model.cache_clear()
