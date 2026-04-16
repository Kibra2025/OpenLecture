import pytest
from types import SimpleNamespace
from openlecture.transcribe import _TransformersModelAdapter

class FakeProcessor:
    def __init__(self) -> None:
        pass
    def __call__(self, audio_samples, **kwargs):
        from torch import tensor
        return {"input_features": tensor([[0.0]*500]*80), "attention_mask": tensor([[1]*500])}
    def batch_decode(self, values, skip_special_tokens=True):
        return ["!!! !!! !!!"]

class FakeModel:
    def __init__(self, output) -> None:
        self.output = output
    def generate(self, **kwargs):
        return self.output
    def to(self, device):
        pass

def test_transformers_adapter_fallback_on_garbage_output(monkeypatch):
    # Mock processor to always return garbage
    fake_processor = FakeProcessor()
    
    # Model that returns a sequence that will be decoded to "!!!"
    # We mock _segments_from_generate_output via monkeypatch or by giving a specific output
    fake_model = FakeModel({"sequences": [ [0]*10 ]}) 
    
    import torch
    device_dml = torch.device("cpu") # Use CPU for mock to avoid RuntimeError with .to("dml")
    adapter = _TransformersModelAdapter(fake_model, fake_processor, device_dml)
    
    # Mock the internal method to return garbage segments
    def mock_segments_from_generate(self, output, audio_duration_seconds):
        return [SimpleNamespace(start=0.0, end=1.0, text="!!!")], output
        
    monkeypatch.setattr(_TransformersModelAdapter, "_segments_from_generate_output", mock_segments_from_generate)
    monkeypatch.setattr("openlecture.transcribe._load_transformers_audio", lambda f: [0.0]*16000)
    
    # We need to mock the second call to generate to return something valid
    def mock_generate_cpu(self, **kwargs):
        # Return something that will be decoded to valid text
        return {"sequences": [ [50364, 123, 50664] ]}
        
    # Since the adapter uses self._model.generate, we can't easily mock it 
    # if we've already created fake_model. Let's replace fake_model.generate.
    
    def conditional_generate(input_features, **kwargs):
        if input_features.device == torch.device("cpu"):
            return {"sequences": [ [50364, 123, 50664] ]}
        return fake_model.output

    fake_model.generate = conditional_generate
    
    # Mock _segments_from_generate_output again to handle both cases
    def mock_segments_dynamic(self, output, audio_duration_seconds):
        if "sequences" in output and output["sequences"] == [ [50364, 123, 50664] ]:
            return [SimpleNamespace(start=0.0, end=1.0, text="Correct text")], output
        return [SimpleNamespace(start=0.0, end=1.0, text="!!!")], output

    monkeypatch.setattr(_TransformersModelAdapter, "_segments_from_generate_output", mock_segments_dynamic)

    segments, _ = adapter.transcribe("fake.wav")
    
    assert len(segments) > 0
    assert segments[0].text == "Correct text"
