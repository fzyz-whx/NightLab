import pytest

torch = pytest.importorskip("torch")

from nightlab.deep.models import AudioCNN, WearableCNN, build_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("task,channels,shape,classes", [
    ("audio", None, (1, 32, 100), 2),
    ("wearable", 2, (2, 30), 5),
    ("wearable", 5, (5, 30), 5),
    ("wearable", 7, (7, 30), 5),
])
def test_shape_gradients_and_cpu_state_dict(task, channels, shape, classes, tmp_path):
    torch.manual_seed(123)
    model = build_model(task, channels).cpu()
    assert sum(p.numel() for p in model.parameters()) < 100_000
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    x = torch.randn(3, *shape, requires_grad=True)
    logits = model(x)
    assert logits.shape == (3, classes)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.features[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        expected = model(x.detach())
        assert model(torch.randn(1, *shape)).shape == (1, classes)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(task, channels).cpu()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x.detach()), expected, rtol=0, atol=0)


@pytest.mark.parametrize("model,shape", [
    (AudioCNN, (2, 32, 100)),
    (AudioCNN, (2, 2, 32, 100)),
    (AudioCNN, (2, 1, 31, 100)),
    (AudioCNN, (2, 1, 32, 99)),
    (AudioCNN, (0, 1, 32, 100)),
    (WearableCNN, (2, 7)),
    (WearableCNN, (2, 2, 30)),
    (WearableCNN, (2, 7, 29)),
    (WearableCNN, (0, 7, 30)),
])
def test_invalid_shapes(model, shape):
    with pytest.raises(ValueError, match="Expected"):
        model()(torch.zeros(shape))


def test_build_defaults_custom_classes_and_invalid_arguments():
    assert isinstance(build_model("audio", 1), AudioCNN)
    assert build_model("wearable").in_channels == 7
    assert WearableCNN(2, 3)(torch.randn(1, 2, 30)).shape == (1, 3)
    with pytest.raises(TypeError, match="torch.Tensor"):
        AudioCNN()([[0]])
    with pytest.raises(ValueError, match="Unknown task"):
        build_model("other")
    with pytest.raises(ValueError, match="audio requires"):
        build_model("audio", 2)
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="in_channels"):
            WearableCNN(invalid)
        with pytest.raises(ValueError, match="num_classes"):
            WearableCNN(num_classes=invalid)
