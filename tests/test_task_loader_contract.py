import ast
from pathlib import Path


def test_train_loader_reads_linear_probe_masks_and_classes():
    source = Path(__file__).parents[1].joinpath("experiments", "train_sequential.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "train_mask" in names
    assert "validation_mask" in names
    assert "test_mask" in names
    assert "classes" in names
