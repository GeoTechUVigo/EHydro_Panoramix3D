import yaml

from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Any, Union


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class MutableModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=False)
    

class _LoaderWithInclude(yaml.SafeLoader):
    def __init__(self, stream, base_dir: Path | None = None):
        super().__init__(stream)
        self._base_dir = Path(base_dir) if base_dir else None


def _construct_include(loader: _LoaderWithInclude, node: yaml.Node) -> Any:
    if not isinstance(node, yaml.ScalarNode):
        raise yaml.constructor.ConstructorError("!include expects a single scalar value (the relative file path).")

    rel_path = Path(loader.construct_scalar(node))
    base = loader._base_dir or Path.cwd()
    path = (base / rel_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"[!include] Does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, lambda stream: _LoaderWithInclude(stream, base_dir=path.parent))

_LoaderWithInclude.add_constructor("!include", _construct_include)


def _load_yaml_with_includes(path: Union[str, Path]) -> dict:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config no encontrada: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.load(f, lambda stream: _LoaderWithInclude(stream, base_dir=p.parent))
