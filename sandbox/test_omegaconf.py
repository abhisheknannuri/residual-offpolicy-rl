from dataclasses import dataclass
from typing import Any
from omegaconf import OmegaConf

@dataclass
class C:
    a: Any

cfg = OmegaConf.structured(C(a=[1, 2]))
obj = OmegaConf.to_object(cfg)
print(type(obj))
print(type(obj.a))
