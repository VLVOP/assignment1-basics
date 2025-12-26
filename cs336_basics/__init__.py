import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .SiLU import SiLU
from .lr_cosine_schedule import lr_cosine_schedule