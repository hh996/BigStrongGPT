"""BigStrongGPT v1：150M 级 Dense（768×12，GQA 8/2，词表 16K）。"""

__version__ = "0.1.0"

from .modeling import BigStrongConfig, BigStrongForCausalLLM

__all__ = ["BigStrongConfig", "BigStrongForCausalLLM", "__version__"]
