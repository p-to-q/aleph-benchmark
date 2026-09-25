"""Pure inputs for the provisional Aleph-Bench portable runtime profile.

This package deliberately contains no scorer.  It only exposes the frozen
Python string semantics that later portable-profile work may consume.
"""

from .string_semantics import FrozenStringSemantics, load_string_semantics

__all__ = ["FrozenStringSemantics", "load_string_semantics"]
