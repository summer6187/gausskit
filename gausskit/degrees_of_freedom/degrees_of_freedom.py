# base class for all degrees of freedom
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gausskit.molecules import Molecules

class DOF:
    def __init__(self, mol: "Molecules"):
        self._mol: Molecules = mol

    @property
    def molecules(self) -> Molecules:
        return self._mol

