"""Hierarchical Taxonomy Mappings for sEMG Gesture Recognition.

Deconstructs flat multi-class gesture classification into coarse functional groups:
  - NinaPro DB2 (50 classes: 0..49):
      Macro 0: Rest (class 0)
      Macro 1: Basic finger & wrist postures (E1: classes 1..17, 17 classes)
      Macro 2: Functional grasps & fine pinches (E2: classes 18..40, 23 classes) -> Specialist CNN
      Macro 3: Finger force patterns (E3: classes 41..49, 9 classes)
  - Ozdemir et al. (10 classes: 0..9):
      Macro 0: Rest (class 0)
      Macro 1: Gross wrist movements (1: Extension, 2: Flexion, 3: Ulnar, 4: Radial, 8: Supination, 9: Pronation)
      Macro 2: Hand/Fine grip movements (5: Grip, 6: Abduction, 7: Adduction) -> Specialist CNN
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MacroGroup:
    """Definition of a macro functional group."""

    macro_id: int
    name: str
    global_labels: tuple[int, ...]
    is_fine_specialist: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.global_labels)


class NinaProDB2Taxonomy:
    """Taxonomic mapping for NinaPro DB2 50-class gestures."""

    MACRO_REST = 0
    MACRO_BASIC_WRIST = 1
    MACRO_FINE_GRASPS = 2
    MACRO_FORCE_PATTERNS = 3

    NUM_MACRO_CLASSES = 4

    GROUPS: dict[int, MacroGroup] = {
        MACRO_REST: MacroGroup(
            macro_id=MACRO_REST,
            name="Rest",
            global_labels=(0,),
            is_fine_specialist=False,
        ),
        MACRO_BASIC_WRIST: MacroGroup(
            macro_id=MACRO_BASIC_WRIST,
            name="Basic_Wrist_Posture",
            global_labels=tuple(range(1, 18)),  # 1..17 (17 classes)
            is_fine_specialist=False,
        ),
        MACRO_FINE_GRASPS: MacroGroup(
            macro_id=MACRO_FINE_GRASPS,
            name="Functional_Fine_Grasps",
            global_labels=tuple(range(18, 41)),  # 18..40 (23 classes)
            is_fine_specialist=True,
        ),
        MACRO_FORCE_PATTERNS: MacroGroup(
            macro_id=MACRO_FORCE_PATTERNS,
            name="Finger_Force_Patterns",
            global_labels=tuple(range(41, 50)),  # 41..49 (9 classes)
            is_fine_specialist=False,
        ),
    }

    @classmethod
    def to_macro_labels(cls, y_global: np.ndarray) -> np.ndarray:
        """Map global 0..49 labels to macro labels 0..3."""
        y_macro = np.zeros_like(y_global, dtype=np.int64)
        y_macro[(y_global >= 1) & (y_global <= 17)] = cls.MACRO_BASIC_WRIST
        y_macro[(y_global >= 18) & (y_global <= 40)] = cls.MACRO_FINE_GRASPS
        y_macro[(y_global >= 41) & (y_global <= 49)] = cls.MACRO_FORCE_PATTERNS
        return y_macro

    @classmethod
    def to_fine_specialist_labels(cls, y_global: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter and remap global labels to local 0..22 indices for Macro 2 (Fine Grasps).

        Returns:
            mask: Boolean array indicating which samples belong to Macro 2.
            y_local: Local class index (0..22) for the masked samples.
        """
        mask = (y_global >= 18) & (y_global <= 40)
        y_local = (y_global[mask] - 18).astype(np.int64)
        return mask, y_local

    @classmethod
    def fine_local_to_global(cls, local_pred: np.ndarray | int) -> np.ndarray | int:
        """Map local 0..22 indices back to global 18..40 classes."""
        return local_pred + 18

    @classmethod
    def to_sub_lda_labels(cls, y_global: np.ndarray, macro_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Filter and remap global labels to 0-indexed local classes for Sub-LDAs."""
        group = cls.GROUPS[macro_id]
        labels = np.array(group.global_labels)
        mask = np.isin(y_global, labels)
        filtered = y_global[mask]
        mapping = {int(g): idx for idx, g in enumerate(group.global_labels)}
        y_local = np.array([mapping[int(v)] for v in filtered], dtype=np.int64)
        return mask, y_local

    @classmethod
    def sub_lda_local_to_global(cls, local_pred: int, macro_id: int) -> int:
        """Map Sub-LDA local index back to global label."""
        return cls.GROUPS[macro_id].global_labels[local_pred]


class OzdemirTaxonomy:
    """Taxonomic mapping for Ozdemir et al. 10-class gestures."""

    MACRO_REST = 0
    MACRO_WRIST_POSTURES = 1
    MACRO_HAND_GRIP = 2

    NUM_MACRO_CLASSES = 3

    # 0: Rest
    # 1: Extension, 2: Flexion, 3: Ulnar, 4: Radial, 8: Supination, 9: Pronation
    # 5: Grip, 6: Abduction, 7: Adduction
    GROUPS: dict[int, MacroGroup] = {
        MACRO_REST: MacroGroup(
            macro_id=MACRO_REST,
            name="Rest",
            global_labels=(0,),
            is_fine_specialist=False,
        ),
        MACRO_WRIST_POSTURES: MacroGroup(
            macro_id=MACRO_WRIST_POSTURES,
            name="Gross_Wrist_Movements",
            global_labels=(1, 2, 3, 4, 8, 9),  # 6 classes
            is_fine_specialist=False,
        ),
        MACRO_HAND_GRIP: MacroGroup(
            macro_id=MACRO_HAND_GRIP,
            name="Hand_Grip_Abduction",
            global_labels=(5, 6, 7),  # 3 classes -> Specialist CNN
            is_fine_specialist=True,
        ),
    }

    @classmethod
    def to_macro_labels(cls, y_global: np.ndarray) -> np.ndarray:
        """Map global 0..9 labels to macro labels 0..2."""
        y_macro = np.zeros_like(y_global, dtype=np.int64)
        wrist_mask = np.isin(y_global, (1, 2, 3, 4, 8, 9))
        hand_mask = np.isin(y_global, (5, 6, 7))
        y_macro[wrist_mask] = cls.MACRO_WRIST_POSTURES
        y_macro[hand_mask] = cls.MACRO_HAND_GRIP
        return y_macro

    @classmethod
    def to_fine_specialist_labels(cls, y_global: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter and remap global labels to local 0..2 indices for Macro 2 (Grip/Hand)."""
        hand_labels = (5, 6, 7)
        mask = np.isin(y_global, hand_labels)
        filtered = y_global[mask]
        mapping = {5: 0, 6: 1, 7: 2}
        y_local = np.array([mapping[int(v)] for v in filtered], dtype=np.int64)
        return mask, y_local

    @classmethod
    def fine_local_to_global(cls, local_pred: np.ndarray | int) -> np.ndarray | int:
        """Map local 0..2 indices back to global 5..7 classes."""
        mapping = np.array([5, 6, 7])
        return mapping[local_pred]

    @classmethod
    def to_sub_lda_labels(cls, y_global: np.ndarray, macro_id: int) -> tuple[np.ndarray, np.ndarray]:
        group = cls.GROUPS[macro_id]
        labels = np.array(group.global_labels)
        mask = np.isin(y_global, labels)
        filtered = y_global[mask]
        mapping = {int(g): idx for idx, g in enumerate(group.global_labels)}
        y_local = np.array([mapping[int(v)] for v in filtered], dtype=np.int64)
        return mask, y_local

    @classmethod
    def sub_lda_local_to_global(cls, local_pred: int, macro_id: int) -> int:
        return cls.GROUPS[macro_id].global_labels[local_pred]
