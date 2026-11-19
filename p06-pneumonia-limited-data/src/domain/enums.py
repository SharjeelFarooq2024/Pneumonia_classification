"""Domain enumerations for pediatric pneumonia classification."""
from __future__ import annotations

from enum import IntEnum, unique
from typing import List, Tuple


@unique
class DiagnosisClass(IntEnum):
    """Mutually exclusive disease classes for pediatric chest radiograph diagnosis."""
    NORMAL = 0
    BACTERIA = 1
    VIRUS = 2

    @property
    def display_name(self) -> str:
        """Standard uppercase token matching MedIMeta labels."""
        return self.name

    @property
    def clinical_title(self) -> str:
        """Formal clinical terminology."""
        titles = {
            DiagnosisClass.NORMAL: "Normal Pediatric Radiograph (No Pneumonia)",
            DiagnosisClass.BACTERIA: "Bacterial Pneumonia",
            DiagnosisClass.VIRUS: "Viral Pneumonia",
        }
        return titles[self]

    @property
    def radiological_hallmark(self) -> str:
        """Characteristic imaging findings on anterior-posterior chest X-ray."""
        hallmarks = {
            DiagnosisClass.NORMAL: "Clear, fully aerated lung fields with sharp costophrenic angles and normal cardiothoracic ratio.",
            DiagnosisClass.BACTERIA: "Focal, dense lobar or segmental consolidation frequently accompanied by visible air bronchograms.",
            DiagnosisClass.VIRUS: "Diffuse, bilateral peribronchial thickening and generalized hazy interstitial/reticular infiltrates without dense lobar opacity.",
        }
        return hallmarks[self]

    @property
    def clinical_pathway(self) -> str:
        """Recommended secondary clinical pathway."""
        pathways = {
            DiagnosisClass.NORMAL: "Routine supportive follow-up; no antimicrobial therapy required.",
            DiagnosisClass.BACTERIA: "Urgent medical review for targeted antibacterial therapy consideration.",
            DiagnosisClass.VIRUS: "Supportive oxygenation/hydration care; perform viral multiplex PCR panel; avoid unnecessary empiric antibiotics.",
        }
        return pathways[self]

    @classmethod
    def from_label(cls, label: int) -> DiagnosisClass:
        """Lookup enum from integer index."""
        try:
            return cls(int(label))
        except ValueError:
            raise ValueError(f"Invalid label index {label}. Valid indices are: {[e.value for e in cls]}")

    @classmethod
    def from_name(cls, name: str) -> DiagnosisClass:
        """Lookup enum case-insensitively from string token."""
        normalized = name.strip().upper()
        for member in cls:
            if member.name == normalized:
                return member
        raise ValueError(f"Unknown class name '{name}'. Expected one of: {[e.name for e in cls]}")

    @classmethod
    def all_names(cls) -> List[str]:
        """Return list of all member names in canonical order."""
        return [e.name for e in cls]

    @classmethod
    def count(cls) -> int:
        """Total number of diagnostic classes."""
        return len(cls)


@unique
class DatasetSplit(IntEnum):
    """Official dataset split partitions."""
    TRAIN = 0
    VAL = 1
    TEST = 2

    @property
    def token(self) -> str:
        return self.name.lower()


@unique
class ModelRegime(IntEnum):
    """Architectural and transfer learning regimes for ablation analysis."""
    BASELINE_SCRATCH_CNN = 0
    RESNET18_SCRATCH = 1
    RESNET18_FROZEN = 2
    RESNET18_FULL_FINETUNE = 3
    RESNET18_STAGED_FINETUNE = 4
    PROPOSED_FULL_RECIPE = 5
