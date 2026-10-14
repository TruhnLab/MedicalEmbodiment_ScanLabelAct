"""Answer formats for the VLM. vLLM enforces the JSON schema while generating (response_format
json_schema), so answers always parse: without it the model sometimes left out the caption or got
stuck repeating boxes (2026-10-05)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ObjectAnswer(BaseModel):
    real_object: bool = Field(description="false if the green box shows only floor, wall, a skirting board, shadow or a reflection")
    same_object: bool = Field(description="do the marked boxes in all crops show the same single object?")
    # max_length: once the model put its whole description here (2026-10-06)
    label: str = Field(max_length=40, description="1-3 words naming the object in the green boxes, e.g. cabinet")
    description: str = Field(description="one sentence: colour, material, shape, anything notable")
    category: Literal["furniture", "door", "appliance", "fixture", "container", "floor_item", "person", "other"]
    movable: Literal["fixed", "movable", "articulated"]
    state: Optional[Literal["open", "closed", "ajar"]] = None
    confidence: Literal["low", "medium", "high"]


class PairAnswer(BaseModel):
    verdict: Literal["same", "fragment", "operable_part", "different"]
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(max_length=200, description="one short sentence")


# detect_frames.py (experiment 2026-10-09): Qwen names the kinds of things in a few views, without boxes;
# an open-vocabulary detector (YOLOE) finds them in every keyframe
Category = Literal["furniture", "door", "appliance", "fixture", "container", "floor_item", "person", "other"]


class VocabItem(BaseModel):
    label: str = Field(max_length=40, description="short noun, e.g. bar stool, door, backpack, cable")
    category: Category
    movable: Literal["fixed", "movable", "articulated"]


class VocabAnswer(BaseModel):
    objects: list[VocabItem] = Field(max_length=60)


class VocabEntry(VocabItem):
    synonyms: list[str] = Field(max_length=8, description="other names used in the list for the same kind of thing")


class VocabList(BaseModel):
    objects: list[VocabEntry] = Field(max_length=150)
