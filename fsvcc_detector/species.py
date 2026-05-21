"""
Species registry for the 13 target species at FSVCC.

Each entry contains:
  - key              : internal snake_case identifier
  - common_name      : display name written to the Google Sheet
  - scientific_name  : written to the Scientific Name column
  - clip_prompts     : text prompts used by the CLIP zero-shot classifier
                       (more prompts = more robust embedding; keep them
                        descriptive of trail-cam imagery)
  - speciesnet_taxa  : SpeciesNet / iNaturalist taxon strings for future
                       integration with the speciesnet package
"""

from __future__ import annotations
from typing import TypedDict


class SpeciesEntry(TypedDict):
    key: str
    common_name: str
    scientific_name: str
    clip_prompts: list[str]
    speciesnet_taxa: list[str]


SPECIES_LIST: list[SpeciesEntry] = [
    {
        "key": "beaver",
        "common_name": "Beaver",
        "scientific_name": "Castor canadensis",
        "clip_prompts": [
            "a beaver photographed by a trail camera",
            "a North American beaver near water",
            "a beaver with a flat tail",
            "a large brown rodent beaver in the wild",
        ],
        "speciesnet_taxa": ["castor canadensis"],
    },
    {
        "key": "bobcat",
        "common_name": "Bobcat",
        "scientific_name": "Lynx rufus",
        "clip_prompts": [
            "a bobcat photographed by a trail camera",
            "a wild bobcat with spotted fur and tufted ears",
            "a bobcat walking in a forest",
            "a lynx rufus wild cat",
        ],
        "speciesnet_taxa": ["lynx rufus"],
    },
    {
        "key": "coyote",
        "common_name": "Coyote",
        "scientific_name": "Canis latrans",
        "clip_prompts": [
            "a coyote photographed by a trail camera",
            "a wild coyote in a field",
            "a coyote with a bushy tail and pointed ears",
            "a North American coyote",
        ],
        "speciesnet_taxa": ["canis latrans"],
    },
    {
        "key": "striped_skunk",
        "common_name": "Striped Skunk",
        "scientific_name": "Mephitis mephitis",
        "clip_prompts": [
            "a striped skunk photographed by a trail camera",
            "a black and white striped skunk",
            "a skunk with white stripes on its back",
            "a mephitis mephitis skunk in the wild",
        ],
        "speciesnet_taxa": ["mephitis mephitis"],
    },
    {
        "key": "opossum",
        "common_name": "Virginia Opossum",
        "scientific_name": "Didelphis virginiana",
        "clip_prompts": [
            "a Virginia opossum photographed by a trail camera",
            "a possum with a white face and gray fur",
            "an opossum with a long pink tail",
            "a didelphis virginiana opossum",
        ],
        "speciesnet_taxa": ["didelphis virginiana"],
    },
    {
        "key": "deer",
        "common_name": "Columbian Black-tailed Deer",
        "scientific_name": "Odocoileus hemionus columbianus",
        "clip_prompts": [
            "a deer photographed by a trail camera",
            "a Columbian black-tailed deer in California",
            "a mule deer with large ears",
            "a deer with antlers in the wild",
            "a female deer doe in brush",
        ],
        "speciesnet_taxa": [
            "odocoileus hemionus columbianus",
            "odocoileus hemionus",
        ],
    },
    {
        "key": "gray_fox",
        "common_name": "Gray Fox",
        "scientific_name": "Urocyon cinereoargenteus",
        "clip_prompts": [
            "a gray fox photographed by a trail camera",
            "a grey fox with a rusty orange face",
            "a small fox with gray fur and black-tipped tail",
            "a urocyon cinereoargenteus gray fox in California",
        ],
        "speciesnet_taxa": ["urocyon cinereoargenteus"],
    },
    {
        "key": "raccoon",
        "common_name": "Raccoon",
        "scientific_name": "Procyon lotor",
        "clip_prompts": [
            "a raccoon photographed by a trail camera",
            "a raccoon with a black mask and ringed tail",
            "a North American raccoon at night",
            "a procyon lotor raccoon foraging",
        ],
        "speciesnet_taxa": ["procyon lotor"],
    },
    {
        "key": "desert_cottontail",
        "common_name": "Desert Cottontail",
        "scientific_name": "Sylvilagus audubonii",
        "clip_prompts": [
            "a desert cottontail photographed by a trail camera",
            "a small wild rabbit with a white fluffy tail",
            "a cottontail rabbit in dry brush",
            "a sylvilagus audubonii cottontail",
        ],
        "speciesnet_taxa": ["sylvilagus audubonii"],
    },
    {
        "key": "squirrel",
        "common_name": "Squirrel",
        "scientific_name": "Sciuridae spp.",
        "clip_prompts": [
            "a squirrel photographed by a trail camera",
            "a California ground squirrel",
            "a western gray squirrel in a tree",
            "a tree squirrel with a bushy tail",
            "a ground squirrel in dry grass",
        ],
        "speciesnet_taxa": [
            "sciurus griseus",
            "otospermophilus beecheyi",
            "sciurus niger",
            "tamiasciurus douglasii",
        ],
    },
    {
        "key": "california_quail",
        "common_name": "California Quail",
        "scientific_name": "Callipepla californica",
        "clip_prompts": [
            "a California quail photographed by a trail camera",
            "a quail bird with a black topknot feather",
            "a California valley quail on the ground",
            "a callipepla californica quail",
        ],
        "speciesnet_taxa": ["callipepla californica"],
    },
    {
        "key": "golden_crowned_sparrow",
        "common_name": "Golden-crowned Sparrow",
        "scientific_name": "Zonotrichia atricapilla",
        "clip_prompts": [
            "a golden-crowned sparrow photographed by a trail camera",
            "a sparrow with a yellow gold crown on its head",
            "a zonotrichia atricapilla sparrow",
            "a small brown bird with a yellow crown in brush",
        ],
        "speciesnet_taxa": ["zonotrichia atricapilla"],
    },
    {
        "key": "river_otter",
        "common_name": "North American River Otter",
        "scientific_name": "Lontra canadensis",
        "clip_prompts": [
            "a river otter photographed by a trail camera",
            "a North American river otter near water",
            "a lontra canadensis otter",
            "a sleek brown otter swimming in a river",
        ],
        "speciesnet_taxa": ["lontra canadensis"],
    },
]

# ── Lookup helpers ─────────────────────────────────────────────────────────────

_BY_KEY: dict[str, SpeciesEntry] = {s["key"]: s for s in SPECIES_LIST}


def by_key(key: str) -> SpeciesEntry | None:
    """Return a species entry by its internal key, or None if unknown."""
    return _BY_KEY.get(key)


def common_name(key: str) -> str:
    entry = _BY_KEY.get(key)
    return entry["common_name"] if entry else "Unknown"


def scientific_name(key: str) -> str:
    entry = _BY_KEY.get(key)
    return entry["scientific_name"] if entry else "Unknown"


# All text prompts flattened, paired with their species index (for CLIP batch)
def all_prompts() -> tuple[list[str], list[int]]:
    """Return (prompts, species_indices) for pre-computing CLIP text features."""
    prompts: list[str] = []
    indices: list[int] = []
    for i, species in enumerate(SPECIES_LIST):
        for p in species["clip_prompts"]:
            prompts.append(p)
            indices.append(i)
    return prompts, indices
