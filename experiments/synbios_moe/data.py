"""Deterministic synthetic-biography generation for the bioS experiment.

The experiment has two representations: ``Profile`` is the underlying facts,
while one or more ``Biography`` records verbalize those same facts with chosen
templates and sentence orders.  Attribute character spans are retained for
later token-level evaluation and probing.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

ATTRIBUTES = ("birth_date", "birth_city", "university", "major", "company", "company_city")
TEMPLATE_COUNTS = (46, 49, 49, 52, 47, 48)

_FIRST = (
    "Alden",
    "Anya",
    "Carlos",
    "Alondra",
    "Aidan",
    "Maya",
    "Noah",
    "Iris",
    "Theo",
    "Lena",
    "Owen",
    "Nora",
    "Eli",
    "Zoe",
    "Milo",
    "Clara",
    "Jonah",
    "Ada",
    "Felix",
    "Mina",
)
_MIDDLE = (
    "James",
    "Alexa",
    "Bennett",
    "Briar",
    "Morgan",
    "Taylor",
    "Jordan",
    "Reese",
    "Quinn",
    "Casey",
    "Rowan",
    "Avery",
    "Sage",
    "Blair",
    "Robin",
    "Emery",
    "Riley",
    "Parker",
    "Drew",
    "Hayden",
)
_LAST = (
    "Forger",
    "Stokes",
    "Rooney",
    "Dennis",
    "Kenny",
    "Wade",
    "Carter",
    "Brooks",
    "Reed",
    "Hayes",
    "Foster",
    "Bennett",
    "Price",
    "Ward",
    "Cole",
    "Gray",
    "Stone",
    "Hart",
    "Lane",
    "Cross",
)
_CITY_ROOTS = (
    "Princeton",
    "Cambridge",
    "Seattle",
    "Austin",
    "Denver",
    "Phoenix",
    "Madison",
    "Portland",
    "Raleigh",
    "Durham",
    "Chicago",
    "Boston",
    "Atlanta",
    "Oakland",
    "Dallas",
    "Miami",
    "Orlando",
    "Fresno",
    "Tacoma",
    "Albany",
)
_STATES = ("CA", "NY", "TX", "MA", "WA", "IL", "PA", "OH", "NC", "GA")
_MAJORS = (
    "Computer Science",
    "Physics",
    "Music",
    "Communications",
    "Economics",
    "Mathematics",
    "History",
    "Sociology",
    "Data Science",
    "Political Science",
    "Chemistry",
    "Philosophy",
    "Architecture",
    "Biology",
    "Finance",
    "Linguistics",
    "Engineering",
    "Psychology",
    "Design",
    "Statistics",
)

_PATTERNS = {
    "birth_date": (
        "{subject} was born on {value}.",
        "{subject}'s birthday is {value}.",
        "{subject} celebrates a birthday on {value}.",
        "{subject} entered the world on {value}.",
        "{subject} has an annual celebration on {value}.",
    ),
    "birth_city": (
        "{pronoun} was born in {value}.",
        "{pronoun} grew up in {value}.",
        "{subject} calls {value} a birthplace.",
        "{pronoun} owes early roots to {value}.",
        "{pronoun} spent early years in {value}.",
    ),
    "university": (
        "{pronoun} attended {value}.",
        "{pronoun} graduated from {value}.",
        "{subject} studied at {value}.",
        "{pronoun} received mentorship at {value}.",
        "{pronoun} completed university at {value}.",
    ),
    "major": (
        "{pronoun} majored in {value}.",
        "{pronoun} studied {value}.",
        "{subject} focused on {value}.",
        "{pronoun} developed a foundation in {value}.",
        "{pronoun} specialized in {value}.",
    ),
    "company": (
        "{pronoun} worked for {value}.",
        "{pronoun} had a professional role at {value}.",
        "{subject} joined {value}.",
        "{pronoun} was employed by {value}.",
        "{pronoun} contributed expertise to {value}.",
    ),
    "company_city": (
        "{pronoun} worked in {value}.",
        "{pronoun} was professionally based in {value}.",
        "{subject} gained work experience in {value}.",
        "{pronoun} pursued a career in {value}.",
        "{pronoun} was employed in {value}.",
    ),
}
_PREFIXES = (
    "",
    "In life, ",
    "According to records, ",
    "Notably, ",
    "Biographical notes say ",
    "Public records show ",
    "It is recorded that ",
    "In particular, ",
    "The profile states ",
    "Historically, ",
    "One account says ",
)


@dataclass(frozen=True)
class Profile:
    person_id: int
    full_name: str
    pronoun: str
    birth_date: str
    birth_city: str
    university: str
    major: str
    company: str
    company_city: str


@dataclass(frozen=True)
class Biography:
    person_id: int
    variant: str
    text: str
    attribute_spans: dict[str, tuple[int, int]]


def _expanded_names(roots: tuple[str, ...], count: int) -> list[str]:
    # Suffixes keep the required pool cardinalities deterministic without an
    # unpublished third-party name list.
    return [f"{roots[i % len(roots)]}{i // len(roots) or ''}" for i in range(count)]


def candidate_pools() -> dict[str, list[str]]:
    """Build deterministic pools with the cardinalities used by this replica."""

    cities = [f"{_CITY_ROOTS[i % 20]}{i // 20 or ''}, {_STATES[i % 10]}" for i in range(200)]
    cities[0] = "New York, NY"
    universities = [f"{_CITY_ROOTS[i % 20]}{i // 20 or ''} University" for i in range(300)]
    majors = [f"{_MAJORS[i % 20]}{i // 20 or ''}" for i in range(100)]
    companies = [f"{_LAST[i % 20]}{i // 20 or ''} Group" for i in range(263)]
    return {
        "first": _expanded_names(_FIRST, 400),
        "middle": _expanded_names(_MIDDLE, 400),
        "last": _expanded_names(_LAST, 1000),
        "birth_city": cities,
        "university": universities,
        "major": majors,
        "company": companies,
        # 36 / 263 = 13.69%, matching the paper's New York headquarters baseline.
        "company_city": [
            "New York, NY" if i < 36 else cities[1 + ((i - 36) * 73) % 199] for i in range(263)
        ],
    }


def generate_profiles(num_people: int, seed: int) -> list[Profile]:
    """Sample unique people and their six facts from the candidate pools."""

    if not 0 < num_people <= 400 * 400 * 1000:
        raise ValueError("num_people exceeds the unique-name space")
    pools = candidate_pools()
    rng = random.Random(seed)
    used: set[str] = set()
    profiles: list[Profile] = []
    while len(profiles) < num_people:
        full_name = " ".join(
            (rng.choice(pools["first"]), rng.choice(pools["middle"]), rng.choice(pools["last"]))
        )
        if full_name in used:
            continue
        used.add(full_name)
        company_index = rng.randrange(263)
        year, month, day = rng.randrange(1900, 2100), rng.randrange(1, 13), rng.randrange(1, 29)
        profiles.append(
            Profile(
                person_id=len(profiles),
                full_name=full_name,
                pronoun=rng.choice(("He", "She", "They")),
                birth_date=date(year, month, day).strftime("%B %-d, %Y")
                if __import__("os").name != "nt"
                else date(year, month, day).strftime("%B %d, %Y").replace(" 0", " "),
                birth_city=rng.choice(pools["birth_city"]),
                university=rng.choice(pools["university"]),
                major=rng.choice(pools["major"]),
                company=pools["company"][company_index],
                company_city=pools["company_city"][company_index],
            )
        )
    return profiles


def _template(attribute: str, index: int) -> str:
    patterns = _PATTERNS[attribute]
    pattern = patterns[index % len(patterns)]
    # Harmless lexical prefixes make the published per-attribute template counts exact.
    prefix = _PREFIXES[index // len(patterns)]
    return prefix + pattern


def render_biography(profile: Profile, *, variant: str, sample: int, seed: int) -> Biography:
    """Render one reproducible augmentation and retain every fact's char span."""

    # A local string-derived RNG prevents iteration order or unrelated random
    # calls from changing an existing person's augmentation.
    rng = random.Random(f"{seed}:{profile.person_id}:{variant}:{sample}")
    fullname = "fullname" in variant
    sentences: list[tuple[str, str, int, int]] = []
    # First render one sentence per fact and remember the fact's sentence-local
    # character offsets.  In the paper's bioS data, a non-fullname biography
    # uses the full name only at the start of its final first sentence, so all
    # sentences begin with the pronoun until any permutation has been applied.
    for attribute, count in zip(ATTRIBUTES, TEMPLATE_COUNTS):
        subject = profile.full_name if fullname else profile.pronoun
        text = _template(attribute, rng.randrange(count)).format(
            subject=subject,
            pronoun=profile.full_name if fullname else profile.pronoun,
            value=getattr(profile, attribute),
        )
        start = text.index(getattr(profile, attribute))
        sentences.append((attribute, text, start, start + len(getattr(profile, attribute))))
    if "permute" in variant:
        # Sentence permutation is data augmentation inside a biography.  It is
        # distinct from DataLoader randomized_documents, which permutes whole
        # biographies before packing them into context windows.
        rng.shuffle(sentences)
    # Match the paper: the full name occurs once, at the start of the final
    # first sentence.  This also means a permuted birthday sentence uses a
    # pronoun when it is no longer first.
    if not fullname:
        attribute, text, start, end = sentences[0]
        old_subject = profile.pronoun
        subject_start = text.find(old_subject)
        if subject_start < 0:
            raise ValueError("first biography sentence does not contain its subject pronoun")
        text = text[:subject_start] + profile.full_name + text[subject_start + len(old_subject) :]
        shift = len(profile.full_name) - len(old_subject)
        if subject_start < start:
            start, end = start + shift, end + shift
        sentences[0] = attribute, text, start, end
    # Convert sentence-local offsets to offsets in the final joined biography.
    offset, chunks, spans = 0, [], {}
    for attribute, sentence, start, end in sentences:
        chunks.append(sentence)
        spans[attribute] = (offset + start, offset + end)
        offset += len(sentence) + 1
    return Biography(profile.person_id, variant, " ".join(chunks), spans)


def entries_per_person(variant: str) -> int:
    """Return how many independently rendered biographies each person owns."""

    for count in (5, 2, 1):
        if f"multi{count}" in variant or f"permute{count}" in variant:
            return count
    return 1


def iter_biographies(profiles: Iterable[Profile], variant: str, seed: int) -> Iterable[Biography]:
    """Stream every augmentation without materializing all biography texts."""

    for profile in profiles:
        for sample in range(entries_per_person(variant)):
            yield render_biography(profile, variant=variant, sample=sample, seed=seed)


def split_for_person(person_id: int, seed: int, validation_fraction: float = 0.5) -> str:
    """Keep every augmentation of one person on the same probe split."""

    digest = hashlib.sha256(f"{seed}:{person_id}".encode()).digest()
    return (
        "validation" if int.from_bytes(digest[:8], "big") / 2**64 < validation_fraction else "train"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(output_dir: str | Path, *, num_people: int, variant: str, seed: int) -> Path:
    """Write profiles, biographies, plain text, and an integrity manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profiles = generate_profiles(num_people, seed)

    # profiles.jsonl is the fact table used to construct labels.  Splitting at
    # person level prevents another augmentation of the same facts leaking into
    # the opposite probe split.
    with (output / "profiles.jsonl").open("w", encoding="utf-8") as handle:
        for profile in profiles:
            row = {**asdict(profile), "split": split_for_person(profile.person_id, seed)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    # biographies.jsonl retains spans/metadata for evaluation; biographies.txt
    # is a convenient plain-text view.  Token shards are generated later by the
    # shared preprocessing pipeline rather than by experiment-specific code.
    biography_count = 0
    with (
        (output / "biographies.jsonl").open("w", encoding="utf-8") as records,
        (output / "biographies.txt").open("w", encoding="utf-8") as text,
    ):
        for biography in iter_biographies(profiles, variant, seed):
            records.write(json.dumps(asdict(biography), ensure_ascii=False) + "\n")
            text.write(biography.text + "\n")
            biography_count += 1
    # Record generator choices and stream file hashes so an evaluation result
    # can identify the exact synthetic dataset it consumed.
    manifest = {
        "format_version": 1,
        "generator": "minitrain.synbios_moe.v1",
        "seed": seed,
        "num_people": num_people,
        "variant": variant,
        "biographies": biography_count,
        "attribute_candidates": {
            "birth_date": 200 * 12 * 28,
            "birth_city": 200,
            "university": 300,
            "major": 100,
            "company": 263,
            "company_city": 200,
        },
        "template_counts": dict(zip(ATTRIBUTES, TEMPLATE_COUNTS)),
        "split_unit": "person",
        "validation_fraction": 0.5,
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in ("profiles.jsonl", "biographies.jsonl", "biographies.txt")
        },
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
