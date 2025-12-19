"""Scan SimpleStories dataset for offensive language and CBRN-related keywords.

Uses external word lists to avoid embedding offensive terms in the repository.
"""

from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm
import re
import urllib.request

# External word list sources
OFFENSIVE_LIST_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
# CBRN terms are technical/scientific, not offensive - fine to include directly
# Sources: CWC Schedules, CDC/USDA Select Agents, IAEA High-Risk Sources, NSG Trigger List
CBRN_KEYWORDS = {
    # Chemical - Nerve Agents
    "sarin", "soman", "tabun", "vx agent", "cyclosarin", "novichok",
    "a-230", "a-232", "a-234", "vr agent",
    # Chemical - Blister Agents
    "mustard gas", "sulfur mustard", "nitrogen mustard", "lewisite",
    "phosgene oxime", "vesicant",
    # Chemical - Blood & Choking
    "hydrogen cyanide", "cyanogen chloride", "arsine", "phosgene",
    "diphosgene", "chlorine gas", "chloropicrin",
    # Chemical - Precursors
    "methylphosphonyl difluoride", "methylphosphonyl dichloride",
    "thiodiglycol", "sodium cyanide", "potassium cyanide", "amiton",
    "3-quinuclidinyl benzilate",
    # Biological - Bacteria
    "bacillus anthracis", "anthrax", "yersinia pestis", "plague",
    "francisella tularensis", "tularemia", "burkholderia mallei", "glanders",
    "burkholderia pseudomallei", "melioidosis", "coxiella burnetii", "q fever",
    "brucella", "rickettsia prowazekii", "typhus", "clostridium botulinum",
    # Biological - Viruses
    "ebolavirus", "ebola", "marburg virus", "variola major", "smallpox",
    "variola minor", "lassa fever", "nipah virus", "hendra virus",
    "crimean-congo hemorrhagic fever", "rift valley fever",
    "venezuelan equine encephalitis", "junin virus", "machupo virus",
    "monkeypox", "1918 influenza",
    # Biological - Toxins
    "ricin", "abrin", "botulinum neurotoxin", "saxitoxin", "tetrodotoxin",
    "staphylococcal enterotoxin", "conotoxin", "t-2 toxin", "microcystin",
    # Radiological
    "cobalt", "cesium", "iridium", "americium", "strontium",
    "radium", "californium", "plutonium", "curium",
    "dirty bomb", "radiological dispersal device", "orphan source",
    # Nuclear - Fissile
    "uranium", "plutonium", "special fissionable material", "yellowcake",
    # Nuclear - Equipment
    "gas centrifuge", "zippe-type centrifuge", "calutron", "gaseous diffusion",
    "atomic vapor laser isotope separation", "heavy water", "deuterium oxide",
    "neutron generator", "krytron", "sprytron", "exploding bridge wire",
    "slapper detonator", "flux compression generator", "explosive lenses",
    "beryllium tamper", "polonium-210 initiator", "tritium gas", "lithium-6 deuteride",
    # Dual-Use Equipment
    "aerosol generation chamber", "biological safety cabinet", "lyophilizer",
    "dna synthesizer",
}


def fetch_wordlist(url: str, fallback: set[str] | None = None) -> set[str]:
    """Fetch a newline-separated word list from URL."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode("utf-8")
        words = {line.strip().lower() for line in text.splitlines() if line.strip()}
        print(f"  Loaded {len(words)} terms from {url.split('/')[-1]}")
        return words


def scan_dataset(dataset_name: str = "SimpleStories/SimpleStories", sample_frac: float = 1.0):
    """Scan dataset for harmful keywords."""
    print("Fetching external word lists...")
    offensive_keywords = fetch_wordlist(OFFENSIVE_LIST_URL)

    print(f"\nLoading {dataset_name}...")
    ds = load_dataset(dataset_name)
    if isinstance(ds, dict):
        ds = ds["train"]

    if sample_frac < 1.0:
        ds = ds.shuffle(seed=42).select(range(int(len(ds) * sample_frac)))

    total = len(ds)
    print(f"Scanning {total} stories...\n")

    offensive_hits: Counter = Counter()
    cbrn_hits: Counter = Counter()
    flagged_stories: list[dict] = []

    for item in tqdm(ds, desc="Scanning"):
        text = item.get("text", "") or ""
        text_lower = text.lower()

        story_offensive = []
        story_cbrn = []

        for kw in offensive_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                offensive_hits[kw] += 1
                story_offensive.append(kw)

        for kw in CBRN_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                cbrn_hits[kw] += 1
                story_cbrn.append(kw)

        if story_offensive or story_cbrn:
            flagged_stories.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "offensive": story_offensive,
                "cbrn": story_cbrn,
            })

    # Report
    print("=" * 60)
    print("HARMFULNESS SCAN RESULTS")
    print("=" * 60)
    print(f"Total stories: {total}")
    print(f"Flagged stories: {len(flagged_stories)} ({100*len(flagged_stories)/total:.4f}%)\n")

    print("OFFENSIVE LANGUAGE HITS:")
    if offensive_hits:
        for kw, cnt in offensive_hits.most_common(20):
            print(f"  {kw}: {cnt}")
    else:
        print("  None found")

    print("\nCBRN KEYWORD HITS:")
    if cbrn_hits:
        for kw, cnt in cbrn_hits.most_common(20):
            print(f"  {kw}: {cnt}")
    else:
        print("  None found")

    if flagged_stories:
        print("\nSAMPLE FLAGGED STORIES (first 5):")
        for i, s in enumerate(flagged_stories[:5]):
            print(f"\n--- Story {i+1} ---")
            print(f"Keywords: offensive={s['offensive']}, cbrn={s['cbrn']}")
            print(s["text"])

    return {"offensive": offensive_hits, "cbrn": cbrn_hits, "flagged": flagged_stories}


if __name__ == "__main__":
    scan_dataset("SimpleStories/SimpleStories")

