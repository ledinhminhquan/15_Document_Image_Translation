"""Built-in seed dataset (offline fallback + tests).

Provides, so everything runs with NO torch, NO tesseract and NO network:
* ``DICT_EN_FR`` - an English->French word dictionary for the dictionary MT baseline;
* ``SEED_PAIRS`` - English->French sentence pairs (MT eval/baseline floor);
* ``SEED_PAGES`` - synthetic **document-page specs**: each page is a list of text blocks
  with the source line, its gold French translation and a bounding box, plus the page
  size. These specs play the role of "ground-truth OCR + gold translation" so the
  agent, the OCR/MT/end-to-end eval, the overlay renderer and the layout-fidelity
  metric can all be exercised end-to-end offline. On Colab the real corpus (opus-100)
  trains the MT core and ``data/synth_render`` rasterises specs like these into real PNGs.

Texts are original/synthetic. Accents are kept in the gold French; the dictionary keys
are lowercased/accent-stripped to match the baseline tokenizer.
"""

from __future__ import annotations

from typing import Dict, List

# English -> French sentence pairs (gold)
SEED_PAIRS: List[Dict[str, str]] = [
    {"id": "p01", "src": "Hello, how are you today?", "tgt": "Bonjour, comment allez-vous aujourd'hui ?"},
    {"id": "p02", "src": "The black cat sleeps on the sofa.", "tgt": "Le chat noir dort sur le canape."},
    {"id": "p03", "src": "The meeting starts tomorrow at nine.", "tgt": "La reunion commence demain a neuf heures."},
    {"id": "p04", "src": "The president signed a new agreement.", "tgt": "Le president a signe un nouvel accord."},
    {"id": "p05", "src": "The children play in the park after school.", "tgt": "Les enfants jouent dans le parc apres l'ecole."},
    {"id": "p06", "src": "This company reported good results this year.", "tgt": "Cette entreprise a annonce de bons resultats cette annee."},
    {"id": "p07", "src": "The researchers published a new study.", "tgt": "Les chercheurs ont publie une nouvelle etude."},
    {"id": "p08", "src": "We must reduce pollution in the cities.", "tgt": "Nous devons reduire la pollution dans les villes."},
    {"id": "p09", "src": "The museum is open every day.", "tgt": "Le musee est ouvert tous les jours."},
    {"id": "p10", "src": "She works as a doctor in a hospital.", "tgt": "Elle travaille comme medecin dans un hopital."},
    {"id": "p11", "src": "Thank you very much for your help.", "tgt": "Merci beaucoup pour votre aide."},
    {"id": "p12", "src": "The new phone has a better battery.", "tgt": "Le nouveau telephone a une meilleure batterie."},
    {"id": "p13", "src": "Click the button to open the file.", "tgt": "Cliquez sur le bouton pour ouvrir le fichier."},
    {"id": "p14", "src": "This guide explains how to install the system.", "tgt": "Ce guide explique comment installer le systeme."},
    {"id": "p15", "src": "The report shows the results of the project.", "tgt": "Le rapport montre les resultats du projet."},
    {"id": "p16", "src": "Save your work before you close the application.", "tgt": "Enregistrez votre travail avant de fermer l'application."},
    {"id": "p17", "src": "Tourists visit the old town in the summer.", "tgt": "Les touristes visitent la vieille ville en ete."},
    {"id": "p18", "src": "The train to the airport leaves every hour.", "tgt": "Le train pour l'aeroport part toutes les heures."},
]

# English -> French word dictionary (lowercased; accents stripped to match the baseline).
DICT_EN_FR: Dict[str, str] = {
    "hello": "bonjour", "how": "comment", "are": "allez", "you": "vous", "today": "aujourd'hui",
    "the": "le", "black": "noir", "cat": "chat", "sleeps": "dort", "on": "sur", "sofa": "canape",
    "meeting": "reunion", "starts": "commence", "tomorrow": "demain", "at": "a", "nine": "neuf",
    "president": "president", "signed": "a signe", "a": "un", "new": "nouveau", "agreement": "accord",
    "children": "enfants", "play": "jouent", "in": "dans", "park": "parc", "after": "apres", "school": "ecole",
    "this": "cette", "company": "entreprise", "reported": "a annonce", "good": "bons", "results": "resultats",
    "year": "annee", "researchers": "chercheurs", "published": "ont publie", "study": "etude",
    "we": "nous", "must": "devons", "reduce": "reduire", "pollution": "pollution", "cities": "villes",
    "museum": "musee", "is": "est", "open": "ouvert", "every": "tous", "day": "jours",
    "she": "elle", "works": "travaille", "as": "comme", "doctor": "medecin", "hospital": "hopital",
    "thank": "merci", "thanks": "merci", "very": "tres", "much": "beaucoup", "for": "pour", "your": "votre",
    "help": "aide", "phone": "telephone", "has": "a", "better": "meilleure", "battery": "batterie",
    "click": "cliquez", "button": "bouton", "to": "pour", "open": "ouvrir", "file": "fichier", "guide": "guide",
    "explains": "explique", "install": "installer", "system": "systeme", "report": "rapport",
    "shows": "montre", "of": "du", "project": "projet", "save": "enregistrez", "work": "travail",
    "before": "avant", "close": "fermer", "application": "application", "and": "et", "with": "avec",
    "tourists": "touristes", "visit": "visitent", "old": "vieille", "town": "ville", "summer": "ete",
    "train": "train", "airport": "aeroport", "leaves": "part", "hour": "heure", "hours": "heures",
    "welcome": "bienvenue", "user": "utilisateur", "name": "nom", "please": "s'il vous plait", "enter": "entrez",
}


def _build_seed_pages(width: int = 1000, margin: int = 60, line_h: int = 70,
                      lines_per_page: int = 6) -> List[Dict]:
    """Lay the seed pairs out as stacked text blocks -> page specs (no PIL needed)."""
    pages: List[Dict] = []
    pairs = SEED_PAIRS
    for start in range(0, len(pairs), lines_per_page):
        chunk = pairs[start:start + lines_per_page]
        blocks = []
        y = margin
        for i, pr in enumerate(chunk):
            # approximate a tight box: width grows with text length, capped to the page
            w = min(width - 2 * margin, int(margin // 2 + len(pr["src"]) * 11))
            blocks.append({
                "text": pr["src"], "translation": pr["tgt"],
                "bbox": [margin, y, w, line_h - 16],
                "block": i, "line": i, "kind": "heading" if i == 0 else "paragraph",
            })
            y += line_h
        height = y + margin
        pages.append({"width": width, "height": height, "src_lang": "en", "tgt_lang": "fr",
                      "blocks": blocks})
    return pages


SEED_PAGES: List[Dict] = _build_seed_pages()


def pairs() -> List[Dict[str, str]]:
    return [dict(x) for x in SEED_PAIRS]


def dictionary() -> Dict[str, str]:
    return dict(DICT_EN_FR)


def seed_pages() -> List[Dict]:
    import copy
    return copy.deepcopy(SEED_PAGES)


def src_texts() -> List[str]:
    return [p["src"] for p in SEED_PAIRS]


def tgt_texts() -> List[str]:
    return [p["tgt"] for p in SEED_PAIRS]


__all__ = ["SEED_PAIRS", "DICT_EN_FR", "SEED_PAGES", "pairs", "dictionary", "seed_pages",
           "src_texts", "tgt_texts"]
