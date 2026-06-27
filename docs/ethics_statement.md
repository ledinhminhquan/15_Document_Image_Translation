# Ethics Statement — P15 Document-Image Machine Translation

> Project: **Document-Image Machine Translation** (package `imgtrans`, folder `15_Document_Image_Translation`)
> Author: Le Dinh Minh Quan (student 23127460)
> Scope: translate the text that appears *inside* an image, scanned document, or born-digital PDF and render the translation **back onto the page** preserving layout (the "camera-translate" overlay), via a cascade **OCR → MT → render** in which the MT core (`facebook/m2m100_418M`, MIT) is the only trained stage.

This document states the ethical commitments of the project, the concrete risks specific to translating *document images*, and the design decisions that mitigate them. It is binding on the shipped system: several of the mitigations below are not aspirations but actual decision points in the agent (`src/imgtrans/agent/`).

---

## 1. Framing: the tool ASSISTS, it does not decide

P15 is a **translation aid**, not an authority. Its single most important ethical stance is that it never asserts certainty about its own output. Every translated page is the product of three lossy/imperfect stages stacked in series:

1. **OCR** (Tesseract / `pytesseract`, pretrained) can misread characters, merge or split words, and fail on degraded scans.
2. **MT** (`facebook/m2m100_418M`, fine-tuned) can mistranslate, drop content, hallucinate fluent-but-wrong text, or truncate.
3. **Render/overlay** (Pillow fit-to-box, algorithmic) can shrink, wrap, or fail to fit text, changing how it reads.

Errors at any stage **propagate forward** — a single OCR character error becomes a wrong word into MT, which becomes a confidently-rendered wrong translation on the page. Because the output *looks* like a finished, professionally typeset document, it carries an unearned air of authority. The whole agentic design exists to counteract that: the system surfaces its own uncertainty rather than hiding it behind a clean render.

---

## 2. Mistranslation risk in high-stakes documents

Document images are exactly the medium of the highest-stakes text a person owns: **passports and national IDs, immigration and asylum paperwork, medical records and prescriptions, contracts, court filings, financial statements, and product/medication labels.** A mistranslation here is not a cosmetic error — it can cause a wrong medication dose, a misunderstood legal obligation, a denied claim, or a misrepresentation to an official.

P15 must never be used as a substitute for a certified human translator in legal or medical contexts. Its design enforces "assist, do not assert" through concrete, inspectable gates rather than a disclaimer alone:

- **Garbage is never translated (agent D3).** Each OCR block is gated on Tesseract per-block mean confidence plus a character-length / alpha-ratio sanity check. Blocks below the low-confidence threshold are **dropped and flagged `needs_review`**, not fed to MT. The system declines to translate text it could not read, instead of inventing a plausible translation of a misread.
- **Hallucination and truncation are caught without a reference (agent D4).** Each MT output is checked by (a) **round-trip back-translation chrF** between the source and its back-translation and (b) a **target/source length ratio** sanity band. A low round-trip score (likely hallucination or drift) or an out-of-band length ratio (truncation, runaway repetition) flags the block `low_confidence`. This uses only the model's own intermediate outputs — no gold translation is required at inference.
- **The system degrades instead of faking success (agent D5).** When a translation cannot be rendered faithfully into its box, the agent does **not** silently clip or distort it. It falls back to **side-by-side** (original untouched + translated caption) or **`needs_review`** (boxes + raw translation, no destructive render). A broken or low-confidence result is shown honestly, never papered over with a clean-looking overlay.

**Round-trip verification is explicitly a *soft* gate.** It can false-flag legitimate free translations and can miss fluent hallucinations; it triggers a single re-decode and then a `low_confidence` flag — it is never a hard reject that silently deletes content. The honest position is that D3/D4 reduce, but do not eliminate, the chance of a confident wrong translation. **High-stakes documents always require a qualified human translator.**

---

## 3. Surveillance, privacy, and consent

Translating a document image is fundamentally different from translating a string a user typed: **the image can contain personal and biometric information the user never intended to expose to a translation system** — names, addresses, ID and passport numbers, photographs, signatures, dates of birth, medical diagnoses, account numbers. Two distinct concerns follow.

**Consent and ownership.** The system should only be used on documents the user has the right to process. Translating a *third party's* private document — a photographed ID, someone else's medical letter or correspondence — without that person's consent is a misuse of the tool, regardless of what the technology permits. Bulk processing of documents to extract or index personal data (i.e., using OCR+MT as a surveillance pipeline over other people's papers) is an out-of-scope and prohibited use.

**Data minimization and local processing (the privacy posture).**

- **No raw-image retention by default.** The pipeline processes a page and returns the translated text and overlay; it does **not** persist the uploaded image, OCR text, or extracted boxes as a default behavior. Any logging that captures document content must be opt-in, documented, and off by default.
- **Local / offline operation is a first-class path, not a degraded one.** The entire cascade runs offline (SeedEngine stub OCR + dictionary MT + DejaVu fonts) with no network calls, so privacy-sensitive documents can be processed without sending pixels to any third party. The optional LLM "brain" (`anthropic`) is **off by default and advisory only** — when off, nothing leaves the machine.
- **The optional LLM advisor never sees what it doesn't need.** Even when enabled, it is advisory and never rewrites translations; it should not be sent raw document images or PII as a matter of course.

Operators who deploy P15 as a service (FastAPI `/translate-image`, the Gradio UI, or the Docker/HF-Space image) inherit a duty to honor this posture: surface a clear notice that uploaded documents may contain personal data, avoid retaining uploads, and disclose any processing that does cross a network boundary.

---

## 4. Bias

Bias enters this cascade in **two** independent places — OCR and MT — and they compound.

### 4.1 MT bias (gender, dialect, register)
The MT core inherits the biases of its training data (`Helsinki-NLP/opus-100` en-fr and similar parallel corpora):

- **Gender bias.** Translating into gendered languages such as French forces gender assignments the source (e.g. English) left unspecified. "the doctor / the nurse / the engineer" can be defaulted to stereotyped genders. P15 cannot resolve underspecified gender from a single page and may render a stereotyped default.
- **Dialect and register.** Models trained on majority-variety, formal corpora translate non-standard dialects, regional vocabulary, and informal register less well, and may flatten them toward a "standard" variety — erasing voice and sometimes meaning.
- **Coverage asymmetry.** `m2m100` is many-to-many across ~100 languages, but quality is **far** from uniform; high-resource pairs (en↔fr) are strong, low-resource languages much weaker. The interface must not imply equal quality across all supported pairs.

### 4.2 OCR bias (scripts and fonts)
OCR quality is **not** uniform across writing systems or typography, and the front-end was *not* trained by us:

- **Script bias.** Tesseract and similar engines are strongest on Latin print and weaker on complex scripts (Arabic, Devanagari, Thai), CJK, handwriting, and historical or decorative type. A document in a less-supported script enters MT already more corrupted — bias in *who gets read accurately*.
- **Font / layout bias.** Unusual fonts, low contrast, multi-column layouts, rotation, and blur raise the error rate. The synthetic generator's degradation suite (rotation, blur, noise, JPEG recompression) is meant to surface this, but it cannot represent every real-world script/condition.

### 4.3 What we do about it
We do not claim to have *removed* bias. We commit to **measuring and disclosing** it: metrics (MT chrF/BLEU, OCR CER/WER, end-to-end chrF, layout fit-rate) are reported **per language pair and per condition**, never as a single headline number that hides disparity. The honest, non-saturated floor is reported on real `opus-100` eval pairs (where the fine-tuned `m2m100` dominates), not only on the seed set where the dictionary baseline saturates (chrF 79.9 vs identity floor 22.4) because seed pairs overlap the dictionary. Where a script or pair is known-weak, that is documented rather than smoothed over, and downstream rendering of a low-confidence block is gated (Section 2) so weak OCR does not silently become an authoritative-looking translation.

> **Non-commercial / licensing note.** Some higher-quality components are **license-flagged and deliberately excluded** from the shipped default: `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0, non-commercial) and Surya OCR (`vikp/surya_rec2`/`surya_det3`, CC-BY-NC-SA-4.0, non-commercial + share-alike). They are documented as research-only upgrades and **never shipped**. The shipped stack is fully permissive (MIT / Apache-2.0 / SIL OFL fonts). Each `opus-100` pair's license must be verified before commercial use. Choosing a fully-permissive default is itself an ethical choice: it keeps the tool freely usable and avoids quietly violating a non-commercial term.

---

## 5. Accessibility benefit (the upside we are trying to deliver)

The reason to build this responsibly is that the upside is real and pro-social. Document-image translation removes a barrier that disproportionately affects people with the least power to overcome it:

- **Newcomers, migrants, and travelers** can read a letter, form, sign, label, or notice in an unfamiliar language without a human translator at hand.
- **Layout preservation matters for comprehension**, not just aesthetics: a translated form whose fields stay where they were is usable; a wall of detached text is not. The overlay keeps the translation anchored to the original structure.
- **Born-digital PDFs are translated losslessly** (agent D2 extracts the embedded text layer and bypasses OCR entirely), so the common digital-document case incurs zero OCR error.
- **Multilingual coverage from one checkpoint** (m2m100 is many-to-many) means the tool can help across many language communities, not only the dominant one.

These benefits are the justification for the project; the safeguards in Sections 2–4 exist so the benefit is delivered **without** the failure modes turning the tool into a source of confident misinformation about someone's legal or medical situation.

---

## 6. Transparency

The user should always be able to see *what the system did and how sure it was*. P15 is built so its uncertainty is legible, not buried:

- **Per-block confidence is reported, not hidden.** OCR confidence (D3), MT round-trip/length-ratio verification (D4), and render-fit feasibility (D5) are computed per block and exposed, so a user can tell which parts of the page are trustworthy and which are flagged.
- **The chosen output mode is explicit.** The agent's final decision — `overlay`, `side-by-side`, or `needs_review` — is part of the result. A `side-by-side` or `needs_review` outcome is itself a transparency signal: "this did not translate/fit cleanly; check it."
- **The pipeline is deterministic and inspectable.** The agent is a deterministic five-decision state machine, not an opaque LLM. The synthetic data is seeded per index and reproducible. Given the same input, the same decisions and outputs follow, so behavior can be audited.
- **Provenance and limits are documented.** Model ids and licenses, the absence of any real public in-image-translation benchmark (research returned null; evaluation rests on the synthetic generator), and the offline-mode limitations (e.g. CJK/Arabic render as tofu without Noto fonts) are stated openly rather than glossed.

---

## 7. Human-in-the-loop review (a requirement, not a suggestion)

For any consequential use, **a competent human must review the output before it is relied upon.** This is structural in P15: the agent's terminal states are designed to *route work to a human* rather than auto-approve.

- Low-confidence OCR blocks → **`needs_review`** (dropped from translation, surfaced for a human, never silently mistranslated).
- Suspected MT hallucination/truncation → **`low_confidence`** flag (re-decoded once, then handed to the user's judgment).
- Translations that cannot be rendered faithfully → **`side-by-side`** or **`needs_review`** (original preserved for human comparison).

The system's job is to do the easy 90% well, **flag** the hard 10% honestly, and **never** pretend the hard 10% is fine. It is a first-pass assistant that hands uncertain cases to a person — and in legal, medical, immigration, or financial settings, the "person" must be a qualified professional, with a certified human translation obtained where accuracy is legally or medically material.

---

## 8. Responsible-use guidance (summary)

**Appropriate uses**
- Understanding the gist of a document, sign, label, form, or letter in another language.
- A first-pass draft translation that a human then reviews and corrects.
- Processing your **own** documents, or documents you are authorized to handle, ideally offline/locally.

**Inappropriate / prohibited uses**
- Relying on output as a **certified or authoritative** translation of legal, medical, immigration, or financial documents without qualified human review.
- Translating **other people's private or identity documents without their consent.**
- Bulk extraction/indexing of personal data from document images (surveillance use).
- Presenting a confident overlay as ground truth while suppressing the confidence/`needs_review` flags.

**Obligations on deployers (API / UI / Docker / Space)**
- Keep raw-image retention **off by default**; disclose any network processing; prefer local/offline operation for sensitive documents.
- Surface per-block confidence and the agent's output mode to end users.
- Report quality **per language pair and condition**; do not imply uniform quality across scripts or pairs.
- Respect component licenses; never ship the non-commercial-flagged models (`nllb-200-distilled-600M`, Surya).

---

### One-line ethic
**Translate what it can read, flag what it cannot, never assert certainty, keep the original for a human — and treat every document image as if it contains someone's private, high-stakes information, because it often does.**
