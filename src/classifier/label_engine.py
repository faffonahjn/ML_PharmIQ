"""
src/classifier/label_engine.py
Rule-seeded therapeutic category labeller.

Design: keyword-in-salt-name matching with priority ordering.
No ML required at labelling stage — labels are then used to train System 3 classifier.
Coverage target: >80% of 273K medicines.
"""

import re
from typing import Dict, List, Optional

# Priority-ordered category → salt keyword mapping
# Order matters: more specific categories are checked first
CATEGORY_RULES: Dict[str, List[str]] = {
    # ── Antibiotics ─────────────────────────────────────────────────────────
    "Antibiotic": [
        "azithromycin", "amoxycillin", "amoxicillin", "ceftriaxone", "cefixime",
        "ofloxacin", "ciprofloxacin", "doxycycline", "metronidazole", "ampicillin",
        "levofloxacin", "clavulanic acid", "sulbactam", "clarithromycin", "erythromycin",
        "cefpodoxime", "cefuroxime", "cefoperazone", "tazobactum", "amikacin",
        "gentamicin", "linezolid", "vancomycin", "meropenem", "imipenem",
        "norfloxacin", "gatifloxacin", "moxifloxacin", "roxithromycin", "nitrofurantoin",
        "tinidazole", "ornidazole", "chloramphenicol", "tetracycline", "minocycline",
        "cephalexin", "cloxacillin", "piperacillin", "colistin", "fosfomycin",
    ],

    # ── Analgesics / NSAIDs ──────────────────────────────────────────────────
    "Analgesic": [
        "paracetamol", "diclofenac", "ibuprofen", "aceclofenac", "nimesulide",
        "mefenamic acid", "naproxen", "ketorolac", "tramadol", "tapentadol",
        "etoricoxib", "celecoxib", "piroxicam", "indomethacin", "meloxicam",
        "flurbiprofen", "ketoprofen", "dexketoprofen", "lornoxicam", "valdecoxib",
        "serratiopeptidase", "trypsin", "bromelain", "rutoside",
    ],

    # ── Anti-diabetics ───────────────────────────────────────────────────────
    "Anti-diabetic": [
        "metformin", "glimepiride", "sitagliptin", "vildagliptin", "pioglitazone",
        "dapagliflozin", "empagliflozin", "gliclazide", "glibenclamide", "teneligliptin",
        "voglibose", "alogliptin", "saxagliptin", "linagliptin", "canagliflozin",
        "repaglinide", "nateglinide", "acarbose", "insulin",
    ],

    # ── Cardiac / Antihypertensive ───────────────────────────────────────────
    "Cardiac": [
        "amlodipine", "atenolol", "telmisartan", "ramipril", "losartan",
        "metoprolol", "bisoprolol", "rosuvastatin", "atorvastatin", "clopidogrel",
        "aspirin", "nitroglycerin", "hydrochlorothiazide", "cilnidipine",
        "olmesartan", "valsartan", "candesartan", "irbesartan", "carvedilol",
        "nebivolol", "prazosin", "nifedipine", "felodipine", "isosorbide",
        "digoxin", "warfarin", "rivaroxaban", "dabigatran", "apixaban",
        "furosemide", "spironolactone", "eplerenone", "torasemide",
        "simvastatin", "pitavastatin", "fenofibrate", "ezetimibe",
    ],

    # ── Respiratory ──────────────────────────────────────────────────────────
    "Respiratory": [
        "salbutamol", "formoterol", "budesonide", "fluticasone", "montelukast",
        "levocetirizine", "guaifenesin", "ambroxol", "bromhexine", "terbutaline",
        "ipratropium", "tiotropium", "salmeterol", "beclomethasone", "ciclesonide",
        "dextromethorphan", "codeine", "phenylephrine", "chlorpheniramine maleate",
        "fexofenadine", "cetirizine", "loratadine", "desloratadine", "rupatadine",
        "menthol", "eucalyptol", "camphor", "zinc sulfate",
    ],

    # ── Gastrointestinal ─────────────────────────────────────────────────────
    "Gastrointestinal": [
        "pantoprazole", "rabeprazole", "domperidone", "ondansetron", "metoclopramide",
        "esomeprazole", "omeprazole", "racecadotril", "drotaverine", "famotidine",
        "ranitidine", "lansoprazole", "levosulpiride", "dicyclomine", "simethicone",
        "lactulose", "bisacodyl", "senna", "mesalamine", "sulfasalazine",
        "pancreatin", "lactobacillus", "saccharomyces", "fructooligosaccharides",
        "hyoscine", "ursodeoxycholic acid", "silymarin",
    ],

    # ── Neurological / Psychiatric ───────────────────────────────────────────
    "Neurological": [
        "pregabalin", "gabapentin", "clonazepam", "alprazolam", "escitalopram",
        "nortriptyline", "amitriptyline", "sertraline", "fluoxetine", "paroxetine",
        "venlafaxine", "duloxetine", "mirtazapine", "olanzapine", "risperidone",
        "quetiapine", "aripiprazole", "zolpidem", "melatonin", "phenytoin",
        "carbamazepine", "valproate", "levetiracetam", "topiramate", "lamotrigine",
        "donepezil", "memantine", "rivastigmine", "piracetam",
    ],

    # ── Vitamins / Supplements ───────────────────────────────────────────────
    "Vitamin/Supplement": [
        "methylcobalamin", "pyridoxine", "thiamine", "folic acid", "cyanocobalamin",
        "ascorbic acid", "cholecalciferol", "alpha lipoic acid", "biotin",
        "vitamin", "ferrous", "iron", "calcium", "magnesium", "zinc",
        "multivitamin", "coenzyme", "omega", "fish oil", "l-carnitine",
        "lycopene", "lutein", "mecobalamin",
    ],

    # ── Hormones / Endocrine ─────────────────────────────────────────────────
    "Hormonal": [
        "testosterone", "progesterone", "estradiol", "levothyroxine", "thyroxine",
        "prednisolone", "dexamethasone", "methylprednisolone", "deflazacort",
        "betamethasone", "hydrocortisone", "triamcinolone", "fludrocortisone",
        "nandrolone", "oxytocin", "somatropin",
    ],

    # ── Dermatology ──────────────────────────────────────────────────────────
    "Dermatology": [
        "clobetasol", "ketoconazole", "clotrimazole", "miconazole", "fluconazole",
        "terbinafine", "itraconazole", "griseofulvin", "mupirocin", "fusidic acid",
        "adapalene", "tretinoin", "benzoyl peroxide", "salicylic acid",
        "calcipotriol", "tacrolimus", "pimecrolimus", "lindane", "permethrin",
        "cetrimide", "chlorhexidine", "povidone iodine",
    ],

    # ── Musculoskeletal ──────────────────────────────────────────────────────
    "Musculoskeletal": [
        "thiocolchicoside", "chlorzoxazone", "methocarbamol", "tizanidine",
        "baclofen", "cyclobenzaprine", "carisoprodol", "diazepam",
        "alendronate", "risedronate", "zoledronic acid", "calcitonin",
        "glucosamine", "chondroitin", "diacerhein",
    ],

    # ── Anti-malarial / Anti-parasitic ───────────────────────────────────────
    "Anti-parasitic": [
        "hydroxychloroquine", "chloroquine", "artemether", "lumefantrine",
        "albendazole", "mebendazole", "ivermectin", "praziquantel",
        "pyrimethamine", "quinine",
    ],

    # ── Ophthalmic ───────────────────────────────────────────────────────────
    "Ophthalmic": [
        "timolol", "latanoprost", "brimonidine", "dorzolamide", "bimatoprost",
        "tobramycin", "moxifloxacin ophthalmic", "ciprofloxacin ophthalmic",
        "dexamethasone ophthalmic", "prednisolone acetate",
    ],
}

# Priority order for conflict resolution
PRIORITY_ORDER = [
    "Anti-diabetic", "Antibiotic", "Cardiac", "Neurological",
    "Hormonal", "Anti-parasitic", "Ophthalmic",
    "Analgesic", "Respiratory", "Gastrointestinal",
    "Vitamin/Supplement", "Dermatology", "Musculoskeletal",
]

CATEGORY_CODES = {cat: i for i, cat in enumerate(PRIORITY_ORDER)}
CATEGORY_CODES["Other"] = len(PRIORITY_ORDER)
CODE_TO_CATEGORY = {v: k for k, v in CATEGORY_CODES.items()}
N_CLASSES = len(PRIORITY_ORDER)  # excludes Other


def _parse_salt_names(composition: str) -> List[str]:
    parts = composition.split("+")
    names = []
    for p in parts:
        name = re.sub(r'\s*\([^)]*\)', '', p).strip().lower()
        name = re.sub(r'\s+', ' ', name)
        if name:
            names.append(name)
    return names


def assign_label(composition: str) -> str:
    """Assign therapeutic category using priority-ordered rule matching."""
    salt_names = _parse_salt_names(composition)
    for category in PRIORITY_ORDER:
        keywords = CATEGORY_RULES[category]
        for salt in salt_names:
            if any(kw in salt for kw in keywords):
                return category
    return "Other"


def assign_label_code(composition: str) -> int:
    return CATEGORY_CODES.get(assign_label(composition), CATEGORY_CODES["Other"])


def coverage_report(compositions) -> Dict:
    labels = [assign_label(c) for c in compositions]
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    labelled = sum(v for k, v in counts.items() if k != "Other")
    return {
        "total": total,
        "labelled": labelled,
        "coverage_pct": round(labelled / total * 100, 2),
        "distribution": dict(sorted(counts.items(), key=lambda x: -x[1])),
    }
