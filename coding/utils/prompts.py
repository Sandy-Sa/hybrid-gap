"""Shared prompt sections used across extraction and gap analysis modules.

Sections that mention an ICD code system (e.g. "ICD-9 dictionary", etiology-
manifestation examples) are provided as factory functions that take a
`code_type` string ("ICD-9" | "ICD-10"). Sections that are code-system-agnostic
remain as plain string constants.
"""

import hashlib

from coding.pipeline.data_loader import _normalize_code_type


def prompt_version(prompt_text: str) -> str:
    """Return a short hash of the prompt text for traceability."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:8]


def _code_system_label(code_type: str) -> str:
    return "ICD-10" if _normalize_code_type(code_type) == "ICD-10" else "ICD-9"

WHAT_TO_SCAN = """WHAT TO SCAN (do a full pass over the entire note)
- Discharge diagnoses / final diagnoses / principal diagnosis
- Hospital course / problems addressed / assessment & plan summaries
- Procedures / operations / interventions
- Complications / adverse events
- Consult summaries (e.g., cardiology, surgery, infectious disease)
- Imaging/procedure reports if present
- Any lines describing treatments that imply an evaluated condition (ONLY if the condition is explicitly named somewhere; do not infer new diagnoses solely from meds)"""

INCLUSION_RULES_DIAGNOSES = """DIAGNOSES: include
- Primary reason for admission / principal diagnosis. Ensure this is always captured as first diagnosis.
- Secondary diagnoses addressed/treated/monitored (comorbidities that impacted care)
- Complications that occurred during admission
- Clinically significant symptoms/findings that were treated or drove workup (e.g., "chest pain", "AKI", "hypoxia") if documented as a problem
- Infections with organism/site if stated; acuity/severity/stage/laterality if stated
- Prefer the MOST SPECIFIC disease/diagnosis wording available in the note (organism, site, acuity, severity, stage/grade, laterality, type, etiology, temporal qualifiers, etc.).
- If the note does not specify the disease beyond a general term, output the general disease name without adding qualifiers like "unspecified".
- Do NOT extract negated findings (e.g., "no evidence of pneumonia", "ruled out for PE") as diagnoses. Only extract conditions that are confirmed or documented as present/active.
- Do NOT extract historical conditions listed as "resolved" unless they impacted current care."""

INCLUSION_RULES_PROCEDURES = """PROCEDURES: include
- Operations/surgeries (and key details if stated: approach, laterality, site, device)
- Invasive bedside procedures (central line, intubation, thoracentesis, dialysis, transfusion, cardioversion, etc.)
- Significant diagnostic procedures if documented as procedures (endoscopy, bronchoscopy, cath, biopsies)
- If an intervention was planned but not done, do not include it."""

DO_NOT_INVENT = """DO NOT INVENT
- Do not add diagnoses or procedures that are not present in the text.
- Do not use outside medical knowledge to infer.
- If a concept is vague, extract it as written and note missing specificity in notes."""

EVIDENCE_REQUIREMENTS = """EVIDENCE REQUIREMENTS
- Evidence quote must be an EXACT substring from the discharge summary.
- Prefer short quotes (roughly 5-25 words). If the note contains identifiers near the concept, quote only the minimal phrase that contains the clinical concept.
- Section should be the closest section header (or best guess: "Discharge Diagnoses", "Hospital Course", "Procedures", "Complications", "Consults", "Imaging", "Medications", "Other")."""

DEDUPLICATION = """DEDUPLICATION
- Do not output duplicates. If the same concept appears multiple times, output ONE item with up to 3 evidence entries."""

def normalization_rules(code_type: str = "ICD-9") -> str:
    """NORMALIZATION RULES block, parameterized by code system."""
    label = _code_system_label(code_type)
    return f"""NORMALIZATION RULES (for the "normalized" field)
The normalized field will be used to search an {label} code dictionary. To maximize search accuracy:
- Expand ALL abbreviations to full terms (e.g., "CHF" -> "congestive heart failure", "CABG" -> "coronary artery bypass graft", "AKI" -> "acute kidney injury", "NSTEMI" -> "non-ST elevation myocardial infarction", "AFib" -> "atrial fibrillation")
- Use standard clinical terminology, not patient-friendly language
- Include anatomical site, laterality, and acuity/chronicity when documented (e.g., "acute systolic heart failure" not just "heart failure")
- For infections, include organism and site if documented (e.g., "urinary tract infection due to Escherichia coli")
- For procedures, include approach and site (e.g., "percutaneous transluminal coronary angioplasty" not just "PTCA")
- Do NOT include ICD codes in the normalized field
- Use lowercase except for proper nouns and organism names
- If a concept is unspecified in the note, use the general term WITHOUT adding "unspecified" -- the search system handles mapping to the appropriate unspecified code"""

FEW_SHOT_EXAMPLE = """EXAMPLE
Input excerpt: "Pt admitted for acute exacerbation of COPD. PMH: HTN, DM2, CKD stage 3. Underwent bronchoscopy with BAL on HD2. Ruled out for PE."

Output:
{
  "diagnoses": [
    {"name": "acute exacerbation of COPD", "normalized": "acute exacerbation of chronic obstructive pulmonary disease", "evidence": [{"quote": "admitted for acute exacerbation of COPD", "section": "History of Present Illness"}]},
    {"name": "HTN", "normalized": "hypertension", "evidence": [{"quote": "PMH: HTN, DM2, CKD stage 3", "section": "Past Medical History"}]},
    {"name": "DM2", "normalized": "type 2 diabetes mellitus", "evidence": [{"quote": "PMH: HTN, DM2, CKD stage 3", "section": "Past Medical History"}]},
    {"name": "CKD stage 3", "normalized": "chronic kidney disease stage III", "evidence": [{"quote": "PMH: HTN, DM2, CKD stage 3", "section": "Past Medical History"}]}
  ],
  "procedures": [
    {"name": "bronchoscopy with BAL", "normalized": "bronchoscopy with bronchoalveolar lavage", "evidence": [{"quote": "Underwent bronchoscopy with BAL on HD2", "section": "Hospital Course"}]}
  ]
}
Note: "Ruled out for PE" is NOT extracted because it is a negated finding."""

OUTPUT_FORMAT = """OUTPUT FORMAT (STRICT)
Return ONLY valid JSON. No markdown. No commentary.
The JSON must have exactly two top-level keys: "diagnoses" and "procedures".
If none found, return empty arrays."""

def output_schema(code_type: str = "ICD-9") -> str:
    """Top-level extraction/gap-analysis output SCHEMA, parameterized by code system."""
    label = _code_system_label(code_type)
    return f"""SCHEMA
{{
  "diagnoses": [
    {{
      "name": "string (clinical phrase exactly as used in the note)",
      "normalized": "string (fully expanded clinical term for {label} dictionary search; no abbreviations, no codes)",
      "evidence": [
        {{ "quote": "string", "section": "string" }}
      ]
    }}
  ],
  "procedures": [
    {{
      "name": "string (procedure/intervention phrase exactly as used in the note)",
      "normalized": "string (fully expanded procedure name with approach/site; no abbreviations, no codes)",
      "evidence": [
        {{ "quote": "string", "section": "string" }}
      ]
    }}
  ]
}}"""


def combination_code_rule(code_type: str = "ICD-9") -> str:
    """Rule line: combination codes do not subsume their components. Example adapts to code system."""
    if _normalize_code_type(code_type) == "ICD-9":
        return (
            '- COMBINATION CODES do NOT subsume their individual components: e.g. if '
            '"hypertensive CKD" (403.xx) is assigned, "essential hypertension" (401.9) '
            'and "CKD" may still be separately reportable and should be listed if '
            'independently documented.'
        )
    return (
        '- COMBINATION CODES do NOT subsume their individual components: e.g. if a '
        'hypertensive CKD combination code (I12.x) is assigned, "essential hypertension" '
        '(I10) and "CKD" (N18.x) may still be separately reportable and should be listed '
        'if independently documented.'
    )


def etiology_manifestation_rule(code_type: str = "ICD-9") -> str:
    """Rule line: etiology-manifestation dual coding. Example adapts to code system."""
    if _normalize_code_type(code_type) == "ICD-9":
        return (
            "- ICD-9 ETIOLOGY-MANIFESTATION DUAL CODING: ICD-9 requires two codes when a "
            "manifestation is caused by an underlying disease. If the note documents a "
            "diabetic complication (e.g., diabetic peripheral neuropathy, diabetic "
            "nephropathy, diabetic retinopathy, diabetic macular edema, diabetic foot ulcer, "
            "diabetic gastroparesis), list the manifestation as a SEPARATE missing item "
            "even if the diabetes code (250.xx) is already assigned. The manifestation "
            "code (e.g., polyneuropathy in diabetes, background diabetic retinopathy, "
            "diabetic nephropathy) is always a separately reportable additional code. "
            "Apply the same rule to other etiology-manifestation pairs documented in the "
            "note (e.g., hypertensive heart disease + heart failure, anemia due to chronic "
            "disease + the chronic disease)."
        )
    return (
        "- ICD-10-CM ETIOLOGY-MANIFESTATION AND COMBINATION CODING: ICD-10-CM bundles many "
        "etiology+manifestation pairs into combination codes (e.g., E11.21 Type 2 diabetes "
        "with diabetic nephropathy, I11.0 hypertensive heart disease with heart failure). "
        "If the note documents a diabetic complication (e.g., diabetic peripheral "
        "neuropathy, diabetic nephropathy, diabetic retinopathy, diabetic macular edema, "
        "diabetic foot ulcer, diabetic gastroparesis), list the manifestation as a SEPARATE "
        "missing item even if a generic diabetes code (e.g., E11.9) is already assigned, so "
        "the verifier can find the correct combination code. Apply the same rule to other "
        "etiology-manifestation pairs (e.g., hypertensive heart disease + heart failure, "
        "anemia due to chronic disease + the chronic disease)."
    )
