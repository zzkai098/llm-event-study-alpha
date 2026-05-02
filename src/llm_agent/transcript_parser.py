"""
transcript_parser.py
====================
Split one earnings call (a slice of `earnings_segments`) into:

  - prepared_remarks: management's scripted statements (Presenter Speech)
  - qa_pairs:         analyst Question paired with the management Answer(s)
                      that immediately follow it
  - speakers:         roster with role inference
                      (executive / analyst / operator / unknown)

Input is the subset of the earnings_segments DataFrame for a single call,
ordered by `componentorder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# componenttypename values seen in the data (per the EDA notebook)
PRESENTER = "Presenter Speech"
QUESTION = "Question"
ANSWER = "Answer"
QA_OPERATOR = "Question and Answer Operator Message"
PRES_OPERATOR = "Presentation Operator Message"


@dataclass
class Speaker:
    name: str
    company: Optional[str]
    role: str  # "executive" | "analyst" | "operator" | "unknown"


@dataclass
class QAPair:
    question_speaker: Speaker
    question_text: str
    answer_speaker: Optional[Speaker]
    answer_text: Optional[str]


@dataclass
class ParsedSession:
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    transcriptid: Optional[int]
    date: Optional[str]
    prepared_remarks: str
    qa_pairs: list[QAPair] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<ParsedSession {self.ticker} {self.fiscal_year} "
            f"Q{self.fiscal_quarter} | prepared={len(self.prepared_remarks)} chars "
            f"| qa_pairs={len(self.qa_pairs)}>"
        )


def _infer_role(
    company_of_person: Optional[str],
    target_company: str,
    component_type: str = "",
) -> str:
    """Infer a speaker's role using companyofperson + component type.

    Priority:
    1. Name/company explicitly contains 'Operator' -> operator
    2. companyofperson is set and matches target company -> executive
    3. companyofperson is set (non-empty, non-operator) -> analyst
    4. companyofperson is None -> fall back to component type:
       Presenter Speech -> executive, Question -> analyst, else -> operator
    """
    co = str(company_of_person).strip().lower() if company_of_person else ""
    if "operator" in co:
        return "operator"
    if co:
        if target_company and target_company.lower() in co:
            return "executive"
        return "analyst"
    # companyofperson is None — use component type as fallback
    if component_type == PRESENTER:
        return "executive"
    if component_type == QUESTION:
        return "analyst"
    if component_type == ANSWER:
        return "executive"
    return "operator"


def _make_speaker(row: pd.Series, target_company: str) -> Speaker:
    name = (row.get("personname") or "Unknown").strip()
    company = row.get("companyofperson")
    component_type = row.get("componenttypename", "")
    role = _infer_role(company, target_company, component_type)
    if "operator" in name.lower():
        role = "operator"
    return Speaker(name=name, company=company, role=role)


def parse_session(
    segments_df: pd.DataFrame,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: int,
    target_company_name: str = "",
) -> ParsedSession:
    """Parse a single call's segments DataFrame into a ParsedSession.

    Args:
        segments_df: subset of earnings_segments for one call
        ticker: company ticker (e.g. "AAPL")
        fiscal_year, fiscal_quarter: which call this is
        target_company_name: company name used to distinguish executives from
            analysts. If empty, role inference falls back to the 'Operator'
            keyword only.
    """
    if segments_df.empty:
        return ParsedSession(
            ticker=ticker, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
            transcriptid=None, date=None, prepared_remarks="",
        )

    df = segments_df.sort_values("componentorder").reset_index(drop=True)
    transcriptid = df["transcriptid"].iloc[0] if "transcriptid" in df else None
    date = df["mostimportantdate"].iloc[0] if "mostimportantdate" in df else None

    # ---- Prepared remarks ----
    prepared_parts: list[str] = []
    for _, r in df[df["componenttypename"] == PRESENTER].iterrows():
        sp = (r.get("personname") or "").strip()
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        prepared_parts.append(f"[{sp}]\n{txt}" if sp else txt)
    prepared_remarks = "\n\n".join(prepared_parts)

    # ---- Q&A pairing ----
    qa_pairs: list[QAPair] = []
    qa_df = df[df["componenttypename"].isin([QUESTION, ANSWER])].reset_index(drop=True)

    i = 0
    while i < len(qa_df):
        row = qa_df.iloc[i]
        if row["componenttypename"] != QUESTION:
            i += 1
            continue
        q_speaker = _make_speaker(row, target_company_name)
        q_text = (row.get("text") or "").strip()

        # Collect any consecutive Answer rows (a question can have multiple answers)
        answer_texts: list[str] = []
        a_speaker: Optional[Speaker] = None
        j = i + 1
        while j < len(qa_df) and qa_df.iloc[j]["componenttypename"] == ANSWER:
            a_row = qa_df.iloc[j]
            if a_speaker is None:
                a_speaker = _make_speaker(a_row, target_company_name)
            sp_name = (a_row.get("personname") or "").strip()
            a_txt = (a_row.get("text") or "").strip()
            if a_txt:
                answer_texts.append(f"[{sp_name}] {a_txt}" if sp_name else a_txt)
            j += 1

        qa_pairs.append(
            QAPair(
                question_speaker=q_speaker,
                question_text=q_text,
                answer_speaker=a_speaker,
                answer_text="\n".join(answer_texts) if answer_texts else None,
            )
        )
        i = j if j > i else i + 1

    # ---- Speaker roster ----
    seen: dict[str, Speaker] = {}
    for _, r in df.iterrows():
        nm = (r.get("personname") or "").strip()
        if not nm or nm in seen:
            continue
        seen[nm] = _make_speaker(r, target_company_name)
    speakers = list(seen.values())

    return ParsedSession(
        ticker=ticker,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        transcriptid=int(transcriptid) if transcriptid is not None else None,
        date=str(date) if date is not None else None,
        prepared_remarks=prepared_remarks,
        qa_pairs=qa_pairs,
        speakers=speakers,
    )
