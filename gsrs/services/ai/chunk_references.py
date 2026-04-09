from __future__ import annotations

from typing import Any

from gsrs.model import Reference, Substance

from .builders.common import build_chunk
from .chunk_models import Chunk
from .chunk_normalize import clean_text, oxford_join, unique_texts


def reference_text(reference: Reference) -> str:
    doc_type = clean_text(reference.docType)
    citation = clean_text(reference.citation)
    if doc_type and citation:
        return f'{doc_type}: {citation}'
    return doc_type or citation or clean_text(reference.id) or clean_text(reference.uuid)


def reference_lookup(substance: Substance) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for reference in substance.references or []:
        label = reference_text(reference)
        if not label:
            continue
        for reference_id in unique_texts([reference.uuid, reference.id]):
            lookup.setdefault(reference_id, label)
    return lookup


def resolve_references(substance: Substance, reference_ids: Any) -> list[str]:
    if not reference_ids:
        return []
    lookup = reference_lookup(substance)
    resolved: list[str] = []
    for reference_id in unique_texts(reference_ids):
        label = clean_text(lookup.get(reference_id))
        if label and label not in resolved:
            resolved.append(label)
    return resolved


def build_reference_index_chunks(substance: Substance) -> list[Chunk]:
    references = [reference_text(reference) for reference in substance.references or []]
    references = unique_texts(references)
    if not references:
        return []
    citations_preview = oxford_join(references[:5])
    text = (
        f"Reference index for {substance.systemName or substance.approvalID or 'this record'}. "
        f'{len(references)} supporting references are attached'
        + (f', including {citations_preview}.' if citations_preview else '.')
    )
    exact_terms: list[str] = []
    for reference in substance.references or []:
        exact_terms.extend(unique_texts([reference.docType, reference.citation, reference.id, reference.uuid, reference.url]))
    return [
        build_chunk(
            substance,
            section='reference_index',
            key='index',
            text=text,
            chunk_role='provenance',
            entity_type='substance',
            group_type='reference_index',
            json_path='$.references',
            hierarchy=['references', 'index'],
            references=references,
            exact_match_terms=exact_terms,
            rank_hint=80,
            metadata={'reference_count': len(references)},
        )
    ]
