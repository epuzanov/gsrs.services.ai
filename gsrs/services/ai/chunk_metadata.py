from __future__ import annotations

from typing import Any

from gsrs.model import Substance

from .chunk_normalize import clean_text, enum_value, unique_texts


def get_display_name(substance: Substance) -> str:
    if clean_text(substance.systemName):
        return clean_text(substance.systemName)
    for name in substance.names or []:
        if getattr(name, 'displayName', False) and clean_text(name.name):
            return clean_text(name.name)
    for name in substance.names or []:
        if getattr(name, 'preferred', False) and clean_text(name.name):
            return clean_text(name.name)
    for name in substance.names or []:
        if clean_text(name.name):
            return clean_text(name.name)
    if clean_text(substance.approvalIDDisplay):
        return clean_text(substance.approvalIDDisplay)
    if clean_text(substance.approvalID):
        return clean_text(substance.approvalID)
    if clean_text(substance.uuid):
        return clean_text(substance.uuid)
    return 'UNKNOWN'


def get_document_id(substance: Substance) -> str:
    return clean_text(substance.uuid) or clean_text(substance.approvalID) or 'UNKNOWN'


def get_approval_id(substance: Substance) -> str | None:
    approval_id = clean_text(substance.approvalID)
    return approval_id or None


def get_substance_class(substance: Substance) -> str:
    return clean_text(enum_value(substance.substanceClass)) or 'unknown'


def make_base_metadata(
    substance: Substance,
    *,
    section: str,
    chunk_role: str,
    entity_type: str,
    group_type: str,
    json_path: str,
    parent_chunk_id: str | None = None,
) -> dict[str, Any]:
    return {
        'entity_name': get_display_name(substance),
        'document_id': get_document_id(substance),
        'approval_id': get_approval_id(substance),
        'substance_class': get_substance_class(substance),
        'section': section,
        'json_path': json_path,
        'hierarchy': [],
        'hierarchy_path': section,
        'chunk_role': chunk_role,
        'entity_type': entity_type,
        'group_type': group_type,
        'references': [],
        'created': getattr(substance, 'created', None),
        'lastEdited': getattr(substance, 'lastEdited', None),
        'access': list(getattr(substance, 'access', None) or []),
        'exact_match_terms': [],
        'parent_chunk_id': parent_chunk_id,
    }


def finalize_metadata(
    metadata: dict[str, Any],
    *,
    hierarchy: list[str] | None = None,
    references: list[str] | None = None,
    exact_match_terms: list[str] | None = None,
    rank_hint: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finalized = dict(metadata)
    finalized['hierarchy'] = unique_texts(hierarchy or [metadata['section']])
    finalized['hierarchy_path'] = ' > '.join(finalized['hierarchy']) or metadata['section']
    finalized['references'] = unique_texts(references or [])
    finalized['exact_match_terms'] = unique_texts(exact_match_terms or [])
    if rank_hint is not None:
        finalized['rank_hint'] = rank_hint
    if extra:
        finalized.update(extra)
    return finalized
