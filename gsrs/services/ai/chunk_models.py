from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChunkRole = Literal[
    'overview',
    'primary_definition',
    'section_summary',
    'atomic_fact',
    'provenance',
]

REQUIRED_METADATA_KEYS = {
    'entity_name',
    'document_id',
    'approval_id',
    'substance_class',
    'section',
    'json_path',
    'hierarchy',
    'hierarchy_path',
    'chunk_role',
    'entity_type',
    'group_type',
    'references',
    'created',
    'lastEdited',
    'access',
    'exact_match_terms',
}

OPTIONAL_METADATA_KEYS = {
    'parent_chunk_id',
    'rank_hint',
}


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkerConfig:
    name_batch_size: int = 30
    emit_atomic_name_chunks: bool = False
    emit_sequence_segments: bool = False
    max_sequence_segment_len: int = 300
    emit_full_sequence_in_text: bool = False
    include_admin_validation_notes: bool = False
    include_reference_index_chunk: bool = True
    include_classification_chunk: bool = True
    include_grouped_relationship_summaries: bool = True
