from __future__ import annotations

from .chunk_models import Chunk, REQUIRED_METADATA_KEYS

ROLE_ORDER = {
    'overview': 0,
    'primary_definition': 1,
    'section_summary': 2,
    'atomic_fact': 3,
    'provenance': 4,
}


def sort_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return sorted(
        chunks,
        key=lambda chunk: (
            int(chunk.metadata.get('rank_hint', 999)),
            ROLE_ORDER.get(chunk.metadata.get('chunk_role', ''), 99),
            chunk.section,
            chunk.chunk_id,
        ),
    )


def validate_chunks(chunks: list[Chunk]) -> None:
    seen_ids: set[str] = set()
    for chunk in chunks:
        if not chunk.chunk_id:
            raise ValueError('Chunk is missing chunk_id')
        if chunk.chunk_id in seen_ids:
            raise ValueError(f'Duplicate chunk_id detected: {chunk.chunk_id}')
        seen_ids.add(chunk.chunk_id)
        if not chunk.text.strip():
            raise ValueError(f'Chunk {chunk.chunk_id} has empty text')
        missing = REQUIRED_METADATA_KEYS - set(chunk.metadata)
        if missing:
            raise ValueError(f'Chunk {chunk.chunk_id} is missing metadata keys: {sorted(missing)}')
