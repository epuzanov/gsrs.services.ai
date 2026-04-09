from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from .common import build_chunk


def build_concept_chunks(substance: Substance) -> list[Chunk]:
    return [
        build_chunk(
            substance,
            section='definition_unavailable_note',
            key='definition',
            text='Definition not currently available for this concept record.',
            chunk_role='primary_definition',
            entity_type='concept',
            group_type='definition_unavailable',
            json_path='$',
            hierarchy=['primary_definition', 'concept'],
            exact_match_terms=[substance.approvalID, substance.systemName],
            rank_hint=20,
        )
    ]
