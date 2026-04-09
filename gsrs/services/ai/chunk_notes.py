from __future__ import annotations

from typing import Any

from gsrs.model import Note, Substance

from .builders.common import build_chunk
from .chunk_models import Chunk
from .chunk_normalize import clean_text, unique_texts
from .chunk_references import resolve_references


def is_admin_note(note: Note) -> bool:
    access = {clean_text(value).lower() for value in (note.access or [])}
    text = clean_text(note.note)
    return 'admin' in access or text.startswith('[Validation]')


def split_notes(notes: list[Any]) -> tuple[list[Any], list[Any]]:
    substantive: list[Any] = []
    admin: list[Any] = []
    for note in notes or []:
        if is_admin_note(note):
            admin.append(note)
        elif clean_text(getattr(note, 'note', None)):
            substantive.append(note)
    return substantive, admin


def _note_chunks(substance: Substance, notes: list[Note], *, section: str, group_type: str, rank_hint: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, note in enumerate(notes, start=1):
        note_text = clean_text(note.note)
        if not note_text:
            continue
        chunks.append(
            build_chunk(
                substance,
                section=section,
                key=f'{index}',
                text=f'{section.replace("_", " ").capitalize()} for {substance.systemName or substance.approvalID or "this record"}: {note_text}.',
                chunk_role='atomic_fact',
                entity_type='note',
                group_type=group_type,
                json_path=f'$.notes[{index - 1}]',
                hierarchy=['notes', group_type],
                references=resolve_references(substance, note.references),
                exact_match_terms=unique_texts([note_text]),
                rank_hint=rank_hint,
                metadata={'note_length': len(note_text)},
            )
        )
    return chunks


def build_substantive_note_chunks(
    substance: Substance,
    *,
    include_admin_validation_notes: bool = False,
) -> list[Chunk]:
    substantive, admin = split_notes(list(substance.notes or []))
    chunks = _note_chunks(
        substance,
        substantive,
        section='substantive_note',
        group_type='public_note',
        rank_hint=70,
    )
    if include_admin_validation_notes:
        chunks.extend(
            _note_chunks(
                substance,
                admin,
                section='admin_validation_note',
                group_type='admin_validation_note',
                rank_hint=71,
            )
        )
    return chunks
