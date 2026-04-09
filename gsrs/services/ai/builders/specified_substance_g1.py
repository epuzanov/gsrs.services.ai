from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from ..chunk_normalize import amount_to_text, clean_text, oxford_join, unique_texts
from ..chunk_references import resolve_references
from .common import build_chunk, display_reference, reference_exact_terms


def build_specified_substance_g1_chunks(substance: Substance) -> list[Chunk]:
    specified_substance = getattr(substance, 'specifiedSubstance', None)
    constituents = list(getattr(specified_substance, 'constituents', None) or [])
    if specified_substance is None:
        return []
    chunks: list[Chunk] = []
    chunks.append(
        build_chunk(
            substance,
            section='specified_substance_definition',
            key='definition',
            text=f'Specified substance G1 definition with {len(constituents)} constituents.',
            chunk_role='primary_definition',
            entity_type='specified_substance',
            group_type='specified_substance_g1',
            json_path='$.specifiedSubstance',
            hierarchy=['primary_definition', 'specified_substance_g1'],
            exact_match_terms=[substance.approvalID, len(constituents)],
            rank_hint=20,
            metadata={'constituent_count': len(constituents)},
        )
    )
    if constituents:
        summary_bits = []
        for constituent in constituents:
            label = display_reference(constituent.substance)
            amount = amount_to_text(constituent.amount)
            role = clean_text(constituent.role)
            bit = label or 'unnamed constituent'
            if role:
                bit = f'{bit} as {role}'
            if amount:
                bit = f'{bit} at {amount}'
            summary_bits.append(bit)
        chunks.append(
            build_chunk(
                substance,
                section='constituents_summary',
                key='summary',
                text=f'Constituent summary: {oxford_join(summary_bits)}.',
                chunk_role='section_summary',
                entity_type='specified_substance',
                group_type='constituents',
                json_path='$.specifiedSubstance.constituents',
                hierarchy=['primary_definition', 'constituents'],
                exact_match_terms=summary_bits,
                rank_hint=21,
                metadata={'constituent_count': len(constituents)},
            )
        )
        for index, constituent in enumerate(constituents):
            label = display_reference(constituent.substance) or 'unnamed constituent'
            amount = amount_to_text(constituent.amount)
            role = clean_text(constituent.role)
            text = f'Constituent {index + 1}: {label}'
            if role:
                text += f' role {role}'
            if amount:
                text += f' amount {amount}'
            text += '.'
            chunks.append(
                build_chunk(
                    substance,
                    section='constituent_atomic',
                    key=f'{index + 1}_{label}',
                    text=text,
                    chunk_role='atomic_fact',
                    entity_type='constituent',
                    group_type='constituent',
                    json_path=f'$.specifiedSubstance.constituents[{index}]',
                    hierarchy=['primary_definition', 'constituents', 'atomic'],
                    references=resolve_references(substance, constituent.references),
                    exact_match_terms=unique_texts([label, role, amount] + reference_exact_terms(constituent.substance)),
                    rank_hint=22,
                )
            )
    return chunks
