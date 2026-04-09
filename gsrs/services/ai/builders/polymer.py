from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from ..chunk_normalize import amount_to_text, clean_text, oxford_join, unique_texts
from .common import build_chunk, display_reference, reference_exact_terms, structure_summary


def build_polymer_chunks(substance: Substance) -> list[Chunk]:
    polymer = getattr(substance, 'polymer', None)
    if polymer is None:
        return []
    classification = polymer.classification
    monomers = list(polymer.monomers or [])
    chunks: list[Chunk] = [
        build_chunk(
            substance,
            section='polymer_definition',
            key='definition',
            text=(
                f'Polymer definition with {len(monomers)} monomers and '
                f'{len(polymer.structuralUnits or [])} structural units. '
                f'Status {clean_text(substance.status) or "unspecified"}.'
            ),
            chunk_role='primary_definition',
            entity_type='polymer',
            group_type='polymer',
            json_path='$.polymer',
            hierarchy=['primary_definition', 'polymer'],
            exact_match_terms=[substance.status, substance.definitionLevel, substance.definitionType],
            rank_hint=20,
        )
    ]
    if classification is not None:
        bits = unique_texts(
            [
                classification.polymerClass,
                classification.polymerGeometry,
                classification.sourceType,
                display_reference(classification.parentSubstance),
            ]
            + list(classification.polymerSubclass or [])
        )
        chunks.append(
            build_chunk(
                substance,
                section='polymer_classification_summary',
                key='classification',
                text=f'Polymer classification summary: {oxford_join(bits)}.',
                chunk_role='section_summary',
                entity_type='polymer',
                group_type='classification',
                json_path='$.polymer.classification',
                hierarchy=['primary_definition', 'classification'],
                exact_match_terms=bits + reference_exact_terms(classification.parentSubstance),
                rank_hint=21,
            )
        )
    if monomers:
        monomer_bits = []
        for monomer in monomers:
            label = display_reference(monomer.monomerSubstance) or 'unnamed monomer'
            amount = amount_to_text(monomer.amount)
            bit = label
            if clean_text(monomer.type):
                bit = f'{bit} ({clean_text(monomer.type)})'
            if amount:
                bit = f'{bit} {amount}'
            monomer_bits.append(bit)
        chunks.append(
            build_chunk(
                substance,
                section='monomers_summary',
                key='summary',
                text=f'Monomers summary: {oxford_join(monomer_bits)}.',
                chunk_role='section_summary',
                entity_type='polymer',
                group_type='monomers',
                json_path='$.polymer.monomers',
                hierarchy=['primary_definition', 'monomers'],
                exact_match_terms=monomer_bits,
                rank_hint=22,
                metadata={'monomer_count': len(monomers)},
            )
        )
        for index, monomer in enumerate(monomers):
            label = display_reference(monomer.monomerSubstance) or f'monomer {index + 1}'
            amount = amount_to_text(monomer.amount)
            text = f'Monomer {index + 1}: {label}'
            if clean_text(monomer.type):
                text += f' type {clean_text(monomer.type)}'
            if amount:
                text += f' amount {amount}'
            text += '.'
            chunks.append(
                build_chunk(
                    substance,
                    section='monomer_atomic',
                    key=f'{index + 1}_{label}',
                    text=text,
                    chunk_role='atomic_fact',
                    entity_type='polymer_monomer',
                    group_type='monomer',
                    json_path=f'$.polymer.monomers[{index}]',
                    hierarchy=['primary_definition', 'monomers', 'atomic'],
                    exact_match_terms=unique_texts([label, monomer.type, amount] + reference_exact_terms(monomer.monomerSubstance)),
                    rank_hint=23,
                )
            )
    structure_bits = []
    if polymer.displayStructure is not None:
        structure_bits.append('display structure ' + structure_summary(polymer.displayStructure))
    if polymer.idealizedStructure is not None:
        structure_bits.append('idealized structure ' + structure_summary(polymer.idealizedStructure))
    if polymer.structuralUnits:
        structure_bits.append(f'{len(polymer.structuralUnits)} structural units')
    if structure_bits:
        chunks.append(
            build_chunk(
                substance,
                section='polymer_structure_summary',
                key='structure',
                text=f'Polymer structure summary: {oxford_join(structure_bits)}.',
                chunk_role='section_summary',
                entity_type='polymer',
                group_type='structure',
                json_path='$.polymer',
                hierarchy=['primary_definition', 'structure'],
                exact_match_terms=structure_bits,
                rank_hint=24,
            )
        )
    return chunks
