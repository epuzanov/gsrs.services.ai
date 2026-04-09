from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from ..chunk_normalize import clean_text, oxford_join, unique_texts
from ..chunk_references import resolve_references
from .common import build_chunk, pk_properties, structure_summary


def build_chemical_chunks(substance: Substance) -> list[Chunk]:
    structure = getattr(substance, 'structure', None)
    moieties = list(getattr(substance, 'moieties', None) or [])
    chunks: list[Chunk] = []
    if structure is not None:
        text = f'Chemical structure for {clean_text(substance.systemName or substance.approvalID or "this substance")}: {structure_summary(structure)}.'
        chunks.append(
            build_chunk(
                substance,
                section='chemical_structure',
                key='primary',
                text=text,
                chunk_role='primary_definition',
                entity_type='chemical_structure',
                group_type='chemical_structure',
                json_path='$.structure',
                hierarchy=['primary_definition', 'chemical_structure'],
                references=resolve_references(substance, getattr(structure, 'references', None)),
                exact_match_terms=unique_texts(
                    [
                        structure.formula,
                        structure.mwt,
                        structure.smiles,
                        structure.inchiKey,
                        structure.inchi,
                        structure.stereochemistry,
                        structure.opticalActivity,
                    ]
                ),
                rank_hint=20,
                metadata={
                    'formula': clean_text(getattr(structure, 'formula', None)) or None,
                    'molecular_weight': clean_text(getattr(structure, 'mwt', None)) or None,
                },
            )
        )
    if moieties:
        labels = [structure_summary(moiety) or f'moiety {index}' for index, moiety in enumerate(moieties, start=1)]
        chunks.append(
            build_chunk(
                substance,
                section='moieties',
                key='summary',
                text=f'{len(moieties)} moieties are described for this chemical record: {oxford_join(labels[:5])}.',
                chunk_role='section_summary',
                entity_type='chemical_structure',
                group_type='moieties',
                json_path='$.moieties',
                hierarchy=['primary_definition', 'moieties'],
                exact_match_terms=labels,
                rank_hint=21,
                metadata={'moiety_count': len(moieties)},
            )
        )
    properties = pk_properties(substance.properties or [])
    if len(properties) >= 2:
        property_bits = [f"{clean_text(item.name)} {clean_text(item.value.to_string() if item.value else '')}".strip() for item in properties]
        chunks.append(
            build_chunk(
                substance,
                section='pk_summary',
                key='grouped_pk',
                text=f'Grouped pharmacokinetic summary: {oxford_join(property_bits[:6])}.',
                chunk_role='section_summary',
                entity_type='property',
                group_type='pharmacokinetics',
                json_path='$.properties',
                hierarchy=['properties', 'pharmacokinetics'],
                exact_match_terms=property_bits,
                rank_hint=22,
                metadata={'property_count': len(properties)},
            )
        )
    return chunks
