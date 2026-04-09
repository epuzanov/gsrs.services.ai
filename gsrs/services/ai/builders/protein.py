from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from ..chunk_normalize import clean_text, oxford_join, site_list_to_text, unique_texts
from ..chunk_references import resolve_references
from .common import build_chunk, sequence_preview, summarize_modifications


def build_protein_chunks(substance: Substance) -> list[Chunk]:
    protein = getattr(substance, 'protein', None)
    if protein is None:
        return []
    subunits = list(protein.subunits or [])
    protein_subtypes = protein.proteinSubType if isinstance(protein.proteinSubType, list) else [protein.proteinSubType]
    chunks: list[Chunk] = [
        build_chunk(
            substance,
            section='protein_definition',
            key='definition',
            text=(
                f'Protein definition: type {clean_text(protein.proteinType) or "unspecified"}, '
                f'subtypes {oxford_join(protein_subtypes) or "none"}, '
                f'sequence origin {clean_text(protein.sequenceOrigin) or "unspecified"}, '
                f'sequence type {clean_text(protein.sequenceType) or "unspecified"}, '
                f'and {len(subunits)} subunits.'
            ),
            chunk_role='primary_definition',
            entity_type='protein',
            group_type='protein',
            json_path='$.protein',
            hierarchy=['primary_definition', 'protein'],
            exact_match_terms=unique_texts([protein.proteinType, protein.sequenceOrigin, protein.sequenceType] + protein_subtypes),
            rank_hint=20,
            metadata={'subunit_count': len(subunits)},
        )
    ]
    if subunits:
        subunit_bits = [f'subunit {subunit.subunitIndex or index + 1} length {subunit.length or len(subunit.sequence)}' for index, subunit in enumerate(subunits)]
        chunks.append(
            build_chunk(
                substance,
                section='subunits_summary',
                key='summary',
                text=f'Protein subunits summary: {oxford_join(subunit_bits)}.',
                chunk_role='section_summary',
                entity_type='protein_subunit',
                group_type='subunits',
                json_path='$.protein.subunits',
                hierarchy=['primary_definition', 'subunits'],
                exact_match_terms=subunit_bits + [subunit.sequence for subunit in subunits],
                rank_hint=21,
                metadata={'subunit_count': len(subunits)},
            )
        )
        for index, subunit in enumerate(subunits):
            text = (
                f'Subunit {subunit.subunitIndex or index + 1}: '
                f'length {subunit.length or len(subunit.sequence)} '
                f'sequence {sequence_preview(subunit.sequence)}.'
            )
            chunks.append(
                build_chunk(
                    substance,
                    section='subunit_atomic',
                    key=f'{subunit.subunitIndex or index + 1}',
                    text=text,
                    chunk_role='atomic_fact',
                    entity_type='protein_subunit',
                    group_type='subunit',
                    json_path=f'$.protein.subunits[{index}]',
                    hierarchy=['primary_definition', 'subunits', 'atomic'],
                    references=resolve_references(substance, subunit.references),
                    exact_match_terms=[subunit.sequence, subunit.subunitIndex, subunit.length],
                    rank_hint=22,
                )
            )
    glycosylation = protein.glycosylation
    if glycosylation is not None:
        glyco_bits = []
        if glycosylation.glycosylationType:
            glyco_bits.append(clean_text(glycosylation.glycosylationType))
        if glycosylation.sitesShorthand:
            glyco_bits.append(clean_text(glycosylation.sitesShorthand))
        for label, sites in (
            ('C-glycosylation', glycosylation.CGlycosylationSites),
            ('N-glycosylation', glycosylation.NGlycosylationSites),
            ('O-glycosylation', glycosylation.OGlycosylationSites),
        ):
            if sites:
                glyco_bits.append(f'{label} at {site_list_to_text(sites)}')
        if glyco_bits:
            chunks.append(
                build_chunk(
                    substance,
                    section='glycosylation_summary',
                    key='summary',
                    text=f'Glycosylation summary: {oxford_join(glyco_bits)}.',
                    chunk_role='section_summary',
                    entity_type='protein',
                    group_type='glycosylation',
                    json_path='$.protein.glycosylation',
                    hierarchy=['primary_definition', 'glycosylation'],
                    exact_match_terms=glyco_bits,
                    rank_hint=23,
                )
            )
    disulfide_links = list(protein.disulfideLinks or [])
    if disulfide_links:
        disulfide_bits = [
            clean_text(link.sitesShorthand) or site_list_to_text(link.sites or [])
            for link in disulfide_links
        ]
        chunks.append(
            build_chunk(
                substance,
                section='disulfide_summary',
                key='summary',
                text=f'Disulfide links: {oxford_join(disulfide_bits)}.',
                chunk_role='section_summary',
                entity_type='protein',
                group_type='disulfide_links',
                json_path='$.protein.disulfideLinks',
                hierarchy=['primary_definition', 'disulfide_links'],
                exact_match_terms=disulfide_bits,
                rank_hint=24,
                metadata={'disulfide_count': len(disulfide_links)},
            )
        )
    modification_summary = summarize_modifications(protein.modifications or substance.modifications)
    if modification_summary['count']:
        chunks.append(
            build_chunk(
                substance,
                section='protein_modifications_summary',
                key='summary',
                text=f'Protein modifications summary: {modification_summary["text"]}.',
                chunk_role='section_summary',
                entity_type='protein',
                group_type='modifications',
                json_path='$.protein.modifications',
                hierarchy=['primary_definition', 'modifications'],
                exact_match_terms=modification_summary['terms'],
                rank_hint=25,
                metadata={'modification_count': modification_summary['count']},
            )
        )
    return chunks
