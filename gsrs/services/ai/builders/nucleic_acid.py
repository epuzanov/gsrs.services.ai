from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk, ChunkerConfig
from ..chunk_normalize import clean_text, oxford_join, site_list_to_text, unique_texts
from ..chunk_references import resolve_references
from .common import (
    build_chunk,
    choose_feature_properties,
    property_exact_terms,
    render_property_value,
    sequence_preview,
    sequence_segments,
    summarize_modifications,
)


def build_nucleic_acid_chunks(substance: Substance, config: ChunkerConfig | None = None) -> list[Chunk]:
    config = config or ChunkerConfig()
    nucleic_acid = getattr(substance, 'nucleicAcid', None)
    if nucleic_acid is None:
        return []
    subunits = list(nucleic_acid.subunits or [])
    chunks: list[Chunk] = [
        build_chunk(
            substance,
            section='nucleic_acid_definition',
            key='definition',
            text=(
                f'Nucleic acid definition: type {clean_text(nucleic_acid.nucleicAcidType) or "unspecified"}, '
                f'subtypes {oxford_join(nucleic_acid.nucleicAcidSubType or []) or "none"}, '
                f'sequence origin {clean_text(nucleic_acid.sequenceOrigin) or "unspecified"}, '
                f'sequence type {clean_text(nucleic_acid.sequenceType) or "unspecified"}, '
                f'and {len(subunits)} subunits.'
            ),
            chunk_role='primary_definition',
            entity_type='nucleic_acid',
            group_type='nucleic_acid',
            json_path='$.nucleicAcid',
            hierarchy=['primary_definition', 'nucleic_acid'],
            exact_match_terms=unique_texts(
                [nucleic_acid.nucleicAcidType, nucleic_acid.sequenceOrigin, nucleic_acid.sequenceType] + list(nucleic_acid.nucleicAcidSubType or [])
            ),
            rank_hint=20,
            metadata={'subunit_count': len(subunits)},
        )
    ]
    if subunits:
        sequence_bits = [f'subunit {subunit.subunitIndex or index + 1} length {subunit.length or len(subunit.sequence)}' for index, subunit in enumerate(subunits)]
        if config.emit_full_sequence_in_text and len(subunits) == 1:
            sequence_text = clean_text(subunits[0].sequence)
        else:
            sequence_text = oxford_join(
                f"{bit} sequence {sequence_preview(subunit.sequence)}"
                for bit, subunit in zip(sequence_bits, subunits, strict=False)
            )
        sequence_summary_chunk = build_chunk(
            substance,
            section='sequence_summary',
            key='summary',
            text=f'Sequence summary: {sequence_text}.',
            chunk_role='section_summary',
            entity_type='nucleic_acid_sequence',
            group_type='sequence',
            json_path='$.nucleicAcid.subunits',
            hierarchy=['primary_definition', 'sequence'],
            exact_match_terms=[subunit.sequence for subunit in subunits],
            rank_hint=21,
            metadata={'subunit_count': len(subunits)},
        )
        chunks.append(sequence_summary_chunk)
        if config.emit_sequence_segments:
            for subunit_index, subunit in enumerate(subunits):
                for segment_index, segment in enumerate(sequence_segments(subunit.sequence, config.max_sequence_segment_len), start=1):
                    chunks.append(
                        build_chunk(
                            substance,
                            section='sequence_segments',
                            key=f'{subunit.subunitIndex or subunit_index + 1}_{segment_index}',
                            text=f'Sequence segment {segment_index} for subunit {subunit.subunitIndex or subunit_index + 1}: {segment}.',
                            chunk_role='atomic_fact',
                            entity_type='nucleic_acid_sequence',
                            group_type='sequence_segment',
                            json_path=f'$.nucleicAcid.subunits[{subunit_index}]',
                            hierarchy=['primary_definition', 'sequence', 'segments'],
                            exact_match_terms=[segment],
                            parent_chunk_id=sequence_summary_chunk.chunk_id,
                            rank_hint=22,
                        )
                    )
    features = choose_feature_properties(substance.properties or [])
    if features:
        feature_names = [clean_text(feature.name) for feature in features]
        chunks.append(
            build_chunk(
                substance,
                section='feature_summary',
                key='summary',
                text=f'Nucleic acid features include {oxford_join(feature_names)}.',
                chunk_role='section_summary',
                entity_type='nucleic_acid_feature',
                group_type='features',
                json_path='$.properties',
                hierarchy=['features'],
                exact_match_terms=feature_names,
                rank_hint=23,
                metadata={'feature_count': len(features)},
            )
        )
        for index, feature in enumerate(features):
            value_text = render_property_value(feature)
            text = f'Feature {index + 1}: {clean_text(feature.name)}'
            if value_text:
                text += f' {value_text}'
            text += '.'
            chunks.append(
                build_chunk(
                    substance,
                    section='feature_atomic',
                    key=f'{index + 1}_{clean_text(feature.name)}',
                    text=text,
                    chunk_role='atomic_fact',
                    entity_type='nucleic_acid_feature',
                    group_type='feature',
                    json_path=f'$.properties[{index}]',
                    hierarchy=['features', 'atomic'],
                    references=resolve_references(substance, feature.references),
                    exact_match_terms=property_exact_terms(feature),
                    rank_hint=24,
                )
            )
    linkages = list(nucleic_acid.linkages or [])
    if linkages:
        linkage_bits = [
            clean_text(linkage.linkage) + (f' at {clean_text(linkage.sitesShorthand) or site_list_to_text(linkage.sites or [])}' if clean_text(linkage.sitesShorthand) or linkage.sites else '')
            for linkage in linkages
        ]
        chunks.append(
            build_chunk(
                substance,
                section='linkages_summary',
                key='summary',
                text=f'Linkages summary: {oxford_join(linkage_bits)}.',
                chunk_role='section_summary',
                entity_type='nucleic_acid',
                group_type='linkages',
                json_path='$.nucleicAcid.linkages',
                hierarchy=['primary_definition', 'linkages'],
                exact_match_terms=linkage_bits,
                rank_hint=25,
                metadata={'linkage_count': len(linkages)},
            )
        )
    sugars = list(nucleic_acid.sugars or [])
    if sugars:
        sugar_bits = [
            clean_text(sugar.sugar) + (f' at {clean_text(sugar.sitesShorthand) or site_list_to_text(sugar.sites or [])}' if clean_text(sugar.sitesShorthand) or sugar.sites else '')
            for sugar in sugars
        ]
        chunks.append(
            build_chunk(
                substance,
                section='sugars_summary',
                key='summary',
                text=f'Sugars summary: {oxford_join(sugar_bits)}.',
                chunk_role='section_summary',
                entity_type='nucleic_acid',
                group_type='sugars',
                json_path='$.nucleicAcid.sugars',
                hierarchy=['primary_definition', 'sugars'],
                exact_match_terms=sugar_bits,
                rank_hint=26,
                metadata={'sugar_count': len(sugars)},
            )
        )
    modification_summary = summarize_modifications(nucleic_acid.modifications or substance.modifications)
    if modification_summary['count']:
        chunks.append(
            build_chunk(
                substance,
                section='na_modifications_summary',
                key='summary',
                text=f'Nucleic acid modifications summary: {modification_summary["text"]}.',
                chunk_role='section_summary',
                entity_type='nucleic_acid',
                group_type='modifications',
                json_path='$.nucleicAcid.modifications',
                hierarchy=['primary_definition', 'modifications'],
                exact_match_terms=modification_summary['terms'],
                rank_hint=27,
                metadata={'modification_count': modification_summary['count']},
            )
        )
    return chunks
