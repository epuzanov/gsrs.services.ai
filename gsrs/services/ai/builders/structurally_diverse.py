from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from ..chunk_normalize import oxford_join, unique_texts
from .common import build_chunk, display_reference, reference_exact_terms


def build_structurally_diverse_chunks(substance: Substance) -> list[Chunk]:
    structurally_diverse = getattr(substance, 'structurallyDiverse', None)
    if structurally_diverse is None:
        return []
    chunks: list[Chunk] = []
    source_bits = unique_texts(
        [
            structurally_diverse.sourceMaterialClass,
            structurally_diverse.sourceMaterialType,
            structurally_diverse.sourceMaterialState,
            structurally_diverse.developmentalStage,
            structurally_diverse.fractionName,
            structurally_diverse.fractionMaterialType,
            display_reference(structurally_diverse.parentSubstance),
        ]
    )
    if source_bits:
        chunks.append(
            build_chunk(
                substance,
                section='source_material_summary',
                key='source',
                text=f'Source material summary: {oxford_join(source_bits)}.',
                chunk_role='primary_definition',
                entity_type='structurally_diverse',
                group_type='source_material',
                json_path='$.structurallyDiverse',
                hierarchy=['primary_definition', 'source_material'],
                exact_match_terms=source_bits + reference_exact_terms(structurally_diverse.parentSubstance),
                rank_hint=20,
            )
        )
    taxonomy_bits = unique_texts(
        [
            structurally_diverse.organismFamily,
            structurally_diverse.organismGenus,
            structurally_diverse.organismSpecies,
            structurally_diverse.organismAuthor,
            structurally_diverse.infraSpecificType,
            structurally_diverse.infraSpecificName,
            display_reference(structurally_diverse.hybridSpeciesPaternalOrganism),
            display_reference(structurally_diverse.hybridSpeciesMaternalOrganism),
        ]
    )
    if taxonomy_bits:
        chunks.append(
            build_chunk(
                substance,
                section='taxonomy_summary',
                key='taxonomy',
                text=f'Taxonomy summary: {oxford_join(taxonomy_bits)}.',
                chunk_role='section_summary',
                entity_type='structurally_diverse',
                group_type='taxonomy',
                json_path='$.structurallyDiverse',
                hierarchy=['primary_definition', 'taxonomy'],
                exact_match_terms=taxonomy_bits,
                rank_hint=21,
            )
        )
    organism_part_bits = unique_texts(list(structurally_diverse.part or []) + [structurally_diverse.partLocation])
    if organism_part_bits:
        chunks.append(
            build_chunk(
                substance,
                section='organism_part_summary',
                key='organism_part',
                text=f'Organism part summary: {oxford_join(organism_part_bits)}.',
                chunk_role='section_summary',
                entity_type='structurally_diverse',
                group_type='organism_part',
                json_path='$.structurallyDiverse.part',
                hierarchy=['primary_definition', 'organism_part'],
                exact_match_terms=organism_part_bits,
                rank_hint=22,
            )
        )
    return chunks
