from __future__ import annotations

from gsrs.model import Substance

from ..chunk_models import Chunk
from ..chunk_normalize import clean_text, oxford_join, unique_texts
from ..chunk_references import resolve_references
from .common import build_chunk, display_reference, reference_exact_terms


def build_mixture_chunks(substance: Substance) -> list[Chunk]:
    mixture = getattr(substance, 'mixture', None)
    if mixture is None:
        return []
    components = list(mixture.components or [])
    parent = mixture.parentSubstance
    chunks: list[Chunk] = [
        build_chunk(
            substance,
            section='mixture_definition',
            key='definition',
            text=f'Mixture definition with {len(components)} components.',
            chunk_role='primary_definition',
            entity_type='mixture',
            group_type='mixture',
            json_path='$.mixture',
            hierarchy=['primary_definition', 'mixture'],
            exact_match_terms=[substance.approvalID, len(components)],
            rank_hint=20,
            metadata={'component_count': len(components)},
        )
    ]
    if components:
        component_labels = [display_reference(component.substance) or f'component {index + 1}' for index, component in enumerate(components)]
        chunks.append(
            build_chunk(
                substance,
                section='components_summary',
                key='summary',
                text=f'Mixture components include {oxford_join(component_labels)}.',
                chunk_role='section_summary',
                entity_type='mixture',
                group_type='components',
                json_path='$.mixture.components',
                hierarchy=['primary_definition', 'components'],
                exact_match_terms=component_labels,
                rank_hint=21,
                metadata={'component_count': len(components)},
            )
        )
        for index, component in enumerate(components):
            label = display_reference(component.substance) or f'component {index + 1}'
            component_type = clean_text(component.type)
            text = f'Component {index + 1}: {label}'
            if component_type:
                text += f' type {component_type}'
            text += '.'
            chunks.append(
                build_chunk(
                    substance,
                    section='component_atomic',
                    key=f'{index + 1}_{label}',
                    text=text,
                    chunk_role='atomic_fact',
                    entity_type='mixture_component',
                    group_type='component',
                    json_path=f'$.mixture.components[{index}]',
                    hierarchy=['primary_definition', 'components', 'atomic'],
                    references=resolve_references(substance, component.references),
                    exact_match_terms=unique_texts([label, component_type] + reference_exact_terms(component.substance)),
                    rank_hint=22,
                )
            )
    if parent is not None:
        parent_label = display_reference(parent)
        chunks.append(
            build_chunk(
                substance,
                section='parent_substance_summary',
                key='parent',
                text=f'Parent substance summary: {parent_label}.',
                chunk_role='section_summary',
                entity_type='mixture',
                group_type='parent_substance',
                json_path='$.mixture.parentSubstance',
                hierarchy=['primary_definition', 'parent_substance'],
                exact_match_terms=reference_exact_terms(parent),
                rank_hint=23,
                metadata={
                    'parent_substance_name': parent_label or None,
                    'parent_substance_id': clean_text(getattr(parent, 'refuuid', None)) or None,
                },
            )
        )
    return chunks
