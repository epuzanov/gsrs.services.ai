from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable

from gsrs.model import Relationship, Substance

from .builders import (
    build_chemical_chunks,
    build_concept_chunks,
    build_mixture_chunks,
    build_nucleic_acid_chunks,
    build_polymer_chunks,
    build_protein_chunks,
    build_specified_substance_g1_chunks,
    build_structurally_diverse_chunks,
)
from .builders.common import (
    build_chunk,
    display_reference,
    name_priority,
    overview_definition_sentence,
    property_exact_terms,
    reference_exact_terms,
    render_property_value,
)
from .chunk_metadata import get_approval_id, get_display_name, get_substance_class
from .chunk_models import Chunk, ChunkerConfig
from .chunk_normalize import clean_text, humanize_token, oxford_join, shorten_name, unique_texts
from .chunk_notes import build_substantive_note_chunks
from .chunk_ranking import sort_chunks, validate_chunks
from .chunk_references import build_reference_index_chunks, resolve_references


def _substance_summary(substance: Substance) -> str:
    substance_class = get_substance_class(substance)
    if substance_class == 'specifiedSubstanceG1':
        constituents = list(getattr(getattr(substance, 'specifiedSubstance', None), 'constituents', None) or [])
        return f'Specified substance with {len(constituents)} constituents.'
    if substance_class == 'mixture':
        components = list(getattr(getattr(substance, 'mixture', None), 'components', None) or [])
        return f'Mixture record with {len(components)} components.'
    if substance_class == 'polymer':
        polymer = getattr(substance, 'polymer', None)
        monomers = list(getattr(polymer, 'monomers', None) or [])
        classification = getattr(polymer, 'classification', None)
        bits = [f'polymer record with {len(monomers)} monomers']
        if classification is not None and clean_text(getattr(classification, 'polymerClass', None)):
            bits.append(f"classification {clean_text(classification.polymerClass)}")
        if clean_text(substance.status):
            bits.append(f"status {clean_text(substance.status)}")
        return ', '.join(bits).capitalize() + '.'
    if substance_class == 'concept':
        return 'Definition not currently available.'
    if substance_class == 'chemical':
        structure = getattr(substance, 'structure', None)
        if structure is not None and clean_text(getattr(structure, 'formula', None)):
            return f"Chemical record with formula {clean_text(structure.formula)}."
        return 'Chemical record.'
    if substance_class == 'protein':
        subunits = list(getattr(getattr(substance, 'protein', None), 'subunits', None) or [])
        return f'Protein record with {len(subunits)} subunits.'
    if substance_class == 'nucleicAcid':
        subunits = list(getattr(getattr(substance, 'nucleicAcid', None), 'subunits', None) or [])
        return f'Nucleic acid record with {len(subunits)} subunits.'
    if substance_class == 'structurallyDiverse':
        structurally_diverse = getattr(substance, 'structurallyDiverse', None)
        genus = clean_text(getattr(structurally_diverse, 'organismGenus', None))
        species = clean_text(getattr(structurally_diverse, 'organismSpecies', None))
        if genus or species:
            return f'Structurally diverse record sourced from {oxford_join([genus, species])}.'
        return 'Structurally diverse record.'
    return f'{humanize_token(substance_class).capitalize()} record.'


def build_overview_chunk(substance: Substance) -> list[Chunk]:
    display_name = get_display_name(substance)
    approval_id = get_approval_id(substance) or 'not assigned'
    exact_terms = [display_name, substance.systemName, substance.approvalID, substance.approvalIDDisplay]
    exact_terms.extend(clean_text(name.name) for name in substance.names or [])
    text = (
        f'{overview_definition_sentence(substance)} '
        f'Approval ID {approval_id}. '
        f'{_substance_summary(substance)}'
    )
    return [
        build_chunk(
            substance,
            section='overview',
            key='overview',
            text=text,
            chunk_role='overview',
            entity_type='substance',
            group_type='overview',
            json_path='$',
            hierarchy=['overview'],
            exact_match_terms=unique_texts(exact_terms),
            rank_hint=0,
            metadata={
                'status': clean_text(substance.status) or None,
                'definition_type': clean_text(substance.definitionType) or None,
                'definition_level': clean_text(substance.definitionLevel) or None,
                'version': clean_text(substance.version) or None,
            },
        )
    ]


def build_core_names_chunk(substance: Substance) -> list[Chunk]:
    names = sorted(list(substance.names or []), key=name_priority)
    if not names:
        return []
    selected = names[:8]
    labels: list[str] = []
    exact_terms: list[str] = []
    for name in selected:
        raw_name = clean_text(name.name)
        label = shorten_name(raw_name)
        roles: list[str] = []
        if getattr(name, 'displayName', False):
            roles.append('display')
        if getattr(name, 'preferred', False):
            roles.append('preferred')
        name_type = clean_text(name.type)
        if name_type:
            roles.append(name_type)
        if roles:
            label = f"{label} ({', '.join(roles)})"
        labels.append(label)
        exact_terms.append(raw_name)
    return [
        build_chunk(
            substance,
            section='core_names',
            key='core',
            text=f'Core names: {oxford_join(labels)}.',
            chunk_role='section_summary',
            entity_type='name',
            group_type='core_names',
            json_path='$.names',
            hierarchy=['names', 'core'],
            exact_match_terms=unique_texts(exact_terms),
            rank_hint=10,
            metadata={'name_count': len(substance.names or [])},
        )
    ]


def build_name_batches(substance: Substance, batch_size: int = 30) -> list[Chunk]:
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for name in sorted(list(substance.names or []), key=name_priority):
        group_key = (
            clean_text(name.type) or 'unspecified',
            '|'.join(unique_texts(getattr(org, 'nameOrg', None) for org in (name.nameOrgs or []))) or 'unspecified',
            '|'.join(unique_texts(name.domains or [])) or 'unspecified',
        )
        grouped[group_key].append(name)
    chunks: list[Chunk] = []
    batch_counter = 0
    for group_key in sorted(grouped):
        name_type, name_orgs, domains = group_key
        values = grouped[group_key]
        for start in range(0, len(values), batch_size):
            batch_counter += 1
            batch = values[start : start + batch_size]
            labels = [shorten_name(name.name) for name in batch]
            exact_terms = [clean_text(name.name) for name in batch]
            text = f'Name batch {batch_counter}: {oxford_join(labels)}.'
            details: list[str] = []
            if name_type != 'unspecified':
                details.append(f'type {name_type}')
            if name_orgs != 'unspecified':
                details.append(f'name organizations {name_orgs}')
            if domains != 'unspecified':
                details.append(f'domains {domains}')
            if details:
                text += ' Grouped by ' + ', '.join(details) + '.'
            chunks.append(
                build_chunk(
                    substance,
                    section='name_batch',
                    key=f'{batch_counter}_{name_type}_{name_orgs}_{domains}',
                    text=text,
                    chunk_role='section_summary',
                    entity_type='name',
                    group_type='name_batch',
                    json_path='$.names',
                    hierarchy=['names', 'batch'],
                    exact_match_terms=exact_terms,
                    rank_hint=11,
                    metadata={
                        'name_type': None if name_type == 'unspecified' else name_type,
                        'name_orgs': None if name_orgs == 'unspecified' else name_orgs.split('|'),
                        'domains': None if domains == 'unspecified' else domains.split('|'),
                        'name_count': len(batch),
                    },
                )
            )
    return chunks


def build_identifier_chunks(substance: Substance) -> list[Chunk]:
    chunks: list[Chunk] = []
    identifiers = [code for code in substance.codes or [] if not getattr(code, 'isClassification', False)]
    for index, code in enumerate(sorted(identifiers, key=lambda item: (clean_text(item.codeSystem), clean_text(item.code)))):
        code_system = clean_text(code.codeSystem) or 'unspecified system'
        code_value = clean_text(code.code)
        code_text = clean_text(code.codeText)
        text = f'Identifier {code_system}: {code_value}.'
        if code_text and code_text != code_value:
            text += f' Code text {code_text}.'
        chunks.append(
            build_chunk(
                substance,
                section='identifier',
                key=f'{code_system}_{code_value}_{index}',
                text=text,
                chunk_role='atomic_fact',
                entity_type='identifier',
                group_type='identifier',
                json_path=f'$.codes[{index}]',
                hierarchy=['codes', 'identifiers'],
                references=resolve_references(substance, code.references),
                exact_match_terms=unique_texts([code_system, code_value, code_text, code.type]),
                rank_hint=12,
                metadata={
                    'code_system': code_system,
                    'code': code_value,
                    'code_text': code_text or None,
                    'code_type': clean_text(code.type) or None,
                },
            )
        )
    return chunks


def build_classification_chunks(substance: Substance) -> list[Chunk]:
    chunks: list[Chunk] = []
    classifications = [code for code in substance.codes or [] if getattr(code, 'isClassification', False)]
    for index, code in enumerate(sorted(classifications, key=lambda item: (clean_text(item.codeSystem), clean_text(item.code)))):
        code_system = clean_text(code.codeSystem) or 'unspecified system'
        code_value = clean_text(code.code)
        hierarchy = unique_texts((code.comments or '').split('|'))
        hierarchy_text = ' > '.join(hierarchy)
        text = f'Classification {code_system}: {code_value}.'
        if hierarchy_text:
            text += f' Hierarchy {hierarchy_text}.'
        chunks.append(
            build_chunk(
                substance,
                section='classification',
                key=f'{code_system}_{code_value}_{index}',
                text=text,
                chunk_role='atomic_fact',
                entity_type='classification',
                group_type='classification',
                json_path=f'$.codes[{index}]',
                hierarchy=['codes', 'classifications'],
                references=resolve_references(substance, code.references),
                exact_match_terms=unique_texts([code_system, code_value, code.comments] + hierarchy),
                rank_hint=13,
                metadata={
                    'code_system': code_system,
                    'code': code_value,
                    'classification_hierarchy': hierarchy,
                    'classification_path': hierarchy_text or None,
                },
            )
        )
    return chunks


def build_atomic_property_chunks(substance: Substance) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, property_ in enumerate(substance.properties or []):
        value_text = render_property_value(property_)
        text = f'Property {clean_text(property_.name)}'
        if value_text:
            text += f': {value_text}'
        text += '.'
        chunks.append(
            build_chunk(
                substance,
                section='atomic_property',
                key=f'{index}_{clean_text(property_.name)}',
                text=text,
                chunk_role='atomic_fact',
                entity_type='property',
                group_type='property',
                json_path=f'$.properties[{index}]',
                hierarchy=['properties', 'atomic'],
                references=resolve_references(substance, property_.references),
                exact_match_terms=property_exact_terms(property_),
                rank_hint=40,
                metadata={
                    'property_name': clean_text(property_.name),
                    'property_type': clean_text(property_.propertyType) or None,
                    'value_type': clean_text(property_.type) or None,
                    'defining': bool(property_.defining),
                },
            )
        )
    return chunks


RELATIONSHIP_GROUPS: list[tuple[str, str]] = [
    ('salt', 'salts'),
    ('impurit', 'impurities'),
    ('metabol', 'metabolites'),
    ('transporter', 'transporters'),
    ('enzyme', 'metabolic_enzymes'),
    ('binder', 'binders'),
    ('active moiety', 'active_moiety'),
    ('target organism', 'target_organisms'),
    ('target', 'targets'),
]


def _relationship_group(value: str) -> str:
    lowered = value.lower()
    for needle, label in RELATIONSHIP_GROUPS:
        if needle in lowered:
            return label
    return 'relationships'


def build_grouped_relationship_summaries(substance: Substance) -> list[Chunk]:
    grouped: dict[str, list[Relationship]] = defaultdict(list)
    for relationship in substance.relationships or []:
        grouped[_relationship_group(clean_text(relationship.type))].append(relationship)
    chunks: list[Chunk] = []
    for index, group_name in enumerate(sorted(grouped), start=1):
        values = grouped[group_name]
        labels = [display_reference(relationship.relatedSubstance) for relationship in values]
        exact_terms: list[str] = []
        for relationship in values:
            exact_terms.extend(unique_texts([relationship.type, relationship.qualification, relationship.interactionType]))
            exact_terms.extend(reference_exact_terms(relationship.relatedSubstance))
        chunks.append(
            build_chunk(
                substance,
                section='relationship_summary',
                key=f'{index}_{group_name}',
                text=f'{group_name.replace("_", " ").capitalize()} relationships include {oxford_join(labels)}.',
                chunk_role='section_summary',
                entity_type='relationship',
                group_type=group_name,
                json_path='$.relationships',
                hierarchy=['relationships', group_name],
                exact_match_terms=unique_texts(exact_terms + labels),
                rank_hint=50,
                metadata={'relationship_count': len(values)},
            )
        )
    return chunks


def build_atomic_relationship_chunks(substance: Substance) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, relationship in enumerate(substance.relationships or []):
        related_name = display_reference(relationship.relatedSubstance)
        bits = [f'Relationship {clean_text(relationship.type) or "unspecified"} with {related_name}']
        if clean_text(relationship.qualification):
            bits.append(f'qualification {clean_text(relationship.qualification)}')
        if clean_text(relationship.interactionType):
            bits.append(f'interaction type {clean_text(relationship.interactionType)}')
        if clean_text(relationship.comments):
            bits.append(f'comments {clean_text(relationship.comments)}')
        if relationship.amount is not None:
            bits.append(f'amount {clean_text(relationship.amount.to_string())}')
        text = '. '.join(bits) + '.'
        exact_terms = unique_texts(
            [
                relationship.type,
                relationship.qualification,
                relationship.interactionType,
                relationship.comments,
                relationship.amount.to_string() if relationship.amount else None,
            ]
            + reference_exact_terms(relationship.relatedSubstance)
            + reference_exact_terms(relationship.mediatorSubstance)
        )
        chunks.append(
            build_chunk(
                substance,
                section='atomic_relationship',
                key=f'{index}_{clean_text(relationship.type)}_{related_name}',
                text=text,
                chunk_role='atomic_fact',
                entity_type='relationship',
                group_type=_relationship_group(clean_text(relationship.type)),
                json_path=f'$.relationships[{index}]',
                hierarchy=['relationships', 'atomic'],
                references=resolve_references(substance, relationship.references),
                exact_match_terms=exact_terms,
                rank_hint=51,
            )
        )
    return chunks


PRIMARY_BUILDERS: dict[str, Callable[..., list[Chunk]]] = {
    'chemical': build_chemical_chunks,
    'specifiedSubstanceG1': build_specified_substance_g1_chunks,
    'mixture': build_mixture_chunks,
    'nucleicAcid': build_nucleic_acid_chunks,
    'protein': build_protein_chunks,
    'polymer': build_polymer_chunks,
    'structurallyDiverse': build_structurally_diverse_chunks,
    'concept': build_concept_chunks,
}


class SubstanceChunker:
    def __init__(
        self,
        class_: type[Any] | None = None,
        config: ChunkerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(class_, ChunkerConfig):
            if config is not None:
                raise TypeError('Provide ChunkerConfig either positionally or via config=, not both')
            config = class_
            class_ = None
        chunk_class = kwargs.pop('class', class_)
        if kwargs:
            unexpected = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected constructor argument(s): {unexpected}')
        self.config = config or ChunkerConfig()
        self._chunk_class: type[Any] = chunk_class or Chunk

    def chunk(self, substance: Substance) -> list[Any]:
        if not isinstance(substance, Substance):
            raise TypeError('SubstanceChunker.chunk expects a gsrs.model.Substance instance')
        chunks: list[Chunk] = []
        chunks += build_overview_chunk(substance)
        chunks += build_core_names_chunk(substance)
        chunks += build_name_batches(substance, batch_size=self.config.name_batch_size)
        chunks += build_identifier_chunks(substance)
        if self.config.include_classification_chunk:
            chunks += build_classification_chunks(substance)
        chunks += self._build_primary_definition_chunks(substance)
        chunks += build_atomic_property_chunks(substance)
        if self.config.include_grouped_relationship_summaries:
            chunks += build_grouped_relationship_summaries(substance)
        chunks += build_atomic_relationship_chunks(substance)
        if self.config.include_reference_index_chunk:
            chunks += build_reference_index_chunks(substance)
        chunks += build_substantive_note_chunks(
            substance,
            include_admin_validation_notes=self.config.include_admin_validation_notes,
        )
        chunks = sort_chunks(chunks)
        validate_chunks(chunks)
        return [self._cast_chunk(chunk) for chunk in chunks]

    def chunk_json(self, payload: dict[str, Any]) -> list[Any]:
        return self.chunk(Substance.model_validate(payload))

    def _build_primary_definition_chunks(self, substance: Substance) -> list[Chunk]:
        builder = PRIMARY_BUILDERS.get(get_substance_class(substance))
        if builder is None:
            return []
        if builder is build_nucleic_acid_chunks:
            return builder(substance, self.config)
        return builder(substance)

    def _cast_chunk(self, chunk: Chunk) -> Any:
        if self._chunk_class is Chunk:
            return chunk
        payload = asdict(chunk)
        if self._chunk_class is dict:
            return payload
        try:
            return self._chunk_class(**payload)
        except TypeError:
            return self._chunk_class(payload)
