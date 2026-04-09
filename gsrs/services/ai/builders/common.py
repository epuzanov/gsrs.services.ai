from __future__ import annotations

from typing import Any, Iterable

from gsrs.model import Modifications, Property, Substance

from ..chunk_metadata import finalize_metadata, get_display_name, get_document_id, make_base_metadata
from ..chunk_models import Chunk
from ..chunk_normalize import (
    amount_to_text,
    clean_text,
    oxford_join,
    shorten_name,
    site_list_to_text,
    slugify,
    unique_texts,
)


def build_chunk(
    substance: Substance,
    *,
    section: str,
    key: str,
    text: str,
    chunk_role: str,
    entity_type: str,
    group_type: str,
    json_path: str,
    hierarchy: list[str] | None = None,
    references: list[str] | None = None,
    exact_match_terms: list[str] | None = None,
    parent_chunk_id: str | None = None,
    rank_hint: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    chunk_id = f'{get_document_id(substance)}:{slugify(section)}:{slugify(key)}'
    base_metadata = make_base_metadata(
        substance,
        section=section,
        chunk_role=chunk_role,
        entity_type=entity_type,
        group_type=group_type,
        json_path=json_path,
        parent_chunk_id=parent_chunk_id,
    )
    return Chunk(
        chunk_id=chunk_id,
        document_id=get_document_id(substance),
        section=section,
        text=clean_text(text),
        metadata=finalize_metadata(
            base_metadata,
            hierarchy=hierarchy,
            references=references,
            exact_match_terms=exact_match_terms,
            rank_hint=rank_hint,
            extra=metadata,
        ),
    )


def display_reference(ref: Any) -> str:
    return (
        clean_text(getattr(ref, 'refPname', None))
        or clean_text(getattr(ref, 'name', None))
        or clean_text(getattr(ref, 'approvalID', None))
        or clean_text(getattr(ref, 'refuuid', None))
    )


def reference_exact_terms(ref: Any) -> list[str]:
    return unique_texts(
        [
            getattr(ref, 'refPname', None),
            getattr(ref, 'name', None),
            getattr(ref, 'approvalID', None),
            getattr(ref, 'refuuid', None),
        ]
    )


def name_priority(name: Any) -> tuple[int, str]:
    priority = 3
    if getattr(name, 'displayName', False):
        priority = 0
    elif getattr(name, 'preferred', False):
        priority = 1
    elif clean_text(getattr(name, 'type', None)).lower() == 'of':
        priority = 2
    return priority, clean_text(getattr(name, 'name', None)).lower()


def summarize_modifications(modifications: Modifications | None) -> dict[str, Any]:
    if modifications is None:
        return {'count': 0, 'text': '', 'terms': []}
    agent_modifications = list(modifications.agentModifications or [])
    physical_modifications = list(modifications.physicalModifications or [])
    structural_modifications = list(modifications.structuralModifications or [])
    parts: list[str] = []
    exact_terms: list[str] = []
    if agent_modifications:
        labels = [
            clean_text(item.agentModificationType)
            + (f' using {display_reference(item.agentSubstance)}' if display_reference(item.agentSubstance) else '')
            for item in agent_modifications
        ]
        parts.append(f'agent modifications include {oxford_join(labels[:4])}')
        for item in agent_modifications:
            exact_terms.extend(
                unique_texts(
                    [
                        item.agentModificationType,
                        item.agentModificationRole,
                        item.agentModificationProcess,
                        display_reference(item.agentSubstance),
                    ]
                )
            )
    if physical_modifications:
        labels = [clean_text(item.physicalModificationRole) for item in physical_modifications]
        parts.append(f'physical modifications include {oxford_join(labels[:4])}')
        for item in physical_modifications:
            exact_terms.extend(unique_texts([item.physicalModificationRole, item.modificationGroup]))
    if structural_modifications:
        labels = []
        for item in structural_modifications:
            label_bits = [clean_text(item.structuralModificationType)]
            if clean_text(item.residueModified):
                label_bits.append(f"residue {clean_text(item.residueModified)}")
            if getattr(item, 'sitesShorthand', None):
                label_bits.append(clean_text(item.sitesShorthand))
            elif getattr(item, 'sites', None):
                label_bits.append(site_list_to_text(item.sites or []))
            labels.append(' '.join(bit for bit in label_bits if bit))
        parts.append(f'structural modifications include {oxford_join(labels[:4])}')
        for item in structural_modifications:
            exact_terms.extend(
                unique_texts(
                    [
                        item.structuralModificationType,
                        item.locationType,
                        item.residueModified,
                        item.sitesShorthand,
                        display_reference(item.molecularFragment),
                        item.molecularFragmentRole,
                    ]
                )
            )
    count = len(agent_modifications) + len(physical_modifications) + len(structural_modifications)
    return {'count': count, 'text': '; '.join(parts), 'terms': unique_texts(exact_terms)}


def render_property_value(property_: Property) -> str:
    parts: list[str] = []
    value_text = amount_to_text(property_.value)
    if value_text:
        parts.append(value_text)
    referenced = display_reference(property_.referencedSubstance)
    if referenced:
        parts.append(f'referenced substance {referenced}')
    parameter_bits: list[str] = []
    for parameter in property_.parameters or []:
        parameter_text = clean_text(parameter.name)
        parameter_type = clean_text(parameter.type)
        parameter_value = amount_to_text(parameter.value)
        part = parameter_text
        if parameter_type:
            part = f'{part} ({parameter_type})' if part else parameter_type
        if parameter_value:
            part = f'{part}: {parameter_value}' if part else parameter_value
        if part:
            parameter_bits.append(part)
    if parameter_bits:
        parts.append('parameters ' + '; '.join(parameter_bits))
    return '. '.join(parts).strip('. ')


def structure_summary(structure: Any) -> str:
    if structure is None:
        return ''
    parts: list[str] = []
    if clean_text(getattr(structure, 'formula', None)):
        parts.append(f"formula {clean_text(structure.formula)}")
    if getattr(structure, 'mwt', None) is not None:
        parts.append(f'molecular weight {clean_text(structure.mwt)}')
    if clean_text(getattr(structure, 'smiles', None)):
        parts.append(f"SMILES {clean_text(structure.smiles)}")
    if clean_text(getattr(structure, 'inchiKey', None)):
        parts.append(f"InChIKey {clean_text(structure.inchiKey)}")
    if clean_text(getattr(structure, 'stereochemistry', None)):
        parts.append(f"stereochemistry {clean_text(structure.stereochemistry)}")
    if clean_text(getattr(structure, 'opticalActivity', None)):
        parts.append(f"optical activity {clean_text(structure.opticalActivity)}")
    return ', '.join(parts)


def sequence_preview(sequence: str, *, max_len: int = 60) -> str:
    cleaned = clean_text(sequence)
    if len(cleaned) <= max_len:
        return cleaned
    return f'{cleaned[:max_len]}...'


def sequence_segments(sequence: str, segment_length: int) -> list[str]:
    cleaned = clean_text(sequence)
    return [cleaned[index : index + segment_length] for index in range(0, len(cleaned), segment_length) if cleaned[index : index + segment_length]]


def choose_feature_properties(properties: Iterable[Property]) -> list[Property]:
    selected: list[Property] = []
    for property_ in properties:
        parameter_names = ' '.join(clean_text(parameter.name).lower() for parameter in property_.parameters or [])
        property_name = clean_text(property_.name).lower()
        if 'feature' in property_name or 'site' in parameter_names or 'range' in parameter_names:
            selected.append(property_)
    return selected


def property_exact_terms(property_: Property) -> list[str]:
    exact_terms: list[str] = [clean_text(property_.name), clean_text(property_.propertyType), clean_text(property_.type)]
    exact_terms.append(amount_to_text(property_.value))
    exact_terms.extend(reference_exact_terms(property_.referencedSubstance))
    for parameter in property_.parameters or []:
        exact_terms.extend(unique_texts([parameter.name, parameter.type, amount_to_text(parameter.value)]))
    return unique_texts(exact_terms)


def overview_definition_sentence(substance: Substance) -> str:
    name = shorten_name(get_display_name(substance), max_words=20)
    substance_class = clean_text(getattr(substance.substanceClass, 'value', substance.substanceClass))
    label = substance_class.replace('structurallyDiverse', 'structurally diverse').replace('specifiedSubstanceG1', 'specified substance G1')
    status = clean_text(substance.status) or 'unspecified status'
    definition_type = clean_text(substance.definitionType) or 'unspecified definition type'
    definition_level = clean_text(substance.definitionLevel) or 'unspecified definition level'
    return (
        f'{name} is a {label} substance. '
        f'Status {status}. Definition type {definition_type}. Definition level {definition_level}.'
    )


def pk_properties(properties: Iterable[Property]) -> list[Property]:
    keywords = (
        'pk',
        'pharmacokinetic',
        'half life',
        'clearance',
        'cmax',
        'auc',
        'bioavailability',
        'volume of distribution',
    )
    matches: list[Property] = []
    for property_ in properties:
        haystack = ' '.join(
            [
                clean_text(property_.name).lower(),
                clean_text(property_.propertyType).lower(),
                clean_text(property_.type).lower(),
            ]
        )
        if any(keyword in haystack for keyword in keywords):
            matches.append(property_)
    return matches
