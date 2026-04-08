from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from gsrs.model import (
    AgentModification,
    ChemicalSubstance,
    Code,
    GinasCommonData,
    GinasCommonSubData,
    MixtureSubstance,
    Name,
    Note,
    NucleicAcidSubstance,
    PolymerSubstance,
    Property,
    ProteinSubstance,
    Reference,
    Relationship,
    SpecifiedSubstanceG1Substance,
    StructurallyDiverseSubstance,
    Substance,
)


class SubstanceChunker:
    """Build embedding chunks for GSRS model objects."""

    _DEFAULT_IDENTIFIERS_ORDER = (
        'FDA UNII',
        'UNII',
        'SMS_ID',
        'SMSID',
        'ASK',
        'ASKP',
        'SVGID',
        'EVMPD',
        'xEVMPD',
        'CAS',
        'DRUG BANK',
        'RXCUI',
        'CHEMBL',
        'PUBCHEM',
    )
    _DEFAULT_CLASSIFICATIONS_ORDER = (
        'WHO-ATC',
        'WHO-VATC',
        'NCI_THESAURUS',
        'EMA ASSESSMENT REPORTS',
        'WHO-ESSENTIAL MEDICINES LIST',
        'NDF-RT',
        'LIVERTOX',
        'FDA ORPHAN DRUG',
        'EU-ORPHAN DRUG',
    )

    def __init__(
        self,
        class_: type[Any] | None = None,
        identifiers_order: list[str] | tuple[str, ...] | None = None,
        classifications_order: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        chunk_class = kwargs.pop('class', class_)
        if kwargs:
            unexpected = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected constructor argument(s): {unexpected}')

        self._chunk_class: type[Any] = chunk_class or dict
        self._identifiers_order = self._normalize_order(
            identifiers_order
            if identifiers_order is not None
            else self._DEFAULT_IDENTIFIERS_ORDER
        )
        self._classifications_order = self._normalize_order(
            classifications_order
            if classifications_order is not None
            else self._DEFAULT_CLASSIFICATIONS_ORDER
        )
        self._root_substance: Substance | None = None
        self._json_paths: dict[int, str] = {}

    def chunk(self, value: Any) -> list[Any]:
        if value is None:
            return []
        self._root_substance = value if isinstance(value, Substance) else getattr(value, '_parent', None)
        if not isinstance(self._root_substance, Substance):
            self._root_substance = None
        self._json_paths = {}
        start_path = self._clean_text(getattr(value, '_json_path', None)) or '$'
        self._annotate(value, self._root_substance, start_path)
        return [self._cast_chunk(chunk) for chunk in self._chunk_tree(value)]

    def _cast_chunk(self, chunk: dict[str, object]) -> Any:
        if self._chunk_class is dict:
            return chunk
        try:
            return self._chunk_class(**chunk)
        except TypeError:
            return self._chunk_class(chunk)

    def _annotate(self, value: Any, parent: Substance | None, json_path: str) -> None:
        current_parent = value if isinstance(value, Substance) else parent
        if isinstance(value, BaseModel):
            self._json_paths[id(value)] = json_path
        if isinstance(value, GinasCommonSubData):
            value._set_parent(current_parent, json_path)
        if isinstance(value, BaseModel):
            for field_name, field in value.__class__.model_fields.items():
                child = value.__dict__.get(field_name)
                if child is None:
                    continue
                alias = self._clean_text(field.alias or field_name)
                child_path = f'{json_path}.{alias}' if alias else json_path
                self._annotate(child, current_parent, child_path)
            return
        if isinstance(value, (list, tuple, set)):
            for index, item in enumerate(value):
                self._annotate(item, current_parent, f'{json_path}[{index}]')

    def _chunk_tree(self, value: Any) -> list[dict[str, object]]:
        if value is None:
            return []
        rows: list[dict[str, object]] = []
        if isinstance(value, BaseModel):
            rows.append(self._build_chunk(value))
            for field_name in value.__class__.model_fields:
                child = value.__dict__.get(field_name)
                if child is not None:
                    rows.extend(self._chunk_tree(child))
            return [row for row in rows if row is not None]
        if isinstance(value, (list, tuple, set)):
            for item in value:
                rows.extend(self._chunk_tree(item))
        return rows

    def _build_chunk(self, value: BaseModel) -> dict[str, object] | None:
        if isinstance(value, Substance):
            return {
                'chunk_id': f'root_uuid:{value.uuid}',
                'document_id': self._clean_text(value.uuid),
                'source_url': self._source_url(value),
                'section': 'summary',
                'text': self.summary_text(value),
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy_metadata('root'),
                    'json_path': '$',
                    'canonical_name': self._stable_name(value),
                    'system_name': self._clean_text(value.systemName) or None,
                    'substance_class': self._substance_class_value(value),
                    'approval_id': self._clean_text(value.approvalID) or None,
                    'status': self._clean_text(value.status) or None,
                    'definition_type': self._clean_text(value.definitionType) or None,
                    'definition_level': self._clean_text(value.definitionLevel) or None,
                    'version': self._clean_text(value.version) or None,
                    'tags': self._clean_list(value.tags) or None,
                    'name_count': len(value.names or []),
                    'code_count': len(value.codes or []),
                    'property_count': len(value.properties or []),
                    'relationship_count': len(value.relationships or []),
                    'note_count': len(value.notes or []),
                    'reference_count': len(value.references or []),
                    **self.substance_class_metadata(value),
                },
            }
        if isinstance(value, Name):
            raw_name = self._clean_text(value.name)
            name_type = self._clean_text(value.type)
            name_type_label = value.get_type_label(name_type)
            std_name = self._clean_text(value.stdName)
            name_orgs = self._clean_list([item.nameOrg for item in (value.nameOrgs or []) if item.nameOrg])
            access = self._access_label(value)
            parts = [f'{raw_name} is a {access} {name_type_label}']
            if value.displayName or value.preferred:
                parts.append('that is used as')
                parts.append('both a display and preferred' if value.displayName and value.preferred else 'a display' if value.displayName else 'a preferred')
                parts.append('name')
            parts.append(f'for {self._embedding_root_name(value)}.')
            if name_orgs:
                parts.append(f'It is registered by {", ".join(name_orgs)} naming organizations as the official name.')
            if std_name and std_name != raw_name:
                parts.append(f'Standardized name {std_name}.')
            return {
                'chunk_id': f'root_names_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'names',
                'text': ' '.join(parts),
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'names'),
                    'json_path': self._embedding_json_path(value, '$'),
                    'name_value': raw_name,
                    'name_type': name_type or None,
                    'name_type_label': name_type_label,
                    'std_name': std_name or None,
                    'preferred': bool(value.preferred),
                    'display_name': bool(value.displayName),
                    'domains': self._clean_list(value.domains) or None,
                    'languages': self._clean_list(value.languages) or None,
                    'name_jurisdiction': self._clean_list(value.nameJurisdiction) or None,
                    'name_orgs': name_orgs or None,
                    'references': self._embedding_references(value) or None,
                },
            }
        if isinstance(value, Code):
            code_system = self._clean_text(value.codeSystem)
            code = self._clean_text(value.code)
            code_type = self._clean_text(value.type)
            code_text = self._clean_text(value.codeText)
            text_parts = [f"{self._embedding_root_name(value)} {self._access_label(value)} {code_type.lower()}"]
            class_metadata = {}
            if value.isClassification:
                text_parts.append(f"classification code {code} in {code_system or 'unknown'} code system:")
                class_parts = [self._clean_text(part) for part in (value.comments or '').split('|') if self._clean_text(part)]
                if class_parts:
                    class_metadata = {'classification_hierarchy': class_parts, 'classification_path': ' > '.join(class_parts)}
                    text_parts.append(f"{class_metadata['classification_path']}.")
            else:
                text_parts.append(f"Identifier code {code} in {code_system or 'unknown'} code system: {code}.")
            if code_text and code_text != code:
                text_parts.append(f'Code text: {code_text}.')
            return {
                'chunk_id': f'root_codes_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'codes',
                'text': ' '.join(text_parts),
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'codes'),
                    **class_metadata,
                    'json_path': self._embedding_json_path(value, '$'),
                    'code_system': code_system or None,
                    'code': code,
                    'code_type': code_type or None,
                    'code_text': code_text or None,
                    'comments': self._clean_text(value.comments) or None,
                    'url': self._clean_text(value.url) or None,
                    'references': self._embedding_references(value) or None,
                },
            }
        if isinstance(value, Reference):
            citation = self._clean_text(value.citation)
            doc_type = self._clean_text(value.docType)
            if not citation and not doc_type:
                return None
            reference_text = self.embedding_reference_text(value)
            parts = [f'{self._access_label(value, True)} reference for {self._embedding_root_name(value)}.']
            if doc_type:
                parts.append(f'Document type {doc_type}.')
            if citation:
                parts.append(f'Citation: {citation}.')
            if value.url:
                parts.append(f'URL: {value.url}.')
            return {
                'chunk_id': f'root_references_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'references',
                'text': ' '.join(parts),
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'references'),
                    'json_path': self._embedding_json_path(value, '$'),
                    'references': [reference_text] if reference_text else None,
                    'doc_type': doc_type or None,
                    'citation': citation or None,
                    'reference_url': self._clean_text(value.url) or None,
                    'reference_id': self._clean_text(value.id or value.uuid) or None,
                    'uploaded_file': self._clean_text(value.uploadedFile) or None,
                    'public_domain': bool(value.publicDomain),
                    'tags': self._clean_list(value.tags) or None,
                },
            }
        if isinstance(value, Property):
            prop_name = self._clean_text(value.name)
            value_text = self._render_property_value(value)
            property_type = self._clean_text(value.propertyType)
            value_type = self._clean_text(value.type)
            referenced_name = value.referencedSubstance.refPname if value.referencedSubstance else ''
            referenced_id = value.referencedSubstance.refuuid if value.referencedSubstance else ''
            parameter_names = self._clean_list([parameter.name for parameter in (value.parameters or []) if parameter.name])
            parts = [f'{self._embedding_root_name(value)} {self._access_label(value)} property {prop_name}.']
            if property_type:
                parts.append(f'Property type {property_type}.')
            if value_type:
                parts.append(f'Value type {value_type}.')
            if value_text:
                parts.append(f'Value {value_text}.')
            return {
                'chunk_id': f'root_properties_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'properties',
                'text': ' '.join(parts),
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'properties'),
                    'json_path': self._embedding_json_path(value, '$'),
                    'property_name': prop_name or None,
                    'property_type': property_type or None,
                    'value_type': value_type or None,
                    'value_text': value_text or None,
                    'defining': bool(value.defining),
                    'referenced_name': referenced_name or None,
                    'referenced_id': referenced_id or None,
                    'parameter_names': parameter_names or None,
                    'references': self._embedding_references(value) or None,
                },
            }
        if isinstance(value, Relationship):
            rel_type = self._clean_text(value.type)
            related_name = value.relatedSubstance.refPname
            related_id = value.relatedSubstance.refuuid
            qualification = self._clean_text(value.qualification)
            interaction_type = self._clean_text(value.interactionType)
            mediator_name = value.mediatorSubstance.refPname if value.mediatorSubstance else ''
            mediator_id = value.mediatorSubstance.refuuid if value.mediatorSubstance else ''
            amount_text = value.amount.to_string() if value.amount else ''
            comments = self._clean_text(value.comments)
            parts = [f'{self._embedding_root_name(value)} has {self._access_label(value)} relationship {rel_type} with {related_name}.']
            if qualification:
                parts.append(f'Qualification {qualification}.')
            if interaction_type:
                parts.append(f'Interaction type {interaction_type}.')
            if mediator_name:
                parts.append(f'Mediator substance {mediator_name}.')
            if amount_text:
                parts.append(f'Amount {amount_text}.')
            if comments:
                parts.append(f'Comments {comments}.')
            return {
                'chunk_id': f'root_relationships_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'relationships',
                'text': ' '.join(parts).strip(),
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'relationships'),
                    'json_path': self._embedding_json_path(value, '$'),
                    'relationship_type': rel_type or None,
                    'related_name': related_name or None,
                    'related_id': related_id or None,
                    'qualification': qualification or None,
                    'interaction_type': interaction_type or None,
                    'mediator_name': mediator_name or None,
                    'mediator_id': mediator_id or None,
                    'amount_text': amount_text or None,
                    'comments': comments or None,
                    'references': self._embedding_references(value) or None,
                },
            }
        if isinstance(value, Note):
            note = self._clean_text(value.note)
            return {
                'chunk_id': f'root_notes_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'notes',
                'text': f'{self._embedding_root_name(value)} {self._access_label(value)} note: {note}',
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'notes'),
                    'json_path': self._embedding_json_path(value, '$'),
                    'note_length': len(note),
                    'references': self._embedding_references(value) or None,
                },
            }
        if isinstance(value, AgentModification):
            return {
                'chunk_id': f'root_modifications_agentModifications_uuid:{value.uuid}',
                'document_id': self._embedding_document_id(value),
                'source_url': self._embedding_source_name(value),
                'section': 'agentModifications',
                'text': f'{self._embedding_root_name(value)} agent modification type {self._clean_text(value.agentModificationType)}.',
                'metadata': {
                    **self._chunk_metadata(value),
                    **self._hierarchy(value, 'agentModifications'),
                    'json_path': self._embedding_json_path(value, '$'),
                    'modification_kind': 'agent',
                },
            }
        return self._generic_chunk(value)

    def _generic_chunk(self, value: BaseModel) -> dict[str, object]:
        section = self._section_name(value)
        parts = [f'{self._subject(value)} {self._access_label(value)} {self._humanize(section)}.']
        details: list[str] = []
        children: list[str] = []
        for field_name, field in value.__class__.model_fields.items():
            child = value.__dict__.get(field_name)
            if child is None:
                continue
            label = self._humanize(self._clean_text(field.alias or field_name))
            if isinstance(child, BaseModel):
                children.append(f'{label} present')
            elif isinstance(child, (list, tuple, set)):
                items = list(child)
                if not items:
                    continue
                if any(isinstance(item, BaseModel) for item in items):
                    children.append(f'{len(items)} {label}')
                else:
                    cleaned = self._clean_list(items)
                    if cleaned:
                        details.append(f'{label} {", ".join(cleaned[:4])}')
            else:
                cleaned = self._clean_text(getattr(child, 'value', child))
                if cleaned:
                    details.append(f'{label} {cleaned}')
        if details:
            parts.append(f'Details include {"; ".join(details[:6])}.')
        if children:
            parts.append(f'Contains {", ".join(children[:6])}.')
        return {
            'chunk_id': self._generic_chunk_id(value, section),
            'document_id': self._document_id(value),
            'source_url': self._source_url(value),
            'section': section,
            'text': ' '.join(parts),
            'metadata': {
                **self._generic_metadata(value, section),
                'object_type': value.__class__.__name__,
            },
        }

    def _hierarchy(self, value: GinasCommonSubData, section: str) -> dict[str, object]:
        json_path = self._embedding_json_path(value, '$')
        parts = ['root'] + [part.split('[', 1)[0] for part in json_path.split('.')[1:] if part.split('[', 1)[0]]
        if len(parts) == 1:
            parts.append(section)
        return self._hierarchy_metadata(*parts)

    def _section_name(self, value: BaseModel) -> str:
        json_path = self._json_path_for(value)
        parts = [part.split('[', 1)[0] for part in json_path.split('.')[1:] if part.split('[', 1)[0]]
        return parts[-1] if parts else self._snake_case(value.__class__.__name__)

    def _generic_chunk_id(self, value: Any, section: str) -> str:
        ident = self._clean_text(getattr(value, 'uuid', None)) or section
        return f'root_{section}_uuid:{ident}'

    def _generic_metadata(self, value: Any, section: str) -> dict[str, object]:
        return {
            'access': self._access_label(value),
            'created': self._clean_text(getattr(value, 'created', None)) or None,
            'lastEdited': self._clean_text(getattr(value, 'lastEdited', None)) or None,
            **self._hierarchy_metadata('root', section),
            'json_path': self._json_path_for(value),
        }

    def _document_id(self, value: Any) -> str:
        if isinstance(value, GinasCommonSubData):
            return self._embedding_document_id(value)
        target = self._root_substance if self._root_substance is not None else value
        return self._clean_text(getattr(target, 'uuid', None))

    def _source_url(self, value: Any) -> str | None:
        if isinstance(value, GinasCommonData):
            return self._embedding_source_name(value)
        target = self._root_substance if self._root_substance is not None else value
        return self._clean_text(getattr(target, 'selfLink', None)) or None

    def _subject(self, value: Any) -> str:
        if isinstance(value, GinasCommonSubData):
            return self._embedding_root_name(value)
        if self._root_substance is not None:
            document_id = self._clean_text(self._root_substance.uuid)
            name = self._clean_text(self._stable_name(self._root_substance))
            return f'Substance {document_id}' if name == document_id else name
        return self._clean_text(getattr(value, 'name', None) or getattr(value, 'uuid', None) or value.__class__.__name__)

    def _json_path_for(self, value: Any) -> str:
        return self._clean_text(getattr(value, '_json_path', None)) or self._json_paths.get(id(value), '$')

    def summary_text(self, substance: Substance) -> str:
        document_name = self._summary_title_name(substance)
        substance_class = self._substance_class_value(substance)
        approval_id_display = self._clean_text(substance.approvalIDDisplay or substance.approvalID)
        definition_type = self._clean_text(substance.definitionType)
        definition_level = self._clean_text(substance.definitionLevel)
        status = self._clean_text(substance.status) or ('approved' if approval_id_display else None)
        parts = [f'{document_name} is a']
        parts.append('protected from access' if substance.access else 'for publicly accessible')
        if substance.deprecated:
            parts.append('deprecated')
        parts.append(f'{substance_class} substance.')
        if status:
            parts.append('Current status is')
            if approval_id_display:
                parts.append(f'{status} with approval ID {approval_id_display}.')
            else:
                parts.append(f'{status}.')
        if definition_type or definition_level:
            parts.append('Definition')
            if definition_type:
                parts.append(f'type {definition_type}' + ('' if definition_level else '.'))
            if definition_level:
                parts.append(f'and definition level {definition_level}.' if definition_type else f'level {definition_level}.')
        for sentence in [
            self.summary_definitional_sentence(substance),
            self._summary_names_sentence(substance),
            self._summary_primary_identifiers_sentence(substance),
            self._summary_classifications_sentence(substance),
            self._summary_content_sentence(substance),
        ]:
            if sentence:
                parts.append(sentence)
        return ' '.join(parts)

    def summary_definitional_sentence(self, substance: Substance) -> str:
        if isinstance(substance, ChemicalSubstance):
            structure = substance.structure
            if structure is None:
                return ''
            formula = self._clean_text(structure.formula)
            molecular_weight = structure.mwt
            stereochemistry = self._clean_text(structure.stereochemistry).lower()
            smiles = self._clean_text(structure.smiles)
            inchi_key = self._clean_text(structure.inchiKey)
            moieties = self._clean_list([getattr(m, 'formula', None) for m in substance.moieties or []])
            access = 'protected' if getattr(substance, 'access', None) else 'public'
            descriptors: list[str] = [f'Structure is a {access} chemical structure']
            if formula:
                descriptors.append(f'molecular formula {formula}')
            if molecular_weight is not None:
                descriptors.append(f'molecular weight {molecular_weight}')
            if stereochemistry:
                descriptors.append('racemic' if stereochemistry == 'racemic' else stereochemistry)
            if smiles and inchi_key:
                descriptors.append(f'with SMILES {smiles} and InChIKey {inchi_key}')
            elif smiles:
                descriptors.append(f'with SMILES {smiles}')
            elif inchi_key:
                descriptors.append(f'with InChIKey {inchi_key}')
            if moieties:
                descriptors.append(f'and moieties {", ".join(moieties)}')
            return f"{', '.join(descriptors)}." if descriptors else ''
        if isinstance(substance, MixtureSubstance):
            if substance.mixture is None:
                return ''
            details: list[str] = []
            component_count = len(substance.mixture.components or [])
            if component_count:
                label = 'component' if component_count == 1 else 'components'
                details.append(f'Mixture with {component_count} {label}')
            parent_name = substance.mixture.parentSubstance.refPname if substance.mixture.parentSubstance else ''
            if parent_name:
                details.append(f'parent substance {parent_name}')
            return f"{', '.join(details)}." if details else ''
        if isinstance(substance, ProteinSubstance):
            details = substance.protein
            if details is None:
                return ''
            parts: list[str] = []
            protein_type = self._clean_text(details.proteinType)
            if protein_type:
                parts.append(f'Protein type {protein_type}')
            protein_subtypes = self._clean_list(details.proteinSubType)
            if protein_subtypes:
                parts.append(f'subtypes {self._oxford_join(protein_subtypes)}')
            subunit_count = len(details.subunits or [])
            if subunit_count:
                label = 'subunit' if subunit_count == 1 else 'subunits'
                parts.append(f'{subunit_count} {label}')
            sequence_origin = self._clean_text(details.sequenceOrigin)
            if sequence_origin:
                parts.append(f'sequence origin {sequence_origin}')
            sequence_type = self._clean_text(details.sequenceType)
            if sequence_type:
                parts.append(f'sequence type {sequence_type}')
            glycosylation_type = self._clean_text(details.glycosylation.glycosylationType if details.glycosylation else None)
            if glycosylation_type:
                parts.append(f'glycosylation type {glycosylation_type}')
            return f"{', '.join(parts)}." if parts else ''
        if isinstance(substance, NucleicAcidSubstance):
            details = substance.nucleicAcid
            if details is None:
                return ''
            parts: list[str] = []
            nucleic_acid_type = self._clean_text(details.nucleicAcidType)
            if nucleic_acid_type:
                parts.append(f'Nucleic acid type {nucleic_acid_type}')
            subtypes = self._clean_list(details.nucleicAcidSubType)
            if subtypes:
                parts.append(f'subtypes {self._oxford_join(subtypes)}')
            subunit_count = len(details.subunits or [])
            if subunit_count:
                label = 'subunit' if subunit_count == 1 else 'subunits'
                parts.append(f'{subunit_count} {label}')
            sequence_origin = self._clean_text(details.sequenceOrigin)
            if sequence_origin:
                parts.append(f'sequence origin {sequence_origin}')
            sequence_type = self._clean_text(details.sequenceType)
            if sequence_type:
                parts.append(f'sequence type {sequence_type}')
            return f"{', '.join(parts)}." if parts else ''
        if isinstance(substance, PolymerSubstance):
            details = substance.polymer
            if details is None:
                return ''
            classification = details.classification
            parts: list[str] = []
            polymer_class = self._clean_text(classification.polymerClass if classification else None)
            if polymer_class:
                parts.append(f'Polymer class {polymer_class}')
            monomer_count = len(details.monomers or [])
            if monomer_count:
                label = 'monomer' if monomer_count == 1 else 'monomers'
                parts.append(f'{monomer_count} {label}')
            structural_unit_count = len(details.structuralUnits or [])
            if structural_unit_count:
                label = 'structural unit' if structural_unit_count == 1 else 'structural units'
                parts.append(f'{structural_unit_count} {label}')
            polymer_geometry = self._clean_text(classification.polymerGeometry if classification else None)
            if polymer_geometry:
                parts.append(f'geometry {polymer_geometry}')
            return f"{', '.join(parts)}." if parts else ''
        if isinstance(substance, SpecifiedSubstanceG1Substance):
            if substance.specifiedSubstance is None:
                return ''
            constituent_count = len(substance.specifiedSubstance.constituents or [])
            if not constituent_count:
                return ''
            label = 'constituent' if constituent_count == 1 else 'constituents'
            return f'Specified substance with {constituent_count} {label}.'
        if isinstance(substance, StructurallyDiverseSubstance):
            details = substance.structurallyDiverse
            if details is None:
                return ''
            parts: list[str] = []
            species = self._clean_text(details.organismSpecies)
            if species:
                parts.append(f'organism species {species}')
            source_material_class = self._clean_text(details.sourceMaterialClass)
            if source_material_class:
                parts.append(f'source material class {source_material_class}')
            source_material_type = self._clean_text(details.sourceMaterialType)
            if source_material_type:
                parts.append(f'source material type {source_material_type}')
            organism_parts = self._clean_list(details.part)
            if organism_parts:
                parts.append(f'parts {self._oxford_join(organism_parts)}')
            return f"Structurally diverse material with {', '.join(parts)}." if parts else ''
        return ''

    def substance_class_metadata(self, substance: Substance) -> dict[str, object]:
        if isinstance(substance, ChemicalSubstance):
            structure = substance.structure
            return {
                'formula': self._clean_text(structure.formula if structure else None) or None,
                'molecular_weight': structure.mwt if structure else None,
                'smiles': self._clean_text(structure.smiles if structure else None) or None,
                'inchi_key': self._clean_text(structure.inchiKey if structure else None) or None,
                'inchi': self._clean_text(structure.inchi if structure else None) or None,
                'structure_digest': self._clean_text(structure.digest if structure else None) or None,
                'structure_hash': self._clean_text(structure.hash if structure else None) or None,
                'stereochemistry': self._clean_text(structure.stereochemistry if structure else None) or None,
                'optical_activity': getattr(structure.opticalActivity, 'value', None) if structure else None,
                'atropisomerism': getattr(structure.atropisomerism, 'value', None) if structure else None,
                'stereo_centers': structure.stereoCenters if structure else None,
                'defined_stereo': structure.definedStereo if structure else None,
                'ez_centers': structure.ezCenters if structure else None,
                'charge': structure.charge if structure else None,
                'structure_references': self._embedding_references(substance, structure.references if structure else None) or None,
                'has_molfile': bool(structure.molfile) if structure else False,
                'moieties': self._clean_list([getattr(m, 'formula', None) for m in substance.moieties or []]) or [],
            }
        if isinstance(substance, MixtureSubstance):
            parent = substance.mixture.parentSubstance if substance.mixture else None
            return {
                'mixture_component_count': len(substance.mixture.components or []) if substance.mixture else 0,
                'mixture_parent_substance': parent.refPname if parent else None,
                'mixture_parent_substance_id': parent.refuuid if parent else None,
            }
        if isinstance(substance, ProteinSubstance):
            details = substance.protein
            glycosylation = details.glycosylation if details else None
            return {
                'protein_subunit_count': len(details.subunits or []) if details else 0,
                'protein_disulfide_link_count': len(details.disulfideLinks or []) if details else 0,
                'protein_other_link_count': len(details.otherLinks or []) if details else 0,
                'protein_type': self._clean_text(details.proteinType if details else None) or None,
                'protein_subtypes': self._clean_list(details.proteinSubType if details else None) or None,
                'protein_sequence_origin': self._clean_text(details.sequenceOrigin if details else None) or None,
                'protein_sequence_type': self._clean_text(details.sequenceType if details else None) or None,
                'glycosylation_type': self._clean_text(glycosylation.glycosylationType if glycosylation else None) or None,
                'c_sites_count': len(glycosylation.CGlycosylationSites or []) if glycosylation else 0,
                'n_sites_count': len(glycosylation.NGlycosylationSites or []) if glycosylation else 0,
                'o_sites_count': len(glycosylation.OGlycosylationSites or []) if glycosylation else 0,
                'has_protein_modifications': bool(details.modifications) if details else False,
            }
        if isinstance(substance, NucleicAcidSubstance):
            details = substance.nucleicAcid
            return {
                'na_subunit_count': len(details.subunits or []) if details else 0,
                'na_linkage_count': len(details.linkages or []) if details else 0,
                'na_sugar_count': len(details.sugars or []) if details else 0,
                'nucleic_acid_type': self._clean_text(details.nucleicAcidType if details else None) or None,
                'nucleic_acid_subtypes': self._clean_list(details.nucleicAcidSubType if details else None) or None,
                'nucleic_acid_sequence_origin': self._clean_text(details.sequenceOrigin if details else None) or None,
                'nucleic_acid_sequence_type': self._clean_text(details.sequenceType if details else None) or None,
                'has_nucleic_acid_modifications': bool(details.modifications) if details else False,
            }
        if isinstance(substance, PolymerSubstance):
            details = substance.polymer
            classification = details.classification if details else None
            parent = classification.parentSubstance if classification else None
            return {
                'polymer_monomer_count': len(details.monomers or []) if details else 0,
                'polymer_structural_unit_count': len(details.structuralUnits or []) if details else 0,
                'polymer_class': self._clean_text(classification.polymerClass if classification else None) or None,
                'polymer_geometry': self._clean_text(classification.polymerGeometry if classification else None) or None,
                'polymer_subclass': self._clean_list(classification.polymerSubclass if classification else None) or None,
                'polymer_source_type': self._clean_text(classification.sourceType if classification else None) or None,
                'polymer_parent_substance': parent.refPname if parent else None,
                'polymer_parent_substance_id': parent.refuuid if parent else None,
                'has_display_structure': bool(details.displayStructure) if details else False,
                'has_idealized_structure': bool(details.idealizedStructure) if details else False,
            }
        if isinstance(substance, SpecifiedSubstanceG1Substance):
            return {
                'specified_substance_constituent_count': len(substance.specifiedSubstance.constituents or []) if substance.specifiedSubstance else 0,
            }
        if isinstance(substance, StructurallyDiverseSubstance):
            details = substance.structurallyDiverse
            paternal = details.hybridSpeciesPaternalOrganism if details else None
            maternal = details.hybridSpeciesMaternalOrganism if details else None
            parent = details.parentSubstance if details else None
            return {
                'source_material_class': self._clean_text(details.sourceMaterialClass if details else None) or None,
                'source_material_state': self._clean_text(details.sourceMaterialState if details else None) or None,
                'source_material_type': self._clean_text(details.sourceMaterialType if details else None) or None,
                'developmental_stage': self._clean_text(details.developmentalStage if details else None) or None,
                'fraction_name': self._clean_text(details.fractionName if details else None) or None,
                'fraction_material_type': self._clean_text(details.fractionMaterialType if details else None) or None,
                'organism_family': self._clean_text(details.organismFamily if details else None) or None,
                'organism_genus': self._clean_text(details.organismGenus if details else None) or None,
                'organism_species': self._clean_text(details.organismSpecies if details else None) or None,
                'organism_author': self._clean_text(details.organismAuthor if details else None) or None,
                'part': self._clean_list(details.part if details else None) or None,
                'part_location': self._clean_text(details.partLocation if details else None) or None,
                'infra_specific_type': self._clean_text(details.infraSpecificType if details else None) or None,
                'infra_specific_name': self._clean_text(details.infraSpecificName if details else None) or None,
                'hybrid_species_paternal_organism': paternal.refPname if paternal else None,
                'hybrid_species_paternal_organism_id': paternal.refuuid if paternal else None,
                'hybrid_species_maternal_organism': maternal.refPname if maternal else None,
                'hybrid_species_maternal_organism_id': maternal.refuuid if maternal else None,
                'parent_substance': parent.refPname if parent else None,
                'parent_substance_id': parent.refuuid if parent else None,
            }
        return {}

    def _access_label(self, value: Any, capitalized: bool = False) -> str:
        label = 'protected' if getattr(value, 'access', None) else 'public'
        return label.capitalize() if capitalized else label

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, BaseModel):
            value = value.model_dump(by_alias=True, exclude_none=True)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, datetime):
            value = value.isoformat()
        text = str(value).replace('\r', ' ').replace('\n', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def _clean_list(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, (str, BaseModel)) or not isinstance(values, (list, tuple, set)):
            values = [values]
        cleaned_values: list[str] = []
        for value in values:
            cleaned = self._clean_text(value)
            if cleaned and cleaned not in cleaned_values:
                cleaned_values.append(cleaned)
        return cleaned_values

    def _references_from_ids(self, reference_ids: Any, reference_text_by_id: dict[str, str] | None) -> list[str]:
        if not reference_text_by_id:
            return []
        references: list[str] = []
        for reference_id in self._clean_list(reference_ids):
            reference_text = self._clean_text(reference_text_by_id.get(reference_id))
            if reference_text and reference_text not in references:
                references.append(reference_text)
        return references

    def _hierarchy_metadata(self, *parts: Any) -> dict[str, Any]:
        hierarchy = [self._clean_text(part) for part in parts if self._clean_text(part)]
        return {'hierarchy': hierarchy, 'hierarchy_path': ' > '.join(hierarchy), 'hierarchy_level': len(hierarchy)}

    def _chunk_metadata(self, value: Any) -> dict[str, Any]:
        return {'access': 'protected' if getattr(value, 'access', None) else 'public', 'created': self._clean_text(getattr(value, 'created', None)) or None, 'lastEdited': self._clean_text(getattr(value, 'lastEdited', None)) or None}

    def _embedding_source_name(self, value: Any) -> str | None:
        self_link = self._clean_text(getattr(value, 'selfLink', None))
        if self_link:
            return self_link
        parent = getattr(value, '_parent', None)
        if parent is not None:
            return self._clean_text(self._embedding_source_name(parent)) or None
        return None

    def _embedding_document_id(self, value: Any) -> str:
        parent = getattr(value, '_parent', None)
        return self._clean_text(parent.uuid if parent else getattr(value, 'uuid', None))

    def _embedding_root_name(self, value: Any) -> str:
        parent = getattr(value, '_parent', None)
        if parent is not None:
            document_id = self._clean_text(parent.uuid)
            root_name = self._clean_text(self._stable_name(parent))
            if root_name:
                return f'Substance {document_id}' if document_id and root_name == document_id else root_name
        root_name = self._clean_text(getattr(value, 'name', None))
        if root_name:
            return root_name
        document_id = self._embedding_document_id(value)
        return f'Substance {document_id}' if document_id else 'Substance'

    def _embedding_json_path(self, value: Any, fallback: str) -> str:
        return self._clean_text(getattr(value, '_json_path', None)) or fallback

    def _embedding_references(self, value: Any, references: Any = None) -> list[str]:
        if isinstance(value, Substance):
            return self._references_from_ids(references, self._reference_text_lookup(value))
        parent = getattr(value, '_parent', None)
        if parent is None:
            return []
        reference_ids = getattr(value, 'references', None) if references is None else references
        return self._embedding_references(parent, reference_ids)

    def embedding_reference_text(self, reference: Reference) -> str:
        doc_type = self._clean_text(reference.docType)
        citation = self._clean_text(reference.citation)
        if doc_type and citation:
            return f'{doc_type}: {citation}'
        return doc_type or citation

    def _stable_name(self, substance: Substance) -> str:
        if substance.systemName:
            return self._clean_text(substance.systemName)
        for item in substance.names:
            if item.displayName and item.name:
                return self._clean_text(item.name)
        for item in substance.names:
            if item.preferred and item.name:
                return self._clean_text(item.name)
        for item in substance.names:
            if item.name:
                return self._clean_text(item.name)
        return self._clean_text(substance.approvalID or substance.uuid or 'Unknown substance')

    def _substance_class_value(self, substance: Substance) -> str:
        substance_class = substance.substanceClass
        if hasattr(substance_class, 'value'):
            return self._clean_text(substance_class.value) or 'unknown'
        return self._clean_text(substance_class) or 'unknown'

    def _reference_text_lookup(self, substance: Substance) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for reference in substance.references or []:
            reference_text = self.embedding_reference_text(reference)
            if not reference_text:
                continue
            for reference_id in (self._clean_text(reference.uuid), self._clean_text(reference.id)):
                if reference_id and reference_id not in lookup:
                    lookup[reference_id] = reference_text
        return lookup

    @staticmethod
    def _render_property_value(property: Property) -> str:
        parts: list[str] = []
        value_text = property.value.to_string() if property.value else ''
        if value_text:
            parts.append(value_text)
        ref_name = property.referencedSubstance.refPname if property.referencedSubstance else ''
        if ref_name:
            parts.append(f'referenced substance {ref_name}')
        param_bits = []
        for parameter in property.parameters or []:
            pname = parameter.name
            ptype = parameter.type
            pvalue = parameter.value.to_string() if parameter.value else ''
            bit = pname
            if ptype:
                bit = f'{bit} ({ptype})' if bit else ptype
            if pvalue:
                bit = f'{bit}: {pvalue}' if bit else pvalue
            if bit:
                param_bits.append(bit)
        if param_bits:
            parts.append('parameters ' + '; '.join(param_bits))
        return '. '.join(parts).strip('. ')

    @staticmethod
    def _oxford_join(values: list[str]) -> str:
        cleaned = [value for value in values if value]
        if not cleaned:
            return ''
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f'{cleaned[0]} and {cleaned[1]}'
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _summary_title_name(self, substance: Substance) -> str:
        name = self._clean_text(self._stable_name(substance))
        letters = ''.join(character for character in name if character.isalpha())
        if letters and letters == letters.upper():
            return name.title()
        return name

    @staticmethod
    def _summary_name_with_languages(item: Name) -> str:
        name = item.name.strip()
        languages = [language.strip() for language in (item.languages or []) if str(language).strip()]
        if not languages:
            return name
        return f"{name} [{'|'.join(languages)}]"

    def _summary_names_sentence(self, substance: Substance) -> str:
        unique_names: list[str] = []
        display_name: str = ''
        preferred_names: list[str] = []
        official_names: dict[str, str] = {}
        for item in substance.names or []:
            name = self._clean_text(item.name)
            if not name or name in unique_names:
                continue
            unique_names.append(name)
            formatted_name = self._summary_name_with_languages(item)
            if item.displayName and not display_name:
                display_name = formatted_name
            if item.preferred:
                preferred_names.append(formatted_name)
            if self._clean_text(item.type) == 'of' and item.nameOrgs:
                official_names[formatted_name] = ', '.join([no.nameOrg for no in item.nameOrgs])

        details: list[str] = []
        if display_name:
            details.append(f'{display_name} as the display name{" and as the preferred name as well" if display_name in preferred_names else ""}')
            if display_name in official_names:
                details[-1] = details[-1] + f' and is registered by {official_names[display_name]} naming organizations as the official name'
        for preferred_name in preferred_names:
            if preferred_name != display_name:
                details.append(f'{preferred_name} as the preferred name')
            if preferred_name in official_names:
                details[-1] = details[-1] + f' and is registered by {official_names[preferred_name]} naming organizations as the official name'
        for official_name, name_orgs in official_names.items():
            if official_name != display_name and official_name not in preferred_names:
                role = 'as the official name' if official_name != display_name else 'as the official name as well'
                details.append(f'{official_name} is registered by {name_orgs} naming organizations as the official name')
        if not details:
            return ''
        return f'The record includes official and alternative names, including {self._oxford_join(details)}.'

    def _summary_primary_identifiers_sentence(self, substance: Substance) -> str:
        order_lookup = {system: order for order, system in enumerate(self._identifiers_order)}
        primary_identifiers: list[tuple[tuple[int, int], str]] = []
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(substance.codes or []):
            if self._clean_text(item.type) != 'PRIMARY' or item.isClassification:
                continue
            system = self._clean_text(item.codeSystem)
            code = self._clean_text(item.code)
            if not system or not code:
                continue
            system_upper = system.upper()
            key = (system_upper, code)
            if key in seen:
                continue
            seen.add(key)
            order = order_lookup.get(system_upper, len(self._identifiers_order) + index)
            primary_identifiers.append(((order, index), f"{system}: {code}"))
        if not primary_identifiers:
            return ''
        labels = [label for _, label in sorted(primary_identifiers)[:8]]
        return f'It also includes primary identifiers such as {self._oxford_join(labels)}.'

    def _summary_classifications_sentence(self, substance: Substance) -> str:
        order_lookup = {system: order for order, system in enumerate(self._classifications_order)}
        classifications: list[tuple[tuple[int, int], str]] = []
        seen: set[str] = set()
        for index, item in enumerate(substance.codes or []):
            if self._clean_text(item.type) != 'PRIMARY':
                continue
            system = self._clean_text(item.codeSystem)
            if not system:
                continue
            system_upper = system.upper()
            if system_upper in seen:
                continue
            if not item.isClassification and system_upper not in order_lookup:
                continue
            seen.add(system_upper)
            order = order_lookup.get(system_upper, len(self._classifications_order) + index)
            classifications.append(((order, index), system))
        if not classifications:
            return ''
        labels = [label for _, label in sorted(classifications)[:9]]
        return f'And classifications such as {self._oxford_join(labels)}.'

    def _summary_content_topics(self, substance: Substance) -> list[str]:
        topics: list[str] = []
        if substance.names:
            topics.append('names')
        if substance.codes:
            topics.append('codes')
        if hasattr(substance, 'moieties') and substance.moieties:
            topics.append('moieties')
        if substance.properties:
            topics.append('properties')
        if substance.relationships:
            topics.append('relationships')
        if hasattr(substance, 'modifications') and substance.modifications:
            if substance.modifications.agentModifications:
                topics.append('agentModifications')
            if substance.modifications.physicalModifications:
                topics.append('physicalModifications')
            if substance.modifications.structuralModifications:
                topics.append('structuralModifications')
        if substance.references:
            topics.append('references')
        if substance.notes:
            topics.append('notes')
        if substance.tags:
            topics.append('tags')
        return topics

    def _summary_content_sentence(self, substance: Substance) -> str:
        topics = self._summary_content_topics(substance)
        if not topics:
            return ''
        return f'Content covers {self._oxford_join(topics)}.'

    @staticmethod
    def _normalize_order(values: list[str] | tuple[str, ...]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            cleaned = str(value).strip().upper()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _snake_case(value: str) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', value).lower()

    @staticmethod
    def _humanize(value: str) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', value).replace('_', ' ').lower()
