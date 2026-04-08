import unittest

import _bootstrap  # noqa: F401

from gsrs.model import ChemicalSubstance, Code, MixtureSubstance, Substance
from gsrs.services.ai import SubstanceChunker


class EmbeddingChunkTests(unittest.TestCase):
    def chunk(self, value):
        return SubstanceChunker().chunk(value)

    def test_chunk_casts_to_configured_class(self):
        class ChunkEnvelope(dict):
            pass

        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        chunks = SubstanceChunker(**{'class': ChunkEnvelope}).chunk(substance)
        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(chunk, ChunkEnvelope) for chunk in chunks))

    def test_chunk_defaults_to_dict_when_class_not_provided(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        chunks = self.chunk(substance)
        self.assertTrue(chunks)
        self.assertIs(type(chunks[0]), dict)

    def test_primary_identifier_preferred_order_is_configurable_from_constructor(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
                'codes': [
                    {'code': 'WK2XYI10QM', 'codeSystem': 'FDA UNII', 'type': 'PRIMARY'},
                    {'code': '15687-27-1', 'codeSystem': 'CAS', 'type': 'PRIMARY'},
                ],
            }
        )
        sentence = SubstanceChunker(identifiers_order=['CAS', 'FDA UNII'])._summary_primary_identifiers_sentence(substance)
        self.assertIn('CAS: 15687-27-1', sentence)
        self.assertIn('FDA UNII: WK2XYI10QM', sentence)
        self.assertLess(sentence.index('CAS: 15687-27-1'), sentence.index('FDA UNII: WK2XYI10QM'))

    def test_classification_preferred_order_is_configurable_from_constructor(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
                'codes': [
                    {'code': 'A', 'codeSystem': 'WHO-ATC', 'type': 'PRIMARY', '_isClassification': True},
                    {'code': 'V', 'codeSystem': 'WHO-VATC', 'type': 'PRIMARY', '_isClassification': True},
                ],
            }
        )
        sentence = SubstanceChunker(classifications_order=['WHO-VATC', 'WHO-ATC'])._summary_classifications_sentence(substance)
        self.assertIn('WHO-VATC', sentence)
        self.assertIn('WHO-ATC', sentence)
        self.assertLess(sentence.index('WHO-VATC'), sentence.index('WHO-ATC'))

    def test_subelement_document_id_uses_parent_uuid(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'approvalID': 'APP-1',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        chunk = next(chunk for chunk in self.chunk(substance) if chunk['section'] == 'names')
        self.assertEqual(chunk['document_id'], '11111111-1111-1111-1111-111111111111')
        self.assertEqual(chunk['chunk_id'], f'root_names_uuid:{substance.names[0].uuid}')
        self.assertEqual(chunk['section'], 'names')
        self.assertEqual(chunk['metadata']['hierarchy'], ['root', 'names'])
        self.assertEqual(chunk['metadata']['hierarchy_path'], 'root > names')
        self.assertEqual(chunk['metadata']['hierarchy_level'], 2)
        self.assertEqual(chunk['metadata']['json_path'], '$.names[0]')

    def test_substance_to_embedding_chunks_returns_summary_and_subelements(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'approvalID': 'APP-1',
                'names': [
                    {
                        'name': 'Example Concept',
                        'type': 'cn',
                        'languages': ['en'],
                        'preferred': True,
                        'references': ['aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'],
                    }
                ],
                'codes': [
                    {
                        'code': 'ABC-123',
                        'codeSystem': 'CAS',
                        'references': ['bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'],
                    }
                ],
                'properties': [{'name': 'Molecular Weight'}],
                'relationships': [{'type': 'ACTIVE MOIETY', 'relatedSubstance': {'name': 'Related', 'refuuid': '33333333-3333-3333-3333-333333333333'}}],
                'notes': [{'note': 'Important note'}],
                'references': [
                    {
                        'uuid': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
                        'docType': 'SYSTEM',
                        'citation': 'generated',
                    },
                    {
                        'uuid': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
                        'docType': 'MANUAL',
                        'citation': 'registry import',
                    },
                ],
                'modifications': {'agentModifications': [{'agentModificationType': 'CHEMICAL', 'agentSubstance': {'refuuid': '22222222-2222-2222-2222-222222222222'}}]},
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        chunks = self.chunk(substance)
        sections = [chunk['section'] for chunk in chunks]
        self.assertIn('summary', sections)
        self.assertIn('names', sections)
        self.assertIn('codes', sections)
        self.assertIn('properties', sections)
        self.assertIn('relationships', sections)
        self.assertIn('notes', sections)
        self.assertIn('references', sections)
        self.assertIn('agentModifications', sections)
        mod_chunk = next(chunk for chunk in chunks if chunk['section'] == 'agentModifications')
        self.assertEqual(mod_chunk['metadata']['hierarchy'], ['root', 'modifications', 'agentModifications'])
        self.assertEqual(chunks[0]['metadata']['approval_id'], 'APP-1')
        self.assertEqual(chunks[0]['metadata']['name_count'], 1)
        self.assertEqual(chunks[0]['source_url'], 'https://example.test/gsrs')
        self.assertEqual(chunks[0]['metadata']['hierarchy'], ['root'])
        self.assertEqual(chunks[0]['metadata']['json_path'], '$')
        name_chunk = next(chunk for chunk in chunks if chunk['section'] == 'names')
        self.assertEqual(name_chunk['metadata']['json_path'], '$.names[0]')
        self.assertEqual(name_chunk['metadata']['references'], ['SYSTEM: generated'])
        self.assertNotIn('reference_ids', name_chunk['metadata'])
        code_chunk = next(chunk for chunk in chunks if chunk['section'] == 'codes')
        self.assertEqual(code_chunk['metadata']['json_path'], '$.codes[0]')
        self.assertEqual(code_chunk['metadata']['references'], ['MANUAL: registry import'])
        self.assertNotIn('reference_ids', code_chunk['metadata'])
        summary_chunk = next(chunk for chunk in chunks if chunk['section'] == 'summary')
        self.assertIn('Content covers names, codes, properties, relationships, agentModifications, references, and notes.', summary_chunk['text'])

    def test_name_chunk_uses_friendly_name_type_labels(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'approvalID': 'APP-1',
                'names': [{'name': 'Example Brand', 'type': 'bn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        chunk = next(chunk for chunk in self.chunk(substance) if chunk['section'] == 'names')
        self.assertIn('Brand Name', chunk['text'])
        self.assertEqual(chunk['metadata']['name_type'], 'bn')
        self.assertEqual(chunk['metadata']['name_type_label'], 'Brand Name')

    def test_root_summary_names_include_languages_for_display_preferred_and_official(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'approvalID': 'APP-1',
                'names': [
                    {
                        'name': 'Display Name',
                        'type': 'cn',
                        'displayName': True,
                        'languages': ['en'],
                    },
                    {
                        'name': 'Preferred Name',
                        'type': 'cn',
                        'preferred': True,
                        'languages': ['de'],
                    },
                    {
                        'name': 'Official Name',
                        'type': 'of',
                        'languages': ['fr'],
                        'nameOrgs': [{'nameOrg': 'EDQM'}],
                    },
                ],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )

        summary_chunk = next(chunk for chunk in self.chunk(substance) if chunk['section'] == 'summary')
        text = summary_chunk['text']

        self.assertIn('Display Name [en] as the display name', text)
        self.assertIn('Preferred Name [de] as the preferred name', text)
        self.assertIn('Official Name [fr] is registered by EDQM naming organizations as the official name', text)

    def test_classification_code_adds_classification_chunk(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'approvalID': 'APP-1',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        code = Code.model_validate(
            {
                'code': 'ABC-123',
                'codeSystem': 'ATC',
                '_isClassification': True,
                'comments': 'Level 1|Level 2|Level 3',
            }
        )
        code._set_parent(substance, '$.codes[0]')

        chunks = self.chunk(code)

        self.assertEqual(len(chunks), 1)
        class_chunk = chunks[0]
        self.assertEqual(class_chunk['chunk_id'], f'root_codes_uuid:{code.uuid}')
        self.assertIn('classification code ABC-123 in ATC code system: Level 1 > Level 2 > Level 3.', class_chunk['text'])
        self.assertEqual(class_chunk['metadata']['classification_hierarchy'], ['Level 1', 'Level 2', 'Level 3'])
        self.assertEqual(class_chunk['metadata']['hierarchy'], ['root', 'codes'])
        self.assertEqual(class_chunk['metadata']['json_path'], '$.codes[0]')

    def test_code_chunk_matches_adapter_style(self):
        substance = Substance.model_validate(
            {
                'substanceClass': 'concept',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'approvalID': 'APP-1',
                'names': [{'name': 'Example Concept', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
            }
        )
        code = Code.model_validate({'code': 'ABC-123', 'codeSystem': 'CAS'})
        code._set_parent(substance, '$.codes[0]')
        chunk = self.chunk(code)[0]
        self.assertTrue(chunk['text'].startswith('Example Concept'))
        self.assertIn('Identifier code ABC-123 in CAS code system: ABC-123.', chunk['text'])
        self.assertEqual(chunk['document_id'], '11111111-1111-1111-1111-111111111111')
        self.assertEqual(chunk['chunk_id'], f'root_codes_uuid:{code.uuid}')
        self.assertEqual(chunk['source_url'], 'https://example.test/gsrs')
        self.assertEqual(chunk['metadata']['hierarchy'], ['root', 'codes'])
        self.assertEqual(chunk['metadata']['json_path'], '$.codes[0]')

    def test_embedding_root_name_falls_back_to_substance_uuid(self):
        code = Code.model_validate({'code': 'ABC-123', 'codeSystem': 'CAS'})
        parent = Substance.model_construct(
            substanceClass='concept',
            uuid='11111111-1111-1111-1111-111111111111',
            names=[],
            references=[],
            version='1',
        )
        code._set_parent(parent, '$.codes[0]')
        chunk = self.chunk(code)[0]
        self.assertTrue(chunk['text'].startswith('Substance 11111111-1111-1111-1111-111111111111'))
        self.assertIn('Identifier code ABC-123 in CAS code system: ABC-123.', chunk['text'])

    def test_chemical_substance_summary_metadata_includes_structure_attributes(self):
        substance = ChemicalSubstance.model_validate(
            {
                'substanceClass': 'chemical',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Example Chemical', 'type': 'cn', 'languages': ['en']}],
                'references': [
                    {
                        'uuid': 'cccccccc-cccc-cccc-cccc-cccccccccccc',
                        'docType': 'SYSTEM',
                        'citation': 'structure source',
                    }
                ],
                '_self': 'https://example.test/gsrs',
                'version': '1',
                'structure': {
                    'stereochemistry': 'ACHIRAL',
                    'opticalActivity': 'NONE',
                    'atropisomerism': 'No',
                    'formula': 'H2O',
                    'references': ['cccccccc-cccc-cccc-cccc-cccccccccccc'],
                },
                'moieties': [{'stereochemistry': 'ACHIRAL', 'opticalActivity': 'NONE', 'atropisomerism': 'No'}],
            }
        )
        chunks = self.chunk(substance)
        self.assertIn('structure', [chunk['section'] for chunk in chunks])
        summary_chunk = next(chunk for chunk in chunks if chunk['section'] == 'summary')
        self.assertIn('molecular formula H2O', summary_chunk['text'])
        self.assertEqual(summary_chunk['metadata']['formula'], 'H2O')
        self.assertEqual(summary_chunk['metadata']['stereochemistry'], 'ACHIRAL')
        self.assertEqual(summary_chunk['metadata']['optical_activity'], 'NONE')
        self.assertEqual(summary_chunk['metadata']['atropisomerism'], 'No')
        self.assertEqual(summary_chunk['metadata']['structure_references'], ['SYSTEM: structure source'])
        self.assertEqual(summary_chunk['metadata']['moieties'], [])

    def test_chemical_substance_uses_enum_values_in_summary_metadata(self):
        substance = ChemicalSubstance.model_validate(
            {
                'substanceClass': 'chemical',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Enum Chemical', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
                'structure': {
                    'stereochemistry': 'ACHIRAL',
                    'opticalActivity': '( + / - )',
                    'atropisomerism': 'No',
                    'formula': 'C2H6O',
                },
                'moieties': [{'stereochemistry': 'ACHIRAL', 'opticalActivity': 'NONE', 'atropisomerism': 'No'}],
            }
        )
        summary_chunk = next(chunk for chunk in self.chunk(substance) if chunk['section'] == 'summary')
        self.assertEqual(summary_chunk['metadata']['optical_activity'], '( + / - )')
        self.assertEqual(summary_chunk['metadata']['atropisomerism'], 'No')

    def test_mixture_substance_summary_metadata_includes_mixture_attributes(self):
        substance = MixtureSubstance.model_validate(
            {
                'substanceClass': 'mixture',
                'uuid': '11111111-1111-1111-1111-111111111111',
                'names': [{'name': 'Example Mixture', 'type': 'cn', 'languages': ['en']}],
                'references': [{'docType': 'SYSTEM'}],
                '_self': 'https://example.test/gsrs',
                'version': '1',
                'mixture': {
                    'components': [
                        {
                            'substance': {
                                'refuuid': '22222222-2222-2222-2222-222222222222',
                                'refPname': 'Water',
                            }
                        }
                    ],
                    'parentSubstance': {
                        'refuuid': '33333333-3333-3333-3333-333333333333',
                        'refPname': 'Parent Mix',
                        'approvalID': 'PARENT-1',
                    },
                },
            }
        )
        chunks = self.chunk(substance)
        self.assertIn('mixture', [chunk['section'] for chunk in chunks])
        summary_chunk = next(chunk for chunk in chunks if chunk['section'] == 'summary')
        self.assertIn('Mixture with 1 component', summary_chunk['text'])
        self.assertIn('parent substance Parent Mix', summary_chunk['text'])
        self.assertEqual(summary_chunk['metadata']['mixture_component_count'], 1)
        self.assertEqual(summary_chunk['metadata']['mixture_parent_substance'], 'Parent Mix')
        self.assertEqual(summary_chunk['metadata']['mixture_parent_substance_id'], '33333333-3333-3333-3333-333333333333')

    def test_sample_substance_summary_is_rich(self):
        payload = {
            'substanceClass': 'chemical',
            'uuid': '0103a288-6eb6-4ced-b13a-849cd7edf028',
            'status': 'pending',
            'definitionType': 'PRIMARY',
            'definitionLevel': 'COMPLETE',
            'names': [
                {
                    'name': 'IBUPROFEN',
                    'type': 'of',
                    'displayName': True,
                    'languages': ['en'],
                }
            ],
            'codes': [
                {'code': 'WK2XYI10QM', 'codeSystem': 'FDA UNII', 'type': 'PRIMARY'},
                {'code': '100000090365', 'codeSystem': 'SMS_ID', 'type': 'PRIMARY'},
                {'code': 'SUB08098MIG', 'codeSystem': 'EVMPD', 'type': 'PRIMARY'},
                {'code': '15687-27-1', 'codeSystem': 'CAS', 'type': 'PRIMARY'},
                {'code': 'DB01050', 'codeSystem': 'DRUG BANK', 'type': 'PRIMARY'},
                {'code': '5640', 'codeSystem': 'RXCUI', 'type': 'PRIMARY'},
                {'code': 'CHEMBL521', 'codeSystem': 'ChEMBL', 'type': 'PRIMARY'},
                {'code': '3672', 'codeSystem': 'PUBCHEM', 'type': 'PRIMARY'},
                {'code': 'X', 'codeSystem': 'WHO-ATC', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'WHO-VATC', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'NCI_THESAURUS', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'EMA ASSESSMENT REPORTS', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'WHO-ESSENTIAL MEDICINES LIST', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'NDF-RT', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'LIVERTOX', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'FDA ORPHAN DRUG', 'type': 'PRIMARY', '_isClassification': True},
                {'code': 'X', 'codeSystem': 'EU-Orphan Drug', 'type': 'PRIMARY', '_isClassification': True},
            ],
            'properties': [{'name': 'Molecular Weight'}],
            'relationships': [{'type': 'ACTIVE MOIETY', 'relatedSubstance': {'refuuid': '33333333-3333-3333-3333-333333333333', 'name': 'Related'}}],
            'notes': [{'note': 'Important note'}],
            'references': [{'docType': 'SYSTEM'}],
            '_self': 'https://example.test/substances/1',
            'version': '1',
            'structure': {
                'stereochemistry': 'RACEMIC',
                'opticalActivity': 'NONE',
                'atropisomerism': 'No',
                'formula': 'C13H18O2',
                'mwt': 206.2813,
                'smiles': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
                '_inchiKey': 'HEFNNWSXXWATRW-UHFFFAOYSA-N',
            },
            'moieties': [{'stereochemistry': 'ACHIRAL', 'opticalActivity': 'NONE', 'atropisomerism': 'No'}],
        }
        substance = ChemicalSubstance.model_validate(payload)
        summary_chunk = next(chunk for chunk in self.chunk(substance) if chunk['section'] == 'summary')
        text = summary_chunk['text']
        self.assertTrue(text.startswith('Ibuprofen is a for publicly accessible chemical substance.'))
        self.assertIn('Definition type PRIMARY and definition level COMPLETE.', text)
        self.assertIn('Structure is a public chemical structure, molecular formula C13H18O2, molecular weight 206.2813, racemic, with SMILES CC(C)Cc1ccc(cc1)C(C)C(=O)O and InChIKey HEFNNWSXXWATRW-UHFFFAOYSA-N.', text)
        self.assertIn('including IBUPROFEN [en] as the display name', text)
        self.assertNotIn('brand names such as', text)
        self.assertIn('FDA UNII: WK2XYI10QM', text)
        self.assertIn('SMS_ID: 100000090365', text)
        self.assertIn('EVMPD: SUB08098MIG', text)
        self.assertIn('CAS: 15687-27-1', text)
        self.assertIn('DRUG BANK: DB01050', text)
        self.assertIn('RXCUI: 5640', text)
        self.assertIn('ChEMBL: CHEMBL521', text)
        self.assertIn('PUBCHEM: 3672', text)
        for classification in ['WHO-ATC', 'WHO-VATC', 'NCI_THESAURUS', 'EMA ASSESSMENT REPORTS', 'WHO-ESSENTIAL MEDICINES LIST', 'NDF-RT', 'LIVERTOX', 'FDA ORPHAN DRUG', 'EU-Orphan Drug']:
            self.assertIn(classification, text)
        self.assertIn('Content covers names, codes, moieties, properties, relationships, references, and notes.', text)


if __name__ == '__main__':
    unittest.main()


