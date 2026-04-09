import unittest

import _bootstrap  # noqa: F401

from gsrs.model import Substance
from gsrs.services.ai import ChunkerConfig, SubstanceChunker
from gsrs.services.ai.chunk_models import REQUIRED_METADATA_KEYS


def base_payload(
    substance_class: str,
    *,
    uuid: str,
    name: str,
    approval_id: str,
) -> dict:
    return {
        'substanceClass': substance_class,
        'uuid': uuid,
        'approvalID': approval_id,
        '_name': name,
        'names': [
            {
                'name': name,
                'type': 'cn',
                'displayName': True,
                'languages': ['en'],
            }
        ],
        'references': [
            {
                'uuid': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
                'docType': 'SYSTEM',
                'citation': 'fixture source',
            },
            {
                'uuid': 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
                'docType': 'LITERATURE',
                'citation': 'literature support',
            },
        ],
        'version': '1',
        'status': 'active',
        'definitionType': 'PRIMARY',
        'definitionLevel': 'COMPLETE',
    }


def by_section(chunks):
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk.section, []).append(chunk)
    return grouped


class EmbeddingChunkTests(unittest.TestCase):
    def assert_metadata_contract(self, chunks):
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(REQUIRED_METADATA_KEYS.issubset(chunk.metadata))
            self.assertEqual(chunk.metadata['document_id'], chunk.document_id)
            self.assertEqual(chunk.metadata['section'], chunk.section)

    def test_chunk_casts_to_configured_class(self):
        class ChunkEnvelope(dict):
            pass

        payload = base_payload(
            'concept',
            uuid='10101010-1010-1010-1010-101010101010',
            name='Castable Concept',
            approval_id='CAST-1',
        )

        chunks = SubstanceChunker(class_=ChunkEnvelope).chunk(Substance.model_validate(payload))

        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(chunk, ChunkEnvelope) for chunk in chunks))
        self.assertEqual(chunks[0]['section'], 'overview')

    def test_chunk_accepts_class_keyword_alias(self):
        payload = base_payload(
            'concept',
            uuid='14141414-1414-1414-1414-141414141414',
            name='Alias Concept',
            approval_id='ALIAS-1',
        )

        chunks = SubstanceChunker(**{'class': dict}).chunk(Substance.model_validate(payload))

        self.assertTrue(chunks)
        self.assertIs(type(chunks[0]), dict)

    def test_constructor_accepts_positional_config_with_class_first_signature(self):
        payload = base_payload(
            'concept',
            uuid='15151515-1515-1515-1515-151515151515',
            name='Configured Concept',
            approval_id='CONFIG-1',
        )

        chunker = SubstanceChunker(ChunkerConfig(include_admin_validation_notes=True))
        chunks = chunker.chunk(Substance.model_validate(payload))

        self.assertTrue(chunks)
        self.assertEqual(chunker.config.include_admin_validation_notes, True)

    def test_chunk_json_dispatches_from_substance_root(self):
        payload = base_payload(
            'concept',
            uuid='11111111-1111-1111-1111-111111111111',
            name='Sparse Concept',
            approval_id='CONCEPT-1',
        )
        chunks = SubstanceChunker().chunk_json(payload)
        self.assert_metadata_contract(chunks)
        self.assertIn('definition_unavailable_note', by_section(chunks))

    def test_core_names_include_display_preferred_official_and_languages(self):
        payload = base_payload(
            'concept',
            uuid='16161616-1616-1616-1616-161616161616',
            name='Display Name',
            approval_id='NAMES-1',
        )
        payload['names'] = [
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
            },
            {
                'name': 'Extra Alias',
                'type': 'sys',
                'languages': ['es'],
            },
        ]

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)
        core_names = sections['core_names'][0]
        sys_batches = [chunk for chunk in sections['name_batch'] if chunk.metadata.get('name_type') == 'sys']

        self.assertIn('Display Name [en] (display, cn)', core_names.text)
        self.assertIn('Preferred Name [de] (preferred, cn)', core_names.text)
        self.assertIn('Official Name [fr] (of)', core_names.text)
        self.assertNotIn('Extra Alias [es] (sys)', core_names.text)
        self.assertIn('Display Name', core_names.metadata['exact_match_terms'])
        self.assertIn('Preferred Name', core_names.metadata['exact_match_terms'])
        self.assertIn('Official Name', core_names.metadata['exact_match_terms'])
        self.assertNotIn('Extra Alias', core_names.metadata['exact_match_terms'])
        self.assertEqual(len(sys_batches), 1)
        self.assertIn('Extra Alias', sys_batches[0].text)
        self.assertNotIn('Extra Alias [es]', sys_batches[0].text)

    def test_name_batches_group_by_type_and_language_and_enrich_official_names_with_orgs(self):
        payload = base_payload(
            'concept',
            uuid='17171717-1717-1717-1717-171717171717',
            name='Batch Concept',
            approval_id='BATCH-1',
        )
        payload['names'] = [
            {
                'name': 'Official English One',
                'type': 'of',
                'languages': ['en'],
                'nameOrgs': [{'nameOrg': 'USAN'}],
            },
            {
                'name': 'Official English Two',
                'type': 'of',
                'languages': ['en'],
                'nameOrgs': [{'nameOrg': 'INN'}],
            },
            {
                'name': 'Official French',
                'type': 'of',
                'languages': ['fr'],
                'nameOrgs': [{'nameOrg': 'EDQM'}],
            },
            {
                'name': 'Systematic English',
                'type': 'sys',
                'languages': ['en', 'de'],
            },
        ]

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        batches = by_section(chunks)['name_batch']
        official_en = next(chunk for chunk in batches if chunk.metadata.get('name_type') == 'of' and chunk.metadata.get('languages') == ['en'])
        official_fr = next(chunk for chunk in batches if chunk.metadata.get('name_type') == 'of' and chunk.metadata.get('languages') == ['fr'])
        sys_en = next(chunk for chunk in batches if chunk.metadata.get('name_type') == 'sys' and chunk.metadata.get('languages') == ['en'])
        sys_de = next(chunk for chunk in batches if chunk.metadata.get('name_type') == 'sys' and chunk.metadata.get('languages') == ['de'])

        self.assertTrue(official_en.text.startswith('English official names:'))
        self.assertIn('Official English One (namingOrg: USAN)', official_en.text)
        self.assertIn('Official English Two (namingOrg: INN)', official_en.text)
        self.assertIn('Details: name organizations USAN, INN.', official_en.text)
        self.assertEqual(official_en.metadata['name_orgs'], ['USAN', 'INN'])
        self.assertTrue(official_fr.text.startswith('French official names:'))
        self.assertIn('Official French (namingOrg: EDQM)', official_fr.text)
        self.assertEqual(official_fr.metadata['languages'], ['fr'])
        self.assertTrue(sys_en.text.startswith('English systematic names:'))
        self.assertIn('Systematic English', sys_en.text)
        self.assertEqual(sys_en.metadata['languages'], ['en'])
        self.assertTrue(sys_de.text.startswith('German systematic names:'))
        self.assertIn('Systematic English', sys_de.text)
        self.assertEqual(sys_de.metadata['languages'], ['de'])
        self.assertNotIn('[en]', official_en.text)
        self.assertNotIn('[fr]', official_fr.text)
        self.assertNotIn('[en]', sys_en.text)
        self.assertNotIn('[de]', sys_de.text)

    def test_chemical_chunking_supports_identifiers_classifications_and_relationship_groups(self):
        payload = base_payload(
            'chemical',
            uuid='22222222-2222-2222-2222-222222222222',
            name='IBUPROFEN',
            approval_id='WK2XYI10QM',
        )
        payload['names'].append(
            {
                'name': '2-(4-isobutylphenyl)propanoic acid',
                'type': 'of',
                'languages': ['en'],
                'preferred': True,
            }
        )
        payload['codes'] = [
            {'code': 'WK2XYI10QM', 'codeSystem': 'FDA UNII', 'type': 'PRIMARY'},
            {'code': '15687-27-1', 'codeSystem': 'CAS', 'type': 'PRIMARY'},
            {'code': 'M01AE01', 'codeSystem': 'WHO-ATC', 'type': 'PRIMARY', '_isClassification': True, 'comments': 'M|M01|M01AE|M01AE01'},
        ]
        payload['structure'] = {
            'stereochemistry': 'RACEMIC',
            'opticalActivity': 'NONE',
            'atropisomerism': 'No',
            'formula': 'C13H18O2',
            'mwt': 206.28,
            'smiles': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
            '_inchiKey': 'HEFNNWSXXWATRW-UHFFFAOYSA-N',
            'references': ['aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'],
        }
        payload['moieties'] = [
            {
                'stereochemistry': 'ACHIRAL',
                'opticalActivity': 'NONE',
                'atropisomerism': 'No',
                'formula': 'C13H18O2',
            }
        ]
        payload['properties'] = [
            {'name': 'Half life', 'propertyType': 'PK', 'value': {'average': 2.0, 'units': 'h'}},
            {'name': 'Cmax', 'propertyType': 'PK', 'value': {'average': 25.0, 'units': 'ug/mL'}},
        ]
        payload['relationships'] = [
            {'type': 'SALT/SOLVATE', 'relatedSubstance': {'refuuid': '33333333-3333-3333-3333-333333333333', 'refPname': 'Ibuprofen Lysine'}},
            {'type': 'METABOLITE', 'relatedSubstance': {'refuuid': '44444444-4444-4444-4444-444444444444', 'refPname': 'Hydroxyibuprofen'}},
            {'type': 'ACTIVE MOIETY', 'relatedSubstance': {'refuuid': '55555555-5555-5555-5555-555555555555', 'refPname': 'Ibuprofen'}},
        ]
        payload['notes'] = [{'note': 'Public monograph note for ibuprofen.'}]

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('chemical_structure', sections)
        self.assertIn('moieties', sections)
        self.assertIn('pk_summary', sections)
        self.assertIn('identifier', sections)
        self.assertIn('classification', sections)
        self.assertIn('relationship_summary', sections)
        self.assertIn('atomic_relationship', sections)
        self.assertIn('substantive_note', sections)
        relationship_groups = {chunk.metadata['group_type'] for chunk in sections['relationship_summary']}
        self.assertIn('salts', relationship_groups)
        self.assertIn('metabolites', relationship_groups)
        self.assertIn('active_moiety', relationship_groups)
        overview = sections['overview'][0]
        self.assertIn('IBUPROFEN', overview.metadata['exact_match_terms'])
        self.assertIn('WK2XYI10QM', overview.metadata['exact_match_terms'])

    def test_specified_substance_g1_emits_constituent_chunks(self):
        payload = base_payload(
            'specifiedSubstanceG1',
            uuid='33333333-3333-3333-3333-333333333333',
            name='DPBS',
            approval_id='DPBS-1',
        )
        payload['specifiedSubstance'] = {
            'constituents': [
                {'role': 'buffer', 'amount': {'average': 137.9, 'units': 'mM'}, 'substance': {'refuuid': '90000000-0000-0000-0000-000000000001', 'refPname': 'Sodium chloride'}},
                {'role': 'buffer', 'amount': {'average': 2.7, 'units': 'mM'}, 'substance': {'refuuid': '90000000-0000-0000-0000-000000000002', 'refPname': 'Potassium chloride'}},
                {'role': 'buffer', 'amount': {'average': 10.0, 'units': 'mM'}, 'substance': {'refuuid': '90000000-0000-0000-0000-000000000003', 'refPname': 'Sodium phosphate'}},
                {'role': 'buffer', 'amount': {'average': 1.8, 'units': 'mM'}, 'substance': {'refuuid': '90000000-0000-0000-0000-000000000004', 'refPname': 'Potassium phosphate'}},
            ]
        }

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('specified_substance_definition', sections)
        self.assertIn('constituents_summary', sections)
        self.assertEqual(len(sections['constituent_atomic']), 4)
        self.assertIn('Specified substance with 4 constituents.', sections['overview'][0].text)

    def test_mixture_emits_components_and_parent_summary(self):
        payload = base_payload(
            'mixture',
            uuid='44444444-4444-4444-4444-444444444444',
            name='Diapon',
            approval_id='DIAPON-1',
        )
        payload['mixture'] = {
            'components': [
                {'substance': {'refuuid': '90000000-0000-0000-0000-000000000005', 'refPname': 'Water'}},
                {'substance': {'refuuid': '90000000-0000-0000-0000-000000000006', 'refPname': 'Glycerol'}},
            ],
            'parentSubstance': {'refuuid': '90000000-0000-0000-0000-000000000007', 'refPname': 'Diapon parent', 'approvalID': 'PARENT-1'},
        }

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('mixture_definition', sections)
        self.assertIn('components_summary', sections)
        self.assertEqual(len(sections['component_atomic']), 2)
        self.assertIn('parent_substance_summary', sections)

    def test_nucleic_acid_supports_sequences_features_linkages_sugars_and_modifications(self):
        payload = base_payload(
            'nucleicAcid',
            uuid='55555555-5555-5555-5555-555555555555',
            name='Tenumomeran',
            approval_id='TENUM-1',
        )
        payload['nucleicAcid'] = {
            'nucleicAcidType': 'RNA',
            'nucleicAcidSubType': ['mRNA'],
            'sequenceOrigin': 'synthetic',
            'sequenceType': 'single stranded',
            'subunits': [
                {
                    'subunitIndex': 1,
                    'sequence': 'AUGGCUAUGGCUAUGGCUAUGGCUAUGGCUAUGGCUAUGGCUAUGGCUAUGGCUAUGGCU',
                }
            ],
            'linkages': [{'linkage': 'phosphorothioate', 'sitesShorthand': '1_1-1_5'}],
            'sugars': [{'sugar': '2-O-methyl-ribose', 'sitesShorthand': '1_2-1_10'}],
            'modifications': {
                'structuralModifications': [
                    {
                        'structuralModificationType': 'AMINO_ACID_SUBSTITUTION',
                        'residueModified': 'A',
                        'sitesShorthand': '1_6-1_8',
                    }
                ]
            },
        }
        payload['properties'] = [
            {
                'name': 'Feature: cap',
                'propertyType': 'feature',
                'parameters': [{'name': 'site-range', 'value': {'nonNumericValue': '1..5'}}],
            },
            {
                'name': 'Feature: guide',
                'propertyType': 'feature',
                'parameters': [{'name': 'site-range', 'value': {'nonNumericValue': '10..20'}}],
            },
        ]

        chunker = SubstanceChunker(ChunkerConfig(emit_sequence_segments=True, max_sequence_segment_len=20))
        chunks = chunker.chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('nucleic_acid_definition', sections)
        self.assertIn('sequence_summary', sections)
        self.assertIn('sequence_segments', sections)
        self.assertIn('feature_summary', sections)
        self.assertEqual(len(sections['feature_atomic']), 2)
        self.assertIn('linkages_summary', sections)
        self.assertIn('sugars_summary', sections)
        self.assertIn('na_modifications_summary', sections)
        self.assertIn('1..5', sections['feature_atomic'][0].metadata['exact_match_terms'])

    def test_protein_supports_subunits_glycosylation_disulfides_and_modifications(self):
        payload = base_payload(
            'protein',
            uuid='66666666-6666-6666-6666-666666666666',
            name='Lumivatamig',
            approval_id='LUMI-1',
        )
        payload['protein'] = {
            'proteinType': 'protein',
            'proteinSubType': ['monoclonal antibody'],
            'sequenceOrigin': 'synthetic',
            'sequenceType': 'complete',
            'subunits': [
                {'subunitIndex': 1, 'sequence': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLE', 'length': 54},
                {'subunitIndex': 2, 'sequence': 'DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLI', 'length': 57},
            ],
            'glycosylation': {
                'glycosylationType': 'N-linked',
                'NGlycosylationSites': [{'subunitIndex': 1, 'residueIndex': 40}],
            },
            'disulfideLinks': [{'sitesShorthand': '1_23-2_88'}],
            'modifications': {
                'structuralModifications': [
                    {
                        'structuralModificationType': 'AMINO_ACID_SUBSTITUTION',
                        'residueModified': 'C',
                        'sitesShorthand': '1_10',
                    }
                ]
            },
        }

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('protein_definition', sections)
        self.assertIn('subunits_summary', sections)
        self.assertEqual(len(sections['subunit_atomic']), 2)
        self.assertIn('glycosylation_summary', sections)
        self.assertIn('disulfide_summary', sections)
        self.assertIn('protein_modifications_summary', sections)

    def test_polymer_emits_classification_monomers_structure_and_skips_admin_notes_by_default(self):
        payload = base_payload(
            'polymer',
            uuid='77777777-7777-7777-7777-777777777777',
            name='Incomplete Polymer',
            approval_id='POLY-1',
        )
        payload['status'] = 'failed'
        payload['polymer'] = {
            'classification': {
                'polymerClass': 'polyamide',
                'polymerGeometry': 'linear',
                'polymerSubclass': ['copolymer'],
                'sourceType': 'synthetic',
            },
            'monomers': [
                {'type': 'SRU', 'amount': {'average': 0.6, 'units': 'ratio'}, 'monomerSubstance': {'refuuid': '90000000-0000-0000-0000-000000000008', 'refPname': 'Monomer A'}},
                {'type': 'SRU', 'amount': {'average': 0.4, 'units': 'ratio'}, 'monomerSubstance': {'refuuid': '90000000-0000-0000-0000-000000000009', 'refPname': 'Monomer B'}},
            ],
            'displayStructure': {
                'stereochemistry': 'ACHIRAL',
                'opticalActivity': 'NONE',
                'atropisomerism': 'No',
                'formula': 'C10H10N2',
            },
            'idealizedStructure': {
                'stereochemistry': 'ACHIRAL',
                'opticalActivity': 'NONE',
                'atropisomerism': 'No',
                'formula': 'C10H10N2',
            },
            'structuralUnits': [{'label': 'A', 'structure': '*CC*', 'type': 'SRU'}],
        }
        payload['notes'] = [
            {'note': 'Public note describing an incomplete polymer definition.'},
            {'note': '[Validation] failed because monomer count is incomplete', 'access': ['admin']},
        ]

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('polymer_definition', sections)
        self.assertIn('polymer_classification_summary', sections)
        self.assertIn('monomers_summary', sections)
        self.assertEqual(len(sections['monomer_atomic']), 2)
        self.assertIn('polymer_structure_summary', sections)
        self.assertIn('substantive_note', sections)
        self.assertNotIn('admin_validation_note', sections)

    def test_admin_validation_notes_can_be_included(self):
        payload = base_payload(
            'concept',
            uuid='88888888-8888-8888-8888-888888888888',
            name='Concept With Validation',
            approval_id='CONCEPT-2',
        )
        payload['notes'] = [{'note': '[Validation] concept is incomplete', 'access': ['admin']}]

        chunker = SubstanceChunker(ChunkerConfig(include_admin_validation_notes=True))
        chunks = chunker.chunk(Substance.model_validate(payload))

        self.assertIn('admin_validation_note', by_section(chunks))

    def test_structurally_diverse_emits_source_taxonomy_and_part_sections(self):
        payload = base_payload(
            'structurallyDiverse',
            uuid='99999999-9999-9999-9999-999999999999',
            name='Influenza Vaccine Material',
            approval_id='FLU-1',
        )
        payload['structurallyDiverse'] = {
            'sourceMaterialClass': 'ORGANISM',
            'sourceMaterialType': 'virus',
            'sourceMaterialState': 'inactivated',
            'organismFamily': 'Orthomyxoviridae',
            'organismGenus': 'Influenzavirus A',
            'organismSpecies': 'A/Puerto Rico/8/1934',
            'part': ['whole organism'],
            'partLocation': 'virion',
        }

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('source_material_summary', sections)
        self.assertIn('taxonomy_summary', sections)
        self.assertIn('organism_part_summary', sections)

    def test_concept_emits_definition_unavailable_note(self):
        payload = base_payload(
            'concept',
            uuid='12121212-1212-1212-1212-121212121212',
            name='Sparse Concept Record',
            approval_id='CONCEPT-3',
        )

        chunks = SubstanceChunker().chunk(Substance.model_validate(payload))
        sections = by_section(chunks)

        self.assert_metadata_contract(chunks)
        self.assertIn('definition_unavailable_note', sections)
        self.assertIn('Definition not currently available.', sections['overview'][0].text)

    def test_chunking_is_deterministic_for_same_payload(self):
        payload = base_payload(
            'concept',
            uuid='13131313-1313-1313-1313-131313131313',
            name='Deterministic Concept',
            approval_id='CONCEPT-4',
        )
        substance = Substance.model_validate(payload)
        chunker = SubstanceChunker()

        first = chunker.chunk(substance)
        second = chunker.chunk(substance)

        self.assertEqual(
            [(chunk.chunk_id, chunk.section, chunk.text) for chunk in first],
            [(chunk.chunk_id, chunk.section, chunk.text) for chunk in second],
        )


if __name__ == '__main__':
    unittest.main()
