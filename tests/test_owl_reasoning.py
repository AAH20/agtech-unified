"""Tests for OWL reasoning and inference on ontology."""

from src.digital_twin.ontology import AGROVOCOntology, Concept


class TestOWLReasoning:
    """Test OWL RL / RDFS reasoning capabilities."""

    def test_subsumption_inference(self):
        """Infer that wheat is a crop via broader hierarchy."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="crop"))
        ont.add_concept(Concept(id="crop", pref_label="Crop"))

        # After reasoning, wheat should be inferred as a crop
        inferred = ont.infer_subsumption("wheat")
        assert "crop" in inferred

    def test_transitive_broader(self):
        """Transitive broader relationships are inferred."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="cereal"))
        ont.add_concept(Concept(id="cereal", pref_label="Cereal", broader="crop"))
        ont.add_concept(Concept(id="crop", pref_label="Crop"))

        all_broader = ont.get_all_broader("wheat")
        assert "cereal" in all_broader
        assert "crop" in all_broader

    def test_transitive_narrower(self):
        """Transitive narrower relationships are inferred."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="crop", pref_label="Crop"))
        ont.add_concept(Concept(id="cereal", pref_label="Cereal", broader="crop"))
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="cereal"))

        all_narrower = ont.get_all_narrower("crop")
        assert "cereal" in all_narrower
        assert "wheat" in all_narrower

    def test_equivalence_inference(self):
        """Equivalent concepts are inferred."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_concept(Concept(id="triticum", pref_label="Triticum"))
        ont.add_equivalence("wheat", "triticum")

        equivalents = ont.get_equivalent_concepts("wheat")
        assert "triticum" in equivalents

    def test_property_chain(self):
        """Property chain inference."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="field1", pref_label="Field 1"))
        ont.add_concept(Concept(id="wheat", pref_label="Wheat"))
        ont.add_concept(Concept(id="flour", pref_label="Flour"))
        # field1 grows wheat, wheat produces flour
        # Infer: field1 indirectly_produces flour
        ont.add_property_chain("grows", "produces", "indirectly_produces")
        inferred = ont.infer_property_chain("field1", "grows", "produces")
        assert inferred is not None

    def test_class_satisfiability(self):
        """Check if a class is satisfiable."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="crop"))
        ont.add_concept(Concept(id="crop", pref_label="Crop"))

        # wheat should be satisfiable (has a valid hierarchy)
        assert ont.is_satisfiable("wheat") is True

    def test_inconsistent_concept(self):
        """Detect inconsistent concept definitions."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="crop"))
        ont.add_concept(Concept(id="crop", pref_label="Crop", broader="wheat"))

        # Circular broader relationship is inconsistent
        assert ont.is_consistent() is False

    def test_infer_instance_types(self):
        """Infer types for instances."""
        ont = AGROVOCOntology()
        ont.add_concept(Concept(id="wheat", pref_label="Wheat", broader="crop"))
        ont.add_concept(Concept(id="crop", pref_label="Crop"))

        # An instance of wheat is also an instance of crop
        types = ont.infer_types("wheat")
        assert "crop" in types
