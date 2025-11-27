from rdflib import RDF, OWL, RDFS
def semantic_extraction(g):
    # On recupere les classes
    classes = set(g.subjects(RDF.type, OWL.Class))


    # On recupere les proprietes et relations
    object_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    datatype_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))

    # On recupere les instances
    all_subjects = set(g.subjects())
    instances = all_subjects - classes - object_props - datatype_props

    #  4. Hiérarchies de classes
    hierarchies = [(s, o) for s, o in g.subject_objects(RDFS.subClassOf)]

    #  5. Domaines et ranges
    domaines = [(p, d) for p, d in g.subject_objects(RDFS.domain)]
    ranges = [(p, r) for p, r in g.subject_objects(RDFS.range)]

    #  6. Équivalences (classes et propriétés)
    equivalent_classes = [(s, o) for s, o in g.subject_objects(OWL.equivalentClass)]
    equivalent_props = [(s, o) for s, o in g.subject_objects(OWL.equivalentProperty)]

    elements = {
        "classes": list(classes),
        "object_properties": list(object_props),
        "datatype_properties": list(datatype_props),
        "instances": list(instances),
        "hierarchies": hierarchies,
        "domaines": domaines,
        "ranges": ranges,
        "equivalent_classes": equivalent_classes,
        "equivalent_properties": equivalent_props,
    }

    return elements