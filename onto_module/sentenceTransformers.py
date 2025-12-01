import asyncio
from typing import Optional

import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from rdflib import URIRef, Namespace, Graph, RDF, XSD, Literal, OWL, RDFS

from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from onto_module.groq import *

model = SentenceTransformer('all-MiniLM-L12-v2')   # all-mpnet-base-v2

class Alignment(BaseModel):
    entityA: str
    entityB: str
    relation: str = "="  # =, <, >
    score: float = 1.0
    entityType: str = "Class" # Class, Property, Instance
    method: str = "SimulatedMatcher"

ALIGN = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#")
MAP = Namespace("http://www.w3.org/2004/0OWL/alignment#")

def embedding(elements: dict) -> dict:
    elements_vectorises = {}

    # 1. Vectorisation des URIs (Classes, Propriétés, Instances)
    uris = set()
    for key in ["classes", "object_properties", "datatype_properties", "instances"]:
        uris.update(elements.get(key, []))

    string_uris = [str(uri) for uri in uris if isinstance(uri, URIRef)]

    if string_uris:
        vectorisations = model.encode(string_uris, convert_to_tensor=True)
        mapping_uri = {uri: embedding.cpu().numpy() for uri, embedding in zip(string_uris, vectorisations)}

        # Stocker les embeddings sous forme (uri, vecteur)
        for key in ["classes", "object_properties", "datatype_properties", "instances"]:
            elements_vectorises[key] = [
                (uri, mapping_uri.get(str(uri))) for uri in elements.get(key, [])
            ]
    else:
        print("Aucune URI trouvée pour la vectorisation.")

    # 2. Vectorisation des relations
    for key in ["hierarchies", "domaines", "ranges", "equivalent_classes", "equivalent_properties"]:
        relations_vectorisees = []
        for s, o in elements.get(key, []):
            s_emb = mapping_uri.get(str(s), np.zeros(model.get_sentence_embedding_dimension()))
            o_emb = mapping_uri.get(str(o), np.zeros(model.get_sentence_embedding_dimension()))
            relation_vector = np.concatenate((s_emb, o_emb))
            relations_vectorisees.append(((s, o), relation_vector))
        elements_vectorises[key] = relations_vectorisees

    return elements_vectorises


async def comparaison_hybride_batch(embA: dict, embB: dict, seuil=0.90, taille_lot=50):
    """
    1. Calcule les similarités cosinus entre embA et embB.
    2. Conserve les paires dont le score >= seuil.
    3. Envoie ces paires au LLM PAR LOTS ET EN PARALLÈLE pour validation détaillée.
    4. Retourne les alignements validés avec Type d'Entité et Relation (6 éléments).
    """

    candidats = []  # (uriA, uriB, score)
    types_elements = ["classes", "object_properties", "datatype_properties", "instances"]

    # --- 1️⃣ Génération des paires par embeddings (SYNCHRONE) ---
    for key in types_elements:
        elements_A = embA.get(key, [])
        elements_B = embB.get(key, [])

        # Filtrage des embeddings None
        vecteurs_A = [(uri, vec) for uri, vec in elements_A if vec is not None]
        vecteurs_B = [(uri, vec) for uri, vec in elements_B if vec is not None]

        if not vecteurs_A or not vecteurs_B:
            continue

        uris_A, vecs_A = zip(*vecteurs_A)
        uris_B, vecs_B = zip(*vecteurs_B)

        sim_matrix = cosine_similarity(list(vecs_A), list(vecs_B))

        for i in range(len(uris_A)):
            for j in range(len(uris_B)):
                score = float(sim_matrix[i][j])
                if score >= seuil:
                    # Stocke le URI A, URI B, et le score
                    candidats.append((uris_A[i], uris_B[j], score))

    print(f"🔍 {len(candidats)} paires candidates (score >= {seuil}) avant validation LLM.")

    # --- 2️⃣ Préparation pour la validation LLM ASYNCHRONE ---

    # Extraire seulement les paires (uriA, uriB) pour les appels LLM
    paires_llm = [(c[0], c[1]) for c in candidats]

    # Découpage des paires en lots
    lots = [
        paires_llm[i:i + taille_lot]
        for i in range(0, len(paires_llm), taille_lot)
    ]

    # Création et exécution des tâches ASYNCHRONES
    print(f"Lancement de {len(lots)} appels Groq en parallèle...")

    taches_llm = [appeler_llm_par_lot(lot) for lot in lots]
    resultats_lots = await asyncio.gather(*taches_llm)

    # --- 4️⃣ Consolidation et Filtrage des résultats (CORRIGÉ) ---

    # Consolider les verdicts des lots en un seul dictionnaire (paire: statut détaillé)
    verdicts_consolides: VerdictsDict = {}
    for verdicts in resultats_lots:
        verdicts_consolides.update(verdicts)

    alignements = []  # Contiendra 6 éléments par tuple

    # Utiliser les verdicts pour filtrer et enrichir les candidats
    for uriA, uriB, score in [(c[0], c[1], c[2]) for c in candidats]:

        # 1. Récupération du verdict détaillé (qui est désormais un Dict, et non un booléen)
        verdict_detaille = verdicts_consolides.get((uriA, uriB))

        # 2. Vérification si le LLM a validé l'alignement ("OUI")
        if verdict_detaille and verdict_detaille.get("statut") == "OUI":
            # Extraction des informations supplémentaires, avec valeurs par défaut
            type_entite = verdict_detaille.get("type", "Inconnu")
            relation = verdict_detaille.get("relation", "↔")

            # AJOUT DES 6 ÉLÉMENTS AU TUPLE (uriA, uriB, score, méthode, type, relation)
            alignements.append((
                uriA,
                uriB,
                score,
                "hybride-LLM",
                type_entite,  # 5ème élément
                relation  # 6ème élément
            ))

    print(f" {len(alignements)} alignements validés par LLM.")
    return alignements


def create_rdf_file(alignments: List[Alignment], ontologies: List[Optional[str]]) -> str:
    """
    Génère un fichier RDF/XML représentant l'alignement des ontologies,
    en utilisant rdflib, conformément aux conventions OAEI/Alignment API.
    """
    g = Graph()
    g.bind("align", ALIGN)
    g.bind("xsd", XSD)

    alignment_uri = URIRef("http://example.org/alignment/result")

    # 1. Créer le nœud principal de l'Alignement (Métadonnées)
    g.add((alignment_uri, RDF.type, ALIGN.Alignment))

    # Ajout des URIs des ontologies
    onto1_name = ontologies[0] if ontologies and ontologies[0] else 'Ontology1'
    onto2_name = ontologies[1] if ontologies and len(ontologies) > 1 and ontologies[1] else 'Ontology2'

    g.add((alignment_uri, ALIGN.onto1, URIRef(f"file:///{onto1_name}")))
    g.add((alignment_uri, ALIGN.onto2, URIRef(f"file:///{onto2_name}")))
    g.add((alignment_uri, ALIGN.level, Literal("partial")))

    # 2. Ajouter chaque Correspondance (Map)
    for i, align in enumerate(alignments):
        map_uri = URIRef(f"http://example.org/alignment/map_{i}")

        g.add((map_uri, RDF.type, ALIGN.map))
        g.add((alignment_uri, ALIGN.map, map_uri))

        # 2a. Entités
        g.add((map_uri, ALIGN.entity1, URIRef(align.entityA)))
        g.add((map_uri, ALIGN.entity2, URIRef(align.entityB)))

        # 2b. Relation (ex: "=", "<", ">")
        g.add((map_uri, ALIGN.relation, Literal(align.relation)))

        # 2c. Mesure (Score)
        # Utilise XSD.float pour s'assurer que le score est typé correctement
        g.add((map_uri, ALIGN.measure, Literal(align.score, datatype=XSD.float)))

    # 3. Sérialisation du graphe en RDF/XML
    return g.serialize(format='xml')





def create_merged_ontology(alignments: List[Alignment], ontologies: List[Optional[str]]) -> str:
    """
    NOUVEAU: Génère l'ONTOLOGIE DE PONT minimale (Bridge Ontology).
    Elle ne contient QUE les déclarations d'équivalence (owl:equivalentClass/Property)
    entre les URIs alignées.
    """
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # Déclaration de l'ontologie de pont
    g.add((URIRef("http://example.org/bridge/ontology"), RDF.type, OWL.Ontology))

    for align in alignments:
        entityA_uri = URIRef(align.entityA)
        entityB_uri = URIRef(align.entityB)

        # On n'ajoute que les relations d'équivalence (relation = "=")
        if align.relation == "=":

            # Définir le type de l'entité A et B (nécessaire pour owl:equivalent)
            if align.entityType == "Class":
                g.add((entityA_uri, RDF.type, OWL.Class))
                g.add((entityB_uri, RDF.type, OWL.Class))
                g.add((entityA_uri, OWL.equivalentClass, entityB_uri))

            elif align.entityType in ["Property", "ObjectProperty", "DatatypeProperty"]:
                # Simplification: on suppose ObjectProperty, mais la relation d'équivalence est la même
                g.add((entityA_uri, RDF.type, OWL.ObjectProperty))
                g.add((entityB_uri, RDF.type, OWL.ObjectProperty))
                g.add((entityA_uri, OWL.equivalentProperty, entityB_uri))

    # Sérialisation en OWL/XML (format d'ontologie standard)
    return g.serialize(format='xml')
