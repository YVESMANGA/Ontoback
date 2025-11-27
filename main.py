from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from fastapi.responses import Response as FastAPIResponse

from rdflib import Graph
from onto_module.onto_importation import *
from onto_module.Extration_semantique import semantic_extraction
from onto_module.sentenceTransformers import *

app = FastAPI(title="Ontology Alignment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComplianceResult(BaseModel):
    filename: str
    is_ecommerce: bool
    details: str = "Simulated check details."

# Schéma pour la requête de génération RDF (reçue du frontend)
class RdfGenerationRequest(BaseModel):
    validatedAlignments: List[Alignment]
    ontologyFiles: List[Optional[str]]
# ---------------------------------------------
# Route 1 : Upload et vérification e-commerce
# ---------------------------------------------
@app.post("/upload_ontology/")
async def upload_ontology(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        # Sauvegarde temporaire du fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Charger le graphe RDF/OWL
        g = charger_ontologie(tmp_path, file.filename)
        # Vérifier si c'est une ontologie e-commerce
        is_ecommerce, reason = check_ecommerce_robust(g)

        results.append({
            "filename": file.filename,
            "is_ecommerce": is_ecommerce,
            "triplet_count": len(g) if g else 0
        })

    return {"results": results}


# ---------------------------------------------
# Route 2 : Alignement des ontologies
# ---------------------------------------------
@app.post("/align_ontologies/")
async def align_ontologies(files: list[UploadFile] = File(...)):
    ontologies_elements = []

    # 1. Traitement et extraction des éléments d'ontologie (Synchrone)
    for file in files:
        # Sauvegarde temporaire
        # NOTE : Les opérations de fichier dans FastAPI peuvent être bloquantes.
        # Pour une haute performance, elles devraient aussi être exécutées dans un
        # thread pool (executor), mais nous gardons le code simple ici.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Parser le fichier RDF/OWL avec rdflib
        g = Graph()
        # Choisir le format selon ton fichier : "xml", "turtle", "json-ld", etc.
        g.parse(tmp_path, format="xml")

        # Extraction sémantique (Supposée synchrone)
        elements = semantic_extraction(g)
        ontologies_elements.append(elements)

    # 2. Vectorisation (Supposée synchrone)
    embeddings_list = [embedding(el) for el in ontologies_elements]

    if len(embeddings_list) < 2:
        return {"error": "Deux fichiers d'ontologie sont requis pour l'alignement."}

    embA = embeddings_list[0]
    embB = embeddings_list[1]

    # 3. Comparaison Hybride Asynchrone (LA CORRECTION)
    # Nous attendons (await) ici que Groq ait traité tous les lots en parallèle.
    alignements = await comparaison_hybride_batch(embA, embB) # <--- FIX APPLIQUÉ

    # Hypothèse : 'alignements' contient maintenant 6 éléments par tuple
    # (a, b, s, m, t, r) -> (entityA, entityB, score, method, entityType, relation)

    results_json = [
        {
            "entityA": a,
            "entityB": b,
            "score": s,
            "method": m,
            "entityType": t,
            "relation": r
        }
        for a, b, s, m, t, r in alignements
    ]
    return {"alignments": results_json}


@app.post("/generate_rdf/")
async def generate_rdf(request: RdfGenerationRequest):
    """Point d'API 3: Génération du fichier RDF d'alignement validé (étape 3)."""
    if not request.validatedAlignments:
        raise HTTPException(status_code=400, detail="Aucun alignement validé fourni.")

    # Le modèle Pydantic garantit que validatedAlignments est une List[Alignment]

    rdf_content = create_rdf_file(request.validatedAlignments, request.ontologyFiles)
    # Retourne le contenu RDF avec le bon type MIME
    return FastAPIResponse(content=rdf_content, media_type="application/rdf+xml")


@app.post("/generate_merged_ontology/")
async def generate_merged_ontology(request: RdfGenerationRequest):
    """Génération de l'ontologie de pont/fusionnée (prête à l'emploi)."""
    if not request.validatedAlignments:
        raise HTTPException(status_code=400, detail="Aucun alignement validé fourni.")

    merged_ontology_content = create_merged_ontology(request.validatedAlignments, request.ontologyFiles)

    # On renvoie le contenu avec le même type MIME pour les ontologies RDF/XML
    return FastAPIResponse(content=merged_ontology_content, media_type="application/rdf+xml")