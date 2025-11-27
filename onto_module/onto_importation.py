from rdflib import Graph, RDFS, RDF, Namespace, URIRef, OWL

ECOMMERCE_KEYWORDS = [
    "price", "currency", "seller", "shipping", "payment", "order",
    "prix", "devise", "vendeur", "livraison", "paiement", "commande",
    "offer", "offre", "produit", "product"
]

GR_NS = Namespace("http://purl.org/goodrelations/v1#")
SCHEMA_NS = Namespace("http://schema.org/")
OWL_NS = Namespace("http://www.w3.org/2002/07/owl#")


def charger_ontologie(file_path: str, filename: str):
    g = Graph()

    # On force le format XML si le fichier contient <?xml
    with open(file_path, "r", encoding="utf-8") as f:
        head = f.read(200).lower()

    if "<?xml" in head or "<rdf:" in head:
        format_name = "xml"
    else:
        # fallback à ton auto-détecteur
        if filename.endswith(".ttl"):
            format_name = "turtle"
        else:
            format_name = "xml"

    try:
        g.parse(file_path, format=format_name)
        print(f"Chargé en format : {format_name}")
        return g
    except Exception as e:
        print("Erreur de parsing :", e)
        return None



def check_ecommerce_robust(g: Graph) -> tuple[bool, str]:
    if g is None:
        return False, "Le graphe n'a pas pu être chargé."

    # 1. Vérifier imports (GoodRelations / Schema.org)
    imports = list(g.objects(None, OWL.imports))
    if URIRef(str(GR_NS)[:-1]) in imports:
        return True, "Importation de GoodRelations détectée."

    # 2. Vérifier usage direct de concepts e-commerce
    if (None, None, GR_NS.Offering) in g:
        return True, "Utilisation de GoodRelations détectée."

    if (None, RDFS.subClassOf, SCHEMA_NS.Product) in g:
        return True, "Schema.org Product détecté."
    if (None, RDFS.subClassOf, SCHEMA_NS.Offer) in g:
        return True, "Schema.org Offer détecté."

    # 3. Scanner TOUTES les URIs du graphe
    found_keywords = set()
    for s, p, o in g:
        for uri in [s, p, o]:
            name = str(uri).split("#")[-1].split("/")[-1]
            if any(k in name.lower() for k in ECOMMERCE_KEYWORDS):
                found_keywords.add(name)

    if len(found_keywords) >= 3:
        return True, f"Détection de concepts e-commerce : {', '.join(list(found_keywords))}"

    return False, "Ontologie non reconnue comme e-commerce."



