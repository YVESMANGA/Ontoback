import os
import aiohttp
import asyncio
import json
from typing import Dict, Tuple, List, Any
# Import de la bibliothèque pour charger les variables d'environnement
from dotenv import load_dotenv

# --- Ligne Cruciale pour l'environnement ---
# Charge les variables du fichier .env (comme GROQ_API_KEY)
load_dotenv()
# ------------------------------------------

# Types pour les verdicts du LLM
Verdict = Dict[str, str]  # {"statut": "OUI"/"NON", "type": "...", "relation": "..."}
VerdictsDict = Dict[Tuple[str, str], Verdict]

# Clé Groq et configuration
# La clé est chargée depuis l'environnement
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # modèle à utiliser
TAILLE_LOT = 50  # nombre de paires par sous-requête
URL_GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"  # Endpoint compatible OpenAI


async def appeler_llm_par_lot(lot: List[Tuple[str, str]]) -> VerdictsDict:
    """
    Appelle Groq pour valider un lot de paires (uriA, uriB) et retourne un dictionnaire de verdicts.
    Les requêtes sont faites de manière asynchrone par sous-lots.
    """
    if not GROQ_API_KEY:
        # Lève une erreur si la clé n'est pas trouvée, même après load_dotenv()
        raise ValueError("La variable d'environnement GROQ_API_KEY n'est pas définie. Vérifiez votre fichier .env.")

    verdicts: VerdictsDict = {}

    async def appeler_groq_sous_lot(sous_lot: List[Tuple[str, str]]):

        # 1. Prépare les paires pour l'inclusion dans le prompt utilisateur
        prompt_lines = []
        for i, (uriA, uriB) in enumerate(sous_lot):
            # Utiliser un index ou un identifiant pour aider le LLM à structurer sa réponse
            prompt_lines.append(f"Paire {i + 1} : Concept A : {uriA} | Concept B : {uriB}")

        # 2. Construction du prompt au format Chat Completions
        prompt_user_content = "Vérifie les paires de concepts suivantes et indique pour chacune si elles sont le même concept.\nRéponds **uniquement** avec un tableau JSON unique (une liste de dictionnaires) dont chaque élément contient les clés : `paire_id` (l'index de la paire), `statut` (OUI/NON), `type`, `relation`.\n\n"
        prompt_user_content += "\n".join(prompt_lines)

        messages = [
            {"role": "system",
             "content": "Tu es un expert en ontologies, concis et précis, répondant uniquement avec le JSON demandé. Le JSON DOIT être une liste valide d'objets."},
            {"role": "user", "content": prompt_user_content}
        ]

        # 3. Construction du payload (format Chat Completions)
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": 200 * len(sous_lot),
            "temperature": 0.0,  # Pour des réponses déterministes
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # 4. Appel à l'API
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(URL_GROQ_CHAT, json=payload, headers=headers) as resp:

                # Gestion des erreurs HTTP (Non-200)
                if resp.status != 200:
                    print(f"⚠️ Erreur API Groq : code {resp.status}, texte: {await resp.text()} pour ce sous-lot.")
                    sous_verdicts = {pair: {"statut": "NON", "type": "Erreur API", "relation": "↔"} for pair in
                                     sous_lot}
                    verdicts.update(sous_verdicts)  # Mise à jour avant de continuer
                    return

                    # Traitement de la réponse 200 OK
                res: Dict[str, Any] = await resp.json()

                try:
                    # 5. Récupération et parsing de la réponse
                    llm_response_text = res["choices"][0]["message"]["content"]
                    json_data = json.loads(llm_response_text)

                    # S'assurer que le résultat est une liste (même si un seul objet est renvoyé)
                    json_list = json_data if isinstance(json_data, list) else [json_data]

                    temp_verdicts: Dict[Tuple[str, str], Verdict] = {}

                    # 6. Mise en forme des verdicts avec vérification de type (CORRECTION)
                    for i, result in enumerate(json_list):
                        if i < len(sous_lot):
                            original_pair = sous_lot[i]

                            # VÉRIFICATION CRUCIALE : assure que 'result' est un dictionnaire
                            if isinstance(result, dict):
                                temp_verdicts[original_pair] = {
                                    "statut": result.get("statut", "NON"),
                                    "type": result.get("type", "Inconnu"),
                                    "relation": result.get("relation", "↔")
                                }
                            else:
                                # Fallback si l'élément n'est pas un dictionnaire
                                print(
                                    f"⚠️ Élément de réponse Groq non-dictionnaire. Verdict par défaut appliqué pour la paire {original_pair}.")
                                temp_verdicts[original_pair] = {
                                    "statut": "NON",
                                    "type": "Parsing Error",
                                    "relation": "↔"
                                }

                    sous_verdicts = temp_verdicts

                except (KeyError, json.JSONDecodeError, IndexError) as e:
                    print(
                        f"⚠️ Erreur de parsing JSON/Structure de réponse Groq : {e}. Texte brut reçu : {llm_response_text[:100]}...")
                    # Fallback pour le sous-lot entier en cas d'erreur de parsing globale
                    sous_verdicts = {pair: {"statut": "NON", "type": "Erreur JSON", "relation": "↔"} for pair in
                                     sous_lot}

        verdicts.update(sous_verdicts)

    # Découpe le lot en sous-lots de TAILLE_LOT et lance les tâches asynchrones
    sous_lots = [lot[i:i + TAILLE_LOT] for i in range(0, len(lot), TAILLE_LOT)]
    tasks = [appeler_groq_sous_lot(sl) for sl in sous_lots]

    # Attend que toutes les requêtes asynchrones soient terminées
    await asyncio.gather(*tasks)

    print(f"✅ Appels API Groq terminés pour {len(lot)} paires")
    return verdicts


# ----------------------------------------------------------------------
# --- Exemple d'utilisation (Décommentez pour exécuter ce script) ---
# ----------------------------------------------------------------------
"""
async def main():
    # Exemple de paires à vérifier
    test_lot = [
        ("http://concept.org/chat", "http://concept.org/cat"),
        ("http://concept.org/chien", "http://concept.org/dog"),
        ("http://concept.org/pomme", "http://concept.org/apple"),
        ("http://concept.org/maison", "http://concept.org/home"),
        ("http://concept.org/rouge", "http://concept.org/vert"), # Différent
        ("http://concept.org/velo", "http://concept.org/bicycle"), 
    ]

    # Exécute la fonction asynchrone
    resultats = await appeler_llm_par_lot(test_lot)

    print("\n--- Résultats finaux ---")
    for (uriA, uriB), verdict in resultats.items():
        print(f"({uriA}, {uriB}) -> {verdict}")

# Exécute la fonction principale
if __name__ == "__main__":
    # Note: Assurez-vous que GROQ_API_KEY est défini dans votre .env
    asyncio.run(main())
"""