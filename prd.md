Voici le **Product Requirements Document (PRD)** complet pour le projet **Travel Order Resolver**.

En tant que ton pair Product Manager, j'ai consolidé toutes nos découvertes, les contraintes techniques du PDF et les bonnes pratiques identifiées dans la phase d'analyse BMAD. Ce document servira de "Single Source of Truth" pour l'équipe de développement.

---

# 📝 PRD: Travel Order Resolver
**Date :** 8 Janvier 2026
**Version :** 1.0 (Initial Draft)
**Statut :** 🟢 Validé pour développement
**Product Owner :** [Toi]
**Facilitator :** Codex (AI Peer)

---

## 1. Executive Summary
Le "Travel Order Resolver" est un moteur de traitement back-end conçu pour combler le fossé entre les demandes de voyage exprimées en langage naturel (désordonné, humain) et les bases de données ferroviaires rigides. L'objectif est de produire un pipeline logiciel capable de transformer une phrase telle que *"jveu allez a Paris depuis Bordeaux demain"* en un itinéraire de train SNCF valide, tout en gérant les fautes, le contexte et la structure grammaticale.

**Note Clé :** Ce projet exclut explicitement toute interface graphique Web. Le focus est 100% sur la performance algorithmique (NLP + Graphes) et l'architecture CLI (Command Line Interface).

---

## 2. Problème et Opportunité

| Le Problème | L'Opportunité |
| :--- | :--- |
| Les utilisateurs s'expriment naturellement, pas avec des codes gares. | Utiliser le NLP moderne (Transformers/CamemBERT) pour interpréter l'intention utilisateur. |
| Les moteurs actuels échouent souvent sur les fautes d'orthographe. | Créer un système robuste qui tolère le "bruit" linguistique (typos, absence d'accents). |
| Les données de transport sont statiques. | Exploiter les graphes pour trouver des itinéraires optimaux dynamiquement. |

---

## 3. Scope & User Stories

### In Scope (Ce qu'on fait)
*   Détection d'intention (Trip vs Not a Trip).
*   Extraction d'Entités Nommées (Villes de Départ/Arrivée) en Français.
*   Calcul d'itinéraire ferroviaire basé sur Open Data SNCF.
*   CLI (Command Line Interface) pour traitement par batch.

### Out of Scope (Ce qu'on ne fait pas pour la V1)
*   Application Web / Front-end React ou autre (INTERDIT par le spec).
*   Achat réel de billets.
*   Gestion temps réel (retards de trains).

### Core User Stories
1.  **US-01 (Valid Request) :** En tant que système, je reçois "Je veux aller de Toulouse à Paris" -> Je retourne l'itinéraire : Toulouse -> Bordeaux -> Paris.
2.  **US-02 (Robustness) :** En tant que système, je reçois "j'aimrai allai a toulon" (fautes) -> Je comprends Départ=Inconnu, Arrivée=Toulon -> Je demande le départ ou je traite.
3.  **US-03 (Noise Filter) :** En tant que système, je reçois "J'aime manger des pommes" -> Je retourne `INVALID` car ce n'est pas une commande de voyage.
4.  **US-04 (Complex phrasing) :** En tant que système, je reçois "Paris est ma destination depuis Lyon" -> Je comprends Départ=Lyon, Arrivée=Paris (inversion grammaticale).

---

## 4. Spécifications Fonctionnelles

### 4.1 Input / Output (Strict requirements)
Le système doit respecter strictement les formats d'échange pour l'évaluation automatique.

*   **Format d'Entrée :** Fichier texte ou `stdin`. Encodage UTF-8.
    *   Ligne type : `sentenceID,Texte de la commande`
*   **Format de Sortie NLP :**
    *   Si Valide : `sentenceID,Départ,Destination`
    *   Si Invalide : `sentenceID,INVALID` (ou code d'erreur spécifique)
*   **Format de Sortie Pathfinder :**
    *   `sentenceID,Départ,Escale1,Escale2,...,Destination`

### 4.2 Composant NLP (Le Cerveau)
*   **Dataset :** Création impérative d'un dataset de **10 000 phrases** via génération synthétique + nettoyage manuel. Doit inclure des structures grammaticales variées (voix passive, interrogative).
*   **Modèle :** Fine-tuning d'un modèle pré-entraîné (ex: CamemBERT) pour la tâche de NER (Named Entity Recognition).
*   **Disambiguation :** Capacité à distinguer les villes homonymes ou noms communs (ex: "Je veux une Orange" vs "Je vais à Orange").

### 4.3 Composant Pathfinder (Les Jambes)
*   **Données :** Importation des CSV "Gares et liste des gares" (SNCF Open Data).
*   **Graph Model :** Les gares sont des nœuds, les connexions sont des arêtes pondérées (par le temps ou la distance).
*   **Algorithme :** Implémentation de A* (A-Star) ou Dijkstra pour le plus court chemin.

---

## 5. Non-Functional Requirements (Contraintes)

*   **Performance :** Le système doit pouvoir ingérer un fichier de test de 1000 lignes en un temps raisonnable (temps d'inférence < 100ms par phrase idéalement).
*   **Architecture :** Le module NLP doit être découplé du Pathfinder (testable indépendamment).
*   **Langue :** Français obligatoire. Anglais/Espagnol en bonus.
*   **Documentation :** PDF requis expliquant l'architecture, l'entraînement et les métriques.

---

## 6. Critères de Succès & Metrics

Nous évaluerons la réussite du projet sur ces KPI :

| Métrique | Target | Description |
| :--- | :--- | :--- |
| **NER F1-Score** | > 0.85 | Moyenne harmonique de la Précision et du Rappel sur l'extraction des villes. |
| **Exact Match Rate** | > 80% | Pourcentage de phrases où (Départ, Arrivée) sont parfaitement identifiés. |
| **Trash Handling** | > 95% | Pourcentage de textes hors-sujet correctement classés comme `INVALID`. |
| **Valid Path Rate** | 100% | Le Pathfinder ne doit jamais proposer un trajet qui n'existe pas physiquement. |

---

## 7. Plan de Risque (Pre-Mortem)

*   **Risque : Le modèle confond "Paris" (Ville) et "Paris" (Prénom dans "Hilton").**
    *   *Mitigation :* Entraîner le modèle avec des phrases contenant des noms de personnes ambigus pour qu'il apprenne le contexte.
*   **Risque : Suroptimisation (Overfitting) sur le dataset généré.**
    *   *Mitigation :* Utiliser une grande variété de templates de génération (au moins 50 modèles de phrases différents) et valider sur des phrases écrites par de vrais humains (cross-testing avec d'autres groupes).
*   **Risque : Graphe non connecté.**
    *   *Mitigation :* S'assurer lors de l'import des données SNCF que les identifiants de gares sont normalisés (par exemple, gérer les différentes "Gare du Nord" sous un même nœud ou des nœuds connectés à 0 distance).

---

## 8. Roadmap & Phasing (Stratégie d'exécution)

### Phase 1 : Infrastructure & Data (Jours 1-3)
*   Scraping/Download SNCF Data.
*   Script de génération de phrases (Dataset V1).
*   Mise en place du repo et du squelette CLI.

### Phase 2 : Baseline Model (Jours 4-7)
*   Implémentation d'une solution simple (Regex/SpaCy) pour valider le pipeline I/O.
*   Implémentation du Pathfinder (Dijkstra basique).
*   Premier test "End-to-End".

### Phase 3 : Advanced NLP & Fine-Tuning (Jours 8-15)
*   Entraînement de CamemBERT sur le dataset V2 (10k lignes).
*   Optimisation des hyperparamètres.
*   Calcul et documentation des métriques (Accuracy/Loss curves).

### Phase 4 : Polish & Bonus (Jours 16-20)
*   Implémentation des features Bonus (Speech-to-Text, "Via").
*   Rédaction de la documentation finale.
*   Tests croisés avec d'autres groupes.

---

## Peer Note (Commentaire du Facilitateur)
*"Ce PRD met l'accent sur la création du **Dataset**, car c'est là que 80% de la performance ML va se jouer. Si les données d'entraînement sont mauvaises, même le meilleur modèle Transformers échouera (Garbage In, Garbage Out). Je recommande fortement de passer les 2-3 premiers jours exclusivement sur la compréhension des données SNCF et la génération de phrases."*