
---

# PRD : Travel Order Resolver (NLP & Pathfinding)

**Version :** 1.0  
**Statut :** En cours de définition  
**Auteur :** Yoann NGUENLO  

---

## 1. Vision & Objectifs
L'objectif est de construire un programme capable d'interpréter des commandes de voyage en langage naturel (français) et de générer un itinéraire de train optimal en s'appuyant sur les données réelles de la SNCF.

### Objectifs principaux :
- Extraire avec précision l'origine et la destination d'une phrase complexe.
- Calculer le meilleur itinéraire (plus court/plus rapide) via un algorithme de graphe.
- Fournir une solution robuste capable de gérer les fautes d'orthographe et les structures de phrases variées.

---

## 2. User Stories
- **En tant qu'utilisateur**, je veux saisir une phrase comme "Je veux aller à Paris depuis Lyon" et obtenir mon trajet.
- **En tant qu'administrateur/évaluateur**, je veux passer un fichier de 1000 phrases au programme et obtenir un fichier de résultats structurés pour évaluer la précision.

---

## 3. Spécifications Fonctionnelles (Requirements)

### 3.1. Module NLP (Cœur du projet)
- [ ] **Détection de validité** : Identifier si une phrase est une commande de voyage valide ou non.
- [ ] **Extraction d'entités (NER)** : Extraire la ville de départ et la ville d'arrivée.
- [ ] **Gestion du Français** : Supporter les spécificités de la langue française (articles, prépositions "de", "à", "vers", etc.).
- [ ] **Robustesse linguistique** : Gérer l'absence de majuscules, d'accents et les fautes d'orthographe.
- [ ] **Désambiguïsation** : Distinguer les noms de villes des noms communs (ex: "Orange", "Port") ou des noms de personnes (ex: "Albert").
- [ ] **Indépendance** : Le module NLP doit pouvoir fonctionner de manière isolée pour les tests.

### 3.2. Traitement des Données & Dataset
- [x] **Importation SNCF** : Utiliser les Open Data SNCF (liste des gares et horaires).
- [ ] **Génération de Dataset** : Créer un jeu de données d'environ 10 000 phrases variées.
- [ ] **Collaboration** : Intégrer des données provenant d'autres groupes pour augmenter la diversité.
- [ ] **Nettoyage (Trash texts)** : Inclure des phrases invalides dans le dataset pour entraîner le modèle à répondre `INVALID`.

### 3.3. Module Pathfinder (Optimisation)
- [ ] **Modélisation de Graphe** : Transformer les horaires SNCF en un graphe (Nœuds = Gares, Arêtes = Trajets).
- [ ] **Algorithme de calcul** : Implémenter Dijkstra, A* ou un algorithme similaire.
- [ ] **Calcul d'itinéraire** : Retourner une séquence de villes (étapes) entre le départ et l'arrivée.

### 3.4. Interface & Echanges (I/O)
- [ ] **Support UTF-8** : Tous les fichiers d'entrée et de sortie doivent être encodés en UTF-8.
- [ ] **Entrée Standard (stdin)** : Lecture des commandes ligne par ligne au format `sentenceID,sentence`.
- [ ] **Sortie NLP** : Format `sentenceID,Departure,Destination` ou `sentenceID,Code_Erreur`.
- [ ] **Sortie Pathfinder** : Format `sentenceID,Departure,Step1,Step2,...,Destination`.

---

## 4. Contraintes Techniques & Livrables

### 4.1. Contraintes
- [ ] **Langage** : Français impératif (autres langues en bonus).
- [ ] **Architecture** : Pas d'application Web (CLI uniquement).
- [ ] **Performance** : Le temps de réponse par phrase doit permettre un traitement par batch.

### 4.2. Livrables obligatoires
- [ ] **Code Source** : Propre et documenté.
- [ ] **Documentation Architecture** : Schéma des différentes couches de l'application.
- [ ] **Rapport d'entraînement** : Description du processus, des jeux de données et des paramètres finaux.
- [ ] **Métriques d'évaluation** : Précision, Rappel (Recall) et exemples détaillés de traitement.
- [ ] **Dataset utilisé** : Fournir le fichier des 10 000 phrases.

---

## 5. Success Metrics (Indicateurs de réussite)
- **Taux de reconnaissance NER** : > 90% sur un dataset de test non vu durant l'entraînement.
- **Robustesse** : > 80% de succès sur des phrases avec fautes d'orthographe.
- **Validité des trajets** : 100% des itinéraires proposés par le Pathfinder doivent exister dans les données SNCF.

---

## 6. Bonus (Nice-to-have)
- [ ] **Speech-to-Text** : Module de reconnaissance vocale.
- [ ] **Multilingue** : Support de l'anglais ou de l'espagnol.
- [ ] **Escales complexes** : "Je veux aller à Paris en passant par Lyon".
- [ ] **Cloud Hosting** : Déploiement du moteur sur Azure/AWS/GCP.
- [ ] **Monitoring** : Mesure de l'empreinte carbone ou de la consommation CPU/RAM par requête.

---

### Étapes recommandées (Roadmap) :
1. **Sprint 1 :** Scraping/Import SNCF + Génération du Dataset initial.
2. **Sprint 2 :** Modèle NLP de base (Baseline) + Structure du Graphe.
3. **Sprint 3 :** Fine-tuning (CamemBERT/BERT) + Algorithme Pathfinder.
4. **Sprint 4 :** Tests intensifs, métriques et rédaction de la documentation.