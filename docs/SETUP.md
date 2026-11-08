# Configuration de NeuroGuard AI

## 1. Vérifier le dépôt sans compte externe

Installer Python 3.12, créer un environnement virtuel et installer les dépendances
avec les commandes du [README](../README.md). Depuis la racine du dépôt :

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python process.py --data sample --out out/setup-check
```

Le dossier de sortie doit être nouveau ou vide. Une exécution réussie produit
deux TSV et un `run.json` avec `status: completed`. Une erreur interrompt le
traitement ; des sorties partielles sans ce manifeste ne sont pas un succès.
Ne pas réutiliser un dossier contenant des résultats.

Le contrat `process(data_dir, out_dir)` de HABS reste disponible. La commande
locale appelle cette même fonction ; le client HABS n'est donc pas nécessaire
pour commencer. `config.toml` est recherché à côté de `process.py`, indépendamment
du répertoire courant. Les chemins `--data`, `--out` et `--config` explicitement
fournis sont relatifs au répertoire courant.

## 2. Accéder aux données nécessaires au premier modèle

L'accès au dataset est personnel et doit être approuvé par HABS :

1. Ouvrir [le dataset Hugging Face](https://huggingface.co/datasets/HABSteam/multimodal-eeg-emotion).
2. Se connecter, accepter les conditions et attendre l'approbation.
3. Installer le client dans un environnement d'outillage séparé du pipeline :

```bash
python3.12 -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install huggingface_hub
hf auth login
hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./data --include "*.csv"
hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./data --include "PREPROCESSED/P001/*emotion-video*" "PREPROCESSED/P001/*metadata.json"
```

Sur Windows, créer et activer `.venv-tools` comme `.venv` dans le README.
Ne pas télécharger les 351 Go pour ce premier contrôle. Le participant P001
sert à développer le chargement ; plusieurs participants seront ensuite requis
pour entraîner et valider un modèle. Les enregistrements vidéo ne sont pas
encore traités par `process.py`, qui vérifie les enregistrements de repos.

Chaque BDF doit conserver son `.record_map`. Lire [DATASET.md](../DATASET.md)
avant de joindre les tables : les événements doivent être recalés sur l'origine
du segment, les vidéos peuvent se répéter et `TESTSUBJECT` est une valeur manquante.
Ne pas publier les données téléchargées, jetons ou liens SAS dans le dépôt.

## 3. Activer les soumissions GPU HABS, si nécessaires

L'enregistrement de l'équipe et la valeur fournie par HABS ne peuvent pas être
déduits du code. Ces étapes sont à effectuer par un membre de l'équipe :

1. Sur le Slack du hackathon, envoyer à `@HABS Bot` le lien exact :
   `https://github.com/sofiazabour/neuroguard-ai`.
2. Récupérer la **team key** et le **results link** retournés par le bot.
3. Dans GitHub, ouvrir **Settings → Secrets and variables → Actions → Variables**.
4. Créer une **Repository variable** nommée **`HABS_KEY`**, avec la team key.
   C'est le client ID lié au dépôt, pas un mot de passe ; le workflow utilise OIDC.
   Ne pas mettre le results link dans cette variable.
5. Vérifier que la CI du dernier commit sur `main` est verte.
6. Seulement lorsqu'un calcul GPU est utile : **Actions → HABS → Run workflow → main**.

Le workflow vérifie la présence de la clé, exécute les tests et le contrôle local,
puis soumet le code. La branche `main` et l'adresse du dépôt doivent rester celles
enregistrées auprès de HABS. Un push ne soumet aucun job GPU.

Conserver le results link hors du dépôt ; suivre les instructions de récupération
des résultats dans le [guide HABS](../HABS.md). Le client `habs-hack` peut être
installé dans `.venv-tools` si nécessaire :

```bash
python -m pip install "habs-hack[storage] @ git+https://github.com/HABS-BRAIN/habs-hack@main"
```

## 4. Modifier les dépendances

`requirements.txt` énumère les dépendances directes et utilise
`requirements-lock.txt` comme contraintes transitives. Pour une mise à jour :
créer un environnement Python 3.12 propre, résoudre les versions souhaitées,
actualiser les deux fichiers, puis exécuter les tests et le contrôle sur `sample`.
Ne pas générer le verrou depuis un environnement contenant les outils HABS ou
les dépendances des notebooks. Les notebooks gardent leur environnement séparé.

## Premier jalon scientifique

Fixer la cible et les métriques, puis obtenir un score reproductible sur des
participants absents de l'entraînement. Les opérations de normalisation et de
sélection de caractéristiques doivent être ajustées sur l'entraînement seulement.
L'extraction alpha actuelle confirme le fonctionnement du pipeline ; elle ne
constitue pas une validation d'un détecteur de stress ou d'émotions.
