# NeuroGuard AI

[![CI](https://github.com/sofiazabour/neuroguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sofiazabour/neuroguard-ai/actions/workflows/ci.yml)

Projet d'exploration EEG et physiologique pour le **HABS Neuro-AI Hackathon 2026**.
La cible de prédiction et le cas d'usage restent à valider par l'équipe.

**État actuel :** installation, tests et extraction de puissance EEG par bande.
Le pipeline ne réalise pas encore de classification émotionnelle ni de diagnostic.
Les notebooks fournis par HABS sont des exemples de départ.

## Démarrer maintenant

Python **3.12**. Depuis un terminal macOS/Linux :

```bash
git clone https://github.com/sofiazabour/neuroguard-ai.git
cd neuroguard-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python process.py --data sample --out out/first-run
```

Sur Windows : `py -3.12 -m venv .venv`, puis
`.venv\Scripts\Activate.ps1` dans PowerShell. Les commandes `python` suivantes
sont identiques. Aucun compte Hugging Face ni GPU n'est requis pour cet échantillon.
Utiliser un nouveau dossier de sortie à chaque exécution.

## Configuration et résultats

[`config.toml`](config.toml) est partagé par la commande locale et HABS :

```toml
dataset_variant = "CURATED"
max_recordings = 3
band_hz = [8.0, 12.0]
```

`dataset_variant` accepte `CURATED` ou `PREPROCESSED`. Le traitement sélectionne
les premiers enregistrements de repos dans l'ordre des noms, jusqu'à la limite
configurée ; ce sous-ensemble sert à vérifier le pipeline, pas à évaluer un modèle.
L'échantillon inclus contient un seul participant. Pour une configuration locale
différente : `python process.py --data sample --out out/other-run --config custom.toml`.
HABS utilise toujours le `config.toml` commité.

| Fichier produit | Contenu |
| --- | --- |
| `band_power.tsv` | Résumé par enregistrement : puissance moyenne, durée, canal maximal |
| `band_power_by_channel.tsv` | Puissance par électrode, en µV² |
| `run.json` | Paramètres, fichiers traités, versions, durée, commit et état Git si disponibles |

Les noms génériques remplacent `alpha.tsv` et `alpha_by_channel.tsv` du template,
car la bande est maintenant configurable. La bande par défaut reste l'alpha 8–12 Hz.
Sur HABS, le code peut être livré sans historique Git ; le commit est alors `null`
dans le manifeste et reste identifiable dans les métadonnées du job HABS.

## Vérifications automatiques et GPU

À chaque push et pull request, **CI** installe les dépendances, vérifie un signal
synthétique de puissance connue et traite l'échantillon réel. Les résultats du
test réel sont conservés sept jours dans les artefacts GitHub Actions.

Le workflow **HABS** reste manuel et limité à `main`. Il vérifie le code avant
la soumission. Pour activer l'accès GPU, l'équipe doit renseigner la variable
GitHub `HABS_KEY` fournie par HABS : voir le [guide de configuration](docs/SETUP.md).
Un push seul ne lance aucun calcul GPU.

## Suite du projet

1. Choisir une cible mesurable à partir des annotations disponibles.
2. Vérifier l'alignement essais/événements/labels et la qualité des signaux.
3. Construire un modèle simple et mesurer ses performances sur des participants
   absents de l'entraînement, avec prétraitement ajusté sur l'entraînement seul.
4. Comparer EEG, physiologie et combinaison, puis préparer une démonstration.

Le dépôt ne contient pas encore de métriques de prédiction.

## Documentation

- [Configuration locale, accès aux données et HABS](docs/SETUP.md)
- [Dataset et précautions de validation](DATASET.md)
- [Notebooks d'exemple](starting_notebooks/README.md)
- [Guide HABS d'origine](HABS.md)

Les données téléchargées, sorties et fichiers `.env` sont ignorés par Git.
Les dépendances directes et transitives du pipeline sont épinglées dans
`requirements.txt` et `requirements-lock.txt` pour Python 3.12.

## NeuroGuard artifact-type prototype (November 2026)

The active `process.py` is an **exploratory multi-type artifact candidate detector**.
The former alpha-band smoke test is preserved as `process_alpha_baseline.py`.
The previous Fp1-only prototypes are preserved for comparison.

```bash
python process.py --data data --out out/neuroguard_run
python -m pytest tests/test_neuroguard_detector.py -q
```

The data directory must contain `CURATED/Pxxx/*_task-artefact_*.bdf` and
`Pxxx_day-1_metadata.json`. Outputs: `artifact_windows.csv`,
`calibration.csv`, `task_interval_summary.csv`, and `run.json`.

**Important:** These are provisional ocular/muscle/movement *candidate* rules,
not validated artifact labels. Experimental task intervals are not event-level
ground truth. Do not report task-interval classification as clinical sensitivity.
The pipeline currently processes the first three recordings and is an **offline
window replay**, not yet a wall-clock-paced real-time streaming implementation.
The previous `tests/test_process.py` tests the former alpha baseline and needs
migration; run `tests/test_neuroguard_detector.py` for the new detector.

Next: validate individual events, establish subject-held-out performance,
benchmark real-time replay and update CI and submission checks.
