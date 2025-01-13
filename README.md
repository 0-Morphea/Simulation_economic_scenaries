# Simulateur de Répartition des Fees & Tokenomics

Ce projet est une application Streamlit permettant de simuler différents scénarios de répartition des frais dans un modèle simplifié de tokenomics. L'objectif principal est d'observer l'évolution des paramètres clés tels que le burn, la trésorerie, la liquidité, le prix des tokens et la supply sur une période donnée, tout en offrant la possibilité de tester des scénarios avancés comme les stress tests.

## Fonctionnalités principales

- **Simulation basique :** Comparez deux scénarios (`S1` et `S2`) avec une visualisation de l'évolution des fees, de la supply, et du prix des tokens.
- **Mode avancé :** Ajoutez des paramètres supplémentaires comme des stress tests, ajustez l'élasticité et la supply initiale, et comparez automatiquement les résultats des scénarios `S1` et `S2`.
- **Visualisations interactives :** Génération de graphiques dynamiques avec Plotly pour analyser les résultats.
- **Synthèse des résultats :** Présentation des données clés (burn total, trésorerie cumulée, prix final, etc.) sous forme de tableau.

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/your-repo/simulator-tokenomics.git
   cd simulator-tokenomics
Installez les dépendances nécessaires :

bash
Copier le code
pip install -r requirements.txt
Lancez l'application Streamlit :

bash
Copier le code
streamlit run main.py
Utilisation
Mode Basique : Idéal pour une analyse rapide des scénarios S1 et S2.

Réglez les paramètres de simulation (volume de départ, nombre de jours, croissance journalière).
Comparez les résultats sous forme de tableaux et graphiques.
Mode Avancé : Explorez des scénarios plus complexes :

Paramétrez un stress test avec un choc de marché.
Ajustez l'élasticité et la supply initiale.
Analysez en profondeur les différences entre les scénarios S1 et S2.
Paramètres de simulation
Volume initial (volume_start) : Volume de transactions de départ en USD.
Nombre de jours (days) : Durée de la simulation.
Croissance journalière (daily_growth) : Taux de variation quotidienne du volume (en %).
Scénario (scenario) :
S1 : Répartition des fees avec 30% burn, 30% liquidité, 40% trésorerie.
S2 : Répartition des fees avec 15% burn, 10% liquidité, 75% trésorerie.
Stress test (do_stress_test) : Application d'un choc sur le volume à un jour spécifique (shock_day).
Elasticité (elasticity) : Relation entre le burn et le prix des tokens.
Supply initiale (supply_start) : Quantité de tokens au début de la simulation.
Visualisations
Évolution journalière des fees : Burn, liquidité et trésorerie.
Supply et prix du token : Impact des burns sur la supply et le prix.
Comparaisons entre scénarios : Analyse graphique des métriques clés (burn, supply, etc.).
Exemples
Exemple de simulation
Volume initial : 100 000 USD
Durée : 30 jours
Croissance journalière : 1%
Scénario : S1
Résultats :

Burn total : 9 123 USD
Trésorerie cumulée : 12 456 USD
Prix final du token : 0.0023 USD
Supply finale : 999 000 tokens
Exemple de stress test
Volume initial : 200 000 USD
Choc au jour 15 : Réduction de 50% du volume
Scénarios comparés : S1 vs S2
Dépendances
streamlit : Interface utilisateur interactive.
pandas : Gestion des données.
plotly : Visualisation des résultats.
numpy : Calculs numériques.
Contribution
Les contributions sont les bienvenues ! Si vous souhaitez améliorer ou ajouter des fonctionnalités, créez une issue ou soumettez une pull request.
