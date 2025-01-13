import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

#  LOGIC / MODEL

def simulate_fees_scenario(volume_start,
                           days,
                           daily_growth,
                           scenario,
                           do_stress_test,
                           shock_day,
                           shock_factor,
                           supply_start=1_000_000,
                           elasticity=0.5):


    # Définition des taux en fonction du SCENARIO (on reste sur 1% total)

    if scenario == 'S1':
        burn_rate = 0.30  # fraction du 1% pour le burn
        liquidity_rate = 0.30
        treasury_rate = 0.40
    else:  # S2
        burn_rate = 0.15
        liquidity_rate = 0.10
        treasury_rate = 0.75

    total_fee_rate = 0.01  # 1%

    # Hypothèses de base pour la modélisation du prix du token
    # On suppose un "prix" initial arbitraire, ou corrélé à supply.
    # Cf. "price ~ 1 / supply" * un facteur k, ici 2.0
    price_token = 2.0 * (1_000_000 / supply_start)
    supply = supply_start

    records = []

    current_volume = volume_start

    for day in range(1, days + 1):
        # EVENTUEL STRESS TEST
        if do_stress_test and day == shock_day:
            # choc => on multiplie le volume par shock_factor
            # (ex. 0.5 = -50%, 1.5 = +50%, etc.)
            current_volume *= shock_factor

        # On applique la croissance journalière
        if day > 1:  # on ne l'applique pas avant le day 1
            current_volume = current_volume * (1 + daily_growth)

        # fees totaux sur la journée
        daily_fees = current_volume * total_fee_rate

        # Burn, liquidity, treasury
        daily_burn = daily_fees * burn_rate
        daily_liquidity = daily_fees * liquidity_rate
        daily_treasury = daily_fees * treasury_rate

        # On décrémente la supply du token via le burn
        # Ici, on suppose 1$ = 1 token, c'est TRÈS simplifié.
        # Pour un vrai modèle, il faudrait convertir daily_burn en "nombre de tokens" rachetés.
        # On fait un mini "approx" : tokens_burned = daily_burn / price_token.
        tokens_burned = daily_burn / price_token
        supply = max(0, supply - tokens_burned)

        # On recalcule un "nouveau prix" hypothétique en se basant sur la new supply
        # price ~ 1 / supply ^ elasticity (ici, un usage artisanal)
        # On scale un peu pour la démo
        if supply > 0:
            price_token = 2.0 * (1_000_000 / supply)**elasticity
        else:
            price_token = 0

        records.append({
            'day': day,
            'volume': current_volume,
            'fees_total': daily_fees,
            'burn_usd': daily_burn,
            'liquidity_usd': daily_liquidity,
            'treasury_usd': daily_treasury,
            'tokens_burned': tokens_burned,
            'supply': supply,
            'price_token': price_token
        })

    df = pd.DataFrame(records)
    return df



#  STREAMLIT APP

def show_basic_mode():
    st.header("Mode Basique : Comparez 2 Scénarios (S1 vs S2)")
    st.write("Ici, on simule la répartition des fees sur un volume de transactions, "
             "avec un simple modèle de burn (sans trop d'artifices).")

    # Inputs
    volume_journalier = st.slider(
        "Volume de transactions de départ (USD / jour)", 
        min_value=10_000, max_value=2_000_000, step=10_000, value=100_000
    )
    days = st.slider("Nombre de jours de simulation", 1, 365, 30)
    daily_growth = st.number_input("Croissance du volume par jour (en %)", -10.0, 10.0, 1.0, 0.1) / 100.0

    # Choix du scénario
    scenario_choice = st.radio("Choisir Scénario", ("S1", "S2"))

    # Bouton pour run
    if st.button("Lancer la simulation"):
        df_result = simulate_fees_scenario(volume_start=volume_journalier,
                                           days=days,
                                           daily_growth=daily_growth,
                                           scenario=scenario_choice,
                                           do_stress_test=False,
                                           shock_day=9999,  # pas de choc
                                           shock_factor=1.0)

        st.subheader("Résultats (extraits)")
        st.dataframe(df_result.head(20))

        # Graph 1 : évolution Burn / Liquidity / Treasury
        fig_fees = px.area(
            df_result,
            x='day',
            y=['burn_usd', 'liquidity_usd', 'treasury_usd'],
            title=f"Évolution journalière des fees (Scénario {scenario_choice})",
            labels={"value": "USD", "day": "Jour"}
        )
        st.plotly_chart(fig_fees, use_container_width=True)

        # Graph 2 : supply vs price
        fig_price_supply = px.line(
            df_result,
            x='day',
            y=['supply', 'price_token'],
            title="Évolution de la supply et du prix (modèle simplifié)",
            labels={"day": "Jour"}
        )
        st.plotly_chart(fig_price_supply, use_container_width=True)

        # Stats cumulées
        total_burn = df_result['burn_usd'].sum()
        total_treasury = df_result['treasury_usd'].sum()
        final_price = df_result['price_token'].iloc[-1]
        final_supply = df_result['supply'].iloc[-1]

        st.markdown(f"**Burn total (USD) :** {total_burn:,.2f}")
        st.markdown(f"**Trésorerie cumulée (USD) :** {total_treasury:,.2f}")
        st.markdown(f"**Prix final du token (approx) :** {final_price:,.4f} $")
        st.markdown(f"**Supply finale (approx) :** {final_supply:,.2f} tokens")


def show_advanced_mode():
    st.header("Mode Avancé : Stress Tests & Comparaison Multi-Scénarios")

    with st.expander("Paramètres Généraux"):
        volume_journalier = st.slider(
            "Volume de transactions de départ (USD / jour)", 
            min_value=10_000, max_value=2_000_000, step=10_000, value=200_000
        )
        days = st.slider("Nombre de jours de simulation", 1, 365, 60)
        daily_growth = st.number_input("Croissance du volume par jour (en %)", -10.0, 10.0, 1.0, 0.1) / 100.0
        elasticity = st.number_input("Elasticité (impact du burn sur le prix)", 0.1, 2.0, 0.5, 0.1)
        supply_start = st.number_input("Supply initiale du token", 100_000, 100_000_000, 1_000_000, 100_000)

    with st.expander("Stress Test"):
        do_stress_test = st.checkbox("Activer un choc de marché ?")
        shock_day = st.slider("Jour du choc", 1, days, 15)
        shock_factor = st.slider("Facteur multiplicatif du volume", 0.1, 2.0, 0.5, 0.05)
        st.write("**Exemple** : 0.5 => -50% du volume, 1.5 => +50% du volume...")

    st.write("---")

    st.write("### Comparaison Automatique S1 vs S2")
    if st.button("Simuler S1 & S2"):
        # On simule S1
        df_s1 = simulate_fees_scenario(
            volume_start=volume_journalier,
            days=days,
            daily_growth=daily_growth,
            scenario='S1',
            do_stress_test=do_stress_test,
            shock_day=shock_day,
            shock_factor=shock_factor,
            supply_start=supply_start,
            elasticity=elasticity
        )
        # On simule S2
        df_s2 = simulate_fees_scenario(
            volume_start=volume_journalier,
            days=days,
            daily_growth=daily_growth,
            scenario='S2',
            do_stress_test=do_stress_test,
            shock_day=shock_day,
            shock_factor=shock_factor,
            supply_start=supply_start,
            elasticity=elasticity
        )

        # On peut concaténer pour faciliter la visualisation
        df_s1['scenario'] = 'S1'
        df_s2['scenario'] = 'S2'
        df_all = pd.concat([df_s1, df_s2], ignore_index=True)

        st.subheader("Aperçu des données")
        st.dataframe(df_all.head(20))

        # Graph - Comparaison Burn / scenario
        fig_burn_compare = px.line(
            df_all,
            x='day',
            y='burn_usd',
            color='scenario',
            title="Comparaison du Burn (USD) jour par jour",
            labels={"day": "Jour", "burn_usd": "Burn (USD)"}
        )
        st.plotly_chart(fig_burn_compare, use_container_width=True)

        # Graph - supply
        fig_supply_compare = px.line(
            df_all,
            x='day',
            y='supply',
            color='scenario',
            title="Comparaison de la Supply Tokens",
            labels={"day": "Jour", "supply": "Supply"}
        )
        st.plotly_chart(fig_supply_compare, use_container_width=True)

        # Calcul cumul
        df_grouped = df_all.groupby('scenario').agg({
            'burn_usd': 'sum',
            'treasury_usd': 'sum',
            'liquidity_usd': 'sum',
            'price_token': 'last',
            'supply': 'last'
        }).rename(columns={
            'burn_usd': 'Burn Total',
            'treasury_usd': 'Trésorerie Totale',
            'liquidity_usd': 'Liquidité Totale',
            'price_token': 'Prix Final (approx)',
            'supply': 'Supply Finale'
        })
        st.subheader("Synthèse par Scénario")
        st.dataframe(df_grouped)

        st.write("**Analyse :**")
        st.write("- On peut voir le total du Burn, la Trésorerie, la Liquidité, etc. sur la période.")
        st.write("- Le prix final et la supply finale (simplement d'après notre petit modèle d'élasticité).")


def main():
    st.title("Simulateur Avancé de Répartition des Fees & Tokenomics")
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Choisissez un mode", ["Basique", "Avancé"])

    if mode == "Basique":
        show_basic_mode()
    else:
        show_advanced_mode()


if __name__ == "__main__":
    main()
