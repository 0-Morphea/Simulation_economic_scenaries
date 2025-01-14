import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
#    LOGIQUE / MODEL
# =========================

def bonding_curve_rbot_for_tokens(m_before, m_after, p0, f):
    """
    Exponentielle :
    B(m,n) = p0 * (f^n - f^m) / (f - 1)
    m_before = old AiToken supply
    m_after  = new AiToken supply
    p0,f = params Bonding Curve
    Retourne la quantité (positive => besoin en $RBOT si on achète,
                           négative => on récupère du RBOT si on vend).
    """
    if abs(f - 1.0) < 1e-9:
        # Cas f ~= 1 => prix constant
        return p0 * (m_after - m_before)
    return p0 * ((f**m_after) - (f**m_before)) / (f - 1)


def simulate_protocol(
    scenario_name,
    # Param généraux
    total_periods,       # nombre total (jour ou mois)
    time_unit,           # "Days" ou "Months"
    nb_agents,           # juste un param contextuel (int)
    # Bonding curve
    p0,                  # prix initial
    f,                   # facteur exponentiel
    # Secondary market
    secondary_market_start,
    portion_secondary_after,
    # Fees
    protocol_fee_percent,
    treasury_fee_percent,
    # Comportements utilisateurs
    treasury_funding_objective_musd_year,
    stakeholders_yearly_sales_percent,
    stakers_lock_months,
    # Table transactions (edité)
    transactions_df,
    # Table vesting (RBOT)
    vesting_df,
    # Param init RBOT
    init_rbot_price,
    init_rbot_supply,
    # Divers
    elasticity_price=0.5
):
    """
    Simulation "avancée" :
    - Sur la période 1..total_periods (unit=jour/mois),
    - Bonding curve + secondary market (activé après 'secondary_market_start'),
    - Ajout de fees (protocol, treasury),
    - Comportements utilisateurs (vente % par an => on répartit par mois),
    - Vesting / unlock de RBOT,
    - Mise à jour du prix RBOT et supply.

    Retourne un DataFrame (une ligne par période).
    """

    # Conversions si needed
    # ex. treasury_funding_objective_musd_year => mensuel
    # Si time_unit = "Months", on divise par 12. S'il s'agit de Days, par 365.
    if time_unit == "Months":
        periods_in_year = 12
    else:
        periods_in_year = 365

    monthly_funding_needed = (treasury_funding_objective_musd_year * 1_000_000) / periods_in_year
    # On transfère en "par période"

    # État initial
    rbot_price = init_rbot_price
    rbot_supply = init_rbot_supply
    ai_token_circ = 0.0

    protocol_fees_usd = 0.0
    treasury_balance_usd = 0.0

    records = []

    # Pour simplifier, le ratio annual => period pour la vente:
    # ex. 10% an => ~0.10/12 par mois ou 0.10/365 par jour
    stakeholder_sell_rate = (stakeholders_yearly_sales_percent / 100.0) / periods_in_year

    # On prépare un accès direct par (period -> row) pour transactions
    # Suppose transactions_df a 'period' comme colonne
    transactions_map = {}
    for _, row in transactions_df.iterrows():
        key = row['period']
        buy_ai = row.get('buy_ai', 0.0)
        sell_ai = row.get('sell_ai', 0.0)
        transactions_map[int(key)] = (buy_ai, sell_ai)

    # Idem pour vesting (month, unlocked)
    vesting_map = {}
    for _, rowv in vesting_df.iterrows():
        vtime = int(rowv['period'])
        vunlocked = rowv['unlocked']
        if vtime not in vesting_map:
            vesting_map[vtime] = 0.0
        vesting_map[vtime] += vunlocked

    # Boucle de simulation
    for t in range(1, total_periods+1):
        # 1) Vesting (RBOT)
        unlocked_today = vesting_map.get(t, 0.0)
        old_supply = rbot_supply
        rbot_supply += unlocked_today  # + tokens sur le marché => pression
        # On modélise un petit effet prix si on veut (voir plus bas)

        # 2) Lecture volume buy/sell depuis transactions_df
        buy_ai = 0.0
        sell_ai = 0.0
        if t in transactions_map:
            buy_ai, sell_ai = transactions_map[t]

        # 3) Stakeholders sales => ex. stakeholder_sell_rate * ai_token_circ
        additional_sell = stakeholder_sell_rate * ai_token_circ
        sell_ai += additional_sell

        # 4) Treasury funding => on simule qu'on vend AiToken ou on vend RBOT ?
        #   Ex.: On vend AiToken pour lever monthly_funding_needed => simplifié
        #   => convert (AiToken) -> $ ? On l'ajoute au sell_ai ?
        #   On fait un "TODO": c'est un exemple
        #   On suppose 1 AiToken = 1$ pour la démo, ou on calcule un prix ?
        #   Pour aller vite, on skip l'intégration dans le BC. Cf. la suite.
        #   (Tu peux l'implémenter selon ta logique.)
        # ...

        # 5) Si t < secondary_market_start => 100% sur BC
        #    Sinon => portion X sur BC, portion (1-X) sur secondary
        portion_bc = 1.0
        portion_sc = 0.0
        if t >= secondary_market_start:
            portion_bc = 1.0 - portion_secondary_after
            portion_sc = portion_secondary_after

        bc_buy = buy_ai * portion_bc
        bc_sell = sell_ai * portion_bc
        sc_buy = buy_ai * portion_sc
        sc_sell = sell_ai * portion_sc

        # 6) Calcul sur Bonding Curve
        net_bc = bc_buy - bc_sell
        old_ai = ai_token_circ
        new_ai = old_ai + net_bc
        if new_ai < 0:
            new_ai = 0
        # Montant RBOT $ requis
        rbot_bc = bonding_curve_rbot_for_tokens(old_ai, new_ai, p0, f)

        # Si rbot_bc>0 => on consomme du $ => prise en compte des fees
        cost_usd = max(0, rbot_bc)
        # Fees
        protoc_fee = cost_usd * (protocol_fee_percent / 100.0)
        treas_fee = cost_usd * (treasury_fee_percent / 100.0)
        protocol_fees_usd += protoc_fee
        treasury_balance_usd += treas_fee

        # maj circ
        ai_token_circ = new_ai

        # 7) Secondary Market (placeholder)
        # sc_buy, sc_sell => on ne calcule pas le slippage exact
        # on peut juste logger
        # On pourrait imaginer un "rbot_sc" +/- si on implémente un AMM

        # 8) Update rbot_price (effet supply + rbot_bc)
        # ex. delta = (rbot_bc / 10000) - (unlocked_today / 20000)
        # ou usage d'elasticsearch
        delta_price = (rbot_bc / 10000.0) - (unlocked_today / 20000.0)
        rbot_price = max(0.000001, rbot_price + delta_price)

        # on peut imaginer un second effet si ai_token_circ bouge, etc.

        # 9) Stocker
        records.append({
            'scenario': scenario_name,
            time_unit: t,
            'Unlocked_RBOT': unlocked_today,
            'RBOT_Supply': rbot_supply,
            'RBOT_Price': rbot_price,
            'AiToken_Circ': ai_token_circ,
            'Buy_BC': bc_buy,
            'Sell_BC': bc_sell,
            'RBOT_BC_Used': rbot_bc,
            'Protocol_Fees_USD': protocol_fees_usd,
            'Treasury_USD': treasury_balance_usd,
            'Secondary_Buy': sc_buy,
            'Secondary_Sell': sc_sell
        })

    return pd.DataFrame(records)


# =========================
#   STREAMLIT UI
# =========================

def app():
    st.title("Simulation : Bonding Curve, Secondary Market, Vesting, Protocol Fees, etc.")

    # 1) Scénario
    scenario_name = st.text_input("Scenario Name", "Scenario_1")

    # 2) Time unit
    time_unit = st.radio("Time Unit", ["Days", "Months"], index=1)

    # 3) Nombre total de périodes
    total_periods = st.number_input(f"Total {time_unit} to simulate", 1, 5000, 36)

    # 4) nb_agents (int)
    nb_agents = st.number_input("Number of AI Agents (int)", 0, 1000000, 10, step=1)

    # Bonding curve
    st.subheader("Bonding Curve Params")
    p0 = st.number_input("p0 (initial price for 1st token)", 0.000001, 1000.0, 1.0, 0.1)
    f = st.number_input("f (exponent factor)", 0.9999, 2.0, 1.0001, 0.0001)

    # Secondary Market
    st.subheader("Secondary Market")
    secondary_market_start = st.number_input(f"Enable Secondary Market after X {time_unit}", 1, 9999, 12)
    portion_secondary_after = st.slider("Fraction going to Secondary once active", 0.0, 1.0, 0.5, 0.1)

    # Fees
    st.subheader("Protocol / Treasury Fees (%)")
    protocol_fee_percent = st.slider("Protocol Fee %", 0.0, 10.0, 1.0, 0.1)
    treasury_fee_percent = st.slider("Treasury Fee %", 0.0, 10.0, 1.0, 0.1)

    # Behaviors
    st.subheader("Users Behaviors / Funding")
    treasury_funding_objective_musd_year = st.number_input("Treasury Funding Objective (M$ / year)", 0.0, 1000.0, 10.0)
    stakeholders_yearly_sales_percent = st.slider("Stakeholders yearly sales (%)", 0.0, 100.0, 10.0, 1.0)
    stakers_lock_months = st.number_input("Stakers average lock time", 0, 60, 12)

    # RBOT init
    st.subheader("RBOT Initial State")
    init_rbot_price = st.number_input("Initial RBOT Price ($)", 0.000001, 100000.0, 1.0)
    init_rbot_supply = st.number_input("Initial RBOT Supply", 0.0, 1e9, 1_000_000.0, 100_000.0)

    st.write("---")

    # Table Transactions
    st.subheader("Monthly (or Daily) Transactions Table")
    # Par défaut, on crée un DF "period, buy_ai, sell_ai"
    default_data = {
        'period': list(range(1, total_periods+1)),
        'buy_ai': [0.0]*total_periods,
        'sell_ai': [0.0]*total_periods
    }
    transactions_init = pd.DataFrame(default_data)
    st.markdown("Modifiez directement les colonnes `buy_ai` / `sell_ai` si besoin.")
    edited_transactions = st.data_editor(transactions_init, key="transaction_editor")

    # Vesting
    st.subheader("RBOT Vesting / Unlock Table")
    vesting_init = pd.DataFrame({'period': [12, 24], 'unlocked': [50000, 100000]})
    st.markdown("Ajoutez ou modifiez les lignes pour décrire le vesting RBOT.")
    edited_vesting = st.data_editor(vesting_init, key="vesting_editor")

    if st.button("Run Simulation"):
        df_result = simulate_protocol(
            scenario_name=scenario_name,
            total_periods=int(total_periods),
            time_unit=time_unit,
            nb_agents=int(nb_agents),
            p0=p0,
            f=f,
            secondary_market_start=secondary_market_start,
            portion_secondary_after=portion_secondary_after,
            protocol_fee_percent=protocol_fee_percent,
            treasury_fee_percent=treasury_fee_percent,
            treasury_funding_objective_musd_year=treasury_funding_objective_musd_year,
            stakeholders_yearly_sales_percent=stakeholders_yearly_sales_percent,
            stakers_lock_months=stakers_lock_months,
            transactions_df=edited_transactions,
            vesting_df=edited_vesting,
            init_rbot_price=init_rbot_price,
            init_rbot_supply=init_rbot_supply
        )

        st.write("## Résultats de la Simulation")
        st.dataframe(df_result.head(2000))  # ou st.dataframe(df_result) selon la taille

        # Graph : Buy / Sell BC
        fig_buy_sell = px.bar(
            df_result,
            x=df_result.columns[1],  # day/month axis
            y=["Buy_BC","Sell_BC"],
            title="Dynamics of Buy vs Sell on Bonding Curve",
            barmode="group"
        )
        st.plotly_chart(fig_buy_sell, use_container_width=True)

        # Graph : RBOT Price
        fig_price = px.line(
            df_result,
            x=df_result.columns[1],
            y="RBOT_Price",
            title="RBOT Price Evolution"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Graph : Protocol Fees & Treasury
        fig_fees = px.line(
            df_result,
            x=df_result.columns[1],
            y=["Protocol_Fees_USD","Treasury_USD"],
            title="Protocol Fees & Treasury Balance (USD)"
        )
        st.plotly_chart(fig_fees, use_container_width=True)

        # Graph : AiToken in circulation
        fig_ai = px.line(
            df_result,
            x=df_result.columns[1],
            y="AiToken_Circ",
            title="AiToken in Circulation"
        )
        st.plotly_chart(fig_ai, use_container_width=True)


def main():
    app()

if __name__ == "__main__":
    main()
