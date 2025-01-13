import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def simulate_scenario(
    scenario_name,
    days,
    daily_growth,
    initial_volume,
    nb_ai_agents,
    bond_curve_slope,
    bond_curve_exponent,
    bc_to_trading_ratio,
    fee_burn_rbot,
    fee_liquidity_bc,
    fee_treasury,
    shock_day=None,
    shock_factor=1.0,
    initial_rbot_supply=1_000_000,
    elasticity=0.5
):
    """
    Simule:
    - Volume de transactions croissant,
    - Création d'AI Agents (impact sur demande RBOT, burn...),
    - Bonding Curve paramétrable (slope, exponent),
    - Répartition fees (burn, BC liquidity, treasury),
    - Seuil (bc_to_trading_ratio) pour l'allocation,
    - Eventuel choc de marché (shock_day, shock_factor),
    - Retourne un DataFrame avec variables clefs (burn, supply, price...).
    """
    df_records = []
    total_fee_rate = 0.01
    rbot_price = 1.0  # Hypothèse simple, ou on peut le modéliser plus complexe
    rbot_supply = initial_rbot_supply

    current_volume = initial_volume
    current_agents = 0

    for d in range(1, days + 1):
        # Croissance volume
        if d > 1:
            current_volume *= (1 + daily_growth)

        # Choc si demandé
        if shock_day and d == shock_day:
            current_volume *= shock_factor

        # Hypothèse: nouveau projet AI Agent chaque X jours => param "nb_ai_agents"
        # Simplification: On répartit la création sur la durée:
        agents_created_today = nb_ai_agents / days
        current_agents += agents_created_today

        # Bonding Curve (très simple) : si AI Agent se crée, on achète du RBOT => on calcule la "demande" sur RBOT
        # Ex: daily_buy_rbot ~ bond_curve_slope * (current_agents+1)**bond_curve_exponent (toy model)
        daily_buy_rbot = bond_curve_slope * (current_agents + 1)**bond_curve_exponent

        # On met à jour la supply RBOT si on suppose qu'une partie est burn:
        # Tout dépend du modèle: ici on imagine un burn direct proportionnel "fee_burn_rbot"
        # sur le volume "current_volume * total_fee_rate"
        daily_fees = current_volume * total_fee_rate
        daily_burn_rbot_usd = daily_fees * fee_burn_rbot
        daily_liquidity_bc_usd = daily_fees * fee_liquidity_bc
        daily_treasury_usd = daily_fees * fee_treasury

        # Converti burn USD => RBOT burné (si rbot_price>0)
        rbot_burned = (daily_burn_rbot_usd / rbot_price) if rbot_price > 0 else 0
        rbot_supply = max(0, rbot_supply - rbot_burned)

        # On imagine que daily_buy_rbot en USD => la demande fait monter un peu le prix:
        # Petit modèle: price ~ (some factor * daily_buy_rbot) / supply^elasticity
        # ou tout autre variation simplifiée.
        rbot_price_change = (daily_buy_rbot) / (rbot_supply**elasticity + 1e-9)
        rbot_price = max(0, rbot_price + rbot_price_change)

        # Trading Pool vs BC Liquidity (ratio bc_to_trading_ratio)
        # Ex: 0.2 => 20% BC, 80% Trading
        bc_liquidity = daily_liquidity_bc_usd * bc_to_trading_ratio
        trading_pool_part = daily_liquidity_bc_usd * (1 - bc_to_trading_ratio)

        df_records.append({
            'scenario': scenario_name,
            'day': d,
            'volume': current_volume,
            'agents_created_cumul': current_agents,
            'fees_total': daily_fees,
            'burn_rbot_usd': daily_burn_rbot_usd,
            'burn_rbot_tokens': rbot_burned,
            'bc_liquidity_usd': bc_liquidity,
            'trading_pool_usd': trading_pool_part,
            'treasury_usd': daily_treasury_usd,
            'rbot_supply': rbot_supply,
            'rbot_price': rbot_price
        })

    return pd.DataFrame(df_records)

def page_app():
    st.title("Digital Twin - Bonding Curve & Fees Simulation")

    st.sidebar.header("Simulation Inputs")

    # Input global
    days = st.sidebar.number_input("Days", 1, 365, 60)
    daily_growth = st.sidebar.slider("Daily Growth %", -5.0, 10.0, 1.0, 0.5) / 100.0
    initial_volume = st.sidebar.number_input("Initial Volume (USD)", 1_000, 10_000_000, 100_000, 10_000)
    nb_ai_agents = st.sidebar.number_input("Number of AI Agents to Create", 0, 10_000, 50, 10)

    # Bonding curve
    st.sidebar.subheader("Bonding Curve")
    bond_curve_slope = st.sidebar.number_input("BC Slope", 0.0, 100.0, 1.0, 0.1)
    bond_curve_exponent = st.sidebar.number_input("BC Exponent", 0.0, 2.0, 1.0, 0.1)

    # Ratio
    st.sidebar.subheader("Liquidity Distribution")
    bc_to_trading_ratio = st.sidebar.slider("BC Liquidity Ratio", 0.0, 1.0, 0.2, 0.05)

    # Fees rates (somme <=1, car total 1%)
    st.sidebar.subheader("Fees Distribution (1% total)")
    fee_burn_rbot = st.sidebar.slider("Burn RBOT fraction", 0.0, 1.0, 0.3, 0.05)
    fee_liquidity_bc = st.sidebar.slider("Liquidity fraction", 0.0, 1.0 - fee_burn_rbot, 0.3, 0.05)
    fee_treasury = st.sidebar.slider("Treasury fraction", 0.0, 1.0, 1.0 - (fee_burn_rbot + fee_liquidity_bc), 0.05)

    # Shock
    st.sidebar.subheader("Shock (Optional)")
    do_shock = st.sidebar.checkbox("Enable Shock")
    shock_day = None
    shock_factor = 1.0
    if do_shock:
        shock_day = st.sidebar.slider("Shock Day", 1, days, 10)
        shock_factor = st.sidebar.slider("Shock Factor", 0.1, 2.0, 0.5, 0.05)

    # Additional advanced
    st.sidebar.subheader("RBOT Model")
    initial_rbot_supply = st.sidebar.number_input("Initial RBOT Supply", 100_000, 10_000_000, 1_000_000, 100_000)
    elasticity = st.sidebar.slider("Price Elasticity", 0.1, 2.0, 0.5, 0.1)

    st.write("## Multiple Scenarios")

    n_scenarios = st.number_input("How many scenarios to compare?", 1, 5, 2)
    scenario_params = []
    for i in range(n_scenarios):
        with st.expander(f"Scenario {i+1} Params"):
            scen_name = st.text_input(f"Name for Scenario {i+1}", value=f"S{i+1}")
            # On peut personnaliser par scenario
            slope = st.number_input(f"BC Slope S{i+1}", 0.0, 100.0, bond_curve_slope, 0.1, key=f"slope_{i}")
            exponent = st.number_input(f"BC Exponent S{i+1}", 0.0, 2.0, bond_curve_exponent, 0.1, key=f"exp_{i}")
            burn_ = st.slider(f"Burn fraction S{i+1}", 0.0, 1.0, fee_burn_rbot, 0.05, key=f"burn_{i}")
            liq_ = st.slider(f"Liquidity fraction S{i+1}", 0.0, 1.0 - burn_, fee_liquidity_bc, 0.05, key=f"liq_{i}")
            trs_ = 1.0 - (burn_ + liq_)
            st.write(f"Treasury fraction S{i+1}: {trs_:.2f}")

            scenario_params.append({
                'scenario_name': scen_name,
                'days': days,
                'daily_growth': daily_growth,
                'initial_volume': initial_volume,
                'nb_ai_agents': nb_ai_agents,
                'bond_curve_slope': slope,
                'bond_curve_exponent': exponent,
                'bc_to_trading_ratio': bc_to_trading_ratio,
                'fee_burn_rbot': burn_,
                'fee_liquidity_bc': liq_,
                'fee_treasury': trs_,
                'shock_day': shock_day if do_shock else None,
                'shock_factor': shock_factor,
                'initial_rbot_supply': initial_rbot_supply,
                'elasticity': elasticity
            })

    if st.button("Run Simulation(s)"):
        df_all = []
        for sp in scenario_params:
            df_scen = simulate_scenario(**sp)
            df_all.append(df_scen)
        df_merged = pd.concat(df_all, ignore_index=True)
        st.write(df_merged.head(20))

        # Graph
        fig_burn = px.line(
            df_merged,
            x='day', 
            y='burn_rbot_usd', 
            color='scenario',
            title="Burn (USD) over time"
        )
        st.plotly_chart(fig_burn, use_container_width=True)

        fig_price = px.line(
            df_merged,
            x='day',
            y='rbot_price',
            color='scenario',
            title="RBOT Price over time"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        fig_supply = px.line(
            df_merged,
            x='day',
            y='rbot_supply',
            color='scenario',
            title="RBOT Supply over time"
        )
        st.plotly_chart(fig_supply, use_container_width=True)

        # Group
        final_stats = df_merged.groupby('scenario').agg({
            'rbot_price': 'last',
            'rbot_supply': 'last',
            'burn_rbot_tokens': 'sum',
            'bc_liquidity_usd': 'sum',
            'trading_pool_usd': 'sum',
            'treasury_usd': 'sum'
        }).rename(columns={
            'rbot_price': 'RBOT_Price_Final',
            'rbot_supply': 'RBOT_Supply_Final',
            'burn_rbot_tokens': 'Burned_RBOT_Tokens_Cumul',
            'bc_liquidity_usd': 'BC_Liq_Cumul',
            'trading_pool_usd': 'TradingPool_Cumul',
            'treasury_usd': 'Treasury_Cumul'
        })

        st.write("### Final Stats")
        st.dataframe(final_stats)

def main():
    page_app()

if __name__ == "__main__":
    main()
