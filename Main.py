import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

#=============================
#    LOGIC / MODEL
#=============================

def bonding_curve_rbot_for_tokens(m_before, m_after, p0, f):
    """
    Bonding Curve exponentielle simplifiée:
    B(m,n) = p0 * (f^n - f^m) / (f - 1)
    Retourne la quantité (positive => RBOT requis pour buy,
                           négative => RBOT remboursé si sell).
    """
    if abs(f - 1.0) < 1e-9:
        return p0 * (m_after - m_before)
    return p0 * ((f**m_after) - (f**m_before)) / (f - 1)

def simulate_scenario(
    scenario_name,
    # Paramètre de temps
    total_periods,
    time_unit,
    # Paramètre RBOT init + vesting
    init_rbot_price,
    init_rbot_supply,
    vesting_df,
    # Bonding Curve
    p0,
    f,
    # Secondary Market
    secondary_market_start,
    portion_secondary_after,
    # Fees
    protocol_fee_percent,
    treasury_fee_percent,
    # Comportements utilisateurs
    treasury_funding_objective_musd_year,
    stakeholders_yearly_sales_percent,
    stakers_lock_months,
    # Table transactions
    transactions_df,
    # Divers
    scenario_id=1, 
):
    """
    Simulation unifiée.
    - total_periods: par ex. 48 (mois sur 4 ans) ou 365 (jours)
    - time_unit: "Days", "Months"
    - init_rbot_price, init_rbot_supply: État initial du token RBOT
    - vesting_df: table vesting RBOT
    - p0,f => Bonding curve
    - secondary_market_start => après X periods on active un portion vers le secondary
    - portion_secondary_after => fraction du volume qu’on route sur le secondary
    - fees => protocol, treasury
    - treasury_funding_objective_musd_year => ventes mensuelles / journalières
    - stakeholders_yearly_sales_percent => part vendue par an
    - stakers_lock_months => on peut en tenir compte si on veut bloquer la vente d’une fraction
    - transactions_df => volumes buy/sell édités (colonnes: period, buy_ai, sell_ai)
    - scenario_id => identifiant pour tracer plus tard
    """

    # Conversion year->period
    if time_unit == "Months":
        periods_in_year = 12
    else:
        periods_in_year = 365

    monthly_funding_needed = (treasury_funding_objective_musd_year * 1_000_000) / periods_in_year
    stakeholder_sell_rate = (stakeholders_yearly_sales_percent / 100.0) / periods_in_year

    # État initial
    rbot_price = init_rbot_price
    rbot_supply = init_rbot_supply
    ai_token_circ = 0.0

    protocol_fees_usd = 0.0
    treasury_usd = 0.0

    # Mapping transactions
    tx_map = {}
    for _, row in transactions_df.iterrows():
        key = int(row['period'])
        buy_ = row.get('buy_ai', 0.0)
        sell_ = row.get('sell_ai', 0.0)
        tx_map[key] = (buy_, sell_)

    # Mapping vesting
    vest_map = {}
    for _, vrow in vesting_df.iterrows():
        t_v = int(vrow['period'])
        unlocked = float(vrow['unlocked'])
        vest_map[t_v] = vest_map.get(t_v, 0.0) + unlocked

    records = []
    for t in range(1, total_periods+1):
        # 1) Vesting RBOT
        unlocked_today = vest_map.get(t, 0.0)
        rbot_supply += unlocked_today  # Pressure sur le prix

        # 2) Transactions (buy/sell AiToken)
        buy_ai, sell_ai = tx_map.get(t, (0.0, 0.0))

        # 3) Stakeholders sales => ex. 0.XX % du AiToken circ => +sell
        stakeholder_extra = stakeholder_sell_rate * ai_token_circ
        sell_ai += stakeholder_extra

        # 4) Treasury funding => on pourrait simuler la vente d’AiToken ou RBOT
        #    pour lever monthly_funding_needed => skip pour simplifier
        #    ou l’ajouter à la dimension "sell_ai" etc.

        # 5) Secondary vs BC
        portion_bc = 1.0
        portion_sc = 0.0
        if t >= secondary_market_start:
            portion_sc = portion_secondary_after
            portion_bc = 1.0 - portion_sc

        bc_buy = buy_ai * portion_bc
        bc_sell = sell_ai * portion_bc
        sc_buy = buy_ai * portion_sc
        sc_sell = sell_ai * portion_sc

        # 6) Bonding Curve
        old_ai = ai_token_circ
        net_bc = bc_buy - bc_sell
        new_ai = old_ai + net_bc
        if new_ai < 0:
            new_ai = 0

        # Montant RBOT $
        rbot_bc = bonding_curve_rbot_for_tokens(old_ai, new_ai, p0, f)
        cost_usd = max(0, rbot_bc)
        protoc_fee = cost_usd * (protocol_fee_percent/100.0)
        treas_fee = cost_usd * (treasury_fee_percent/100.0)
        protocol_fees_usd += protoc_fee
        treasury_usd += treas_fee

        ai_token_circ = new_ai

        # 7) Update rbot_price (ex. toy model)
        #    On imagine un delta corrélé à rbot_bc + vesting
        delta_price = (rbot_bc / 10000.0) - (unlocked_today / 20000.0)
        rbot_price = max(0.000001, rbot_price + delta_price)

        records.append({
            'scenario_id': scenario_id,
            'scenario_name': scenario_name,
            time_unit: t,
            'Unlocked_RBOT': unlocked_today,
            'RBOT_Supply': rbot_supply,
            'RBOT_Price': rbot_price,
            'AiToken_Circ': ai_token_circ,
            'Buy_BC': bc_buy,
            'Sell_BC': bc_sell,
            'Secondary_Buy': sc_buy,
            'Secondary_Sell': sc_sell,
            'RBOT_BC_Amount': rbot_bc,
            'Protocol_Fees_USD': protocol_fees_usd,
            'Treasury_USD': treasury_usd
        })

    return pd.DataFrame(records)

#=============================
#    STREAMLIT APP (UI)
#=============================

def page_app():
    # Titre + style
    st.title("⚙️ Advanced Tokenomics Simulator")

    # Barre latérale (verticale)
    st.sidebar.title("Configuration Globale")
    # Nombre de scénarios
    n_scenarios = st.sidebar.number_input("Number of Scenarios", 1, 5, 1)

    # On va stocker les param de scénarios
    scenarios_params = []
    for i in range(n_scenarios):
        with st.sidebar.expander(f"Scenario {i+1}"):
            scenario_name = st.text_input(f"Scenario Name {i+1}", value=f"Scen_{i+1}")
            time_unit = st.selectbox(f"Time Unit (Scen {i+1})", ["Days","Months"], index=1, key=f"tu_{i}")
            total_periods = st.number_input(f"Total {time_unit} to simulate (S{i+1})", 1, 5000, 36, key=f"tp_{i}")

            st.subheader("Token RBOT Init & Vesting")
            init_rbot_price = st.number_input(f"Initial RBOT Price (S{i+1})", 0.000001, 100000.0, 1.0, key=f"rprice_{i}")
            init_rbot_supply = st.number_input(f"Initial RBOT Supply (S{i+1})", 0.0, 1e9, 1_000_000.0, key=f"rsup_{i}")

            st.subheader("Bonding Curve")
            p0 = st.number_input(f"p0 (S{i+1})", 0.000001, 1_000.0, 1.0, key=f"p0_{i}")
            f_ = st.number_input(f"f factor (S{i+1})", 0.9999, 2.0, 1.0001, 0.0001, key=f"f_{i}")

            st.subheader("Secondary Market")
            secondary_market_start = st.number_input(f"Enable SM after X {time_unit}", 1, 9999, 12, key=f"ss_{i}")
            portion_sc = st.slider(f"Portion to Secondary (S{i+1})", 0.0, 1.0, 0.5, 0.1, key=f"psc_{i}")

            st.subheader("Protocol & Treasury Fees (%)")
            proto_fee = st.slider(f"Protocol Fee (S{i+1})", 0.0, 10.0, 1.0, 0.1, key=f"pfe_{i}")
            treas_fee = st.slider(f"Treasury Fee (S{i+1})", 0.0, 10.0, 1.0, 0.1, key=f"tre_{i}")

            st.subheader("User Behaviors")
            tf_obj = st.number_input(f"Treasury Funding Objective M$/year (S{i+1})", 0.0, 1e3, 10.0, key=f"tfm_{i}")
            stake_sales = st.slider(f"Stakeholders yearly sales % (S{i+1})", 0.0, 100.0, 10.0, key=f"stak_{i}")
            lock_m = st.number_input(f"Stakers Lock {time_unit} (S{i+1})", 0, 60, 12, key=f"lock_{i}")

            scenarios_params.append({
                'scenario_id': i+1,
                'scenario_name': scenario_name,
                'time_unit': time_unit,
                'total_periods': int(total_periods),
                'init_rbot_price': float(init_rbot_price),
                'init_rbot_supply': float(init_rbot_supply),
                'p0': float(p0),
                'f': float(f_),
                'secondary_market_start': int(secondary_market_start),
                'portion_secondary_after': float(portion_sc),
                'protocol_fee_percent': float(proto_fee),
                'treasury_fee_percent': float(treas_fee),
                'treasury_funding_objective_musd_year': float(tf_obj),
                'stakeholders_yearly_sales_percent': float(stake_sales),
                'stakers_lock_months': int(lock_m),
            })

    st.write("---")
    st.subheader("Monthly (or Daily) Transactions Table")
    maxT = max(sp['total_periods'] for sp in scenarios_params)
    # On crée un DF "period, buy_ai, sell_ai"
    default_data = {
        'period': list(range(1, maxT+1)),
        'buy_ai': [0.0]*maxT,
        'sell_ai': [0.0]*maxT
    }
    tx_init_df = pd.DataFrame(default_data)
    st.markdown("Feel free to edit buy/sell volumes for each period.")
    edited_tx_df = st.data_editor(tx_init_df, key="editor_tx")

    st.subheader("Vesting Table for RBOT")
    vest_init_df = pd.DataFrame({'period': [12,24,36], 'unlocked': [50000,100000,200000]})
    st.markdown("Add or adjust lines for TGE, Team vesting, etc.")
    edited_vest_df = st.data_editor(vest_init_df, key="editor_vest")

    # Bouton
    if st.button("Run Simulation"):
        st.success("Simulation in progress...")

        all_results = []
        for sp in scenarios_params:
            # Tronquer les tables aux "total_periods"
            sub_tx = edited_tx_df[ edited_tx_df['period'] <= sp['total_periods'] ]
            sub_vest = edited_vest_df[ edited_vest_df['period'] <= sp['total_periods'] ]

            df_res = simulate_scenario(
                scenario_name=sp['scenario_name'],
                total_periods=sp['total_periods'],
                time_unit=sp['time_unit'],
                init_rbot_price=sp['init_rbot_price'],
                init_rbot_supply=sp['init_rbot_supply'],
                vesting_df=sub_vest,
                p0=sp['p0'],
                f=sp['f'],
                secondary_market_start=sp['secondary_market_start'],
                portion_secondary_after=sp['portion_secondary_after'],
                protocol_fee_percent=sp['protocol_fee_percent'],
                treasury_fee_percent=sp['treasury_fee_percent'],
                treasury_funding_objective_musd_year=sp['treasury_funding_objective_musd_year'],
                stakeholders_yearly_sales_percent=sp['stakeholders_yearly_sales_percent'],
                stakers_lock_months=sp['stakers_lock_months'],
                transactions_df=sub_tx,
                scenario_id=sp['scenario_id']
            )
            all_results.append(df_res)

        df_merged = pd.concat(all_results, ignore_index=True)

        st.write("### Simulation Results (All Scenarios)")
        st.dataframe(df_merged.head(1000))

        # On s’inspire de tes UI exemples pour tracer divers graphes

        # 1) Token Price & Market Cap
        # On simule un "market cap" = RBOT_Price * RBOT_Supply (toy model)
        df_merged['MarketCap_USD'] = df_merged['RBOT_Price'] * df_merged['RBOT_Supply']
        # On peut imaginer un FDV = Price * total supply max. 
        # Pour la démo, FDV ~ Price*(init supply+some big supply)
        df_merged['FDV'] = df_merged['RBOT_Price'] * (df_merged['RBOT_Supply'].max())

        time_col = df_merged.columns[2]  # "Days" ou "Months"

        fig_token_mcap = px.line(
            df_merged,
            x=time_col,
            y=['RBOT_Price','MarketCap_USD','FDV'],
            color='scenario_name',
            title="Token Price & Market Cap"
        )
        st.plotly_chart(fig_token_mcap, use_container_width=True)

        # 2) Token Buying & Selling Dynamics
        # net = (Buy_BC + Secondary_Buy) - (Sell_BC + Secondary_Sell)
        df_merged['Total_Buy'] = df_merged['Buy_BC'] + df_merged['Secondary_Buy']
        df_merged['Total_Sell'] = df_merged['Sell_BC'] + df_merged['Secondary_Sell']
        df_merged['Net_Buy_Sell'] = df_merged['Total_Buy'] - df_merged['Total_Sell']

        fig_dynamics = px.area(
            df_merged,
            x=time_col,
            y=['Total_Buy','Total_Sell','Net_Buy_Sell'],
            color='scenario_name',
            title="Token Buying & Selling Dynamics"
        )
        st.plotly_chart(fig_dynamics, use_container_width=True)

        # 3) Token Supply 
        # On a "AiToken_Circ" -> st.plot
        fig_supply = px.line(
            df_merged,
            x=time_col,
            y=['RBOT_Supply','AiToken_Circ'],
            color='scenario_name',
            title="Token Supply (RBOT) vs AiToken in Circulation"
        )
        st.plotly_chart(fig_supply, use_container_width=True)

        # 4) Annualized inflation (toy)
        # ex. yoy inflation = difference ratio
        # On peut calculer un yoy glissant, mais c'est plus complexe
        # On fait un simple ratio: inflation[t] = (Supply[t] - Supply[t-12]) / Supply[t-12]
        df_merged['Inflation'] = 0.0
        # Skipping real yoy approach for time
        # ...
        fig_inflation = px.line(
            df_merged,
            x=time_col,
            y='Inflation',
            color='scenario_name',
            title="Annualized YoY Inflation (toy example)"
        )
        st.plotly_chart(fig_inflation, use_container_width=True)

        # 5) Staked Tokens (optionnel)
        # Cf. user can interpret stakers_lock, etc. - skipping actual modeling
        # ...

        # 6) Protocol Fees, Treasury
        fig_fees = px.area(
            df_merged,
            x=time_col,
            y=['Protocol_Fees_USD','Treasury_USD'],
            color='scenario_name',
            title="Protocol Fees & Treasury PnL"
        )
        st.plotly_chart(fig_fees, use_container_width=True)

        st.write("Simulation Completed ✔️")


def main():
    page_app()


if __name__=="__main__":
    main()
