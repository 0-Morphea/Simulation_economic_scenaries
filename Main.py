import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ======================================================
#   1) MODULE VESTING: CALCULER LA SUPPLY CIRC
# ======================================================

def compute_vesting_schedule(vesting_alloc_df, total_periods, time_unit):
    """
    Calcule, pour chaque période 1..total_periods, la quantité de tokens
    effectivement "circulants" (unlocked) selon la table d'allocation/vesting.

    vesting_alloc_df: table contenant
      - 'allocation_name'
      - 'total_tokens'
      - 'tge_percent'
      - 'lock_period' (nb de périodes)
      - 'vesting_period' (après lock, sur combien de périodes on libère)
    total_periods: nombre de périodes (mois ou jours)
    time_unit: "Days" ou "Months" (juste pour l'info)

    Retourne un array ou DataFrame de taille [1..total_periods],
    'circ_supply[t]' = somme de tous les tokens unlockés jusqu'à t.
    """

    # On construit un array "unlocks_category[categ][t]"
    # puis on cumulera pour faire la supply totale.
    cat_list = vesting_alloc_df['allocation_name'].unique()
    max_t = total_periods

    # Dictionnaire: category -> array(t) of tokens unlocked each period
    unlocks = {}
    for cat in cat_list:
        unlocks[cat] = np.zeros(max_t+1)  # index 1..max_t

    for i, row in vesting_alloc_df.iterrows():
        cat = row['allocation_name']
        total_tokens = float(row['total_tokens'])
        tge_percent = float(row['tge_percent'])/100.0
        lock_p = int(row['lock_period'])
        vest_p = int(row['vesting_period'])

        # TGE unlocked
        tge_amount = total_tokens * tge_percent
        # Le reste => locked, puis vested sur vest_p
        locked_amount = total_tokens - tge_amount

        # On pose t=1 comme "post TGE". Donc on débloque tge_amount à t=1
        if 1 <= max_t:
            unlocks[cat][1] += tge_amount

        # lock_p périodes sans rien débloquer de la partie "locked_amount".
        # Ensuite, on répartit "locked_amount" sur vest_p périodes, ex. linéairement
        # Soit "locked_amount / vest_p" à chaque step.
        if vest_p > 0:
            monthly_unlocked = locked_amount / vest_p
            start_vest = lock_p+1  # après lock_p périodes => la 1ère vest
            end_vest = lock_p + vest_p

            # On répartit sur [start_vest..end_vest]
            for t in range(start_vest, end_vest+1):
                if t <= max_t:
                    unlocks[cat][t] += monthly_unlocked
        else:
            # si vest_p=0 => tout locked_amount n'est jamais libéré (ou c'est auto?)
            # ou alors c'est direct après lock si c'est le design ?
            pass

    # Ensuite, on calcule "cumulative sum" par cat => supply cat[t]
    # Puis on somme par cat => total supply[t]
    supply_each_cat = {}
    for cat in cat_list:
        cumsum_cat = np.cumsum(unlocks[cat])
        supply_each_cat[cat] = cumsum_cat

    total_supply = np.zeros(max_t+1)
    for cat in cat_list:
        total_supply += supply_each_cat[cat]

    # On fait un DataFrame final
    data = {'period': range(1, max_t+1), 'circ_supply': total_supply[1:]}
    df_supply = pd.DataFrame(data)
    return df_supply  # 2 colonnes: period, circ_supply


# ======================================================
#   2) MODULE AMM / HELPER
# ======================================================
def amm_buy_sc_tokens(buy_tokens, pool_rbot, pool_ai, invariant_k):
    # identique à l'exemple précédent
    if buy_tokens <= 0 or pool_ai <= 0:
        return (0.0, pool_rbot, pool_ai, 0.0, None, None)
    old_price = pool_rbot/pool_ai if pool_ai>0 else 1e9

    new_pool_ai = pool_ai - buy_tokens
    if new_pool_ai < 1e-9:
        buy_tokens = pool_ai
        new_pool_ai = 1e-9
    new_pool_rbot = invariant_k / new_pool_ai
    rbot_in = new_pool_rbot - pool_rbot
    new_price = new_pool_rbot / new_pool_ai
    slippage = (new_price - old_price)/old_price if old_price>0 else 0

    return (rbot_in, new_pool_rbot, new_pool_ai, slippage, old_price, new_price)

def amm_sell_sc_tokens(sell_tokens, pool_rbot, pool_ai, invariant_k):
    if sell_tokens <= 0 or pool_ai <= 0:
        return (0.0, pool_rbot, pool_ai, 0.0, None, None)
    old_price = pool_rbot/pool_ai if pool_ai>0 else 1e9

    new_pool_ai = pool_ai + sell_tokens
    new_pool_rbot = invariant_k / new_pool_ai
    rbot_out = pool_rbot - new_pool_rbot
    new_price = new_pool_rbot / new_pool_ai
    slippage = (new_price - old_price)/old_price if old_price>0 else 0

    return (rbot_out, new_pool_rbot, new_pool_ai, slippage, old_price, new_price)

def bonding_curve_rbot_for_tokens(m_before, m_after, p0, f):
    if abs(f-1.0)<1e-9:
        return p0*(m_after-m_before)
    return p0*((f**m_after)-(f**m_before)) / (f-1)


# ======================================================
#   3) SIMULATION SCENARIO
# ======================================================
def simulate_scenario(
    scenario_name,
    scenario_id,
    total_periods,
    time_unit,
    # On reçoit la courbe 'circulating_supply_df'
    # => On saura combien de tokens sont "max" en circulation
    circ_supply_df,
    # Bonding Curve
    p0, f,
    # Activation secondary
    bc_liquidity_threshold,
    pool_percentage,
    secondary_market_start,
    portion_secondary_after,
    # Fees
    protocol_fee_percent,
    treasury_fee_percent,
    # Behaviors
    treasury_funding_objective_musd_year,
    stakeholders_yearly_sales_percent,
    # Table transactions
    transactions_df,
):
    if time_unit=="Months":
        periods_in_year = 12
    else:
        periods_in_year = 365
    monthly_funding_needed = (treasury_funding_objective_musd_year*1_000_000)/periods_in_year
    stakeholder_sell_rate = (stakeholders_yearly_sales_percent/100.0)/periods_in_year

    # ETAT
    ai_token_circ = 0.0   # on va se caler sur circ_supply_df[t] = maximum possible
    protocol_fees_usd = 0.0
    treasury_usd = 0.0
    bc_liquidity_usd = 0.0
    secondary_active = False
    pool_rbot = 0.0
    pool_ai = 0.0
    invariant_k = 0.0

    # Mapping Tx
    tx_map = {}
    for _, row in transactions_df.iterrows():
        per = int(row['period'])
        buy_ = float(row.get('buy_ai', 0.0))
        sell_ = float(row.get('sell_ai',0.0))
        tx_map[per] = (buy_, sell_)

    records = []
    # Au lieu d'un "rbot_supply", on est sur AiToken => "circ_supply_df"
    # On suppose la Bonding Curve index m_before -> m_after sur AiToken

    # On stocke un "m_current" = AiTokens "vendus" sur la BC
    # => c'est le cumul minted via BC, potentiellement.
    m_current = 0.0

    for t in range(1, total_periods+1):
        # 1) On lit la supply "max" autorisée ce mois/ jour
        # => ex. circ_supply_df.at[t, 'circ_supply']
        #   si t n'existe pas, on fait un get ?
        # ...
        row_supp = circ_supply_df[circ_supply_df['period']==t]
        if row_supp.empty:
            # plus grand que maxT => 0
            max_circ = 0.0
        else:
            max_circ = float(row_supp.iloc[0]['circ_supply'])

        # L'AI Token "maximum" en circulation qu'on NE dépasse pas => m_current <= max_circ
        # => On va simuler buy/sell sur BC => m_after <= max_circ
        # 2) Transactions
        buy_ai, sell_ai = tx_map.get(t, (0.0,0.0))

        # 3) On applique un stakeholder sale rate => p.ex. +sell
        #   Mais ! On ne peut pas vendre plus que la portion "circulante".
        #   Si m_current = 500k minted, on ne peut pas exceed that if not staker locked
        #   A simplifier => on rajoute un extra % de m_current
        extra_sell = stakeholder_sell_rate * m_current
        total_sell = sell_ai + extra_sell

        # 4) On calcule la part BC vs SC
        portion_bc = 1.0
        portion_sc = 0.0
        if secondary_active:
            portion_sc = portion_secondary_after
            portion_bc = 1.0 - portion_sc

        bc_buy = buy_ai * portion_bc
        bc_sell = total_sell * portion_bc
        sc_buy = buy_ai * portion_sc
        sc_sell = total_sell * portion_sc

        # === BONDING CURVE
        old_m = m_current
        net_bc = bc_buy - bc_sell
        new_m = old_m + net_bc
        # On borne new_m <= max_circ
        if new_m<0: new_m=0
        if new_m>max_circ: new_m=max_circ

        rbot_bc = bonding_curve_rbot_for_tokens(old_m, new_m, p0, f)
        cost_bc_usd = max(rbot_bc,0)
        # fees
        protoc_fee = cost_bc_usd*(protocol_fee_percent/100.0)
        treas_fee = cost_bc_usd*(treasury_fee_percent/100.0)
        protocol_fees_usd += protoc_fee
        treasury_usd += treas_fee
        bc_liquidity_usd += cost_bc_usd
        m_current = new_m  # minted AiTokens via BC

        # TRIGGER secondary if conditions
        # - bc_liquidity_usd >= threshold
        # - t >= secondary_market_start
        if (not secondary_active) and (bc_liquidity_usd>=bc_liquidity_threshold) and (t>=secondary_market_start):
            secondary_active = True
            portion_for_pool = bc_liquidity_usd*pool_percentage
            bc_liquidity_usd -= portion_for_pool
            # init AMM => moit/moit
            # ex. Ai price = p0*f^m_current (approx)
            ai_price = p0*(f**(m_current)) if m_current>0 else p0
            pool_rbot = portion_for_pool/2
            pool_ai = (portion_for_pool/2)/ai_price if ai_price>0 else 0
            invariant_k = pool_rbot*pool_ai

        # === SECONDARY AMM
        slippage_sc=0.0
        if secondary_active:
            # sc_buy => amm_buy_sc_tokens
            if sc_buy>0:
                (rbot_in,new_rbot,new_ai,slip,oldp, newp) = amm_buy_sc_tokens(
                    sc_buy, pool_rbot, pool_ai, invariant_k
                )
                if rbot_in>0:
                    pool_rbot, pool_ai = new_rbot, new_ai
                    slippage_sc = slip
            if sc_sell>0:
                (rbot_out,new_rbot,new_ai,slip,oldp,newp) = amm_sell_sc_tokens(
                    sc_sell, pool_rbot, pool_ai, invariant_k
                )
                if rbot_out>0:
                    pool_rbot, pool_ai = new_rbot, new_ai
                    slippage_sc = slip

        # On stocke
        records.append({
            'scenario_id': scenario_id,
            'scenario_name': scenario_name,
            time_unit: t,
            'BC_minted': m_current,
            'BC_Liquidity_USD': bc_liquidity_usd,
            'Secondary_Active': secondary_active,
            'Pool_RBOT': pool_rbot,
            'Pool_AI': pool_ai,
            'Slippage_SC': slippage_sc,
            'Protocol_Fees_USD': protocol_fees_usd,
            'Treasury_USD': treasury_usd,
            'MaxCirc': max_circ
        })

    return pd.DataFrame(records)


# ======================================================
#   4) STREAMLIT APP
# ======================================================
def page_app():
    st.title("Tokenomics Simulator with Allocation/Vesting + BC + Secondary")

    # --------------------------------------------------
    # 4.1 Table d'Allocation/Vesting (Input initial)
    # --------------------------------------------------
    st.header("1) Token Allocation & Vesting Table")
    st.write("Définir la répartition (allocation_name, total_tokens, tge_percent, lock_period, vesting_period, etc.).")

    default_vest_alloc = pd.DataFrame({
        'allocation_name': ["Team","Private Sale","Public Sale","Treasury"],
        'total_tokens': [1000000, 2000000, 3000000, 4000000],
        'tge_percent': [10,25,100,100],
        'lock_period': [6,0,0,0],      # en nb de périodes
        'vesting_period':[18,12,0,0], # en nb de périodes
    })
    st.markdown("**Table d'allocation/vesting** (exemple) :")
    edited_alloc_df = st.data_editor(default_vest_alloc, key="alloc_editor")

    st.markdown("En cliquant sur `Run Vesting`, on génère la courbe de supply circulante.")

    colA, colB = st.columns(2)
    with colA:
        time_unit_vest = st.selectbox("Time Unit for Vesting (Days / Months)", ["Days","Months"], index=1)
    with colB:
        total_periods_vest = st.number_input("Nb total periods for vesting schedule", 1, 5000, 36)

    if st.button("Run Vesting Computation"):
        df_supply = compute_vesting_schedule(edited_alloc_df, total_periods_vest, time_unit_vest)
        st.write("**Résultat - Courbe de supply** (circulante) :")
        st.dataframe(df_supply.head(50))

        fig_vest = px.line(df_supply, x='period', y='circ_supply', title="Circulating Supply Over Time")
        st.plotly_chart(fig_vest, use_container_width=True)

        st.success("Vesting schedule computed. We'll reuse this for the scenario simulation.")

        # On stocke dans session_state
        st.session_state['df_supply_vesting'] = df_supply
        st.session_state['time_unit_vest'] = time_unit_vest

    st.write("---")

    # --------------------------------------------------
    # 4.2 Multi-Scénarios
    # --------------------------------------------------
    st.header("2) Configure Multi-Scenarios & Transactions")

    if 'df_supply_vesting' not in st.session_state:
        st.warning("Compute the vesting schedule first to have a 'df_supply_vesting' available.")
        return

    df_vest_ready = st.session_state['df_supply_vesting']
    time_unit_vest_ready = st.session_state['time_unit_vest']

    st.sidebar.title("Scenarios Config")
    n_scen = st.sidebar.number_input("Number of Scenarios", 1, 5, 1)

    scenarios_param = []
    for i in range(n_scen):
        with st.sidebar.expander(f"Scenario {i+1}"):
            scenario_name = st.text_input(f"Name (S{i+1})", value=f"S{i+1}")
            # On peut choisir un "total_periods" pour la simulation
            total_periods_sim = st.number_input(f"Sim Periods (S{i+1})", 1, 5000, int(df_vest_ready['period'].max()), key=f"simp_{i}")
            bc_thresh = st.number_input(f"BC Liquidity Threshold $ (S{i+1})", 0.0, 1e9, 1_000_000.0, key=f"thresh_{i}")
            pool_perc = st.slider(f"Pool % for AMM (S{i+1})", 0.0,1.0,0.2,0.05, key=f"pp_{i}")
            sm_start = st.number_input(f"Secondary Market earliest start (S{i+1})", 1, 9999, 12, key=f"ss_{i}")
            sc_portion = st.slider(f"Portion trades to SC after activation (S{i+1})", 0.0,1.0,0.5,0.05,key=f"scp_{i}")

            p0_ = st.number_input(f"p0 (S{i+1})", 0.000001, 1000.0, 1.0, key=f"p0_{i}")
            f_ = st.number_input(f"f exponent (S{i+1})", 0.9999, 2.0, 1.0001, 0.0001, key=f"f_{i}")

            proto_fee = st.slider(f"Protocol Fee % (S{i+1})", 0.0, 10.0, 1.0, 0.1, key=f"pf_{i}")
            treas_fee = st.slider(f"Treasury Fee % (S{i+1})", 0.0, 10.0, 1.0, 0.1, key=f"tf_{i}")

            tf_obj = st.number_input(f"Treasury Funding M$/year (S{i+1})", 0.0, 1e3, 10.0, key=f"tf_{i}")
            stake_sales = st.slider(f"Stakeholders yearly sales % (S{i+1})", 0.0,100.0,10.0, key=f"st_{i}")

            scenarios_param.append({
                'scenario_id': i+1,
                'scenario_name': scenario_name,
                'total_periods_sim': int(total_periods_sim),
                'bc_liquidity_threshold': float(bc_thresh),
                'pool_percentage': float(pool_perc),
                'secondary_market_start': int(sm_start),
                'portion_secondary_after': float(sc_portion),
                'p0': float(p0_),
                'f': float(f_),
                'protocol_fee_percent': float(proto_fee),
                'treasury_fee_percent': float(treas_fee),
                'treasury_funding_objective_musd_year': float(tf_obj),
                'stakeholders_yearly_sales_percent': float(stake_sales),
            })

    st.subheader("Transactions Table (buy/sell AiToken)")
    maxT_possible = max(sp['total_periods_sim'] for sp in scenarios_param)
    default_tx = {
        'period': list(range(1, maxT_possible+1)),
        'buy_ai': [0.0]*maxT_possible,
        'sell_ai': [0.0]*maxT_possible
    }
    tx_init_df = pd.DataFrame(default_tx)
    st.markdown("Modifier si besoin :")
    edited_tx_df = st.data_editor(tx_init_df, key="tx_editor")

    if st.button("Run Scenarios"):
        # On run la simulation pour chaque scenario
        # On a un df_vest_ready => (period, circ_supply)
        # On peut le tronquer si needed
        all_results = []

        for sp in scenarios_param:
            # on tronque df_vest_ready au 'sp['total_periods_sim']'
            sub_vest = df_vest_ready[df_vest_ready['period']<=sp['total_periods_sim']].copy()
            sub_tx = edited_tx_df[edited_tx_df['period']<=sp['total_periods_sim']].copy()

            df_res = simulate_scenario(
                scenario_name=sp['scenario_name'],
                scenario_id=sp['scenario_id'],
                total_periods=sp['total_periods_sim'],
                time_unit=time_unit_vest_ready,  # on réutilise celui défini pour la vesting
                circ_supply_df=sub_vest,
                p0=sp['p0'],
                f=sp['f'],
                bc_liquidity_threshold=sp['bc_liquidity_threshold'],
                pool_percentage=sp['pool_percentage'],
                secondary_market_start=sp['secondary_market_start'],
                portion_secondary_after=sp['portion_secondary_after'],
                protocol_fee_percent=sp['protocol_fee_percent'],
                treasury_fee_percent=sp['treasury_fee_percent'],
                treasury_funding_objective_musd_year=sp['treasury_funding_objective_musd_year'],
                stakeholders_yearly_sales_percent=sp['stakeholders_yearly_sales_percent'],
                transactions_df=sub_tx,
            )
            all_results.append(df_res)

        df_merged = pd.concat(all_results, ignore_index=True)
        st.write("### Résultats Simulation (Tous Scénarios)")
        st.dataframe(df_merged.head(1000))

        # Graphs
        time_col = df_merged.columns[2]  # 'Days' or 'Months'
        fig_bc_liq = px.line(
            df_merged, x=time_col, y='BC_Liquidity_USD',
            color='scenario_name',
            title="Bonding Curve Liquidity (USD) Over Time"
        )
        st.plotly_chart(fig_bc_liq, use_container_width=True)

        fig_slip = px.line(
            df_merged, x=time_col, y='Slippage_SC',
            color='scenario_name',
            title="Slippage on Secondary AMM"
        )
        st.plotly_chart(fig_slip, use_container_width=True)

        st.success("Simulation terminée. Explorez les tableaux et graphiques !")


def main():
    page_app()

if __name__=="__main__":
    main()
