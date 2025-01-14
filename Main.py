import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ======================
#   AMM / Helper
# ======================

def amm_buy_sc_tokens(buy_tokens, pool_rbot, pool_ai, invariant_k):
    """
    L'utilisateur veut ACQUÉRIR "buy_tokens" AiTokens
    AMM x * y = k => new_pool_ai = pool_ai - buy_tokens
                  new_pool_rbot = k / new_pool_ai
    rbot_in = new_pool_rbot - pool_rbot (positif si user doit apporter)
    Retourne (rbot_in, new_pool_rbot, new_pool_ai, slippage, price_before, price_after).
    """
    if buy_tokens <= 0 or pool_ai <= 0:
        return (0.0, pool_rbot, pool_ai, 0.0, None, None)
    old_price = pool_rbot / pool_ai if pool_ai>0 else 1e9

    new_pool_ai = pool_ai - buy_tokens
    if new_pool_ai < 1e-9:
        # On limite buy_tokens = pool_ai
        buy_tokens = pool_ai
        new_pool_ai = 1e-9

    new_pool_rbot = invariant_k / new_pool_ai
    rbot_in = new_pool_rbot - pool_rbot

    new_price = new_pool_rbot / new_pool_ai if new_pool_ai>0 else 1e9
    slippage = (new_price - old_price)/old_price if old_price>0 else 0

    return (rbot_in, new_pool_rbot, new_pool_ai, slippage, old_price, new_price)

def amm_sell_sc_tokens(sell_tokens, pool_rbot, pool_ai, invariant_k):
    """
    L'utilisateur VEUT VENDRE "sell_tokens" AiTokens
    new_pool_ai = pool_ai + sell_tokens
    new_pool_rbot = k / new_pool_ai
    rbot_out = pool_rbot - new_pool_rbot
    Retourne (rbot_out, new_pool_rbot, new_pool_ai, slippage, price_before, price_after).
    """
    if sell_tokens <= 0 or pool_ai <= 0:
        return (0.0, pool_rbot, pool_ai, 0.0, None, None)
    old_price = pool_rbot / pool_ai if pool_ai>0 else 1e9

    new_pool_ai = pool_ai + sell_tokens
    new_pool_rbot = invariant_k / new_pool_ai
    rbot_out = pool_rbot - new_pool_rbot

    new_price = new_pool_rbot / new_pool_ai if new_pool_ai>0 else 1e9
    slippage = (new_price - old_price)/old_price if old_price>0 else 0

    return (rbot_out, new_pool_rbot, new_pool_ai, slippage, old_price, new_price)

def bonding_curve_rbot_for_tokens(m_before, m_after, p0, f):
    """
    B(m,n) = p0 * (f^n - f^m) / (f - 1)
    Montant en $ requis (positif) ou rendu (négatif) selon si on buy ou sell.
    """
    if abs(f - 1.0) < 1e-9:
        return p0 * (m_after - m_before)
    return p0 * ((f**m_after) - (f**m_before)) / (f - 1)


# ======================
#  Simulation
# ======================

def simulate_scenario(
    scenario_name,
    scenario_id,
    total_periods,
    time_unit,
    # RBOT init
    init_rbot_price,
    init_rbot_supply,
    vesting_df,
    # Bonding Curve
    p0,
    f,
    # Activation secondary
    bc_liquidity_threshold,    # Montant en $ requis pour activer la pool
    pool_percentage,           # fraction de bc_liquidity_usd qu'on dépose une fois
    # once activated
    secondary_market_start,    # par ex. un start minimal en plus
    # portion sc (répartition du flux)
    portion_secondary_after,
    # Fees
    protocol_fee_percent,
    treasury_fee_percent,
    # Behaviors
    treasury_funding_objective_musd_year,
    stakeholders_yearly_sales_percent,
    stakers_lock_months,
    # Table transactions
    transactions_df,
):
    if time_unit == "Months":
        periods_in_year = 12
    else:
        periods_in_year = 365

    monthly_funding_needed = (treasury_funding_objective_musd_year * 1_000_000)/periods_in_year
    stakeholder_sell_rate = (stakeholders_yearly_sales_percent/100.0)/periods_in_year

    # État initial
    rbot_price = init_rbot_price
    rbot_supply = init_rbot_supply
    ai_token_circ = 0.0

    protocol_fees_usd = 0.0
    treasury_usd = 0.0

    # Liquidity sur BC
    bc_liquidity_usd = 0.0
    secondary_active = False

    # Pool AMM
    pool_rbot = 0.0
    pool_ai = 0.0
    invariant_k = 0.0

    # Mapping transactions
    tx_map = {}
    for _, row in transactions_df.iterrows():
        key = int(row['period'])
        buy_ = float(row.get('buy_ai', 0.0))
        sell_ = float(row.get('sell_ai', 0.0))
        tx_map[key] = (buy_, sell_)

    # Vesting
    vest_map = {}
    for _, vrow in vesting_df.iterrows():
        t_v = int(vrow['period'])
        unlocked = float(vrow['unlocked'])
        vest_map[t_v] = vest_map.get(t_v, 0.0) + unlocked

    records = []

    for t in range(1, total_periods+1):

        # 1) Vesting
        unlocked_today = vest_map.get(t, 0.0)
        rbot_supply += unlocked_today

        # 2) Transactions
        buy_ai, sell_ai = tx_map.get(t, (0.0, 0.0))

        # 3) Stakeholders sales
        extra_sell = stakeholder_sell_rate * ai_token_circ
        sell_ai += extra_sell

        # 4) part sur BC vs SC
        portion_bc = 1.0
        portion_sc = 0.0
        # on check si (t >= secondary_market_start) et if secondary_active => qu'on enverra portion_sc
        # Mais on doit d'abord check si bc_liquidity_usd >= bc_liquidity_threshold pour activer
        if t >= secondary_market_start and secondary_active:
            portion_sc = portion_secondary_after
            portion_bc = 1.0 - portion_sc

        # =============== BONDING CURVE PART ===============
        bc_buy = buy_ai * portion_bc
        bc_sell = sell_ai * portion_bc

        old_ai = ai_token_circ
        net_bc = bc_buy - bc_sell
        new_ai = old_ai + net_bc
        if new_ai < 0:
            new_ai = 0

        # Montant $ RBOT pour BC
        rbot_bc = bonding_curve_rbot_for_tokens(old_ai, new_ai, p0, f)
        cost_bc_usd = max(rbot_bc, 0)
        # Fees
        protoc_fee = cost_bc_usd*(protocol_fee_percent/100.0)
        treas_fee = cost_bc_usd*(treasury_fee_percent/100.0)
        protocol_fees_usd += protoc_fee
        treasury_usd += treas_fee

        # On accumule la "liquidité BC" => Suppose 100% du cost_bc_usd y va
        bc_liquidity_usd += cost_bc_usd

        # maj circ
        ai_token_circ = new_ai

        # =============== TRIGGER ACTIVATION SECONDARY ===============
        if (not secondary_active) and (bc_liquidity_usd >= bc_liquidity_threshold) and (t >= secondary_market_start):
            # On active
            secondary_active = True
            portion_for_pool = bc_liquidity_usd * pool_percentage
            bc_liquidity_usd -= portion_for_pool
            # On convertit moit / moit => RBOT vs Ai
            # Il nous faut un "Ai price" => on peut prendre la BC price ou rbot_price
            # ex. bc_price = derivation ? On fait un approximatif:
            # P(n) = p0 * f^n => prix du dernier token => "apparent" price
            # on prend ai_price ~ p0 * f^(ai_token_circ) => simplification
            ai_price = p0*(f**(ai_token_circ)) if ai_token_circ>0 else p0
            pool_rbot = portion_for_pool / 2.0
            pool_ai = (portion_for_pool / 2.0) / ai_price if ai_price>0 else 0
            invariant_k = pool_rbot * pool_ai

        # =============== SECONDARY AMM PART ===============
        sc_buy = buy_ai*portion_sc
        sc_sell = sell_ai*portion_sc

        slippage_sc = 0.0
        price_before_sc = None
        price_after_sc = None

        if secondary_active:
            # sc_buy => amm_buy_sc_tokens
            if sc_buy>0:
                (rbot_in, new_rbot, new_ai_pool, slip, pbefore, pafter) = amm_buy_sc_tokens(
                    sc_buy, pool_rbot, pool_ai, invariant_k
                )
                if rbot_in>0:
                    # user doit dépenser rbot_in => c'est "des $" ?
                    pool_rbot, pool_ai = new_rbot, new_ai_pool
                    slippage_sc = slip
                    price_before_sc = pbefore
                    price_after_sc = pafter

            # sc_sell => amm_sell_sc_tokens
            if sc_sell>0:
                (rbot_out, new_rbot, new_ai_pool, slip, pbefore, pafter) = amm_sell_sc_tokens(
                    sc_sell, pool_rbot, pool_ai, invariant_k
                )
                if rbot_out>0:
                    pool_rbot, pool_ai = new_rbot, new_ai_pool
                    slippage_sc = slip
                    price_before_sc = pbefore
                    price_after_sc = pafter

        # 5) Mettre à jour rbot_price (toy model)
        delta_price = (rbot_bc/10000.0) - (unlocked_today/20000.0)
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
            'BC_Liquidity_USD': bc_liquidity_usd,
            'Secondary_Active': secondary_active,
            'Secondary_Buy': sc_buy,
            'Secondary_Sell': sc_sell,
            'Slippage_SC': slippage_sc,
            'PriceBeforeSC': price_before_sc,
            'PriceAfterSC': price_after_sc,
            'Pool_RBOT': pool_rbot,
            'Pool_AI': pool_ai,
            'Protocol_Fees_USD': protocol_fees_usd,
            'Treasury_USD': treasury_usd
        })

    df = pd.DataFrame(records)
    return df


# ======================
#   STREAMLIT APP
# ======================

def page_app():
    st.title("Tokenomics Simulator with BC + Secondary Market Activation")

    st.sidebar.title("Global Configuration")
    n_scenarios = st.sidebar.number_input("Number of Scenarios", 1, 5, 1)

    scenarios_data = []
    for i in range(n_scenarios):
        with st.sidebar.expander(f"Scenario {i+1}"):
            scenario_name = st.text_input(f"Scenario Name {i+1}", value=f"Scen_{i+1}")
            time_unit = st.selectbox(f"Time Unit (Scen {i+1})", ["Days","Months"], index=1, key=f"tu_{i}")
            total_periods = st.number_input(f"Total {time_unit} (Scen {i+1})", 1, 5000, 36, key=f"tp_{i}")

            init_rbot_price = st.number_input(f"Init RBOT Price (Scen {i+1})", 0.000001, 1e5, 1.0, key=f"ir_{i}")
            init_rbot_supply = st.number_input(f"Init RBOT Supply (Scen {i+1})", 0.0, 1e9, 1_000_000.0, key=f"is_{i}")

            p0_ = st.number_input(f"p0 (Scen {i+1})", 0.000001, 1_000.0, 1.0, key=f"p0_{i}")
            f_ = st.number_input(f"f exponent (Scen {i+1})", 0.9999, 2.0, 1.0001, 0.0001, key=f"f_{i}")

            bc_thresh = st.number_input(f"BC Liquidity Threshold ($) (Scen {i+1})", 0.0, 1e9, 1_000_000.0, key=f"bct_{i}")
            pool_perc = st.slider(f"Pool % for AMM (Scen {i+1})", 0.0, 1.0, 0.2, 0.05, key=f"pp_{i}")
            sm_start = st.number_input(f"Secondary Market earliest start (Scen {i+1})", 1, 9999, 12, key=f"ss_{i}")
            sc_portion = st.slider(f"Portion trades to SC after activation (Scen {i+1})", 0.0,1.0,0.5,0.05,key=f"scp_{i}")

            proto_fee = st.slider(f"Protocol Fee % (Scen {i+1})", 0.0, 10.0, 1.0, 0.1, key=f"pfee_{i}")
            treas_fee = st.slider(f"Treasury Fee % (Scen {i+1})", 0.0, 10.0, 1.0, 0.1, key=f"tfee_{i}")

            tf_obj = st.number_input(f"Treasury Funding M$/year (Scen {i+1})", 0.0, 1e3, 10.0, key=f"tf_{i}")
            stake_sales = st.slider(f"Stakeholders yearly sales % (Scen {i+1})", 0.0, 100.0, 10.0, key=f"sss_{i}")
            lock_m = st.number_input(f"Stakers Lock (Scen {i+1})", 0, 60, 12, key=f"lm_{i}")

            scenarios_data.append({
                'scenario_id': i+1,
                'scenario_name': scenario_name,
                'time_unit': time_unit,
                'total_periods': int(total_periods),
                'init_rbot_price': float(init_rbot_price),
                'init_rbot_supply': float(init_rbot_supply),
                'p0': float(p0_),
                'f': float(f_),
                'bc_liquidity_threshold': float(bc_thresh),
                'pool_percentage': float(pool_perc),
                'secondary_market_start': int(sm_start),
                'portion_secondary_after': float(sc_portion),
                'protocol_fee_percent': float(proto_fee),
                'treasury_fee_percent': float(treas_fee),
                'treasury_funding_objective_musd_year': float(tf_obj),
                'stakeholders_yearly_sales_percent': float(stake_sales),
                'stakers_lock_months': int(lock_m)
            })

    st.write("---")
    st.subheader("Transactions Table (buy/sell AiToken)")
    maxT = max(sd['total_periods'] for sd in scenarios_data)
    default_tx = {
        'period': list(range(1, maxT+1)),
        'buy_ai': [0.0]*maxT,
        'sell_ai': [0.0]*maxT
    }
    tx_init_df = pd.DataFrame(default_tx)
    st.write("Edit below (ex. monthly volumes).")
    edited_tx_df = st.data_editor(tx_init_df, key="editor_tx")

    st.subheader("Vesting Table (RBOT unlocks)")
    vest_init_df = pd.DataFrame({'period':[12,24,36],'unlocked':[50000,100000,200000]})
    st.write("Adjust if needed.")
    edited_vest_df = st.data_editor(vest_init_df, key="editor_vest")

    if st.button("Run Simulation"):
        results_all = []
        for sp in scenarios_data:
            sub_tx = edited_tx_df[ edited_tx_df['period'] <= sp['total_periods'] ]
            sub_vest = edited_vest_df[ edited_vest_df['period'] <= sp['total_periods'] ]

            df_res = simulate_scenario(
                scenario_name=sp['scenario_name'],
                scenario_id=sp['scenario_id'],
                total_periods=sp['total_periods'],
                time_unit=sp['time_unit'],
                init_rbot_price=sp['init_rbot_price'],
                init_rbot_supply=sp['init_rbot_supply'],
                vesting_df=sub_vest,
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
                stakers_lock_months=sp['stakers_lock_months'],
                transactions_df=sub_tx
            )
            results_all.append(df_res)

        df_merged = pd.concat(results_all, ignore_index=True)
        st.write("### Simulation Results")
        st.dataframe(df_merged.head(1000))

        # Plot ex
        time_col = df_merged.columns[2]  # "Days" or "Months"
        # (scenario_name col is [1], depends on indexing)

        # Price
        fig_price = px.line(
            df_merged,
            x=time_col,
            y='RBOT_Price',
            color='scenario_name',
            title="RBOT Price"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # BC Liquidity
        fig_bcliq = px.line(
            df_merged,
            x=time_col,
            y='BC_Liquidity_USD',
            color='scenario_name',
            title="Bonding Curve Liquidity (USD)"
        )
        st.plotly_chart(fig_bcliq, use_container_width=True)

        # Secondary Active
        # We can see when it becomes True
        st.write("Secondary Active Over Time (just a quick check):")
        st.write(df_merged[['scenario_name',time_col,'Secondary_Active']].drop_duplicates())

        # Slippage
        fig_slip = px.line(
            df_merged,
            x=time_col,
            y='Slippage_SC',
            color='scenario_name',
            title="Slippage sur le Secondary AMM"
        )
        st.plotly_chart(fig_slip, use_container_width=True)

        st.success("Simulation done.")


def main():
    page_app()

if __name__ == "__main__":
    main()
