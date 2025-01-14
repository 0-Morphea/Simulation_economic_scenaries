import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================
#     VESTING MODULE
# ============================
def compute_vesting_schedule(vesting_alloc_df, total_periods, time_unit):
    """
    Calcule la supply circulante par période selon la table d'allocation/vesting.
    """
    cat_list = vesting_alloc_df['allocation_name'].unique()
    max_t = total_periods

    unlocks = {}
    for cat in cat_list:
        unlocks[cat] = np.zeros(max_t+1)  # index 1..max_t

    for i, row in vesting_alloc_df.iterrows():
        cat = row['allocation_name']
        total_tokens = float(row['total_tokens'])
        tge_percent = float(row['tge_percent'])/100.0
        lock_p = int(row['lock_period'])
        vest_p = int(row['vesting_period'])

        tge_amount = total_tokens * tge_percent
        locked_amount = total_tokens - tge_amount

        # TGE => t=1
        if 1 <= max_t:
            unlocks[cat][1] += tge_amount

        if vest_p > 0:
            monthly_unlocked = locked_amount/vest_p
            start_vest = lock_p+1
            end_vest = lock_p+vest_p
            for t in range(start_vest, end_vest+1):
                if t<=max_t:
                    unlocks[cat][t] += monthly_unlocked

    supply_each_cat = {}
    for cat in cat_list:
        cumsum_cat = np.cumsum(unlocks[cat])
        supply_each_cat[cat] = cumsum_cat

    total_supply = np.zeros(max_t+1)
    for cat in cat_list:
        total_supply += supply_each_cat[cat]

    df_supply = pd.DataFrame({
        'period': range(1, max_t+1),
        'circ_supply': total_supply[1:]
    })
    return df_supply


# ============================
#   FEES & BURN MODULE
# ============================
def distribute_fees(volume_usd, burn_ratio, liquidity_ratio, treasury_ratio, price_token):
    """
    volume_usd => le volume subissant 1% de fees
    burn_ratio + liqu_ratio + treas_ratio = 1.0 (somme <= 1)
    price_token => pour convertir burn_usd -> burn_tokens
    Retourne: burn_usd, burn_tokens, liquidity_usd, treasury_usd
    """
    fee_1pct = volume_usd*0.01
    burn_usd = fee_1pct*burn_ratio
    liqu_usd = fee_1pct*liquidity_ratio
    treas_usd = fee_1pct*treasury_ratio
    burn_tokens = 0.0
    if price_token>0:
        burn_tokens = burn_usd/price_token
    return burn_usd, burn_tokens, liqu_usd, treas_usd


# ============================
#   SIMULATION
# ============================
def simulate_scenario(
    scenario_name, scenario_id,
    total_periods,
    time_unit,
    # supply from vesting
    df_vest_supply,  # columns: period, circ_supply
    # volume & growth
    init_volume_usd,
    growth_percent,
    # fees distribution
    burn_ratio,
    liquidity_ratio,
    treasury_ratio,
    # to keep it short, we skip BC/AMM logic here, but you can re-insert them
):
    """
    Ex: On simule un volume de trading en $ qui grandit de growth_percent % par période,
    On applique 1% fees => burn, liqu, treasury. On calcule le burn nominal et
    on retire les tokens => impact sur supply => impact sur price (toy).
    """
    records = []
    supply_current = 0.0
    if not df_vest_supply.empty:
        supply_current = float(df_vest_supply.iloc[0]['circ_supply'])

    # On suppose un prix initial
    price_token = 1.0

    volume = init_volume_usd
    volume_list = []

    max_t = total_periods
    # on convert df_vest_supply to a map
    vest_map = {row['period']: row['circ_supply'] for _, row in df_vest_supply.iterrows()}

    for t in range(1, max_t+1):
        # 1) Maj supply circ (depuis vesting) => la supply possible
        supply_circ = vest_map.get(t, supply_current)
        supply_current = supply_circ

        # 2) On applique un volume => volume subit 1% fee
        # distribution fees
        burn_usd, burn_tokens, liqu_usd, treas_usd = distribute_fees(volume, burn_ratio, liquidity_ratio, treasury_ratio, price_token)

        # On retire burn_tokens de la supply => supply_current = supply_current - burn_tokens
        # si c'est <0 => on borne
        new_supply = supply_current - burn_tokens
        if new_supply<0:
            burn_tokens = supply_current  # on ne peut pas bruler plus
            new_supply = 0

        # Impact sur price (toy model): ex. Delta = burn_tokens / 1_000_000 ...
        # On peut faire un ratio d'offre & demande
        # On fait un truc simpliste
        price_delta = (burn_tokens/1e6) 
        new_price = max(0.000001, price_token + price_delta)

        # On stocke
        records.append({
            'scenario_name': scenario_name,
            'scenario_id': scenario_id,
            time_unit: t,
            'volume_usd': volume,
            'burn_usd': burn_usd,
            'burn_tokens': burn_tokens,
            'liquidity_usd': liqu_usd,
            'treasury_usd': treas_usd,
            'supply_before': supply_current,
            'supply_after': new_supply,
            'price_before': price_token,
            'price_after': new_price,
        })

        # update
        supply_current = new_supply
        price_token = new_price
        volume *= (1.0 + growth_percent/100.0)

    df_res = pd.DataFrame(records)
    return df_res


# ============================
#  STREAMLIT APP
# ============================
def page_app():
    st.title("Tokenomics with Vesting + Fees Distribution + Volume Growth + Burn Price Impact")

    # 1) Table vesting
    st.header("Step 1: Vesting Table (allocations)")
    default_vest_alloc = pd.DataFrame({
        'allocation_name': ["Team","Private","Public","Treasury"],
        'total_tokens': [1000000, 2000000, 3000000, 4000000],
        'tge_percent': [10,25,100,100],
        'lock_period':[6,0,0,0],
        'vesting_period':[18,12,0,0],
    })
    st.markdown("Modifiez la table si nécessaire")
    vest_alloc_edited = st.data_editor(default_vest_alloc, key="vesting_alloc_editor")

    colA, colB = st.columns(2)
    with colA:
        time_unit_vest = st.selectbox("Time Unit (Days or Months) for vesting",["Days","Months"], index=1)
    with colB:
        total_periods_vest = st.number_input("Total Periods for vesting schedule",1,5000,36)

    if st.button("Compute Vesting Schedule", key="compute_vesting"):
        df_vest = compute_vesting_schedule(vest_alloc_edited, total_periods_vest, time_unit_vest)
        st.session_state['df_vest'] = df_vest
        st.session_state['time_unit_vest'] = time_unit_vest
        st.success("Vesting schedule computed & stored.")
        st.dataframe(df_vest.head(50))
        fig_vest = px.line(df_vest, x='period', y='circ_supply', title="Vesting-based Circulating Supply")
        st.plotly_chart(fig_vest, use_container_width=True)

    st.write("---")

    # 2) Multi-scenarios
    st.header("Step 2: Multi-Scenarios & Fees distribution")

    if 'df_vest' not in st.session_state:
        st.warning("Compute the vesting schedule first.")
        return

    df_vesting_ready = st.session_state['df_vest']
    time_unit_ready = st.session_state['time_unit_vest']

    st.sidebar.title("Scenarios config")
    n_scen = st.sidebar.number_input("Number of Scenarios",1,5,1,key="nscen_key")

    scenarios_param = []
    for i in range(n_scen):
        with st.sidebar.expander(f"Scenario {i+1}"):
            sc_name = st.text_input(f"Name S{i+1}", value=f"Scenario_{i+1}", key=f"sc_name_{i}")
            periods_sim = st.number_input(f"Total Periods (S{i+1})",1,5000, int(df_vesting_ready['period'].max()), key=f"per_{i}_{sc_name}")
            # fees distribution: burn / liquidity / treasury => sum <=1
            st.markdown("**Repartition of 1% Fee**")
            burn_ratio = st.slider(f"Burn ratio (S{i+1})",0.0,1.0,0.3,0.05,key=f"b_{i}_{sc_name}")
            liqu_ratio = st.slider(f"Liquidity ratio (S{i+1})",0.0,1.0,0.3,0.05,key=f"l_{i}_{sc_name}")
            # on calcule 1 - burn - liqu => treasury
            treas_ratio = 1.0 - burn_ratio - liqu_ratio
            st.write(f"Treasury ratio ~ {treas_ratio:.2f} (auto)")

            init_volume = st.number_input(f"Initial Volume $ (S{i+1})",100.0,1e9,100000.0, key=f"iv_{i}_{sc_name}")
            growth_percent = st.slider(f"Growth volume % (S{i+1})", -10.0,50.0,5.0, key=f"g_{i}_{sc_name}")

            scenarios_param.append({
                'scenario_id': i+1,
                'scenario_name': sc_name,
                'total_periods_sim': int(periods_sim),
                'burn_ratio': burn_ratio,
                'liquidity_ratio': liqu_ratio,
                'treasury_ratio': treas_ratio,
                'init_volume_usd': float(init_volume),
                'growth_percent': float(growth_percent)
            })

    # 3) Run
    if st.button("Run All Scenarios", key="run_scen"):
        all_dfs = []
        for sp in scenarios_param:
            sub_vest = df_vesting_ready[df_vesting_ready['period']<=sp['total_periods_sim']].copy()

            df_res = simulate_scenario(
                scenario_name=sp['scenario_name'],
                scenario_id=sp['scenario_id'],
                total_periods=sp['total_periods_sim'],
                time_unit=time_unit_ready,
                df_vest_supply=sub_vest,
                init_volume_usd=sp['init_volume_usd'],
                growth_percent=sp['growth_percent'],
                burn_ratio=sp['burn_ratio'],
                liquidity_ratio=sp['liquidity_ratio'],
                treasury_ratio=sp['treasury_ratio']
            )
            all_dfs.append(df_res)

        df_merged = pd.concat(all_dfs, ignore_index=True)
        st.write("### Simulation Results")
        st.dataframe(df_merged.head(2000))

        tcol = df_merged.columns[2]  # 'Days' or 'Months'
        fig_burn = px.line(
            df_merged,
            x=tcol,
            y='burn_tokens',
            color='scenario_name',
            title="Burn Tokens (nominal) Over Time"
        )
        st.plotly_chart(fig_burn, use_container_width=True)

        fig_price = px.line(
            df_merged,
            x=tcol,
            y=['price_before','price_after'],
            color='scenario_name',
            title="Token Price Impact"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        st.success("Done simulation. Explore above table/graphs.")


def main():
    page_app()

if __name__=="__main__":
    main()
