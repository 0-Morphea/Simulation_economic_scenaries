import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =======================
#  1) VESTING MODULE
# =======================
def compute_vesting_schedule(vesting_alloc_df, total_periods, time_unit):
    """
    Calcule la supply circulante (circ_supply) sur 1..total_periods,
    selon le tableau d'allocation/vesting.

    Table columns:
      - 'allocation_name'
      - 'total_tokens'
      - 'tge_percent'
      - 'lock_period'
      - 'vesting_period'

    TGE => t=1, puis lock, puis vesting linéaire.
    """
    cat_list = vesting_alloc_df['allocation_name'].unique()
    max_t = total_periods

    unlocks = {}
    for cat in cat_list:
        unlocks[cat] = np.zeros(max_t+1)  # index de 1..max_t

    for i, row in vesting_alloc_df.iterrows():
        cat = row['allocation_name']
        total_tokens = float(row['total_tokens'])
        tge_p = float(row['tge_percent'])/100.0
        lock_p = int(row['lock_period'])
        vest_p = int(row['vesting_period'])

        tge_amount = total_tokens * tge_p
        locked_amount = total_tokens - tge_amount

        # TGE => t=1
        if 1<=max_t:
            unlocks[cat][1] += tge_amount

        if vest_p>0 and locked_amount>0:
            monthly_unlocked = locked_amount/vest_p
            start_vest= lock_p+1
            end_vest= lock_p+vest_p
            for t in range(start_vest, end_vest+1):
                if t<=max_t:
                    unlocks[cat][t]+= monthly_unlocked

    supply_cat = {}
    for cat in cat_list:
        cumsum_cat = np.cumsum(unlocks[cat])
        supply_cat[cat]= cumsum_cat

    total_supply = np.zeros(max_t+1)
    for cat in cat_list:
        total_supply += supply_cat[cat]

    df_supply= pd.DataFrame({
        'period': range(1,max_t+1),
        'circ_supply': total_supply[1:]
    })
    return df_supply


# =======================
#  2) AMM / BONDING CURVE
# =======================
def amm_buy_sc_tokens(buy_tokens, pool_rbot, pool_ai, invariant_k):
    """AMM x*y=k. L'utilisateur veut acquérir buy_tokens AiTokens."""
    if buy_tokens<=0 or pool_ai<=0:
        return (0.0, pool_rbot, pool_ai, 0.0, None, None)
    old_price= pool_rbot/pool_ai if pool_ai>0 else 1e9

    new_pool_ai= pool_ai - buy_tokens
    if new_pool_ai<1e-9:
        buy_tokens=pool_ai
        new_pool_ai=1e-9
    new_pool_rbot= invariant_k/new_pool_ai
    rbot_in= new_pool_rbot- pool_rbot
    new_price= new_pool_rbot/new_pool_ai
    slip= (new_price - old_price)/old_price if old_price>0 else 0
    return (rbot_in, new_pool_rbot, new_pool_ai, slip, old_price,new_price)

def amm_sell_sc_tokens(sell_tokens, pool_rbot, pool_ai, invariant_k):
    """AMM x*y=k. L'utilisateur vend sell_tokens AiTokens."""
    if sell_tokens<=0 or pool_ai<=0:
        return (0.0,pool_rbot,pool_ai,0.0,None,None)
    old_price= pool_rbot/pool_ai if pool_ai>0 else 1e9

    new_pool_ai= pool_ai+ sell_tokens
    new_pool_rbot= invariant_k/new_pool_ai
    rbot_out= pool_rbot- new_pool_rbot
    new_price= new_pool_rbot/new_pool_ai
    slip= (new_price- old_price)/old_price if old_price>0 else 0
    return (rbot_out,new_rbot,new_ai,slip,old_price,new_price)

def bonding_curve_rbot_for_tokens(m_before, m_after, p0, f):
    """ B(m,n)= p0*(f^n- f^m)/(f-1). Montant $ (positif ou négatif) """
    if abs(f-1.0)<1e-9:
        return p0*(m_after-m_before)
    return p0*((f**m_after)-(f**m_before))/(f-1)


# =======================
#   3) SIMULATION
# =======================
def simulate_scenario(
    scenario_name, scenario_id,
    total_periods,
    time_unit,
    # supply from vesting
    df_vest,
    # volume init + growth
    init_volume_usd,
    growth_percent,
    # fees distribution
    burn_ratio, liqu_ratio, treas_ratio,
    # Bonding Curve + secondary
    p0,f,
    bc_liquidity_threshold, pool_percentage,
    secondary_market_start,
    portion_secondary_after
):
    """
    - On a un 'm_current' AiToken minted via BC
    - On accumule bc_liquidity_usd => si > threshold => on ouvre AMM
    - On route 'portion_secondary_after' vers AMM (ex: sc_buy, sc_sell)
    - On applique volume => net_buy => cost_bc => fees => burn => price effect
    """
    vest_map= {int(r['period']): float(r['circ_supply']) for _,r in df_vest.iterrows()}
    bc_liquidity_usd=0.0
    secondary_active=False
    pool_rbot=0.0
    pool_ai=0.0
    invariant_k=0.0
    m_current=0.0
    price_token=1.0

    # On simule un param => "liquidity / sc trades"? => On fait un simple net_buy = volume/price
    volume = init_volume_usd

    protocol_usd=0.0
    treasury_usd=0.0

    records=[]
    for t in range(1,total_periods+1):
        circ_max = vest_map.get(t,m_current)
        if circ_max< m_current:
            circ_max=m_current

        # net_buy
        net_buy_ai= volume/price_token
        old_m= m_current
        new_m= old_m+ net_buy_ai
        if new_m> circ_max:
            new_m= circ_max
        cost_bc= bonding_curve_rbot_for_tokens(old_m, new_m, p0, f)
        cost_bc_usd= max(cost_bc,0.0)

        # 1% fees => burn/liqu/treas
        fee_1pct= cost_bc_usd*0.01
        burn_usd= fee_1pct* burn_ratio
        liqu_usd= fee_1pct* liqu_ratio
        treas_usd= fee_1pct* treas_ratio

        protocol_usd+= liqu_usd
        treasury_usd+= treas_usd

        # burn => tokens = burn_usd / price_token
        burn_tokens=0.0
        if price_token>0:
            burn_tokens= burn_usd/ price_token
        minted_after= new_m- burn_tokens
        if minted_after<0:
            burn_tokens= new_m
            minted_after=0

        bc_liquidity_usd+= cost_bc_usd
        m_current= minted_after

        # activation secondary
        if (not secondary_active) and bc_liquidity_usd>= bc_liquidity_threshold and t>= secondary_market_start:
            secondary_active=True
            portion_for_pool= bc_liquidity_usd* pool_percentage
            bc_liquidity_usd-= portion_for_pool
            # init AMM => moit/ moit
            approx_price= p0*(f**m_current) if m_current>0 else p0
            pool_rbot= portion_for_pool/2
            pool_ai= (portion_for_pool/2)/ approx_price if approx_price>0 else 0
            invariant_k= pool_rbot* pool_ai

        # impact on price
        price_delta= burn_tokens/1e6
        new_price= max(0.000001, price_token+ price_delta)

        records.append({
            'scenario_id': scenario_id,
            'scenario_name': scenario_name,
            time_unit: t,
            'circ_max': circ_max,
            'old_m': old_m,
            'new_m': new_m,
            'burn_tokens': burn_tokens,
            'bc_liquidity_usd': bc_liquidity_usd,
            'secondary_active': secondary_active,
            'pool_rbot': pool_rbot,
            'pool_ai': pool_ai,
            'price_before': price_token,
            'price_after': new_price,
            'cost_bc_usd': cost_bc_usd,
            'protocol_usd': protocol_usd,
            'treasury_usd': treasury_usd
        })

        price_token= new_price
        volume*= (1.0+ growth_percent/100.0)

    df_res= pd.DataFrame(records)
    return df_res


# =======================
#  4) STREAMLIT APP
# =======================
def page_app():
    st.title("Full Tokenomics Simulator (Vesting + Bonding Curve + Secondary + Fees)")

    st.header("Step 1: Allocation/Vesting Table")
    default_vest= pd.DataFrame({
        'allocation_name':["Team","Private","Public","Treasury"],
        'total_tokens':[1_000_000,2_000_000,3_000_000,4_000_000],
        'tge_percent':[10,25,100,100],
        'lock_period':[6,0,0,0],
        'vesting_period':[18,12,0,0]
    })
    st.markdown("**Editez la table** :")
    # On utilise st.experimental_data_editor pour éviter l'erreur sur versions plus anciennes
    # Et on supprime le param help ici.
    vest_edited= st.experimental_data_editor(default_vest, key="vest_alloc_editor")

    colA, colB= st.columns(2)
    with colA:
        time_unit_vest= st.selectbox("Time Unit (Days/Months) for vesting", ["Days","Months"], index=1,
                                     help="Choisissez la granularité temporelle pour le vesting.")
    with colB:
        total_periods_vest= st.number_input("Total Périodes (pour vesting schedule)",
                                           1,5000,36,
                                           key="total_vest_key",
                                           help="Sur combien de pas de temps on calcule la courbe de déblocage.")

    if st.button("Compute Vesting", key="compute_vesting_btn"):
        df_vest_result= compute_vesting_schedule(vest_edited, total_periods_vest, time_unit_vest)
        st.session_state['df_vest']= df_vest_result
        st.session_state['time_unit_vest']= time_unit_vest
        st.success("Vesting computed & stored in session state.")
        st.dataframe(df_vest_result.head(50))
        fig_vest= px.line(df_vest_result,x='period',y='circ_supply', title="Courbe Circulante (Vesting)")
        st.plotly_chart(fig_vest,use_container_width=True)

    st.write("---")

    st.header("Step 2: Multi-Scenarios (Bonding Curve, Fees, Volume)")

    if 'df_vest' not in st.session_state:
        st.warning("Compute the vesting schedule first!")
        return

    df_vest_ready= st.session_state['df_vest']
    time_unit_ready= st.session_state['time_unit_vest']

    st.sidebar.title("Scenarios Config")
    n_scen= st.sidebar.number_input("Number of Scenarios",1,5,1, key="n_scen_key",
                                    help="Combien de scénarios à comparer ?")

    scenarios_data=[]
    for i in range(n_scen):
        with st.sidebar.expander(f"Scenario {i+1}", expanded=(i==0)):
            sc_name= st.text_input(f"Scen Name {i+1}", value=f"S{i+1}",
                                   key=f"sc_name_{i}",
                                   help="Nom du scénario, ex: 'Bull' / 'Bear'")
            per_sim= st.number_input(f"Total Periods (S{i+1})",1,5000,
                                     int(df_vest_ready['period'].max()),
                                     key=f"per_{i}_{sc_name}",
                                     help="Durée de simulation (cohérente avec vesting)")
            init_vol= st.number_input(f"Initial Volume $ (S{i+1})",100.0,1e9,100000.0,
                                      key=f"iv_{i}_{sc_name}",
                                      help="Volume en $ au départ sur la Bonding Curve.")
            growth_pct= st.slider(f"Volume Growth % (S{i+1})", -10.0,50.0,5.0,
                                  key=f"g_{i}_{sc_name}",
                                  help="Croissance du volume par période (%)")

            st.markdown("**Répartition du 1% Fee** (burn/liqu/treasury)")
            burn_ratio= st.slider(f"Burn ratio (S{i+1})",0.0,1.0,0.3,0.05,
                                  key=f"br_{i}_{sc_name}",
                                  help="Fraction du 1% allouée au burn.")
            liqu_ratio= st.slider(f"Liquidity ratio (S{i+1})",0.0,1.0-burn_ratio,0.3,0.05,
                                  key=f"lr_{i}_{sc_name}",
                                  help="Fraction du 1% allouée à la liquidity (reste ira treasury).")
            treas_ratio= 1.0- burn_ratio- liqu_ratio
            st.write(f"Treasury ratio = {treas_ratio:.2f}")

            st.markdown("**Bonding Curve**")
            p0_ = st.number_input(f"p0 (S{i+1})",0.000001,1e5,1.0,
                                  key=f"p0_{i}_{sc_name}",
                                  help="Prix du token pour le 1er minted sur la BC.")
            f_ = st.number_input(f"f exponent (S{i+1})",0.9999,5.0,1.0001,0.0001,
                                 key=f"f_{i}_{sc_name}",
                                 help="Exponent pour la Bonding Curve exponentielle.")

            st.markdown("**Secondary Market** (Seuil + Param)")
            bc_threshold= st.number_input(f"BC Liquidity threshold $ (S{i+1})",
                                          0.0,1e9,1_000_000.0,
                                          key=f"bct_{i}_{sc_name}",
                                          help="Montant $ accumulé via BC pour activer le marché secondaire.")
            pool_perc= st.slider(f"pool % AMM (S{i+1})",0.0,1.0,0.2,0.05,
                                 key=f"poolp_{i}_{sc_name}",
                                 help="Fraction de bc_liquidity qu'on dépose dans le pool AMM.")
            sm_start= st.number_input(f"Secondary Market earliest start (S{i+1})",1,9999,12,
                                      key=f"sst_{i}_{sc_name}",
                                      help="Période min avant de pouvoir activer l'AMM.")
            sc_part= st.slider(f"Portion trades SC after activation (S{i+1})",0.0,1.0,0.5,0.05,
                               key=f"scp_{i}_{sc_name}",
                               help="Fraction des trades routés sur le secondary once actif.")

            scenarios_data.append({
                'scenario_id': i+1,
                'scenario_name': sc_name,
                'total_periods_sim': int(per_sim),
                'init_volume_usd': float(init_vol),
                'growth_percent': float(growth_pct),
                'burn_ratio': float(burn_ratio),
                'liquidity_ratio': float(liqu_ratio),
                'treasury_ratio': float(treas_ratio),
                'p0': float(p0_),
                'f': float(f_),
                'bc_liquidity_threshold': float(bc_threshold),
                'pool_percentage': float(pool_perc),
                'secondary_market_start': int(sm_start),
                'portion_secondary_after': float(sc_part)
            })

    if st.button("Run All Scenarios", key="run_scen_btn"):
        all_dfs=[]
        for sp in scenarios_data:
            sub_vest= df_vest_ready[df_vest_ready['period']<= sp['total_periods_sim']].copy()

            df_res= simulate_scenario(
                scenario_name= sp['scenario_name'],
                scenario_id= sp['scenario_id'],
                total_periods= sp['total_periods_sim'],
                time_unit= time_unit_ready,
                df_vest= sub_vest,
                init_volume_usd= sp['init_volume_usd'],
                growth_percent= sp['growth_percent'],
                burn_ratio= sp['burn_ratio'],
                liqu_ratio= sp['liquidity_ratio'],
                treas_ratio= sp['treasury_ratio'],
                p0= sp['p0'],
                f= sp['f'],
                bc_liquidity_threshold= sp['bc_liquidity_threshold'],
                pool_percentage= sp['pool_percentage'],
                secondary_market_start= sp['secondary_market_start'],
                portion_secondary_after= sp['portion_secondary_after']
            )
            all_dfs.append(df_res)

        df_merged= pd.concat(all_dfs, ignore_index=True)
        st.write("### Résultats Simulation (tous scénarios)")
        st.dataframe(df_merged.head(1000))

        tcol= df_merged.columns[2] # 'Days'/'Months'
        fig_bc= px.line(
            df_merged, x=tcol,y='bc_liquidity_usd',
            color='scenario_name',
            title="Bonding Curve Liquidity ($) Over Time"
        )
        st.plotly_chart(fig_bc,use_container_width=True)

        fig_burn= px.line(
            df_merged, x=tcol, y='burn_tokens',
            color='scenario_name',
            title="Burn Tokens (nominal) Over Time"
        )
        st.plotly_chart(fig_burn,use_container_width=True)

        fig_price= px.line(
            df_merged, x=tcol, y=['price_before','price_after'],
            color='scenario_name',
            title="Token Price: before/after each period"
        )
        st.plotly_chart(fig_price,use_container_width=True)

        st.success("Simulation terminée. Explorez le tableau & graphiques!")


def main():
    page_app()

if __name__=="__main__":
    main()
