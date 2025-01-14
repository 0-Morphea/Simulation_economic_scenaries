import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =======================
#   MODULE : VESTING
# =======================
def compute_vesting_schedule(vesting_alloc_df, total_periods, time_unit):
    """
    Calcule, pour chaque période 1..total_periods, la quantité de tokens
    débloqués (=circ_supply) selon la table d'allocation/vesting.

    vesting_alloc_df columns:
      - 'allocation_name'
      - 'total_tokens'
      - 'tge_percent'
      - 'lock_period'
      - 'vesting_period'
    On suppose un unlock TGE% à t=1,
    puis linéaire sur 'vesting_period' après 'lock_period'.
    """
    cat_list = vesting_alloc_df['allocation_name'].unique()
    max_t = total_periods

    # unlocks[cat][t] = tokens libérés à la période t
    unlocks = {}
    for cat in cat_list:
        unlocks[cat] = np.zeros(max_t+1)  # index 1..max_t

    for i, row in vesting_alloc_df.iterrows():
        cat = row['allocation_name']
        total_tokens = float(row['total_tokens'])
        tge_percent = float(row['tge_percent'])/100.0
        lock_p = int(row['lock_period'])
        vest_p = int(row['vesting_period'])

        # TGE
        tge_amount = total_tokens*tge_percent
        locked_amount = total_tokens - tge_amount
        if 1<=max_t:
            unlocks[cat][1] += tge_amount
        # linéaire vesting
        if vest_p>0 and locked_amount>0:
            monthly_unlocked = locked_amount/vest_p
            start_vest = lock_p+1
            end_vest = lock_p+vest_p
            for t in range(start_vest, end_vest+1):
                if t<=max_t:
                    unlocks[cat][t]+=monthly_unlocked

    # cumul par cat
    supply_cat = {}
    for cat in cat_list:
        cumsum_cat = np.cumsum(unlocks[cat])
        supply_cat[cat] = cumsum_cat

    # somme
    total_supply = np.zeros(max_t+1)
    for cat in cat_list:
        total_supply += supply_cat[cat]

    df_supply = pd.DataFrame({
        'period': range(1,max_t+1),
        'circ_supply': total_supply[1:]
    })
    return df_supply


# =======================
#  HELPER : AMM / BC
# =======================
def amm_buy_sc_tokens(buy_tokens, pool_rbot, pool_ai, invariant_k):
    """L'utilisateur veut ACQUÉRIR 'buy_tokens' AiTokens sur l'AMM (x*y=k).
       Retourne (rbot_in, new_pool_rbot, new_pool_ai, slippage, price_before, price_after).
    """
    if buy_tokens<=0 or pool_ai<=0:
        return (0.0, pool_rbot, pool_ai,0.0,None,None)
    old_price = (pool_rbot/pool_ai) if pool_ai>0 else 1e9

    new_pool_ai = pool_ai - buy_tokens
    if new_pool_ai<1e-9:
        buy_tokens=pool_ai
        new_pool_ai=1e-9
    new_pool_rbot = invariant_k/new_pool_ai
    rbot_in = new_pool_rbot - pool_rbot
    new_price = new_pool_rbot/new_pool_ai
    slip=(new_price-old_price)/old_price if old_price>0 else 0
    return (rbot_in, new_pool_rbot, new_pool_ai, slip, old_price, new_price)

def amm_sell_sc_tokens(sell_tokens, pool_rbot, pool_ai, invariant_k):
    """L'utilisateur VEUT VENDRE 'sell_tokens' AiTokens.
       Retourne (rbot_out,new_pool_rbot,new_pool_ai,slip,price_before,price_after).
    """
    if sell_tokens<=0 or pool_ai<=0:
        return(0.0,pool_rbot,pool_ai,0.0,None,None)
    old_price=(pool_rbot/pool_ai) if pool_ai>0 else 1e9

    new_pool_ai = pool_ai+sell_tokens
    new_pool_rbot = invariant_k/new_pool_ai
    rbot_out = pool_rbot-new_pool_rbot
    new_price = new_pool_rbot/new_pool_ai
    slip=(new_price-old_price)/old_price if old_price>0 else 0
    return (rbot_out,new_pool_rbot,new_pool_ai,slip,old_price,new_price)

def bonding_curve_rbot_for_tokens(m_before, m_after, p0, f):
    """
    B(m,n) = p0*(f^n - f^m)/(f-1)
    Montant ($) requis (positif) ou rendu (négatif) selon buy/sell sur BC.
    """
    if abs(f-1.0)<1e-9:
        return p0*(m_after-m_before)
    return p0*((f**m_after)-(f**m_before))/(f-1)


# =======================
#  MAIN SIMULATION
# =======================
def simulate_scenario(
    scenario_name, scenario_id,
    total_periods,
    time_unit,
    # vesting supply
    df_vest,
    # volume + growth
    init_volume_usd, growth_percent,
    # fees (burn/liqu/treasury)
    burn_ratio, liqu_ratio, treas_ratio,
    # BC + AMM
    p0, f,
    bc_liquidity_threshold, pool_percentage,
    secondary_market_start,
    portion_secondary_after
):
    """
    On simule, par période:
      - On lit la supply circ: m_current <= circ
      - On applique un volume => (ex. 'buy_ai','sell_ai' => simplifié)
      - 1% fee => burn, liqu, treasury => burn => tokens out => effet prix
      - BC => minted AiToken => bc liquidity => si surpass threshold => AMM
      - portion SC => on calcule slippage (toy).
    """
    # Simplifions la partie "transactions" => On utilise un "volume" qu'on convertit en AiTokens ?
    # pour avoir la BC minted => on fera un 'net_buy' = volume / prix ? => etc.
    # Code d'exemple (toy)...

    # On map vest df
    vest_map = {int(r['period']): float(r['circ_supply']) for _,r in df_vest.iterrows()}

    # BC
    bc_liquidity_usd=0.0
    secondary_active=False
    pool_rbot=0.0
    pool_ai=0.0
    invariant_k=0.0

    m_current=0.0  # minted via BC
    price_token=1.0
    protocol_usd=0.0
    treasury_usd=0.0

    volume = init_volume_usd

    records=[]
    for t in range(1, total_periods+1):
        # supply circ max
        circ_max = vest_map.get(t,m_current)
        if circ_max<m_current:
            # si la supply max baisse => keep m_current
            circ_max=m_current

        # On simule un "net_buy" => par ex. volume_usd / price_token => 'AiTokens'
        net_buy_ai = volume/price_token
        old_m = m_current
        new_m = old_m+net_buy_ai
        if new_m>circ_max:
            new_m=circ_max
        cost_bc = bonding_curve_rbot_for_tokens(old_m,new_m,p0,f)
        cost_bc_usd=max(cost_bc,0)
        # fees
        fee_1pct = cost_bc_usd*0.01
        burn_usd = fee_1pct*burn_ratio
        liqu_usd = fee_1pct*liqu_ratio
        treas_usd = fee_1pct*treas_ratio
        protocol_usd += liqu_usd
        treasury_usd += treas_usd

        # burn => tokens = burn_usd/price ?
        burn_tokens=0.0
        if price_token>0:
            burn_tokens=burn_usd/price_token
        # On retire ces burn tokens => new_m - burn_tokens
        minted_after = new_m - burn_tokens
        if minted_after<0:
            burn_tokens=new_m
            minted_after=0

        # bc liquidity
        bc_liquidity_usd+=cost_bc_usd
        m_current=minted_after

        # Trigger AMM
        if (not secondary_active) and bc_liquidity_usd>=bc_liquidity_threshold and t>=secondary_market_start:
            secondary_active=True
            portion_pool=bc_liquidity_usd*pool_percentage
            bc_liquidity_usd-=portion_pool
            # moit moit
            # Ai price = ?
            # On approx "p0*f^m_current"
            approx_price = p0*(f**(m_current)) if m_current>0 else p0
            pool_rbot = portion_pool/2
            pool_ai = (portion_pool/2)/approx_price if approx_price>0 else 0
            invariant_k = pool_rbot*pool_ai

        # si secondary_active => portion_sc = portion_secondary_after => ex: sc_buy = net_buy_ai*portion_sc ?
        # ... => on skip detail pour concision

        # Impact sur price => toy
        # ex price = price + (burn_tokens/1e6)
        price_delta=(burn_tokens/1e6)
        new_price=max(0.000001, price_token+price_delta)

        records.append({
            'scenario_id':scenario_id,
            'scenario_name':scenario_name,
            time_unit:t,
            'circ_max':circ_max,
            'old_m':old_m,
            'new_m':new_m,
            'burn_tokens':burn_tokens,
            'bc_liquidity_usd':bc_liquidity_usd,
            'secondary_active':secondary_active,
            'pool_rbot':pool_rbot,
            'pool_ai':pool_ai,
            'price_before':price_token,
            'price_after':new_price,
            'cost_bc_usd': cost_bc_usd,
            'protocol_usd': protocol_usd,
            'treasury_usd': treasury_usd,
        })

        # update
        price_token=new_price
        volume*= (1.0+growth_percent/100.0)

    return pd.DataFrame(records)


# =======================
#   APP
# =======================
def page_app():
    st.title("Full Tokenomics Sim: Vesting + BC + Secondary + Fees + Volume Growth")

    # ----------------------
    # 1) Table Vesting
    # ----------------------
    st.header("1) Configure Vesting & Allocations")
    st.markdown("Définissez la répartition (TGE%, lock, vest...).")

    default_vest = pd.DataFrame({
        'allocation_name':["Team","Private","Public","Treasury"],
        'total_tokens':[1_000_000,2_000_000,3_000_000,4_000_000],
        'tge_percent':[10,25,100,100],
        'lock_period':[6,0,0,0],
        'vesting_period':[18,12,0,0],
    })

    vest_edited = st.data_editor(default_vest, key="vest_alloc_editor", help="All allocations with TGE %, lock, vesting.\nWill define the max circ supply over time.")
    colA, colB= st.columns(2)
    with colA:
        time_unit_vest=st.selectbox("Time Unit pour Vesting ?",
                                    ["Days","Months"], index=1,
                                    help="Choisissez si la vesting table est en jours ou mois.")
    with colB:
        total_periods_vest=st.number_input("Nombre total de périodes (pour la courbe vesting)",
                                           1,5000,36,
                                           help="Sur combien de périodes on veut calculer le déblocage total?")

    if st.button("Compute Vesting", help="Cliquez pour générer la courbe de supply circulante."):
        df_vest_result=compute_vesting_schedule(vest_edited, total_periods_vest,time_unit_vest)
        st.session_state['df_vest'] = df_vest_result
        st.session_state['time_unit_vest'] = time_unit_vest
        st.success("Vesting computed & stored.")
        st.dataframe(df_vest_result.head(50))
        fig_vest= px.line(df_vest_result,x='period',y='circ_supply',title="Courbe Circulante (Vesting)")
        st.plotly_chart(fig_vest,use_container_width=True)

    st.write("---")

    # ----------------------
    # 2) Multi-Scen
    # ----------------------
    st.header("2) Configurer Multi-Scenarios (Bonding Curve, Fees, etc.)")

    if 'df_vest' not in st.session_state:
        st.warning("Compute Vesting first!")
        return

    df_vest_ready= st.session_state['df_vest']
    time_unit_ready= st.session_state['time_unit_vest']

    st.sidebar.title("Scenarios")
    n_scen= st.sidebar.number_input("Number of Scenarios",1,5,1,help="Combien de scénarios différents à comparer?")
    scenarios_data=[]
    for i in range(n_scen):
        with st.sidebar.expander(f"Scenario {i+1}"):
            sc_name= st.text_input(f"Name scenario {i+1}", value=f"S{i+1}",
                                   key=f"scn_name_{i}",
                                   help="Nom du scénario (ex: 'Bull run').")

            # Period sim
            per_sim= st.number_input(f"Periods sim (S{i+1})",1,5000, int(df_vest_ready['period'].max()),
                                     key=f"ps_{i}_{sc_name}",
                                     help="Durée totale de la simulation (jours/mois)")

            # Volume init / growth
            init_vol= st.number_input(f"Initial Volume $ (S{i+1})",100.0,1e9,100000.0,
                                      key=f"iv_{i}_{sc_name}",
                                      help="Volume initial de trading en $ sur la BC.")
            growth_pct= st.slider(f"Volume Growth % (S{i+1})", -10.0,50.0,5.0, key=f"g_{i}_{sc_name}",
                                  help="Croissance du volume par période")

            # Fees
            st.markdown("**Répartition du 1% Fee**")
            burn_ratio= st.slider(f"Burn ratio (S{i+1})", 0.0,1.0,0.3,0.05,
                                  key=f"br_{i}_{sc_name}",
                                  help="Fraction du 1% allouée au burn.")
            liqu_ratio= st.slider(f"Liquidity ratio (S{i+1})",0.0,1.0,0.3,0.05,
                                  key=f"lr_{i}_{sc_name}",
                                  help="Fraction du 1% allouée à la liquidity.")
            treas_ratio= 1.0- burn_ratio - liqu_ratio
            st.write(f"Treasury ratio = {treas_ratio:.2f}")

            # BC exponent
            p0_ = st.number_input(f"p0 (S{i+1})",0.000001,1e5,1.0,
                                  key=f"p0_{i}_{sc_name}",
                                  help="Prix initial du token BC (premier minted).")
            f_ = st.number_input(f"f exponent (S{i+1})",0.9999,5.0,1.0001,0.0001,
                                 key=f"f_{i}_{sc_name}",
                                 help="Facteur exponentiel f dans la formule BC exponentielle")

            # Activation secondary
            bc_threshold= st.number_input(f"BC Liquidity threshold $ (S{i+1})",0.0,1e9,1_000_000.0,
                                          key=f"bc_th_{i}_{sc_name}",
                                          help="Montant $ accumulé via BC pour activer le marché secondaire.")
            pool_perc= st.slider(f"pool % for AMM (S{i+1})",0.0,1.0,0.2,0.05,
                                 key=f"poolp_{i}_{sc_name}",
                                 help="Fraction de la bc_liquidity injectée à l'AMM.")
            sm_start= st.number_input(f"Secondary Market earliest start (S{i+1})",1,9999,12,
                                      key=f"smst_{i}_{sc_name}",
                                      help="Période min avant de pouvoir activer le marché secondaire.")
            sc_part= st.slider(f"Portion trades SC after activation (S{i+1})",0.0,1.0,0.5,0.05,
                               key=f"scp_{i}_{sc_name}",
                               help="Fraction de trades routés vers le secondary market quand actif.")

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

    if st.button("Run All Scenarios",help="Lance la simulation pour chaque scénario."):
        all_dfs=[]
        for sp in scenarios_data:
            sub_vest= df_vest_ready[df_vest_ready['period']<=sp['total_periods_sim']].copy()

            df_res= simulate_scenario(
                scenario_name=sp['scenario_name'],
                scenario_id=sp['scenario_id'],
                total_periods=sp['total_periods_sim'],
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

        df_merged= pd.concat(all_dfs,ignore_index=True)
        st.write("### Résultats Simulation (Tous Scénarios)")
        st.dataframe(df_merged.head(1000))

        tcol= df_merged.columns[2] # days or months col
        fig_bc = px.line(
            df_merged, x=tcol, y='bc_liquidity_usd',
            color='scenario_name',
            title="Bonding Curve Liquidity ($) Over Time"
        )
        st.plotly_chart(fig_bc,use_container_width=True)

        fig_burn = px.line(
            df_merged, x=tcol, y='burn_tokens',
            color='scenario_name',
            title="Burn Tokens nominal Over Time"
        )
        st.plotly_chart(fig_burn,use_container_width=True)

        fig_price = px.line(
            df_merged, x=tcol, y=['price_before','price_after'],
            color='scenario_name',
            title="Price Impact (before/after) each period"
        )
        st.plotly_chart(fig_price,use_container_width=True)

        st.success("Simulation done. Explore the table & charts.")


def main():
    page_app()

if __name__=="__main__":
    main()
