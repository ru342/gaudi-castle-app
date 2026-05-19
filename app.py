import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")

# --- 形状生成関数 ---
def get_random_params():
    span = 8.0
    strength = float(np.random.uniform(1.5, 4.0))
    count = np.random.randint(2, 6) 
    m_list = np.random.uniform(2.0, 7.0, size=count).tolist()
    p_list = np.sort(np.random.uniform(-3.0, 3.0, size=count)).tolist()
    return {"span": span, "strength": strength, "m_list": m_list, "p_list": p_list}

# --- 物理計算関数 ---
def calc_funicular(params):
    span, strength = params["span"], params["strength"]
    m_list, p_list = params["m_list"], params["p_list"]
    x = np.linspace(-span/2, span/2, 100)
    y_cat = strength * np.cosh(x / strength) - (strength * np.cosh((span/2) / strength))
    y_load = np.zeros_like(x)
    for p, m in zip(p_list, m_list):
        y_l = np.where(x < p, (-m/(p+span/2))*(x+span/2), (-m/(span/2-p))*(span/2-x))
        y_load += y_l
    return x, y_cat + y_load

# --- セッション状態の初期化 ---
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.history = []
    st.session_state.current_left = get_random_params()
    st.session_state.current_right = get_random_params()

# --- メインロジック ---
if st.session_state.step <= 4:
    st.subheader(f"パーツ選び ({st.session_state.step}/4)")
    col1, col2 = st.columns(2)
    with col1:
        x, y = calc_funicular(st.session_state.current_left)
        fig, ax = plt.subplots(figsize=(3, 4)); ax.plot(x, y, color='gray'); ax.axis('off')
        st.pyplot(fig)
        if st.button("これを使う", key=f"L_{st.session_state.step}"):
            st.session_state.history.append(st.session_state.current_left)
            st.session_state.step += 1
            st.session_state.current_left, st.session_state.current_right = get_random_params(), get_random_params()
            st.rerun()
    with col2:
        x, y = calc_funicular(st.session_state.current_right)
        fig, ax = plt.subplots(figsize=(3, 4)); ax.plot(x, y, color='gray'); ax.axis('off')
        st.pyplot(fig)
        if st.button("これを使う", key=f"R_{st.session_state.step}"):
            st.session_state.history.append(st.session_state.current_right)
            st.session_state.step += 1
            st.session_state.current_left, st.session_state.current_right = get_random_params(), get_random_params()
            st.rerun()

else:
    # --- 組み上げられた構造体の生成 ---
    # スケッチのように少しずつずらして重ねるパラメータ
    offsets = [0, 0.5, -0.5, 0.2] # 左右のズレ
    depths = [1.0, 0.8, 0.9, 1.1]  # 深さの倍率（遠近感）
    alphas = [0.8, 0.5, 0.4, 0.6]  # 透明度
    
    # ヒストリーに基づいて、重ね合わせと装飾を計算
    funicular_curves = []
    for params in st.session_state.history:
        funicular_curves.append(calc_funicular(params))

    # --- セクション: 上下反転（アーチ）構造体（圧縮） ---
    # st.subheader("🛠️ 選ばれた4つのパーツを組み上げたアーチ構造体")  # タイトルを調整
    fig_inv, ax_inv = plt.subplots(figsize=(8, 10))
    fig_inv.patch.set_facecolor('white')

    for i, (x, y) in enumerate(funicular_curves):
        # スケッチ風に重ね合わせ、Y軸を反転 (-(y * depths[i]))
        ax_inv.plot(x + offsets[i], -(y * depths[i]), color='#333333', linewidth=2.5, alpha=alphas[i])
        
        # おもりの描画（アーチの上に載っているように見せるため、Y軸を反転）
        for p_idx in range(0, len(x), 15): 
             if np.random.rand() > 0.5: # ランダムにおもりっぽく表示
                ax_inv.plot(x[p_idx] + offsets[i], -(y[p_idx] * depths[i]), 'o', color='black', markersize=3, alpha=alphas[i])

    # Y軸の範囲を調整（反転させるため、既存の-25から25、5から-5にする）
    ax_inv.set_ylim(-5, 25) 
    ax_inv.set_xlim(-6, 6)
    ax_inv.axis('off')
    st.pyplot(fig_inv)

    if st.button("🔄 最初から"):
        del st.session_state.step
        del st.session_state.history
        st.rerun()