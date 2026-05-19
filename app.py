import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="centered", page_title="Gaudi Arch Generator")

def get_random_params():
    # パーツごとに横幅を 3.0 〜 10.0 の間でランダムに決定
    span = float(np.random.uniform(3.0, 10.0))
    strength = float(np.random.uniform(1.5, 4.0))
    count = np.random.randint(2, 6) 
    m_list = np.random.uniform(2.0, 7.0, size=count).tolist()
    p_limit = span / 2 * 0.8
    p_list = np.sort(np.random.uniform(-p_limit, p_limit, size=count)).tolist()
    return {"span": span, "strength": strength, "m_list": m_list, "p_list": p_list}

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

if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.history = []
    st.session_state.current_left = get_random_params()
    st.session_state.current_right = get_random_params()

Y_LIMIT_BOTTOM = -2
Y_LIMIT_TOP = 30
X_LIMIT = 12

if st.session_state.step <= 4:
    st.subheader(f"選択 ({st.session_state.step}/4)")
    
    col1, col2 = st.columns(2)
    choices = [st.session_state.current_left, st.session_state.current_right]
    keys = [f"L_{st.session_state.step}", f"R_{st.session_state.step}"]
    
    for i, col in enumerate([col1, col2]):
        with col:
            x, y = calc_funicular(choices[i])
            fig, ax = plt.subplots(figsize=(4, 5))
            ax.plot(x, -y, color='#333333', linewidth=2.5)
            ax.set_ylim(Y_LIMIT_BOTTOM, Y_LIMIT_TOP)
            ax.set_xlim(-X_LIMIT, X_LIMIT)
            ax.axis('off')
            st.pyplot(fig)
            
            if st.button("選択", key=keys[i], use_container_width=True):
                st.session_state.history.append(choices[i])
                st.session_state.step += 1
                st.session_state.current_left = get_random_params()
                st.session_state.current_right = get_random_params()
                st.rerun()

else:
    st.subheader("完成")
    fig_master, ax_master = plt.subplots(figsize=(10, 8))
    fig_master.patch.set_facecolor('white')

    # 高さを変えずに左右のズレと透け感だけを調整
    offsets = [0, 3.0, -6.0, -3.0]
    alphas = [0.7, 0.5, 0.4, 0.6] 
    
    for i, params in enumerate(st.session_state.history):
        x, y = calc_funicular(params)
        y_arch = -y # 選んだ時の高さをそのまま使用
        ax_master.plot(x + offsets[i], y_arch, color='#333333', linewidth=3, alpha=alphas[i])
        
        for p_idx in range(0, len(x), 15): 
             if np.random.rand() > 0.4: 
                ax_master.plot(x[p_idx] + offsets[i], y_arch[p_idx], 'o', color='black', markersize=4, alpha=alphas[i])

    ax_master.set_ylim(Y_LIMIT_BOTTOM, Y_LIMIT_TOP)
    ax_master.set_xlim(-X_LIMIT, X_LIMIT)
    ax_master.axis('off')
    st.pyplot(fig_master)

    if st.button("リセット"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
