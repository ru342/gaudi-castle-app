import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import requests
from PIL import Image

st.set_page_config(layout="centered")

# --- 設定値 ---
NUM_ANCHORS = 8
# -17.5 から 17.5 の間に等間隔で8個の点を配置
anchors_x = np.linspace(-17.5, 17.5, NUM_ANCHORS)
anchors = [{"x": float(x), "y": 0.0} for x in anchors_x]

NUM_NEW_NODES = 41 
NUM_INTERNAL_NODES = NUM_NEW_NODES - 2
MID_NODE_OFFSET = NUM_INTERNAL_NODES // 2 # ひものちょうど真ん中のインデックス

# --- サイドバー：API設定とプロンプト ---
st.sidebar.title("⚙️ AI生成設定")
st.sidebar.markdown("本物の画像を生成するにはAPIキーを入力してください。")
api_key = st.sidebar.text_input("APIキー (Stability AI)", type="password", help="sk- から始まるキーを入力してください")

st.sidebar.markdown("---")
user_prompt = st.sidebar.text_area(
    "🎨 生成プロンプト (呪文)", 
    value="A majestic medieval fantasy castle designed by Antoni Gaudi, intricate stone carving, sunset lighting, epic scale, highly detailed, masterpiece, 8k resolution",
    height=150
)

# --- 【変更】ノード番号をわかりやすい名前に変換する関数 ---
def format_node_label(idx):
    if idx < NUM_ANCHORS:
        positions = ["一番左", "左から2番目", "左から3番目", "中央の左", "中央の右", "右から3番目", "右から2番目", "一番右"]
        return f"天井 {idx + 1} ({positions[idx]})"
    else:
        # ひもの頂点は番号で呼ぶ
        s_id = (idx - NUM_ANCHORS) // NUM_INTERNAL_NODES
        return f"ひも {s_id + 1} の頂点"

# --- 使用中のノードを動的に取得する関数 ---
def get_used_nodes():
    used = []
    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if not s.get("is_deleted", False):
                used.extend([s["start_node"], s["end_node"]])
    return used

# --- 共通関数：ひもを新しく生成して追加する ---
def add_string(idx1, idx2):
    nodes = st.session_state.nodes
    links = st.session_state.links
    if "string_data" not in st.session_state:
        st.session_state.string_data = []
        
    p1, p2 = nodes[idx1], nodes[idx2]
    dist = np.sqrt((p2["x"] - p1["x"])**2 + (p2["y"] - p1["y"])**2)
    total_len = dist * 1.002
    seg_len = total_len / (NUM_NEW_NODES - 1)
    new_x = np.linspace(p1["x"], p2["x"], NUM_NEW_NODES)
    new_y = np.zeros(NUM_NEW_NODES)
    sag_depth = dist * 0.01
    
    for i in range(NUM_NEW_NODES):
        t = i / (NUM_NEW_NODES - 1)
        linear_y = p1["y"] + (p2["y"] - p1["y"]) * t
        sag = sag_depth * (4.0 * t * (1.0 - t))
        new_y[i] = linear_y - sag
        
    start_idx = len(nodes)
    first_link_idx = len(links)
    
    for i in range(1, NUM_NEW_NODES - 1):
        nodes.append({"x": new_x[i], "y": new_y[i], "px": new_x[i], "py": new_y[i], "fixed": False})
            
    prev_idx = idx1
    current_new_idx = start_idx
    for i in range(1, NUM_NEW_NODES - 1):
        links.append((prev_idx, current_new_idx, seg_len))
        prev_idx = current_new_idx
        current_new_idx += 1
    links.append((prev_idx, idx2, seg_len))
    
    st.session_state.string_data.append({
        "id": len(st.session_state.string_data),
        "start_node": idx1, "end_node": idx2,
        "first_link_idx": first_link_idx, "last_link_idx": len(links) - 1,
        "is_deleted": False
    })

# --- 共通関数：次につなぐ候補点を選定する ---
def update_next_nodes():
    candidates = list(range(NUM_ANCHORS)) 
    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if not s.get("is_deleted", False):
                base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                candidates.append(base + MID_NODE_OFFSET)
        
    used_nodes = get_used_nodes()
    available_candidates = [c for c in candidates if c not in used_nodes]
    if len(available_candidates) >= 2:
        st.session_state.next_nodes = np.random.choice(available_candidates, 2, replace=False).tolist()
    else:
        st.session_state.next_nodes = None

# --- AI画像生成関数 ---
def generate_castle_image(image_bytes, prompt, key):
    if not key:
        time.sleep(1.5)
        img = Image.new('RGB', (800, 800), color=(150, 160, 170))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.warning("⚠️ APIキーが入力されていないため、ダミー画像を表示しています。")
        return buf.getvalue()

    st.info("🌐 Stability AIのサーバーに通信中...")
    api_host = "https://api.stability.ai"
    engine_id = "stable-diffusion-xl-1024-v1-0" 
    
    try:
        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/image-to-image",
            headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
            files={"init_image": image_bytes},
            data={
                "image_strength": 0.5, 
                "init_image_mode": "IMAGE_STRENGTH",
                "text_prompts[0][text]": prompt,
                "text_prompts[0][weight]": 1.0,
                "cfg_scale": 7,
                "samples": 1,
                "steps": 30,
            }
        )
        if response.status_code != 200:
            st.error(f"APIエラーが発生しました: {response.status_code} - {response.text}")
            return None
        data = response.json()
        import base64
        return base64.b64decode(data["artifacts"][0]["base64"])
    except Exception as e:
        st.error(f"通信中にエラーが発生しました: {e}")
        return None

# --- 1. 初期化 ---
if "app_phase" not in st.session_state:
    st.session_state.app_phase = "building"

if "nodes" not in st.session_state:
    st.session_state.nodes = [{"x": p["x"], "y": p["y"], "px": p["x"], "py": p["y"], "fixed": True} for p in anchors]
    st.session_state.links = []
    st.session_state.string_data = [] 
    st.session_state.selected_string_id = None
    st.session_state.selected_end_type = "start"
    st.session_state.preview_new_target = None
    
    # 最初はランダムに3本のひもを掛ける
    for _ in range(3):
        update_next_nodes()
        if st.session_state.next_nodes is not None:
            i1_init, i2_init = st.session_state.next_nodes
            add_string(i1_init, i2_init)
    update_next_nodes()

nodes = st.session_state.nodes
links = st.session_state.links

# ==========================================
# --- 構築フェーズ ---
# ==========================================
if st.session_state.app_phase == "building":

    gravity = 0.08  
    stiffness = 0.95 

    for _ in range(300):
        for n in nodes:
            if not n.get("fixed", False):
                vx = (n["x"] - n["px"]) * 0.75  
                vy = (n["y"] - n["py"]) * 0.75
                n["px"], n["py"] = n["x"], n["y"]
                n["x"] += vx
                n["y"] += vy - gravity

        for _ in range(6):  
            for l in links:
                if l == (0, 0, 0): continue 
                idx1, idx2, target_dist = l
                n1, n2 = nodes[idx1], nodes[idx2]
                dx = n2["x"] - n1["x"]
                dy = n2["y"] - n1["y"]
                d = np.sqrt(dx**2 + dy**2)
                if d == 0: continue
                diff = (target_dist - d) / d * 0.5
                if not n1.get("fixed", False):
                    n1["x"] -= dx * diff * stiffness
                    n1["y"] -= dy * diff * stiffness
                if not n2.get("fixed", False):
                    n2["x"] += dx * diff * stiffness
                    n2["y"] += dy * diff * stiffness

    fig, ax = plt.subplots(figsize=(8, 8))
    # 天井の点を描画
    ax.scatter([p["x"] for p in anchors], [0]*NUM_ANCHORS, color="#555555", s=100, zorder=10)

    selected_s_data = None
    if st.session_state.selected_string_id is not None and "string_data" in st.session_state:
         selected_s_data = st.session_state.string_data[st.session_state.selected_string_id]

    # 【改良】フォーカス方式の描画処理
    for l_idx, l in enumerate(links):
        if l == (0, 0, 0): continue 
        n1, n2 = nodes[l[0]], nodes[l[1]]
        
        s_data_for_link = None
        for s in st.session_state.string_data:
             if not s.get("is_deleted", False) and s["first_link_idx"] <= l_idx <= s["last_link_idx"]:
                 s_data_for_link = s
                 break
        
        # デフォルトは細くて目立たないグレー
        color = "#C0C0C0" 
        lw = 2
        z_order = 4
        
        if s_data_for_link:
            # 選択中のひもだけ、太い鮮やかな青色にして前面に持ってくる
            if selected_s_data and s_data_for_link["id"] == selected_s_data["id"]:
                color = "#1C83E1"
                lw = 6
                z_order = 6
                
        ax.plot([n1["x"], n2["x"]], [n1["y"], n2["y"]], color=color, lw=lw, solid_capstyle="round", zorder=z_order)

    # プレビューの描画（現在の端＝オレンジ、新しい接続先＝緑）
    if selected_s_data and st.session_state.preview_new_target is not None:
        current_idx = selected_s_data["start_node"] if st.session_state.selected_end_type == "start" else selected_s_data["end_node"]
        curr_n = nodes[current_idx]
        ax.scatter(curr_n["x"], curr_n["y"], color="none", edgecolor="#FF9F43", s=300, linewidth=3, zorder=12, label="現在の端")
        
        new_n = nodes[st.session_state.preview_new_target]
        ax.scatter(new_n["x"], new_n["y"], color="none", edgecolor="#32CD32", s=600, linewidth=4, linestyle="--", zorder=13, label="新しい接続先")

    active_node_indices = set(range(NUM_ANCHORS))
    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if not s.get("is_deleted", False):
                base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                active_node_indices.update(range(base, base + NUM_INTERNAL_NODES))

    all_y = [nodes[i]["y"] for i in active_node_indices]
    min_y = min(all_y) if all_y else -15.0
    bottom_limit = min_y - 5.0
    half_width = max(20.0, abs(bottom_limit) * 0.8)

    ax.set_ylim(bottom_limit, 5)
    ax.set_xlim(-half_width, half_width)
    ax.axis("off")
    st.pyplot(fig)

    st.markdown("---")
    if st.button("✨ ガウディ建築を生成する (上下反転してAIに入力)", type="primary", use_container_width=True):
        st.session_state.app_phase = "generating"
        st.rerun()
        
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("➕ ひもを追加")
        if st.session_state.next_nodes is not None:
            i1, i2 = st.session_state.next_nodes
            st.write(f"候補: {format_node_label(i1)} ↔ {format_node_label(i2)}")
            if st.button("このひもを確定して次へ"):
                add_string(i1, i2)
                st.session_state.selected_string_id = None 
                update_next_nodes()
                st.rerun()
        else:
            st.write("選べる点がありません。")

        st.write("---")
        if st.button("すべてリセット"):
            st.session_state.clear()
            st.rerun()

    with col2:
        st.subheader("⚙️ ひもの操作")
        active_strings = [s for s in st.session_state.string_data if not s.get("is_deleted", False)]
        if active_strings:
            # 【変更】シンプルな番号表記に戻す
            string_options = {}
            for s in active_strings:
                label = f"ひも {s['id'] + 1}"
                string_options[label] = s['id']
                
            selected_string_label = st.selectbox("1. 操作するひもを選ぶ", list(string_options.keys()))
            s_id = string_options[selected_string_label]
            st.session_state.selected_string_id = s_id
            current_s_data = st.session_state.string_data[s_id]
            
            st.markdown("#### 🔄 つなぎ直す")
            end_options = {"始点": "start", "終点": "end"}
            end_type = st.radio("変更する端", list(end_options.keys()), horizontal=True)
            type_key = end_options[end_type]
            st.session_state.selected_end_type = type_key
            current_target = current_s_data["start_node"] if type_key == "start" else current_s_data["end_node"]
            
            reconnect_cands = list(range(NUM_ANCHORS))
            for s in active_strings:
                if s["id"] != s_id: 
                    base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                    reconnect_cands.append(base + MID_NODE_OFFSET)
            
            used_nodes = get_used_nodes()
            if current_target in used_nodes:
                used_nodes.remove(current_target) 
                
            available_reconnects = [c for c in reconnect_cands if c not in used_nodes]
            if available_reconnects:
                default_idx = available_reconnects.index(current_target) if current_target in available_reconnects else 0
                new_target = st.selectbox("新しい接続先", available_reconnects, index=default_idx, format_func=format_node_label)
                st.session_state.preview_new_target = new_target
                
                if current_target != new_target:
                    if st.button("🔵 この場所につなぎ直す", use_container_width=True):
                        if st.session_state.selected_end_type == "start":
                            l_idx = current_s_data["first_link_idx"]
                            old_l = st.session_state.links[l_idx]
                            st.session_state.links[l_idx] = (new_target, old_l[1], old_l[2])
                            st.session_state.string_data[s_id]["start_node"] = new_target
                        else:
                            l_idx = current_s_data["last_link_idx"]
                            old_l = st.session_state.links[l_idx]
                            st.session_state.links[l_idx] = (old_l[0], new_target, old_l[2])
                            st.session_state.string_data[s_id]["end_node"] = new_target
                        
                        st.session_state.selected_string_id = None
                        st.session_state.preview_new_target = None
                        update_next_nodes()
                        st.rerun() 
            
            st.markdown("---")
            st.markdown("#### 🗑️ 削除する")
            deps = []
            base = NUM_ANCHORS + s_id * NUM_INTERNAL_NODES
            internal_nodes = set(range(base, base + NUM_INTERNAL_NODES))
            for s in active_strings:
                if s["id"] == s_id: continue
                if s["start_node"] in internal_nodes or s["end_node"] in internal_nodes:
                    deps.append(s["id"])
                    
            if deps:
                st.warning(f"⚠️ 接続されている別のひもがあるため削除できません。")
            else:
                if st.button("🗑️ このひもを削除", use_container_width=True):
                    st.session_state.string_data[s_id]["is_deleted"] = True
                    l_start = current_s_data["first_link_idx"]
                    l_end = current_s_data["last_link_idx"]
                    for i in range(l_start, l_end + 1):
                        st.session_state.links[i] = (0, 0, 0)
                    for i in range(base, base + NUM_INTERNAL_NODES):
                        st.session_state.nodes[i]["fixed"] = True
                        st.session_state.nodes[i]["y"] = -9999
                        
                    st.session_state.selected_string_id = None
                    st.session_state.preview_new_target = None
                    update_next_nodes()
                    st.rerun()
        else:
            st.write("操作できるひもがありません。")

# ==========================================
# --- 生成フェーズ ---
# ==========================================
elif st.session_state.app_phase == "generating":
    
    st.title("🏰 ガウディ建築 生成中...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter([p["x"] for p in anchors], [0]*NUM_ANCHORS, color="gray", s=150, zorder=10)
    ax.plot([-20, 20], [0, 0], color="gray", lw=4, zorder=5)

    for l in links:
        if l == (0, 0, 0): continue 
        n1, n2 = nodes[l[0]], nodes[l[1]]
        ax.plot([n1["x"], n2["x"]], [n1["y"], n2["y"]], color="black", lw=4, solid_capstyle="round", zorder=5)

    active_node_indices = set(range(NUM_ANCHORS))
    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if not s.get("is_deleted", False):
                base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                active_node_indices.update(range(base, base + NUM_INTERNAL_NODES))

    all_y = [nodes[i]["y"] for i in active_node_indices]
    min_y = min(all_y) if all_y else -15.0
    bottom_limit = min_y - 5.0
    half_width = max(20.0, abs(bottom_limit) * 0.8)

    ax.set_ylim(5, bottom_limit) 
    ax.set_xlim(-half_width, half_width)
    ax.axis("off")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    raw_img = Image.open(buf)
    resized_img = raw_img.resize((1024, 1024), Image.LANCZOS)
    out_buf = io.BytesIO()
    resized_img.save(out_buf, format="PNG")
    image_bytes = out_buf.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 骨組み (入力画像)")
        st.image(image_bytes, use_container_width=True)
        
    with col2:
        st.subheader("🎨 AI生成結果")
        with st.spinner("AIがレンダリングしています..."):
            generated_img_bytes = generate_castle_image(image_bytes, user_prompt, api_key)
            
            if generated_img_bytes:
                st.image(generated_img_bytes, use_container_width=True)
            
    st.markdown("---")
    if st.button("⬅️ 編集画面に戻る"):
        st.session_state.app_phase = "building"
        st.rerun()
