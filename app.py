import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import requests
import random
import itertools
from PIL import Image

st.set_page_config(layout="centered")

# --- 設定値 ---
NUM_ANCHORS = 8
anchors_x = np.linspace(-17.5, 17.5, NUM_ANCHORS)
anchors = [{"x": float(x), "y": 0.0} for x in anchors_x]

NUM_NEW_NODES = 41 
NUM_INTERNAL_NODES = NUM_NEW_NODES - 2
MID_NODE_OFFSET = NUM_INTERNAL_NODES // 2 

st.sidebar.title("⚙️ AI生成設定")
st.sidebar.markdown("本物の画像を生成するにはAPIキーを入力してください。")
api_key = st.sidebar.text_input("APIキー (Stability AI)", type="password", help="sk- から始まるキーを入力してください")

st.sidebar.markdown("---")
user_prompt = st.sidebar.text_area(
    "🎨 AIへの指示 (プロンプト)", 
    value="A majestic medieval fantasy castle designed by Antoni Gaudi, intricate stone carving, sunset lighting, epic scale, highly detailed, masterpiece, 8k resolution",
    height=150
)

def format_node_label(idx):
    if idx < NUM_ANCHORS:
        positions = ["一番左", "左から2番目", "左から3番目", "中央の左", "中央の右", "右から3番目", "右から2番目", "一番右"]
        return f"天井 {idx + 1} ({positions[idx]})"
    else:
        s_id = (idx - NUM_ANCHORS) // NUM_INTERNAL_NODES
        return f"ひも {s_id + 1} の頂点"

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

# --- 【改良】繋がっていない点をちょっと確率高めに選出するロジック ---
def update_next_nodes():
    candidates = list(range(NUM_ANCHORS)) 
    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if not s.get("is_deleted", False):
                base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                candidates.append(base + MID_NODE_OFFSET)
        
    st.session_state.candidate_pairs = []
    if len(candidates) >= 2:
        all_pairs = list(itertools.combinations(candidates, 2))
        
        existing_pairs = set()
        if "string_data" in st.session_state:
            for s in st.session_state.string_data:
                if not s.get("is_deleted", False):
                    p = tuple(sorted([s["start_node"], s["end_node"]]))
                    existing_pairs.add(p)
                    
        valid_pairs = [p for p in all_pairs if tuple(sorted(p)) not in existing_pairs]
        if not valid_pairs: 
            valid_pairs = all_pairs
            
        # 各点の現在の接続数をカウント
        node_usage = {c: 0 for c in candidates}
        if "string_data" in st.session_state:
            for s in st.session_state.string_data:
                if not s.get("is_deleted", False):
                    if s["start_node"] in node_usage: node_usage[s["start_node"]] += 1
                    if s["end_node"] in node_usage: node_usage[s["end_node"]] += 1

        # まだどこにも繋がっていない点（接続数0）を含むペアを分離
        unused_contained_pairs = [p for p in valid_pairs if node_usage[p[0]] == 0 or node_usage[p[1]] == 0]
        both_used_pairs = [p for p in valid_pairs if node_usage[p[0]] > 0 and node_usage[p[1]] > 0]
        
        random.shuffle(unused_contained_pairs)
        random.shuffle(both_used_pairs)
        
        # 3つの選択肢のうち、最大2つは「繋がっていない点を含むペア」を優先的に配置するブレンド方式
        selected_pairs = []
        if unused_contained_pairs:
            n_unused = min(2, len(unused_contained_pairs))
            selected_pairs.extend(unused_contained_pairs[:n_unused])
            remaining = 3 - n_unused
            selected_pairs.extend(both_used_pairs[:remaining])
            if len(selected_pairs) < 3:
                selected_pairs.extend(unused_contained_pairs[n_unused:])
        else:
            selected_pairs = both_used_pairs
            
        st.session_state.candidate_pairs = selected_pairs[:3]

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
    
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    for _ in range(3):
        update_next_nodes()
        if "candidate_pairs" in st.session_state and st.session_state.candidate_pairs:
            i1, i2 = st.session_state.candidate_pairs[0]
            add_string(i1, i2)
    update_next_nodes()

nodes = st.session_state.nodes
links = st.session_state.links

# ==========================================
# --- 構築フェーズ ---
# ==========================================
if st.session_state.app_phase == "building":

    gravity = 0.08  
    stiffness = 0.95 

    for _ in range(500):
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
    ax.scatter([p["x"] for p in anchors], [0]*NUM_ANCHORS, color="#555555", s=100, zorder=10)

    selected_s_data = None
    if st.session_state.selected_string_id is not None and "string_data" in st.session_state:
         selected_s_data = st.session_state.string_data[st.session_state.selected_string_id]

    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if s.get("is_deleted", False): continue
            
            base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
            node_indices = [s["start_node"]] + list(range(base, base + NUM_INTERNAL_NODES)) + [s["end_node"]]
            xs = [nodes[i]["x"] for i in node_indices]
            ys = [nodes[i]["y"] for i in node_indices]

            color = "#C0C0C0" 
            lw = 3
            z_order = 4
            
            if selected_s_data and s["id"] == selected_s_data["id"]:
                color = "#1C83E1"
                lw = 7
                z_order = 6
                
            ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="round", solid_capstyle="round", zorder=z_order)

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

    # --- 【改良】新規ひも追加後の形に合わせて動的にカメラ枠を調整するプレビュー生成 ---
    def create_preview_image(idx1, idx2):
        p_fig, p_ax = plt.subplots(figsize=(3, 3))
        p_ax.scatter([p["x"] for p in anchors], [0]*NUM_ANCHORS, color="#555555", s=30, zorder=10)

        sim_nodes = [{"x": n["x"], "y": n["y"], "px": n["x"], "py": n["y"], "fixed": n.get("fixed", False)} for n in nodes]
        sim_links = list(links)

        p1, p2 = sim_nodes[idx1], sim_nodes[idx2]
        dist = np.sqrt((p2["x"] - p1["x"])**2 + (p2["y"] - p1["y"])**2)
        total_len = dist * 1.002
        seg_len = total_len / (NUM_NEW_NODES - 1)
        
        new_x = np.linspace(p1["x"], p2["x"], NUM_NEW_NODES)
        new_y = np.linspace(p1["y"], p2["y"], NUM_NEW_NODES) - (dist * 0.01)
        
        start_idx = len(sim_nodes)
        
        for i in range(1, NUM_NEW_NODES - 1):
            sim_nodes.append({"x": new_x[i], "y": new_y[i], "px": new_x[i], "py": new_y[i], "fixed": False})
            
        prev_idx = idx1
        current_new_idx = start_idx
        for i in range(1, NUM_NEW_NODES - 1):
            sim_links.append((prev_idx, current_new_idx, seg_len))
            prev_idx = current_new_idx
            current_new_idx += 1
        sim_links.append((prev_idx, idx2, seg_len))

        gravity = 0.08  
        stiffness = 0.95 
        for _ in range(200):
            for n in sim_nodes:
                if not n["fixed"]:
                    vx = (n["x"] - n["px"]) * 0.75  
                    vy = (n["y"] - n["py"]) * 0.75
                    n["px"], n["py"] = n["x"], n["y"]
                    n["x"] += vx
                    n["y"] += vy - gravity

            for _ in range(4):  
                for l in sim_links:
                    if l == (0, 0, 0): continue 
                    i1, i2, target_dist = l
                    n1, n2 = sim_nodes[i1], sim_nodes[i2]
                    dx = n2["x"] - n1["x"]
                    dy = n2["y"] - n1["y"]
                    d = np.sqrt(dx**2 + dy**2)
                    if d == 0: continue
                    diff = (target_dist - d) / d * 0.5
                    if not n1["fixed"]:
                        n1["x"] -= dx * diff * stiffness
                        n1["y"] -= dy * diff * stiffness
                    if not n2["fixed"]:
                        n2["x"] += dx * diff * stiffness
                        n2["y"] += dy * diff * stiffness

        if "string_data" in st.session_state:
            for s in st.session_state.string_data:
                if s.get("is_deleted", False): continue
                base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                n_idx = [s["start_node"]] + list(range(base, base + NUM_INTERNAL_NODES)) + [s["end_node"]]
                xs = [sim_nodes[i]["x"] for i in n_idx]
                ys = [sim_nodes[i]["y"] for i in n_idx]
                p_ax.plot(xs, ys, color="#E0E0E0", lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=5)

        new_n_idx = [idx1] + list(range(start_idx, start_idx + NUM_INTERNAL_NODES)) + [idx2]
        new_xs = [sim_nodes[i]["x"] for i in new_n_idx]
        new_ys = [sim_nodes[i]["y"] for i in new_n_idx]
        p_ax.plot(new_xs, new_ys, color="#32CD32", lw=2.5, solid_joinstyle="round", solid_capstyle="round", zorder=8)

        # 【修正】この画像内のシミュレーション後の最小yを取得し、はみ出さないカメラ枠を動的に計算
        sim_active_y = [n["y"] for n in sim_nodes if n["y"] > -9000]
        sim_min_y = min(sim_active_y) if sim_active_y else -15.0
        p_bottom_limit = sim_min_y - 3.0
        p_half_width = max(20.0, abs(p_bottom_limit) * 0.8)

        p_ax.set_ylim(p_bottom_limit, 5)
        p_ax.set_xlim(-p_half_width, p_half_width)
        p_ax.axis("off")
        
        buf = io.BytesIO()
        p_fig.savefig(buf, format="png", bbox_inches='tight', dpi=70)
        plt.close(p_fig) 
        return buf.getvalue()

    st.markdown("---")
    if st.button("✨ ガウディ建築を生成する (上下反転してAIに入力)", type="primary", use_container_width=True):
        st.session_state.app_phase = "generating"
        st.rerun()
        
    st.markdown("---")
    
    st.subheader("➕ 次のひもを追加 (画像を見て選択)")
    
    if "candidate_pairs" in st.session_state and st.session_state.candidate_pairs:
        cols = st.columns(len(st.session_state.candidate_pairs))
        for idx, (i1, i2) in enumerate(st.session_state.candidate_pairs):
            with cols[idx]:
                st.caption(f"{format_node_label(i1)} ↔ {format_node_label(i2)}")
                img_bytes = create_preview_image(i1, i2)
                st.image(img_bytes, use_container_width=True)
                
                if st.button(f"✨ 案 {idx+1} を追加", key=f"add_{i1}_{i2}", use_container_width=True):
                    add_string(i1, i2)
                    st.session_state.selected_string_id = None 
                    update_next_nodes()
                    st.rerun()
    else:
        st.write("現在追加できるひもの候補がありません。")

    st.write("---")

    st.subheader("⚙️ 既存のひもの操作")
    active_strings = [s for s in st.session_state.string_data if not s.get("is_deleted", False)]
    if active_strings:
        string_options = {}
        for s in active_strings:
            label = f"ひも {s['id'] + 1}"
            string_options[label] = s['id']
            
        colA, colB = st.columns([1, 1.5])
        
        with colA:
            selected_string_label = st.selectbox("1. 編集対象を選択", list(string_options.keys()))
            s_id = string_options[selected_string_label]
            st.session_state.selected_string_id = s_id
            current_s_data = st.session_state.string_data[s_id]
            
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

        with colB:
            st.markdown("#### 🔄 つなぎ直す")
            end_options = {"始点": "start", "終点": "end"}
            end_type = st.radio("変更する端", list(end_options.keys()), horizontal=True)
            type_key = end_options[end_type]
            st.session_state.selected_end_type = type_key
            current_target = current_s_data["start_node"] if type_key == "start" else current_s_data["end_node"]
            
            reconnect_cands = list(range(NUM_ANCHORS))
            for s in active_strings:
                base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
                reconnect_cands.append(base + MID_NODE_OFFSET)
            
            opposite_end = current_s_data["end_node"] if type_key == "start" else current_s_data["start_node"]
            available_reconnects = [c for c in reconnect_cands if c != opposite_end]
            
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
            else:
                 st.write("空いている接続先がありません。")
                 
    else:
        st.write("操作できるひもがありません。")
        
    st.write("---")
    if st.button("🚨 すべてリセットしてやり直す"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# --- 生成フェーズ ---
# ==========================================
elif st.session_state.app_phase == "generating":
    
    st.title("🏰 ガウディ建築 生成中...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter([p["x"] for p in anchors], [0]*NUM_ANCHORS, color="gray", s=150, zorder=10)
    ax.plot([-20, 20], [0, 0], color="gray", lw=4, zorder=5)

    if "string_data" in st.session_state:
        for s in st.session_state.string_data:
            if s.get("is_deleted", False): continue
            base = NUM_ANCHORS + s["id"] * NUM_INTERNAL_NODES
            node_indices = [s["start_node"]] + list(range(base, base + NUM_INTERNAL_NODES)) + [s["end_node"]]
            xs = [nodes[i]["x"] for i in node_indices]
            ys = [nodes[i]["y"] for i in node_indices]
            ax.plot(xs, ys, color="black", lw=5, solid_joinstyle="round", solid_capstyle="round", zorder=5)

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
