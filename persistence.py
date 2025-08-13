import numpy as np
import matplotlib.pyplot as plt
import gudhi
from gudhi.representations import PersistenceImage

# ========================
# 0) 可调参数
# ========================
SEED = 42
N_VERTS = 10           # 顶点数
PI_RES = (50, 50)      # Persistence Image 分辨率
PI_BW  = 0.05          # PI 高斯带宽
WEIGHT_MODE = "p"      # "p": 使用持久度(d-b)做权重; "1": 常数权重

# ========================
# 1) 构造 lower-star 过滤的 2-骨架单纯复形
# ========================
def build_lower_star_complex_with_triangles(n_verts=10, seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n_verts, 2))
    f = pts[:, 1]  # 用 y 坐标作为标量场

    st = gudhi.SimplexTree()
    # 0-单形
    for i in range(n_verts):
        st.insert([i], filtration=float(f[i]))
    # 1-单形
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            st.insert([i, j], filtration=float(max(f[i], f[j])))
    # 2-单形
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            for k in range(j + 1, n_verts):
                st.insert([i, j, k], filtration=float(max(f[i], f[j], f[k])))

    st.make_filtration_non_decreasing()
    return st, pts, f

# ========================
# 2) 自检：打印 extended_persistence 原始结构摘要
# ========================
def inspect_ext_structure(ext):
    try:
        top_len = len(ext)
    except Exception:
        top_len = None
    print(f"[Debug] type(ext)={type(ext)}, len(ext)={top_len}")

    # 如果是 list，打印前4个子块长度
    if isinstance(ext, list):
        for idx, blk in enumerate(ext[:4]):
            try:
                print(f"  [Debug] block {idx}: type={type(blk)}, len={len(blk)}")
            except Exception:
                print(f"  [Debug] block {idx}: type={type(blk)}, len=?")
            # 打印子块的前2项示例
            if isinstance(blk, list) and len(blk) > 0:
                for j, it in enumerate(blk[:2]):
                    print(f"    [Debug] sample item {j}: {repr(it)}")
    else:
        # 扁平或迭代器：打印前 6 项
        try:
            it = iter(ext)
            for j in range(6):
                try:
                    sample = next(it)
                    print(f"  [Debug] sample item {j}: {repr(sample)}")
                except StopIteration:
                    break
        except TypeError:
            print("  [Debug] ext is not iterable")

# ========================
# 3) 解析 extended_persistence（多格式兼容）
# ========================
def parse_extended_items(ext):
    pairs = []

    def handle_item(item):
        dim = b = d = t = None
        # ((dim,(b,d)), type)
        if (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple)):
            (dim, bd), t = item
            if isinstance(bd, tuple) and len(bd) == 2:
                b, d = bd
        # (dim,(b,d), type)
        elif (isinstance(item, tuple) and len(item) == 3 and isinstance(item[1], tuple)):
            dim, bd, t = item
            if isinstance(bd, tuple) and len(bd) == 2:
                b, d = bd
        else:
            return False

        try:
            pairs.append({
                "dim": int(dim),
                "birth": float(b),
                "death": float(d),
                "type": str(t)
            })
            return True
        except Exception:
            return False

    # 情况 A：四个子图构成的 list
    if isinstance(ext, list):
        for blk in ext:
            if isinstance(blk, list):
                for item in blk:
                    handle_item(item)
            else:
                # 扁平混入
                handle_item(blk)
    else:
        # 情况 B：迭代器或扁平列表
        try:
            for item in ext:
                handle_item(item)
        except TypeError:
            # 单个对象（极少见）
            handle_item(ext)

    return pairs

# ========================
# 4) 绘图（带空/无穷检查）
# ========================
def plot_extended_pd(pairs, outfile="extended_persistence_diagram.png"):
    clean = [p for p in pairs
             if np.isfinite(p["birth"]) and np.isfinite(p["death"])]
    if len(clean) == 0:
        raise ValueError("No finite (birth, death) pairs to plot. "
                         "Check extended_persistence() output & filtration.")

    births = np.array([p["birth"] for p in clean], dtype=float)
    deaths = np.array([p["death"] for p in clean], dtype=float)

    lo = float(min(births.min(), deaths.min()))
    hi = float(max(births.max(), deaths.max()))
    margin = 0.05 * (hi - lo + 1e-8)
    lo -= margin; hi += margin

    mask_ord = births < deaths
    mask_ext = deaths < births
    mask_eq  = ~mask_ord & ~mask_ext

    plt.figure(figsize=(5, 5))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="diagonal")
    if np.any(mask_ord):
        plt.scatter(births[mask_ord], deaths[mask_ord], marker="o", label="ordinary (b<d)")
    if np.any(mask_ext):
        plt.scatter(births[mask_ext], deaths[mask_ext], marker="x", label="extended (d<b)")
    if np.any(mask_eq):
        plt.scatter(births[mask_eq], deaths[mask_eq], marker="s", label="equal (b=d)")

    plt.xlabel("Birth"); plt.ylabel("Death")
    plt.title("Extended Persistence Diagram (Gudhi)")
    plt.legend()
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"[Saved] {outfile}")

# ========================
# 5) PI（仅 ordinary；带空检查）
# ========================
def compute_and_plot_pi_from_ordinary(pairs,
                                      resolution=(50, 50),
                                      bandwidth=0.05,
                                      weight_mode="p",
                                      outfile="persistence_image.png"):
    # 仅取 ordinary & 有限
    ord_pts = np.array([[p["birth"], p["death"]]
                        for p in pairs
                        if (p["birth"] < p["death"]
                            and np.isfinite(p["birth"])
                            and np.isfinite(p["death"]))],
                       dtype=float)
    if ord_pts.size == 0:
        raise ValueError("No ordinary finite points (birth<death) for PI.")

    # 兼容 1D/2D 输入的权重函数（Gudhi 会按点逐个调用，传入 shape=(2,)）
    if weight_mode == "p":
        def weight_fn(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return max(float(x[1] - x[0]), 0.0)
            else:
                return (x[:, 1] - x[:, 0]).clip(min=0.0)
    elif weight_mode == "1":
        def weight_fn(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return 1.0
            else:
                return np.ones((x.shape[0],), dtype=float)
    else:
        raise ValueError("weight_mode must be 'p' or '1'.")

    H, W = resolution
    PI = PersistenceImage(resolution=resolution, bandwidth=bandwidth, weight=weight_fn)

    # 注意：Gudhi 返回的是展平向量（H*W,）
    img_flat = PI.fit_transform([ord_pts])[0]
    if img_flat.ndim != 1 or img_flat.size != H * W:
        raise ValueError(f"Unexpected PI shape: {img_flat.shape}, expected {(H*W,)}")

    img = img_flat.reshape(H, W)

    # 可视化
    plt.figure(figsize=(5, 4.5))
    plt.imshow(img, origin="lower", aspect="auto")
    plt.xlabel("Birth-axis (discretized)")
    plt.ylabel("Persistence-axis (discretized)")
    plt.title("Persistence Image (from ordinary pairs)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"[Saved] {outfile}")

# ========================
# 6) 主流程：先尝试扩展持久性；若空则回退到普通持久性
# ========================
if __name__ == "__main__":
    print(f"[Info] Gudhi version: {getattr(gudhi, '__version__', 'unknown')}")
    st, pts, f = build_lower_star_complex_with_triangles(n_verts=N_VERTS, seed=SEED)

    # 有些版本/构建需要此步为扩展持久性准备 lower-star 扩展
    try:
        if hasattr(st, "extend_filtration"):
            st.extend_filtration()
            print("[Info] extend_filtration() called.")
    except Exception as e:
        print(f"[Warn] extend_filtration() failed: {e}")

    # ---- 1) 试 extended_persistence() ----
    parsed_pairs = []
    try:
        ext = st.extended_persistence()
        print("[Info] extended_persistence() returned. Inspecting structure ...")
        inspect_ext_structure(ext)
        parsed_pairs = parse_extended_items(ext)
    except Exception as e:
        print(f"[Warn] extended_persistence() raised: {e}")

    # ---- 2) 如解析后仍为空，回退到普通持久性 ----
    if len(parsed_pairs) == 0:
        print("[Info] No pairs parsed from extended_persistence(). Falling back to ordinary persistence.")
        st.compute_persistence()
        ord_list = st.persistence()  # [(dim,(b,d)), ...]
        # 解析 ordinary：赋一个统一的 type 标签 'ordinary'
        for item in ord_list:
            try:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple):
                    dim, bd = item
                    b, d = bd
                    parsed_pairs.append({
                        "dim": int(dim),
                        "birth": float(b),
                        "death": float(d),
                        "type": "ordinary"
                    })
            except Exception:
                continue

    # ---- 3) 统计并绘图 ----
    n_all = len(parsed_pairs)
    n_ord = sum(1 for p in parsed_pairs if (p["birth"] < p["death"]
                                            and np.isfinite(p["birth"])
                                            and np.isfinite(p["death"])))
    n_ext = sum(1 for p in parsed_pairs if (p["death"] < p["birth"]
                                            and np.isfinite(p["birth"])
                                            and np.isfinite(p["death"])))
    n_eq  = sum(1 for p in parsed_pairs if (p["birth"] == p["death"]
                                            and np.isfinite(p["birth"])))

    print(f"[Summary] Total pairs (parsed): {n_all} | ordinary: {n_ord} | extended: {n_ext} | equal: {n_eq}")

    # 画 PD（此时 parsed_pairs 至少来自 ordinary 持久化，不会为空）
    plot_extended_pd(parsed_pairs, outfile="extended_persistence_diagram.png")

    # 用 ordinary 点做 PI
    compute_and_plot_pi_from_ordinary(
        parsed_pairs,
        resolution=PI_RES,
        bandwidth=PI_BW,
        weight_mode=WEIGHT_MODE,
        outfile="persistence_image.png"
    )

    print("[Done]")