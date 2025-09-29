"""
使用 gudhi
PYG graph 对于节点挖取其子图，得到每个节点的拓扑特征
"""

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph, to_undirected

import gudhi


class TopoPointCloudFeatures:
    """对点云计算简单的 H0 / H1 拓扑统计特征（共 20 维）。"""
    small = 0.01

    def __init__(self, Cut: float = 8.0):
        self.Cut = Cut

    def feature_h0(self, persis):
        # 返回 [sum, min, max, mean, std]（过滤掉极短条）
        Feature_b0 = np.zeros(5, dtype=np.float64)
        if len(persis) == 0:
            return Feature_b0
        tmpbars = np.array([(int(d), float(b[0]), float(b[1])) for d, b in persis],
                           dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[(tmpbars['death'] <= self.Cut) &
                       (tmpbars['dim'] == 0) &
                       (tmpbars['death'] - tmpbars['birth'] >= self.small)]
        if len(bars) > 0:
            lengths = bars['death'] - bars['birth']
            Feature_b0[:] = [np.sum(lengths), np.min(lengths), np.max(lengths),
                             np.mean(lengths), np.std(lengths)]
        return Feature_b0

    def feature_h1h2(self, persis):
        # 返回 H1 的三组统计：len/birth/death 各 5 个，共 15 维
        Feature_b1 = np.zeros(15, dtype=np.float64)
        if len(persis) == 0:
            return Feature_b1
        tmpbars = np.array([(int(d), float(b[0]), float(b[1])) for d, b in persis],
                           dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[(tmpbars['death'] - tmpbars['birth'] >= self.small)]
        betti1 = bars[bars['dim'] == 1]
        if len(betti1) > 0:
            lengths = betti1['death'] - betti1['birth']
            births, deaths = betti1['birth'], betti1['death']
            Feature_b1[0:5]   = [np.sum(lengths), np.min(lengths), np.max(lengths), np.mean(lengths), np.std(lengths)]
            Feature_b1[5:10]  = [np.sum(births),  np.min(births),  np.max(births),  np.mean(births),  np.std(births)]
            Feature_b1[10:15] = [np.sum(deaths),  np.min(deaths),  np.max(deaths),  np.mean(deaths),  np.std(deaths)]
        return Feature_b1

    def compute(self, coords):
        """coords: (N, d) ndarray 或 tensor；返回 20 维特征（5 + 15）"""
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        coords = np.asarray(coords, dtype=np.float64)

        # 若邻域点过少，直接返回全零
        if coords.ndim != 2 or coords.shape[0] < 2:
            return np.zeros(20, dtype=np.float64)

        # Rips -> H0
        rips = gudhi.RipsComplex(points=coords)
        st_rips = rips.create_simplex_tree(max_dimension=1)
        persis_rips = st_rips.persistence()
        f0 = self.feature_h0(persis_rips)

        # Alpha -> H1/H2（这里只用 H1 的统计）
        try:
            alpha = gudhi.AlphaComplex(points=coords)
            st_alpha = alpha.create_simplex_tree()
            persis_alpha = st_alpha.persistence()
        except Exception:
            # 退化情形（例如重复点等），直接给空
            persis_alpha = []
        f1 = self.feature_h1h2(persis_alpha)

        return np.concatenate([f0, f1], dtype=np.float64)


def compute_node_topo_features(
    data,
    k: int = 1,
    pos_key: str = "coor",   # 也可用 "pos"
    cut: float = 8.0,
    undirected: bool = True,
    cap_neighbors: int = None,   # 可选：对非常大的 k-hop 邻域做随机下采样
) -> torch.Tensor:
    """
    给 PyG Data 逐节点计算拓扑特征（基于节点 k-hop 邻域的点云），返回 (num_nodes, 20) 的 tensor。
    你可以随后： data.topo = feats 或 data.x = torch.cat([data.x, feats], dim=1)

    参数：
      data: torch_geometric.data.Data，需要有 data.edge_index 和 data[pos_key] (N, d)
      k:    k-hop 邻域半径
      pos_key: 节点坐标的键（例如 'pos' 或你的 'coor'）
      cut:  H0 截断（与类中一致）
      undirected: 是否先把 edge_index 无向化
      cap_neighbors: 若邻域过大，随机下采样到该数量（保持速度；可为 None）

    返回：
      feats: torch.FloatTensor，形状 (N, 20)
    """
    assert hasattr(data, "edge_index"), "data 需要包含 edge_index"
    assert hasattr(data, pos_key), f"data 需要包含坐标张量 '{pos_key}'"

    edge_index = data.edge_index
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    pos = getattr(data, pos_key)
    if isinstance(pos, torch.Tensor):
        pos_cpu = pos.detach().cpu()
    else:
        pos_cpu = torch.as_tensor(pos, dtype=torch.float32)

    extractor = TopoPointCloudFeatures(Cut=cut)
    N = data.num_nodes
    out = np.zeros((N, 20), dtype=np.float64)

    for v in range(N):
        # 取 v 的 k-hop 邻域（包含 v 本身），并拿到映射
        subset, _, _, _ = k_hop_subgraph(
            node_idx=v,
            num_hops=k,
            edge_index=edge_index,
            relabel_nodes=False  # 我们直接用原始索引拿坐标
        )
        coords = pos_cpu[subset]

        # 如需要可做邻域点随机下采样，避免大图过慢
        if cap_neighbors is not None and coords.size(0) > cap_neighbors:
            perm = torch.randperm(coords.size(0))[:cap_neighbors]
            coords = coords[perm]

        out[v, :] = extractor.compute(coords)

    return torch.from_numpy(out).float()



if __name__ == "__main__":

    import torch
    import networkx as nx
    import numpy as np
    from torch_geometric.utils import from_networkx


    # 1) 构造一个 5x5 的无向网格图，并给每个节点一个 3D 坐标（x, y, 0）
    G = nx.grid_2d_graph(5, 5)  # 25 个节点
    for (x, y) in G.nodes():
        # 作为节点属性存起来，from_networkx 会自动拼成 tensor
        G.nodes[(x, y)]['coor'] = torch.tensor([float(x), float(y), 0.0], dtype=torch.float32)

    # 2) 转成 PyG Data；节点属性 'coor' 会被保留下来
    data = from_networkx(G)
    # 这时 data.coor 的形状应是 [25, 3]
    print("Initial coords shape:", data.coor.shape)   # torch.Size([25, 3])
    print("Num nodes:", data.num_nodes)

    # 3) 计算每个节点的拓扑特征（基于 k-hop 邻域的点云）
    #    - k=2：取 2-hop 邻域
    #    - pos_key="coor"：我们把节点坐标放在 data.coor
    #    - cap_neighbors=64：邻域过大时随机下采样到 64 个点（可按需关闭或调整）
    topo_feats = compute_node_topo_features(
        data,
        k=2,
        pos_key="coor",
        cut=8.0,
        undirected=True,
        cap_neighbors=64
    )

    print("Topo feats shape:", topo_feats.shape)  # torch.Size([25, 20])

    # 4) 把拓扑特征挂到图上（你也可以拼到 data.x）
    data.topo = topo_feats

    # 没有原始 data.x 就直接用 topo 作为节点特征
    data.x = topo_feats.clone()

    # 5) 看看前 3 个节点的拓扑特征
    print("First 3 nodes topo features:")
    print(data.topo[:3])

    