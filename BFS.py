import torch

def compute_hop_iteration_with_path(edge, num_nodes, start_node, device="cuda:0"):
    edge = edge.to(device)
    start_node = start_node.to(device)
    if edge.shape[0] == 2:
        edge = edge.t()
    src = edge[:, 0]
    dst = edge[:, 1]
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    ones = torch.ones(edge.shape[0], device=device)
    weight_all = torch.cat([ones, ones])
    f = torch.full((int(num_nodes),), float('inf'), device=device)
    f[start_node] = 0.0
    pred = torch.full((int(num_nodes),), -1, dtype=torch.long, device=device)
    current = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    current[start_node] = True
    while True:
        mask = current[src_all]
        if not mask.any():
            break
        cand_src = src_all[mask]
        cand_dst = dst_all[mask]
        cand_values = f[cand_src] + weight_all[mask]
        unique_nodes, inv = torch.unique(cand_dst, return_inverse=True)
        new_values = torch.full((unique_nodes.shape[0],), float('inf'), device=device)
        new_values = new_values.scatter_reduce(0, inv, cand_values, reduce="amin", include_self=False)
        old_values = f[unique_nodes]
        update_mask = new_values < old_values
        if not update_mask.any():
            break
        updated_nodes = unique_nodes[update_mask]
        f[updated_nodes] = new_values[update_mask]
        update_mask_b = update_mask[inv]
        cand_dst_upd = cand_dst[update_mask_b]
        cand_src_upd = cand_src[update_mask_b]
        best = torch.full((int(num_nodes),), -1, dtype=torch.long, device=device)
        best_values = torch.full((int(num_nodes),), float('inf'), device=device)
        best_values.index_copy_(0, cand_dst_upd, cand_values[update_mask_b])
        best.index_copy_(0, cand_dst_upd, cand_src_upd)
        pred[updated_nodes] = best[updated_nodes]
        current = torch.zeros_like(current)
        current[updated_nodes] = True
    return f, pred

def reconstruct_path(pred, target):
    if isinstance(target, int):
        target = torch.tensor(target, dtype=torch.long)
    elif isinstance(target, torch.Tensor) and target.dim() == 0:
        target = target.unsqueeze(0)
    path = []
    while target != -1:
        path.append(target.item())
        target = pred[target]
    return path[::-1]