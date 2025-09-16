import torch

def identify_s3_shared_edges(shell, device="cuda:0"):
    """
    Shell 정보가 주어지면 각 변이 공유되는 shell 정보를 구하는 함수
    아웃풋은 [S, 2, 2]이며, 총 S개의 변에 대해 [2, 2] 연결 정보를 제공한다.
    [2, 2]:[[shell, edge index],[shell id, edge index]]

    Inputs:
        shell (torch.Tensor): Shell connectivity [M, 3] 

    Outputs:
        shared_edge_indices (torch.Tensor): Indices of shells sharing each edge [S, 2, 2]
    """
    M = shell.shape[0]
    shell = shell.to(device)
    
    edge_node_indices = torch.tensor([
        [0, 1],  # Edge 0
        [1, 2],  # Edge 1
        [2, 0],  # Edge 2
    ], device=device)  # [3, 2]
    
    edge = shell[:, edge_node_indices]  # [M, 3, 2]
    edges_sorted, _ = torch.sort(edge, dim=2)  # [M, 3, 2]
    edges_flat = edges_sorted.view(-1, 2)  # [M*3, 2]
    
    shell_ids = torch.arange(M, device=device).repeat_interleave(3)  # [M*3]
    edge_indices = torch.tile(torch.arange(3, device=device), (M,))  # [M*3]
    
    _, inverse_indices, counts = torch.unique(
        edges_flat, return_inverse=True, return_counts=True, dim=0
    )
    
    shared_mask = counts == 2
    shared_edge_ids = torch.nonzero(shared_mask, as_tuple=True)[0]  # [S]
    
    if shared_edge_ids.numel() == 0:
        return torch.empty((0, 2, 2), dtype=torch.long, device=device)
    
    sorted_inverse, sorted_order = torch.sort(inverse_indices)
    sorted_shell_ids = shell_ids[sorted_order]
    sorted_edge_indices = edge_indices[sorted_order]
    
    positions = torch.searchsorted(sorted_inverse, shared_edge_ids)
    
    shell1 = sorted_shell_ids[positions]
    edge1 = sorted_edge_indices[positions]
    shell2 = sorted_shell_ids[positions + 1]
    edge2 = sorted_edge_indices[positions + 1]
    
    shared_edge_indices = torch.stack([
        torch.stack([shell1, edge1], dim=1),
        torch.stack([shell2, edge2], dim=1)
    ], dim=1)  # [S, 2, 2]
    
    return shared_edge_indices  # [S, 2, 2]

def compute_s3_normal(coords, shell, device="cuda:0"):
    """
    삼각형 면적에 대응되는 노멀 벡터를 계산합니다.

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        shell (torch.Tensor): Shell connectivity [M, 3] (3 node IDs for each triangle)
        
    Output:
        normal (torch.Tensor): Shell Normal [M, 3]
    """
    coords = coords.to(device)
    shell  = shell.to(device)

    a = coords[shell[:, 1]] - coords[shell[:, 0]] # [M,3]
    b = coords[shell[:, 2]] - coords[shell[:, 0]] # [M,3]

    normal = torch.linalg.cross(a, b, dim=1) * 0.5 # [M,3]

    return normal # [M,3]