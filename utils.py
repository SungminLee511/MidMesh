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




import torch

def compute_triangle_surface_faces_with_third_node(shell, device="cuda:0"):
    """
    삼각형 정보가 주어지면 1번만 등장하는 변(테두리)을 구하고, 이러한 테두리 삼각형의 테두리가 아닌 3번째 노드 번호 또한 구하는 함수
    3번째 노드 정보는 바깥방향을 구하기 위해 사용된다.

    Input:
        shell (torch.Tensor): Shell connectivity [M, 3] 
    
    Returns:
        surfaces (torch.Tensor): surface connectivity [K, 2]
        3rd id (torch.Tensor): 3rd node id [K]
        triangle_indices (torch.Tensor): triangle index for each surface edge [K]
    """
    shell = shell.to(device)

    edges = torch.cat([
        shell[:, [0, 1]],  # edge between node 0 and node 1
        shell[:, [1, 2]],  # edge between node 1 and node 2
        shell[:, [2, 0]],  # edge between node 2 and node 0
    ], dim=0)  # [3*M, 2]

    third_nodes = torch.cat([
        shell[:, 2],  # third node for edge [0, 1]
        shell[:, 0],  # third node for edge [1, 2]
        shell[:, 1],  # third node for edge [2, 0]
    ], dim=0)  # [3*M]

    triangle_indices = torch.cat([
        torch.arange(shell.shape[0], device=device),  # triangle indices for edges [0,1]
        torch.arange(shell.shape[0], device=device),  # triangle indices for edges [1,2] 
        torch.arange(shell.shape[0], device=device),  # triangle indices for edges [2,0]
    ], dim=0)  # [3*M]

    sorted_edges, _ = torch.sort(edges, dim=1)
    
    _, inverse_indices, counts = torch.unique(sorted_edges, dim=0, return_inverse=True, return_counts=True)
    
    surface_edge_mask = counts[inverse_indices] == 1
    surface_edges = edges[surface_edge_mask]
    surface_third_nodes = third_nodes[surface_edge_mask]
    surface_triangle_indices = triangle_indices[surface_edge_mask] 
    
    return surface_edges, surface_third_nodes, surface_triangle_indices # [K, 2], [K], [K]

def check_water_tightness(faces, device="cuda:0"):
    surface_edges, _, __ = compute_triangle_surface_faces_with_third_node(faces, device=device)
    if len(surface_edges) == 0:
        return True
    else:
        return False

def eliminate_non_water_tight_triangles(faces, device="cuda:0"):
    surface_edges, _, triangle_indices = compute_triangle_surface_faces_with_third_node(faces.clone(), device=device)
    while len(surface_edges) > 0:
        dangling_triangles = torch.unique(triangle_indices)
        not_dangling_triangles_mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        not_dangling_triangles_mask[dangling_triangles] = False
        faces = faces[not_dangling_triangles_mask, :]
        surface_edges, _, triangle_indices = compute_triangle_surface_faces_with_third_node(faces.clone(), device=device)
    
    return faces



def triangle_to_edge(faces):
    """
    삼각형 면 정보가 주어지면, 각 삼각형 면을 구성하는 변 정보를 구하는 함수

    Input:
        faces (torch.Tensor): Triangle connectivity [M, 3] 

    Returns:
        edges (torch.Tensor): Edge connectivity [3*M, 2]
    """
    edges = torch.cat([
        faces[:, [0, 1]],  # edge between node 0 and node 1
        faces[:, [1, 2]],  # edge between node 1 and node 2
        faces[:, [2, 0]],  # edge between node 2 and node 0
    ], dim=0)  # [3*M, 2]

    edges, _ = torch.sort(edges, dim=1)

    edges = torch.unique(edges, dim=0, return_inverse=False, return_counts=False) # [E, 2]

    return edges.t() # [2, E]

def tetrahedral_to_edge(tetrahedral):
    """
    사면체 정보가 주어지면, 각 사면체를 구성하는 변 정보를 구하는 함수

    Input:
        tetrahedral (torch.Tensor): Tetrahedral connectivity [M, 4] 

    Returns:
        edges (torch.Tensor): Edge connectivity [6*M, 2]
    """
    edges = torch.cat([
        tetrahedral[:, [0, 1]],  # edge between node 0 and node 1
        tetrahedral[:, [0, 2]],  # edge between node 0 and node 2
        tetrahedral[:, [0, 3]],  # edge between node 0 and node 3
        tetrahedral[:, [1, 2]],  # edge between node 1 and node 2
        tetrahedral[:, [1, 3]],  # edge between node 1 and node 3
        tetrahedral[:, [2, 3]],  # edge between node 2 and node 3
    ], dim=0)  # [6*M, 2]

    edges, _ = torch.sort(edges, dim=1)

    edges = torch.unique(edges, dim=0, return_inverse=False, return_counts=False) # [E, 2]

    return edges.t() # [2, E]