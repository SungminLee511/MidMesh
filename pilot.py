import torch
import numpy as np
import pyvista as pv
import math
from .utils import identify_s3_shared_edges

def create_triangular_mesh(xmin, xmax, ymin, ymax, side_length):
    """
    Create a mesh of equilateral triangles.
    
    Args:
        xmin, xmax, ymin, ymax: Bounds of the region
        side_length: Side length of the equilateral triangles
    
    Returns:
        nodes: (N, 2) tensor of node coordinates
        triangles: (M, 3) tensor of triangle connectivity
    """
    dx = side_length
    dy = side_length * math.sqrt(3) / 2
    
    nx = int((xmax - xmin) / dx) + 1
    ny = int((ymax - ymin) / dy) + 1
    
    nodes_list = []
    
    for i in range(ny):
        y = ymin + i * dy
        x_offset = (dx / 2) if (i % 2 == 1) else 0
        
        for j in range(nx):
            x = xmin + j * dx + x_offset
            if x <= xmax + dx/2:  
                nodes_list.append([x, y])
    
    nodes = torch.tensor(nodes_list, dtype=torch.float32)
    
    node_map = {}
    idx = 0
    for i in range(ny):
        x_offset = (dx / 2) if (i % 2 == 1) else 0
        for j in range(nx):
            x = xmin + j * dx + x_offset
            if x <= xmax + dx/2:
                node_map[(i, j)] = idx
                idx += 1
    
    triangles_list = []
    
    for i in range(ny - 1):
        if i % 2 == 0:  
            for j in range(nx - 1):
                if (i, j) in node_map and (i, j+1) in node_map and (i+1, j) in node_map:
                    triangles_list.append([
                        node_map[(i, j)],
                        node_map[(i, j+1)],
                        node_map[(i+1, j)]
                    ])
                
                if (i+1, j) in node_map and (i, j+1) in node_map and (i+1, j+1) in node_map:
                    triangles_list.append([
                        node_map[(i+1, j)],
                        node_map[(i, j+1)],
                        node_map[(i+1, j+1)]
                    ])
        else:  
            for j in range(nx):
                if j > 0:
                    if (i, j-1) in node_map and (i, j) in node_map and (i+1, j) in node_map:
                        triangles_list.append([
                            node_map[(i, j-1)],
                            node_map[(i, j)],
                            node_map[(i+1, j)]
                        ])
                
                if j < nx - 1:
                    if (i, j) in node_map and (i+1, j) in node_map and (i+1, j+1) in node_map:
                        triangles_list.append([
                            node_map[(i, j)],
                            node_map[(i+1, j+1)],
                            node_map[(i+1, j)]
                        ])
    
    triangles = torch.tensor(triangles_list, dtype=torch.long)

    return nodes, triangles


import torch

def find_overlapping_triangles(triangles1, triangles2, grid_size=10, batch_size=100):
    device = triangles1.device
    N, _, _ = triangles1.shape
    N_prime, _, _ = triangles2.shape

    if device.type == 'cuda':
        import torch
        torch.cuda.empty_cache() 
        
        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        free_memory_gb = free_memory / (1024**3)
        
        bytes_per_triangle = 3 * 2 * 4
        
        memory_multiplier = 50  
        
        safety_factor = 0.8
        max_memory_per_batch = free_memory * safety_factor
        
        estimated_memory_per_item = N * bytes_per_triangle * memory_multiplier
        
        max_safe_batch_size = max(1, int(max_memory_per_batch / estimated_memory_per_item))
        
        grid_memory_per_cell = N * 4 
        max_grid_cells = max_memory_per_batch / (grid_memory_per_cell * 10) 
        max_safe_grid_size = max(1, int(max_grid_cells ** 0.5))
        
        if batch_size > max_safe_batch_size:
            print(f"Reducing batch_size from {batch_size} to {max_safe_batch_size} due to memory constraints")
            batch_size = max_safe_batch_size
            
        if grid_size > max_safe_grid_size:
            print(f"Reducing grid_size from {grid_size} to {max_safe_grid_size} due to memory constraints")
            grid_size = max_safe_grid_size
            
        print(f"Memory info: {free_memory_gb:.2f}GB free, using batch_size={batch_size}, grid_size={grid_size}")
    
    
    def sign(p1, p2, p3):
        return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - \
               (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])
    
    def segments_intersect(p1, p2, p3, p4):
        d1 = sign(p3, p4, p1)
        d2 = sign(p3, p4, p2)
        d3 = sign(p1, p2, p3)
        d4 = sign(p1, p2, p4)
        return (d1 * d2 < 0) & (d3 * d4 < 0)
    
    def check_triangle_overlaps_batch(batch_tri2, batch_tri1):
        n2 = batch_tri2.shape[0]
        n1 = batch_tri1.shape[0]
        
        tri2_exp = batch_tri2.unsqueeze(1)
        tri1_exp = batch_tri1.unsqueeze(0)
        
        tri2_verts = tri2_exp.reshape(n2, 1, 3, 2).expand(n2, n1, 3, 2)
        
        d1 = sign(tri2_verts, tri1_exp[..., 0:1, :], tri1_exp[..., 1:2, :])
        d2 = sign(tri2_verts, tri1_exp[..., 1:2, :], tri1_exp[..., 2:3, :])
        d3 = sign(tri2_verts, tri1_exp[..., 2:3, :], tri1_exp[..., 0:1, :])
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        v2_in_t1 = ~(has_neg & has_pos)
        
        tri1_verts = tri1_exp.reshape(n2, n1, 3, 2)
        
        d1 = sign(tri1_verts, tri2_exp[..., 0:1, :], tri2_exp[..., 1:2, :])
        d2 = sign(tri1_verts, tri2_exp[..., 1:2, :], tri2_exp[..., 2:3, :])
        d3 = sign(tri1_verts, tri2_exp[..., 2:3, :], tri2_exp[..., 0:1, :])
        
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        v1_in_t2 = ~(has_neg & has_pos)
        
        edge_pairs = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        
        edges2_start = tri2_exp[:, :, edge_pairs[:, 0]].unsqueeze(3)
        edges2_end = tri2_exp[:, :, edge_pairs[:, 1]].unsqueeze(3)
        edges1_start = tri1_exp[:, :, edge_pairs[:, 0]].unsqueeze(2)
        edges1_end = tri1_exp[:, :, edge_pairs[:, 1]].unsqueeze(2)
        
        intersects = segments_intersect(edges2_start, edges2_end, edges1_start, edges1_end)
        edge_ints = intersects.any(dim=2).any(dim=2)
        
        overlaps = v2_in_t1.any(dim=2) | v1_in_t2.any(dim=2) | edge_ints
        
        return overlaps
    
    all_coords = torch.cat([triangles1.reshape(-1, 2), triangles2.reshape(-1, 2)], dim=0)
    min_coords = all_coords.min(dim=0)[0]
    max_coords = all_coords.max(dim=0)[0]
    
    bbox1_min = triangles1.min(dim=1)[0]
    bbox1_max = triangles1.max(dim=1)[0]
    bbox2_min = triangles2.min(dim=1)[0]
    bbox2_max = triangles2.max(dim=1)[0]
    
    cell_size = (max_coords - min_coords) / grid_size
    cell_size = torch.maximum(cell_size, torch.tensor(1e-6, device=device))
    
    cells1_min = ((bbox1_min - min_coords) / cell_size).long().clamp(0, grid_size - 1)
    cells1_max = ((bbox1_max - min_coords) / cell_size).long().clamp(0, grid_size - 1)
    cells2_min = ((bbox2_min - min_coords) / cell_size).long().clamp(0, grid_size - 1)
    cells2_max = ((bbox2_max - min_coords) / cell_size).long().clamp(0, grid_size - 1)
    
    result = [torch.tensor([], dtype=torch.long, device=device) for _ in range(N_prime)]
    
    n_batches = (N_prime + batch_size - 1) // batch_size
    
    print(f"Processing {N_prime} triangles in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, N_prime)
        batch_indices2 = torch.arange(start_idx, end_idx, device=device)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{n_batches}")
        
        batch_cells2_min = cells2_min[start_idx:end_idx]
        batch_cells2_max = cells2_max[start_idx:end_idx]
        
        cell_x = torch.arange(grid_size, device=device)
        cell_y = torch.arange(grid_size, device=device)
        cell_x, cell_y = torch.meshgrid(cell_x, cell_y, indexing='ij')
        cell_coords = torch.stack([cell_x.flatten(), cell_y.flatten()], dim=1)
        
        in_cell1 = (cell_coords.unsqueeze(0) >= cells1_min.unsqueeze(1)) & \
                   (cell_coords.unsqueeze(0) <= cells1_max.unsqueeze(1))
        in_cell1 = in_cell1[..., 0] & in_cell1[..., 1]
        
        batch_in_cell2 = (cell_coords.unsqueeze(0) >= batch_cells2_min.unsqueeze(1)) & \
                         (cell_coords.unsqueeze(0) <= batch_cells2_max.unsqueeze(1))
        batch_in_cell2 = batch_in_cell2[..., 0] & batch_in_cell2[..., 1]
        
        shared_cells = batch_in_cell2.unsqueeze(1) & in_cell1.unsqueeze(0)
        potential_overlaps = shared_cells.any(dim=2)
        
        batch_tri2 = triangles2[start_idx:end_idx]
        
        for i, idx2 in enumerate(batch_indices2):
            print(f"{str(i)}/{len(batch_indices2)}", end='\r')
            candidates = torch.where(potential_overlaps[i])[0]
            
            if candidates.numel() == 0:
                continue
            
            chunk_size = min(10000, candidates.numel())
            n_chunks = (candidates.numel() + chunk_size - 1) // chunk_size
            
            overlapping = []
            
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, candidates.numel())
                chunk_candidates = candidates[chunk_start:chunk_end]
                
                overlaps = check_triangle_overlaps_batch(
                    batch_tri2[i:i+1], 
                    triangles1[chunk_candidates]
                )
                
                overlapping.append(chunk_candidates[overlaps[0]])
            
            if overlapping:
                result[idx2] = torch.cat(overlapping)
    
    return [r.cpu().tolist() for r in result]


def main_pilot(node, triangles, face='max', new_mesh_side_length=3.0, cuda_device='cuda:0'):
    triangle_xz_node = node[triangles][:,:,[0,2]]

    xmin = float(torch.min(triangle_xz_node[:, :, 0]))
    xmax = float(torch.max(triangle_xz_node[:, :, 0]))
    zmin = float(torch.min(triangle_xz_node[:, :, 1]))
    zmax = float(torch.max(triangle_xz_node[:, :, 1]))

    new_node, new_triangle = create_triangular_mesh(xmin, xmax, zmin, zmax, new_mesh_side_length)

    result = find_overlapping_triangles(triangle_xz_node.to(cuda_device), new_node[new_triangle].to(cuda_device), grid_size=15, batch_size=400)

    new_triangle_centroid = torch.mean(new_node[new_triangle], dim=1)
    y_original_node = torch.mean(node[triangles][:, :, 1], dim=-1)

    values_list = [y_original_node[sublist].tolist() for sublist in result]

    new_triangle_edge = identify_s3_shared_edges(new_triangle, device=cuda_device)
    new_triangle_edge = new_triangle_edge[:, :, 0]

    edges = new_triangle_edge.long()
    edges_reversed = torch.stack([edges[:, 1], edges[:, 0]], dim=1)
    all_edges = torch.cat([edges, edges_reversed], dim=0)

    sorted_edges, indices = torch.sort(all_edges[:, 0])
    sorted_second = all_edges[indices, 1]

    unique_nodes, counts = torch.unique_consecutive(sorted_edges, return_counts=True)
    boundaries = torch.cat([torch.tensor([0]).to(counts.device), counts.cumsum(0)])

    neighbors = {}
    for i, node in enumerate(unique_nodes):
        start, end = boundaries[i], boundaries[i+1]
        neighbors[node.item()] = sorted_second[start:end].tolist()

    final_new_triangle_centroid = torch.cat((new_triangle_centroid[:, 0].unsqueeze(-1), torch.zeros(new_triangle_centroid.shape[0]).unsqueeze(-1), new_triangle_centroid[:, 1].unsqueeze(-1)), dim=-1)
    for idx, z_can in enumerate(values_list):
        if z_can:
            if face == 'max':
                final_new_triangle_centroid[idx, 1] = max(z_can)
            elif face == 'min':
                final_new_triangle_centroid[idx, 1] = min(z_can)
            else:
                final_new_triangle_centroid[idx, 1] = sum(z_can) / len(z_can)

    final_new_node = torch.zeros((new_node.shape[0], 3))
    final_new_node[:, [0, 2]] = new_node

    nonzero_mask = final_new_triangle_centroid[:, 1] != 0
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

    if nonzero_indices.numel() > 0:
        vertices = new_triangle[nonzero_indices].flatten()
        y_values = final_new_triangle_centroid[nonzero_indices, 1].unsqueeze(1).repeat(1, 3).flatten()
        
        final_new_node[:, 1].scatter_add_(0, vertices, y_values)
        count = torch.zeros(new_node.shape[0])
        count.scatter_add_(0, vertices, torch.ones_like(y_values))
        
        valid = count > 0
        final_new_node[valid, 1] /= count[valid]
    
    y_values = final_new_node[new_triangle][:, :, 1]
    valid_triangle_mask = y_values.min(dim=1)[0] != 0
    final_new_triangle = new_triangle[valid_triangle_mask]

    return final_new_node, final_new_triangle, neighbors

def make_mid_pilot(max_node, min_node, max_triangle, max_neighbors, iterations=5, lambda_factor=0.5):
    mid_node = max_node.clone()

    unique_indices = torch.unique(max_triangle.flatten())
    ydiff = max_node[unique_indices, 1] - min_node[unique_indices, 1]

    small_diff_mask = ydiff < 4
    small_indices = unique_indices[small_diff_mask]
    mid_node[small_indices, 1] = (max_node[small_indices, 1] + min_node[small_indices, 1]) / 2

    large_diff_indices = set(unique_indices[~small_diff_mask].tolist())

    print(f"Small diff count: {small_indices.shape[0]}, Large diff count: {len(large_diff_indices)}")

    from collections import deque

    valid_nodes = set(max_triangle.flatten().tolist())
    new_neighbors = {
        node: [n for n in max_neighbors[node] if n in valid_nodes]
        for node in valid_nodes
        if node in max_neighbors
    }

    new_neighbors = {k: v for k, v in new_neighbors.items() if v}

    for idx in large_diff_indices:
        visited = {idx}
        queue = deque(new_neighbors.get(idx, []))
        
        found = False
        while queue and not found:
            nei = queue.popleft()
            if nei in visited:
                continue
            visited.add(nei)
            
            if nei not in large_diff_indices:
                mid_node[idx, 1] = mid_node[nei, 1]
                found = True
            else:
                queue.extend(new_neighbors.get(nei, []))
        
        if not found:
            mid_node[idx, 1] = (max_node[idx, 1] + min_node[idx, 1]) / 2

    def smooth_mesh(vertices, triangles, iterations=5, lambda_factor=0.5):
        """
        Smooth mesh using Laplacian smoothing
        vertices: [N, 3] vertex positions
        triangles: [T, 3] triangle indices
        """
        edges = torch.cat([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]]
        ], dim=0)
        
        edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
        
        edges = torch.unique(edges, dim=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        
        smoothed = vertices.clone()
        
        for _ in range(iterations):
            new_positions = torch.zeros_like(smoothed)
            counts = torch.zeros(len(smoothed))
            
            sources = edges[:, 0]
            targets = edges[:, 1]
            new_positions.index_add_(0, sources, smoothed[targets])
            counts.index_add_(0, sources, torch.ones(len(targets)))
            
            mask = counts > 0
            new_positions[mask] /= counts[mask].unsqueeze(1)
            smoothed[mask] = (1 - lambda_factor) * smoothed[mask] + lambda_factor * new_positions[mask]
        
        return smoothed

    mid_node_smooth = smooth_mesh(mid_node, max_triangle, iterations=iterations, lambda_factor=lambda_factor)

    return mid_node_smooth, max_triangle