import torch
import pyvista as pv
import numpy as np

def save_mesh_to_vtp(node: torch.Tensor, triangle: torch.Tensor, filename: str):
    points = node.cpu().numpy() if isinstance(node, torch.Tensor) else node
    triangle = triangle.cpu() if isinstance(triangle, torch.Tensor) else triangle
    faces = np.hstack([np.full((triangle.shape[0], 1), 3), triangle]).astype(np.int64)
    faces = faces.flatten()
    mesh = pv.PolyData(points, faces)
    mesh.save(filename)

def read_mesh(file_path):
    if file_path.endswith('.k'):
        return read_k_file(file_path)
    elif file_path.endswith('.vtk') or file_path.endswith('.vtu') or file_path.endswith('.vtp'):
        mesh = pv.read(file_path)
        nodes = torch.tensor(mesh.points, dtype=torch.float32)
        elements = {}
        for cell_type in mesh.celltypes:
            cell_data = mesh.cells_dict[cell_type]
            if cell_type == pv.CellType.TRIANGLE:
                elements['S3'] = torch.tensor(cell_data, dtype=torch.int32)
            elif cell_type == pv.CellType.QUAD:
                elements['S4'] = torch.tensor(cell_data, dtype=torch.int32)
            elif cell_type == pv.CellType.TETRA:
                elements['C3D4'] = torch.tensor(cell_data, dtype=torch.int32)
            elif cell_type == pv.CellType.HEXAHEDRON:
                elements['C3D8'] = torch.tensor(cell_data, dtype=torch.int32)
        return nodes, elements
    elif file_path.endswith('.stl'):
        mesh = pv.read(file_path)
        mesh = mesh.triangulate()
        points = torch.tensor(mesh.points, dtype=torch.float32)
        cells = torch.tensor(mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.int32)
        return points, {'S3': cells}
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
        return None

def read_k_file(file_path):
    '''
    Only Reads *NODE and *ELEMENT from .k format
    '''
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_section = False
    element_section_shell = False
    element_section_solid = False
    nodes = []
    element_shell = {}
    element_solid = {}
    for line in lines:
        if line.strip() == '' or line.startswith('$'):
            continue

        if line.startswith('*NODE'):
            node_section = True
            element_section_shell = False
            element_section_solid = False
            continue

        elif line.startswith('*ELEMENT_SHELL'):
            node_section = False
            element_section_shell = True
            element_section_solid = False
            continue
        
        elif line.startswith('*ELEMENT_SOLID'):
            node_section = False
            element_section_shell = False
            element_section_solid = True
            continue

        elif line.startswith('*'):
            node_section = False
            element_section_shell = False
            element_section_solid = False
            continue

        if node_section:
            parts1 = line.split()
            parts2 = line.split(',')
            parts = parts1 if len(parts1) > len(parts2) else parts2
            if len(parts) >= 4:
                node_id = int(parts[0].strip())
                x = float(parts[1].strip())
                y = float(parts[2].strip())
                z = float(parts[3].strip())
                nodes.append([node_id, x, y, z])
        
        elif element_section_shell:
            parts1 = line.split()
            parts2 = line.split(',')
            parts = parts1 if len(parts1) > len(parts2) else parts2
            if len(parts) >= 3:
                elem_type = len(parts) - 2
                elem_id = int(parts[0].strip())
                node_ids = [int(pid.strip()) for pid in parts[2:]]
                if elem_type not in element_shell:
                    element_shell[elem_type] = []
                element_shell[elem_type].append([elem_id] + node_ids)
        
        elif element_section_solid:
            parts1 = line.split()
            parts2 = line.split(',')
            parts = parts1 if len(parts1) > len(parts2) else parts2
            if len(parts) >= 3:
                elem_type = len(parts) - 2
                elem_id = int(parts[0].strip())
                node_ids = [int(pid.strip()) for pid in parts[2:]]
                if elem_type not in element_solid:
                    element_solid[elem_type] = []
                element_solid[elem_type].append([elem_id] + node_ids)
    
    elements = {}
    for key in element_shell:
        dict = {3:'S3', 4:'S4', 6:'S6', 8:'S8'}
        elements[dict[key]] = element_shell[key]
    for key in element_solid:
        dict = {4:'C3D4', 8:'C3D8', 10:'C3D10', 20:'C3D20'}
        elements[dict[key]] = element_solid[key]
    
    nid = {node[0]: idx for idx, node in enumerate(nodes)}

    nodes = torch.tensor(nodes, dtype=torch.float32)
    for elem_type in elements:
        for i in range(len(elements[elem_type])):
            elements[elem_type][i] = [nid[nid_] for nid_ in elements[elem_type][i][1:]]
        elements[elem_type] = torch.tensor(elements[elem_type], dtype=torch.int32)
    
    return nodes, elements