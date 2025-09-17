import plotly.graph_objects as go

def vis(node):
    fig = go.Figure(data=[go.Scatter3d(
        x=node[:, 0].numpy(),
        y=node[:, 1].numpy(),
        z=node[:, 2].numpy(),
        mode='markers',
        marker=dict(size=2)
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'  
    ))
    fig.show()



import torch
import numpy as np
import plotly.graph_objects as go

# plotly 3D mesh visualization

def visualize_mesh(coords, elements, element_type, title='Mesh Visualization'):
    """
    Visualize a 3D mesh using Plotly go.Mesh3d.
    
    Parameters:
      coords (torch.Tensor or array-like): Node coordinates [N, 3].
      elements (torch.Tensor or array-like): Element connectivity.
         For volume elements the connectivity is assumed to be:
           - c3d6: shape (M, 6) wedge
           - c3d8: shape (M, 8) hexahedron
           - c3d4: shape (M, 4) tetrahedron
         For shell elements:
           - s3: shape (M, 3) (triangles)
           - s4: shape (M, 4) (quadrilaterals)
      element_type (str): One of 'c3d6', 'c3d8', 'c3d4', 's3', or 's4'.
      title (str): Title for the Plotly figure.
    """
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    elements_np = elements.cpu().numpy() if isinstance(elements, torch.Tensor) else np.array(elements)
    
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    
    if element_type in ['s3', 's4']:
        if element_type == 's3':
            triangles = elements_np
        elif element_type == 's4':
            tris1 = elements_np[:, [0, 1, 2]]
            tris2 = elements_np[:, [0, 2, 3]]
            triangles = np.concatenate([tris1, tris2], axis=0)
    
    elif element_type in ['c3d6', 'c3d8', 'c3d4']:
        if element_type == 'c3d4':
            face_defs = np.array([
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ]) 
            all_faces = np.take(elements_np, face_defs, axis=1).reshape(-1, 3)
            
        elif element_type == 'c3d8':
            face_defs = np.array([
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # side 1
                [1, 2, 6, 5],  # side 2
                [2, 3, 7, 6],  # side 3
                [3, 0, 4, 7]   # side 4
            ]) 
            all_faces = np.take(elements_np, face_defs, axis=1).reshape(-1, 4)
            tris1 = all_faces[:, [0, 1, 2]]
            tris2 = all_faces[:, [0, 2, 3]]
            all_faces = np.concatenate([tris1, tris2], axis=0) 
            
        elif element_type == 'c3d6':
            tri_bottom = elements_np[:, [0, 1, 2]]
            tri_top    = elements_np[:, [3, 4, 5]]
            quad1 = elements_np[:, [0, 1, 4, 3]]
            quad2 = elements_np[:, [1, 2, 5, 4]]
            quad3 = elements_np[:, [2, 0, 3, 5]]
            quad1_tri1 = quad1[:, [0, 1, 2]]
            quad1_tri2 = quad1[:, [0, 2, 3]]
            quad2_tri1 = quad2[:, [0, 1, 2]]
            quad2_tri2 = quad2[:, [0, 2, 3]]
            quad3_tri1 = quad3[:, [0, 1, 2]]
            quad3_tri2 = quad3[:, [0, 2, 3]]
            all_faces = np.concatenate([
                tri_bottom, tri_top,
                quad1_tri1, quad1_tri2,
                quad2_tri1, quad2_tri2,
                quad3_tri1, quad3_tri2
            ], axis=0)
        
        all_faces_sorted = np.sort(all_faces, axis=1)
        unique_faces, counts = np.unique(all_faces_sorted, axis=0, return_counts=True)
        boundary_faces = unique_faces[counts == 1]
        triangles = boundary_faces

    else:
        raise ValueError(f"Unsupported element type: {element_type}")
    
    i = triangles[:, 0].astype(np.int64)
    j = triangles[:, 1].astype(np.int64)
    k = triangles[:, 2].astype(np.int64)
    
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        color='lightgrey',
        flatshading=True,
    )
    
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data')
    )
    fig.show()

def visualize_node_with_value(coords, value, size=0.5, colorscale='Viridis',
                               value_name = 'Stress',
                               save_path = None,
                               show_axis=False, camera = {
                                    "eye": {"x": 0, "y": 0, "z": 2},   
                                    "center": {"x": 0, "y": 0, "z": 0},    
                                    "up": {"x": 0, "y": 1, "z": 0}         
                                }):
    """
    노드 단위 시각화

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        value (torch.Tensor): Node-wise von Mises stress [N]
        size (float): Marker size
        colorscale (str): Plotly color scale
        title (str): Plot title
        bar_title (str): Colorbar title
        show_axis (bool): Show grid and axis lines if True
        camera (dict): Plotly camera setting
    """
    coords = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)
    value = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.array(value)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    stress = value

    hover_text = [f"Node {i}<br>{value_name}: {s:.2f}" for i, s in enumerate(stress)]

    fig = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=size,
            color=stress,
            colorscale=colorscale,
            colorbar=dict(
                orientation='h',
                x=0.5,
                y=-0.3,
                xanchor='center',
                len=0.35,
                thickness=10),
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text'
    ))

    axis_style = dict(showbackground=show_axis,
                      showgrid=show_axis,
                      showline=show_axis,
                      zeroline=show_axis,
                      visible=show_axis,
                      showticklabels=show_axis,
                      ticks='' if show_axis else None,
                      title='' if not show_axis else None)

    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=camera
        ),
        title={
            'text': f"Node-Wise {value_name} Visualization",
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    if save_path is None:
        fig.show()

    if save_path is not None:
        fig.write_image(save_path)
        print(f"Figure saved to {save_path}")


def visualize_target_nodes(coords, node_ids=None, target_name="Boundary", marker_size=1,
                           target_marker_size=3,
                           show_axis=False, camera = {
                                "eye": {"x": 0, "y": 0, "z": 2},   
                                "center": {"x": 0, "y": 0, "z": 0},    
                                "up": {"x": 0, "y": 1, "z": 0}         
                            },
                            save_path=None):
    """
    Plot nodes in 3D with plotly.

    coords: [N, 3] tensor of node coordinates
    node_ids: [M] tensor of node indices to be colored red
    marker_size: size of the markers in the plot
    target_marker_size: size for highlighted nodes
    show_axis: whether to show grid/axes
    camera: camera dict to control view
    """
    coords = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords)

    N = coords.shape[0]
    colors = ['blue'] * N
    marker_sizes = [marker_size] * N
    if node_ids is not None:
        for node_id in node_ids:
            colors[node_id] = 'red'
            marker_sizes[node_id] = target_marker_size

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=colors
        )
    ))

    axis_style = dict(showbackground=show_axis,
                      showgrid=show_axis,
                      showline=show_axis,
                      zeroline=show_axis,
                      visible=show_axis,
                      showticklabels=show_axis,
                      ticks='' if show_axis else None,
                      title='' if not show_axis else None)

    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='data',
            camera=camera
        ),
        title={
            "text":f"{target_name} Node Visualization",
            'x': 0.5,
            'xanchor': 'center'}
    )

    if save_path is None:
        fig.show()

    if save_path is not None:
        fig.write_image(save_path)
        print(f"Figure saved to {save_path}")

## matplotlib 3D scatter visualization

import matplotlib.pyplot as plt

def plot_values(coordinates, values, size=0.5, value_name="Value", title="Title", save_path=None):
    coords_np = coordinates.detach().cpu().numpy() if isinstance(coordinates, torch.Tensor) else np.array(coordinates)
    values_np = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.array(values)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], 
                        c=values_np, cmap='turbo', s=size, alpha=0.7)
    
    ax.set_xlim(coords_np[:, 0].min(), coords_np[:, 0].max())
    ax.set_ylim(coords_np[:, 1].min(), coords_np[:, 1].max())
    ax.set_zlim(coords_np[:, 2].min(), coords_np[:, 2].max())
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label(value_name, rotation=270, labelpad=20)

    x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
    y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()
    z_min, z_max = coords_np[:, 2].min(), coords_np[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_margin = x_range * 0.02
    y_margin = y_range * 0.02
    z_margin = z_range * 0.02
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    ax.set_box_aspect([x_range, y_range, z_range])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_groups(coordinates, group_ids, size=[0.1, 1], title="Title", save_path=None):
    coords_np = coordinates.detach().cpu().numpy() if isinstance(coordinates, torch.Tensor) else np.array(coordinates)
    groups_np = group_ids.detach().cpu().numpy() if isinstance(group_ids, torch.Tensor) else np.array(group_ids)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_groups = np.unique(groups_np)
    
    # Custom standout colors: blue, red, green, yellow, purple, then others
    standout_colors = [
        '#1f77b4',  # Blue
        '#d62728',  # Red  
        '#2ca02c',  # Green
        '#ffff00',  # Yellow (bright)
        '#9467bd',  # Purple
        '#ff7f0e',  # Orange
        '#e377c2',  # Pink
        '#17becf',  # Cyan
        '#bcbd22',  # Olive
        '#8c564b',  # Brown
        '#000000',  # Black
        '#ff1493',  # Deep pink
        '#00ff00',  # Lime
        '#ff4500',  # Orange red
        '#4169e1'   # Royal blue
    ]
    
    for i, group in enumerate(unique_groups):
        color = standout_colors[i % len(standout_colors)]  # Cycle through colors if more groups
        mask = groups_np == group
        ax.scatter(coords_np[mask, 0], coords_np[mask, 1], coords_np[mask, 2],
                  c=color, label=f'Group {int(group)}', s=size[i if i < len(size) else -1], alpha=0.7)

    x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
    y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()
    z_min, z_max = coords_np[:, 2].min(), coords_np[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_margin = x_range * 0.02
    y_margin = y_range * 0.02
    z_margin = z_range * 0.02
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    ax.set_box_aspect([x_range, y_range, z_range])
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()