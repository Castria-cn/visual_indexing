import io
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import networkx as nx
import plotly.io as pio
from typing import List, Union
import plotly.graph_objects as go

def split_text(text, line_length):
    return "<br>".join([text[i:i+line_length] for i in range(0, len(text), line_length)])

def weight2color(weight: float):
    if weight > 0.4:
        return 'red'
    if weight > 0.3:
        return 'orange'
    if weight > 0.2:
        return 'yellow'
    return '#888'

def images_to_video(images: List[Image.Image], output_path: str, fps: int=30, duration: Union[int, float]=2):
    assert len(images) != 0

    width, height = images[0].size
    print(images[0].size)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_per_image = int(fps * duration)

    for img in images:
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        for _ in range(frames_per_image):
            video_writer.write(open_cv_image)

    video_writer.release()

def visualize_tree(data, max_line_length=30, weight_threshold=0.5, show: bool=False, export_path: str=None, return_img: bool=False):
    if isinstance(data, str):
        with open(data, 'r') as fp:
            data = json.load(fp)

    G = nx.Graph()

    colors = list()
    # 1. 添加节点
    id_to_desc = {}
    for node in data['nodes']:
        node_id = node['id']
        desc = node['desc']
        layer = node['layer']
        
        if 'selected' in node:
            colors.append('red' if node['selected'] else 'blue')
        G.add_node(node_id, layer=layer)
        id_to_desc[node_id] = split_text(desc, max_line_length)

    # 2. 添加边
    for edge in data['edges']:
        id0, id1, weight = edge
        G.add_edge(id0, id1, weight=weight)

    layers = nx.get_node_attributes(G, 'layer')

    # 3. 创建布局
    pos = {}
    layer_nodes = {}
    for node, layer in layers.items():
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)

    for layer, nodes_in_layer in layer_nodes.items():
        x_positions = list(range(len(nodes_in_layer)))
        y_position = -layer
        for i, node in enumerate(nodes_in_layer):
            pos[node] = (x_positions[i], y_position)

    # 4. 绘制每条边并设置颜色
    edge_traces = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        weight = edge[2]['weight']
        color = weight2color(weight)

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=color),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    # 5. 提取节点的位置
    node_x = []
    node_y = []
    node_descriptions = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_descriptions.append(id_to_desc[node])

    # 6. 创建节点
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        hoverinfo='text',
        hovertext=node_descriptions,
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=colors if len(colors) else None,
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
        ))

    # 7. 创建图表
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(title='Multi-layer Tree-like Layout Visualization (Top-Down)',
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=0, l=0, r=0, t=40))

    if show:
        fig.show()
    
    if export_path:
        fig.write_image(export_path, scale=2.0)
    
    if return_img:
        image = pio.to_image(fig, format='png', scale=2.0)
        image = Image.open(io.BytesIO(image))

        return image
    
def animate(data, video_path: str, weight_threshold=0.5, duration: Union[int, float]=1.5):
    if isinstance(data, str):
        with open(data, 'r') as fp:
            data = json.load(fp)
    assert "init" in data, "Animate retrieval must have `init` in data!"
    assert "traj" in data, "Animate retrieval must have `traj` in data!"
    
    selected = set(data['init'])
    data['nodes'] = [dct | {"selected": dct["id"] in selected} for dct in data['nodes']]

    frames = [visualize_tree(data, return_img=True)]
    
    for new_node in tqdm(data['traj'], 'Exporting video...'):
        selected.add(new_node)
        data['nodes'] = [dct | {"selected": dct["id"] in selected} for dct in data['nodes']]
        frames.append(visualize_tree(data, return_img=True))
    
    images_to_video(frames, output_path=video_path, fps=30, duration=duration)