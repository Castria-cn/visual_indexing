import io
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
import plotly.io as pio
import plotly.graph_objects as go
from typing import List, Union, Dict, Tuple

def split_text(text, line_length, splitor: str="<br>", return_list: bool=False) -> Union[List, str]:
    if return_list:
        return [text[i:i+line_length] for i in range(0, len(text), line_length)]
    return splitor.join([text[i:i+line_length] for i in range(0, len(text), line_length)])

def weight2color(weight: float):
    if weight > 0.4:
        return 'red'
    if weight > 0.3:
        return 'orange'
    if weight > 0.2:
        return 'yellow'
    return '#888'

def draw_text_on_image(image: Image.Image,
                       text: str,
                       rect: Tuple[float, float],
                       font_size: int=20,
                       max_length: int=20) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    text = split_text(text, line_length=max_length, splitor="\n")
    draw.text((int(w * rect[0]), int(h * rect[1])), text, fill=(0, 0, 0), font_size=font_size)

    return image

def images_to_video(images: List[Image.Image], output_path: str, fps: int=30, duration: Union[int, float]=2) -> None:
    """
    Transform list of images to video.
    - images: List[PIL.Image.Image]
    - output_path: path of the target video
    - fps: frame per second
    - duration: duration time of each image
    """
    assert len(images) != 0

    width, height = images[0].size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_per_image = int(fps * duration)

    for img in images:
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        for _ in range(frames_per_image):
            video_writer.write(open_cv_image)

    video_writer.release()

def visualize_tree(data: Union[str, Dict],
                   max_line_length: int=30,
                   weight_threshold: float=0.02,
                   show: bool=False,
                   export_path: str=None,
                   return_img: bool=False) -> Union[None, Image.Image]:
    """
    Visualize the given garlic tree.
    - data: data exported by `GarlicTree.as_json`
    - max_line_length: max line length when showing text on the node
    - weight_threshold: edge weight less than this threshold will be hidden
    - show: whether to show the image
    - export_path: path to save the image
    - return_img: whether to return a `PIL.Image.Image` object
    """
    if isinstance(data, str):
        with open(data, 'r') as fp:
            data = json.load(fp)

    G = nx.Graph()
    if 'gt' not in data:
        data['gt'] = -1

    colors = list()
    id_to_desc = {}
    id2node = {}
    for node in data['nodes']:
        node_id = node['id']
        desc = node['desc']
        layer = node['layer']
        id2node[node_id] = node
        
        if 'selected' in node: # `select` first, when drawing traj
            if 'from' in node and node['from'] != -1:
                from_color = 'red' if node['from'] % 2 == 0 else 'pink'
                if node['from'] == data['gt']:
                    from_color = 'yellow'
            else:
                from_color = 'red'
            colors.append('green' if node['selected'] else from_color)
        elif 'from' in node and node['from'] != -1:
            colors.append('red' if node['from'] % 2 == 0 else 'pink')
            if node['from'] == data['gt']:
                colors[-1] = 'yellow'
        else:
            colors.append('red')
        G.add_node(node_id, layer=layer)
        id_to_desc[node_id] = split_text(desc, max_line_length)

    for edge in data['edges']:
        id0, id1, weight = edge
        if weight_threshold < weight_threshold:
            continue
        # assert id2node[id0]['layer'] != id2node[id1]['layer'], f"Assertion: layer of node ({id0}, {id1}) should not be same!"
        G.add_edge(id0, id1, weight=weight)

    layers = nx.get_node_attributes(G, 'layer')

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

    node_x = []
    node_y = []
    node_descriptions = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_descriptions.append(id_to_desc[node])

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
    
def animate(data: Union[str, Dict],
            video_path: str,
            weight_threshold: float=0.02,
            duration: Union[int, float]=1.5,
            gt: Union[int, List[int], None]=None):
    """
    Animate the query process.
    - data: data exported by `GarlicTree.query`
    - video_path: path to save the video
    - weight_threshold: edge weight less than this threshold will be hidden
    - duration: duration time of each step
    """
    if isinstance(data, str):
        with open(data, 'r') as fp:
            data = json.load(fp)
    node2desc = {}
    for node in data['nodes']:
        node2desc[node['id']] = node['desc']
    assert "init" in data, "Animate retrieval must have `init` in data!"
    assert "traj" in data, "Animate retrieval must have `traj` in data!"

    if gt is not None and 'gt' not in data:
        data["gt"] = gt
    if "gt" not in data:
        data["gt"] = -1
    
    selected = set(data['init'])
    data['nodes'] = [dct | {"selected": dct["id"] in selected} for dct in data['nodes']]

    frames = [visualize_tree(data, return_img=True, weight_threshold=weight_threshold)]
    
    for new_node in tqdm(data['traj'], 'Exporting video...'):
        selected.add(new_node)
        data['nodes'] = [dct | {"selected": dct["id"] in selected} for dct in data['nodes']]
        img = visualize_tree(data, return_img=True, weight_threshold=weight_threshold)
        img = draw_text_on_image(img, node2desc[new_node], (0.6, 0.6), font_size=30, max_length=30)
        frames.append(img)
    
    images_to_video(frames, output_path=video_path, fps=30, duration=duration)

if __name__ == '__main__':
    visualize_tree("tmp/traj.json", show=True, weight_threshold=0.1)
    # animate("tmp/traj.json", video_path="10.mp4", duration=2.5, gt=2)