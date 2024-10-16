import plotly.graph_objects as go
import networkx as nx

def split_text(text, line_length):
    return "<br>".join([text[i:i+line_length] for i in range(0, len(text), line_length)])

def visualize_tree(data, max_line_length=30, weight_threshold=0.5):
    G = nx.Graph()

    # 1. 添加节点
    id_to_desc = {}
    for node in data['nodes']:
        node_id = node['id']
        desc = node['desc']
        layer = node['layer']
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
    cnt = 0
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        weight = edge[2]['weight']
        color = 'red' if weight > weight_threshold else '#888'
        cnt += (weight > weight_threshold)

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=color),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    print(cnt)
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

    fig.show()