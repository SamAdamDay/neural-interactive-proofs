"""Visualiser for graph isomorphism rollout samples."""

from typing import Optional
from functools import partial
from math import sqrt

import plotly.graph_objects as go
import plotly.express as px

import networkx as nx

from pvg.parameters import ScenarioType
from pvg.scenario_base import RolloutSamples, register_rollout_samples_class


@register_rollout_samples_class(ScenarioType.GRAPH_ISOMORPHISM)
class GraphIsomorphismRolloutSamples(RolloutSamples):
    """A message exchange in the graph isomorphism task."""

    def visualise(
        self,
        graph_layout_function: Optional[callable] = None,
        graph_layout_seed: Optional[int] = None,
        colour_sequence: str = "Dark24",
        node_text_colour: str = "white",
    ):
        """Visualize the rollout as a plotly graph.

        Parameters
        ----------
        graph_layout_function : callable, default=None
            A function which takes a networkx graph and returns a dictionary of node
            positions. Best to use a function from networkx.layout, possibly partially
            applied with some arguments. If None, uses networkx.spring_layout with
            `k=4/sqrt(n)`, where `n` is the number of nodes in the graph.
        graph_layout_seed : int, default=None
            The seed to use for the graph layout function. If None, the random number
            generator is the `RandomState` instance used by `numpy.random`.
        colour_sequence : str, default="Dark24"
            The name of the colour sequence to use to colour the nodes. Must be one of
            the colour sequences from plotly.express.colors.qualitative.
        node_text_colour : str, default="white"
            The colour of the node labels.
        """

        if graph_layout_function is None:

            def graph_layout_function(graph, *args, **kwargs):
                return nx.spring_layout(graph, k=4 / sqrt(len(graph)), *args, **kwargs)

        graph_layout_function = partial(graph_layout_function, seed=graph_layout_seed)

        # Get the colour sequence
        colour_list = px.colors.qualitative.__getattribute__(colour_sequence)

        def get_colour(i):
            return colour_list[i % len(colour_list)]

        # Add titles for the two graphs
        graph_title_trace = go.Scatter(
            text=["Graph A", "Graph B"],
            x=[0, 2.2],
            y=[1.2, 1.2],
            textfont=dict(size=20),
            mode="text",
            visible=True,
        )

        # Add labels for the exchange of messages at the bottom
        message_label_trace = go.Scatter(
            text=[f"{name.capitalize()} messages:" for name in self.agent_names],
            x=[-0.15] * len(self.agent_names),
            y=[-1.3 - 0.3 * i for i in range(len(self.agent_names))],
            mode="text",
            textposition="middle left",
            visible=True,
        )

        traces = [
            graph_title_trace,
            message_label_trace,
        ]
        buttons = []

        for idx, rollout in enumerate(self):
            # Only show the first batch item initially
            traces_visible = idx == 0

            # The max num nodes in the whole batch
            max_num_nodes = rollout["adjacency"].shape[-1]

            for graph_id in range(2):
                # Get the graph as a networkx graph
                node_mask = rollout["node_mask"][0, graph_id]
                adjacency = rollout["adjacency"][0, graph_id]
                adjacency = adjacency[node_mask, :][:, node_mask]
                graph = nx.from_numpy_array(adjacency)

                # Generate the layouts for the graph
                graph_pos = graph_layout_function(graph)

                # Add an offset to the x coordinates of the nodes in graph B so that it
                # is to the right of graph A
                x_add = 0 if graph_id == 0 else 2.2

                # Add the trace for the edges
                edge_x = []
                edge_y = []
                for edge in graph.edges():
                    x0, y0 = graph_pos[edge[0]]
                    x1, y1 = graph_pos[edge[1]]
                    edge_x.append(x0 + x_add)
                    edge_x.append(x1 + x_add)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)
                traces.append(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        line=dict(width=0.5, color="#888"),
                        hoverinfo="none",
                        mode="lines",
                        visible=traces_visible,
                    )
                )

                # Add the trace for the nodes
                node_x = []
                node_y = []
                node_colour = []
                node_size = []
                node_text = []
                for node in graph.nodes():
                    x, y = graph_pos[node]
                    x += x_add
                    selected_indices = (
                        rollout["x"][graph_id, node, -1].nonzero()[0].tolist()
                    )
                    if len(selected_indices) == 0:
                        node_x.append(x)
                        node_y.append(y)
                        node_colour.append("grey")
                        node_size.append(20)
                        node_text.append(str(node))
                    else:
                        for i, message_index in reversed(
                            list(enumerate(selected_indices))
                        ):
                            node_x.append(x)
                            node_y.append(y)
                            node_colour.append(get_colour(message_index))
                            node_size.append(10 * i + 20)
                            node_text.append(str(node))
                traces.append(
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        text=node_text,
                        textfont=dict(color=node_text_colour),
                        mode="markers+text",
                        hoverinfo="text",
                        marker=dict(
                            color=node_colour,
                            size=node_size,
                            line_width=0,
                            opacity=1,
                        ),
                        visible=traces_visible,
                    )
                )

            # Add the trace for the timeline of the messages exchanged
            timeline_node_x = []
            timeline_node_y = []
            timeline_node_text = []
            timeline_node_colour = []
            for round_id, message in enumerate(rollout["message"].flat):
                x = round_id * 0.125
                y = -1.3 - 0.3 * (round_id % 2)
                timeline_node_x.append(x)
                timeline_node_y.append(y)
                if message < max_num_nodes:
                    graph_letter = "A"
                    node_num = message
                else:
                    graph_letter = "B"
                    node_num = message - max_num_nodes
                timeline_node_text.append(f"{graph_letter}{node_num}")
                timeline_node_colour.append(get_colour(round_id))
            traces.append(
                go.Scatter(
                    x=timeline_node_x,
                    y=timeline_node_y,
                    text=timeline_node_text,
                    textfont=dict(color=node_text_colour),
                    mode="markers+text",
                    hoverinfo="text",
                    marker=dict(
                        color=timeline_node_colour,
                        size=30,
                        line_width=0,
                        opacity=1,
                    ),
                    visible=traces_visible,
                )
            )

            # Add a button to show this batch item
            buttons.append(
                dict(
                    label=f"Batch item {idx}",
                    method="update",
                    args=[
                        {
                            "visible": [True, True]
                            + [False] * (5 * idx)
                            + [True] * 5
                            + [False] * (5 * (len(self) - idx - 1))
                        }
                    ],
                )
            )

        layout = go.Layout(
            title="Graph Isomorphism Rollout Samples",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=5, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[dict(buttons=buttons, active=0, showactive=True)],
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()
