import ast

from numpy import mean
import higra as hg
import plotly.io as pio
from docutils import nodes
from docutils.parsers.rst import Directive

PREDEFINED_GRAPHS = {
    "GRAPH_4_ADJACENCY": lambda **kw: hg.get_4_adjacency_graph(
        kw.get("shape", (3, 3))
    ),
    "GRAPH_8_ADJACENCY": lambda **kw: hg.get_8_adjacency_graph(
        kw.get("shape", (3, 3))
    ),
    "GRAPH_RANDOM": lambda **kw: hg.random_undirected_graph_erdos_renyi(
        num_vertices=kw.get("num_vertices", 10),
        mean_degree=kw.get("mean_degree", 0.9),
        allow_non_connected=kw.get("allow_non_connected", True),
        seed=kw.get("seed", 11),
    ),
}

class PlotGraphDirective(Directive):
    has_content = True

    def run(self):
        lines = [line.strip() for line in self.content if line.strip()]
        if not lines:
            return [
                nodes.error(
                    "",
                    nodes.paragraph(
                        text="[display_graph] Error : no graph specified."
                    ),
                )
            ]

        graph_key = lines[0]
        if graph_key not in PREDEFINED_GRAPHS:
            return [
                nodes.error(
                    "",
                    nodes.paragraph(
                        text=f"[display_graph] Unknown graph '{graph_key}'."
                    ),
                )
            ]

        graph_param_names = {"num_vertices", "mean_degree", "allow_non_connected", "shape"}
        graph_kwargs = {}
        plot_kwargs = {"show_fig": False}

        for line in lines[1:]:
            if "=" not in line:
                continue
            key, val_str = line.split("=", 1)
            key, val_str = key.strip(), val_str.strip()

            try:
                parsed_val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                parsed_val = val_str

            if key == "seed":
                graph_kwargs[key] = parsed_val
                plot_kwargs[key] = parsed_val
            elif key in graph_param_names:
                graph_kwargs[key] = parsed_val
            else:
                plot_kwargs[key] = parsed_val

        try:
            graph = PREDEFINED_GRAPHS[graph_key](**graph_kwargs)

            fig = hg.plot_graph(graph, **plot_kwargs)
            html_div = pio.to_html(
                fig,
                full_html=True,
                include_plotlyjs="cdn",
                default_height="400px",
            )
            return [nodes.raw("", html_div, format="html")]
        except Exception as e:
            return [
                nodes.error(
                    "",
                    nodes.paragraph(
                        text=f"[display_graph] Error during generation : {e}"
                    ),
                )
            ]

def setup(app):
    app.add_directive("display_graph", PlotGraphDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }