import networkx as nx
import os
import re
import torch

def generate_circuit_graph(netlist_file):
    """Parses a Verilog netlist and converts it into a graph representation for PyTorch Geometric."""
    
    if not os.path.exists(netlist_file):
        raise FileNotFoundError(f"Netlist file not found: {netlist_file}")

    G = nx.DiGraph()  
    nodes_set = set()
    edges = []

    gate_patterns = {
        "and": "AND",
        "or": "OR",
        "nand": "NAND",
        "nor": "NOR",
        "xor": "XOR",
        "not": "NOT"
    }

    with open(netlist_file, 'r') as f:
        for line in f:
            line = line.strip().lower()
            for keyword, gate in gate_patterns.items():
                if keyword in line:
                    match = re.findall(r'\((.*?)\)', line)
                    if match:
                        ports = match[0].split(',')
                        ports = [p.strip() for p in ports]
                        if len(ports) >= 2: 
                            output_node = ports[0]
                            input_nodes = ports[1:]
                            G.add_node(output_node, gate_type=gate)
                            nodes_set.add(output_node)
                            for inp in input_nodes:
                                G.add_edge(inp, output_node)
                                nodes_set.add(inp)
                                edges.append((inp, output_node))

    #Map nodes to indices
    node_mapping = {node: i for i, node in enumerate(nodes_set)}
    
    #Convert edges to numerical format for PyTorch Geometric
    edge_index = torch.tensor(
        [[node_mapping[src], node_mapping[dst]] for src, dst in edges], dtype=torch.long
    ).t().contiguous()

    return G, edge_index


if __name__ == "__main__":
    netlist_path = "netlists/Simple_Netlist.v" 
    circuit_graph, edge_index = generate_circuit_graph(netlist_path)

    print("Nodes:", circuit_graph.nodes(data=True))
    print("Edges:", list(circuit_graph.edges()))
    print("Edge Index Tensor:", edge_index)

    
