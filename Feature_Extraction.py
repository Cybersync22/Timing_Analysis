import networkx as nx
import pandas as pd

def extract_features(G):
    """Extracts node-level features from a circuit graph."""
    features = []

    #Compute node depth individually
    try:
        longest_paths = nx.single_source_shortest_path_length(G, source=list(G.nodes())[0])
    except nx.NetworkXNoPath:
        raise ValueError("Graph is not a DAG! Ensure netlist conversion is correct.")

    for node in G.nodes():
        fan_in = len(list(G.predecessors(node)))
        fan_out = len(list(G.successors(node)))
        depth = longest_paths.get(node, 0)

        features.append([node, fan_in, fan_out, depth])

    return pd.DataFrame(features, columns=["Node", "Fan-In", "Fan-Out", "Depth"])

if __name__ == "__main__":
    from Graph_Generation import generate_circuit_graph
    
    #Pass a **netlist file path** instead of an integer
    netlist_path = "netlists/Simple_Netlist.v"  
    G, _ = generate_circuit_graph(netlist_path)  
    
    features_df = extract_features(G)
    #print(features_df)
    print(features_df.describe()) 
    print(features_df.sort_values(by="Depth", ascending=False)) 
