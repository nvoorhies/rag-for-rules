import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyvis.network import Network
import os

class RuleGraphVisualizer:
    def __init__(self, processed_srd_path):
        """Initialize the visualizer with processed SRD data."""
        # Add edges to the network
        for source, target in subgraph.edges():
            edge_data = subgraph.get_edge_data(source, target)
            
            # Determine edge label based on relationship type
            edge_type = edge_data.get('type', '')
            term = edge_data.get('term', '')
            
            if edge_type == 'REFERENCES':
                label = 'references'
            elif edge_type == 'USES_TERM':
                label = f'uses "{term}"'
            else:
                label = edge_type.lower().replace('_', ' ')
            
            # Add edge to network
            net.add_edge(source, target, label=label, arrows='to')
        
        # Set physics options for better layout
        net.set_options('''
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            }
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": true
          }
        }
        ''')
        
        # Save the network
        net.save_graph(output_path)
        print(f"Query subgraph saved to {output_path}")
    
    def generate_dependency_tree(self, rule_id, max_depth=2, output_path="dependency_tree.html"):
        """Generate a tree visualization of rule dependencies."""
        # Find all rules that are prerequisites for understanding this rule
        tree_nodes = {rule_id}
        tree_edges = []
        
        # BFS to find prerequisite rules
        queue = [(rule_id, 0)]  # (node, depth)
        visited = {rule_id}
        
        while queue:
            node, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Check all predecessor nodes
            for pred in self.G.predecessors(node):
                if pred not in visited:
                    visited.add(pred)
                    tree_nodes.add(pred)
                    tree_edges.append((pred, node))
                    queue.append((pred, depth + 1))
            
            # Also add successor nodes
            for succ in self.G.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    tree_nodes.add(succ)
                    tree_edges.append((node, succ))
                    queue.append((succ, depth + 1))
        
        # Create a tree layout
        net = Network(height="600px", width="100%", directed=True, notebook=False)
        
        # Define node colors by type
        color_map = {
            'CORE_RULE': '#FF5733',  # Red
            'DERIVED_RULE': '#33FF57',  # Green
            'EXCEPTION': '#3357FF',  # Blue
            'DEFINITION': '#FFFF33',  # Yellow
            'EXAMPLE': '#FF33FF',  # Magenta
            'TABLE': '#33FFFF',  # Cyan
            'UNKNOWN': '#AAAAAA'  # Grey
        }
        
        # Add nodes to the network
        for node_id in tree_nodes:
            node_data = self.G.nodes[node_id]
            
            # Determine if this is the root rule
            is_root = node_id == rule_id
            
            # Get rule text (truncated for tooltip)
            rule = next((r for r in self.rules if r['id'] == node_id), None)
            tooltip = rule['title'] if rule else node_data.get('label', '')
            if rule and rule.get('text'):
                tooltip += ": " + rule['text'][:100] + "..."
            
            # Determine node color based on type
            node_type = node_data.get('type', 'UNKNOWN')
            color = color_map.get(node_type, '#AAAAAA')
            
            # Make the root node larger
            size = 20 if is_root else 10
            
            # Add node to network
            net.add_node(
                node_id, 
                label=node_data.get('label', node_id),
                title=tooltip,
                color=color,
                size=size,
                borderWidth=3 if is_root else 1
            )
        
        # Add edges to the network
        for source, target in tree_edges:
            # Get edge data from the original graph
            edge_data = self.G.get_edge_data(source, target)
            
            # Determine edge label based on relationship type
            edge_type = edge_data.get('type', '')
            term = edge_data.get('term', '')
            
            if edge_type == 'REFERENCES':
                label = 'references'
            elif edge_type == 'USES_TERM':
                label = f'uses "{term}"'
            else:
                label = edge_type.lower().replace('_', ' ')
            
            # Add edge to network
            net.add_edge(source, target, label=label, arrows='to')
        
        # Use hierarchical layout for tree
        net.set_options('''
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "nodeSpacing": 150,
              "levelSeparation": 150
            }
          },
          "physics": {
            "enabled": false
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": true
          }
        }
        ''')
        
        # Save the network
        net.save_graph(output_path)
        print(f"Dependency tree saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rule_graph_visualizer.py <path_to_processed_srd.json> [output_directory]")
        sys.exit(1)
    
    srd_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = RuleGraphVisualizer(srd_path)
    
    # Create visualizations
    visualizer.plot_rule_distribution(os.path.join(output_dir, "rule_distribution.png"))
    visualizer.create_interactive_graph(os.path.join(output_dir, "full_rule_graph.html"))
    
    # Example: Create a subgraph for some specific rules
    # Find the first few rules as an example
    example_rule_ids = [rule['id'] for rule in visualizer.rules[:5]]
    visualizer.visualize_query_subgraph(example_rule_ids, os.path.join(output_dir, "example_subgraph.html"))
    
    # Example: Generate a dependency tree for the first rule
    if visualizer.rules:
        first_rule_id = visualizer.rules[0]['id']
        visualizer.generate_dependency_tree(first_rule_id, os.path.join(output_dir, "example_dependency_tree.html"))
 Load the processed SRD
        with open(processed_srd_path, 'r', encoding='utf-8') as f:
            self.srd_data = json.load(f)
        
        # Extract graph data
        self.graph_data = self.srd_data['graph']
        self.rules = self.srd_data['rules']
        
        # Create NetworkX graph
        self.G = self._build_networkx_graph()
    
    def _build_networkx_graph(self):
        """Build a NetworkX graph from the processed data."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.graph_data['nodes']:
            G.add_node(node['id'], 
                       label=node['label'], 
                       type=node['type'],
                       scope=node.get('scope', 'UNKNOWN'),
                       complexity=node.get('complexity', 'MEDIUM'))
        
        # Add edges
        for edge in self.graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], 
                      type=edge['type'],
                      term=edge.get('term', None))
        
        return G
    
    def plot_rule_distribution(self, output_path=None):
        """Plot distribution of rules by type and scope."""
        # Count rule types and scopes
        rule_types = {}
        rule_scopes = {}
        
        for rule in self.rules:
            # Count rule types
            rule_type = rule.get('type', 'UNKNOWN')
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
            
            # Count rule scopes
            rule_scope = rule.get('scope', 'UNKNOWN')
            rule_scopes[rule_scope] = rule_scopes.get(rule_scope, 0) + 1
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rule types
        ax1.bar(rule_types.keys(), rule_types.values(), color='skyblue')
        ax1.set_title('Rule Distribution by Type')
        ax1.set_xlabel('Rule Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot rule scopes
        ax2.bar(rule_scopes.keys(), rule_scopes.values(), color='lightgreen')
        ax2.set_title('Rule Distribution by Scope')
        ax2.set_xlabel('Rule Scope')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Distribution plot saved to {output_path}")
        else:
            plt.show()
    
    def create_interactive_graph(self, output_path="rule_graph.html", height="800px", width="1200px"):
        """Create an interactive visualization of the rule graph."""
        # Create pyvis network
        net = Network(height=height, width=width, directed=True, notebook=False)
        
        # Define node colors by type
        color_map = {
            'CORE_RULE': '#FF5733',  # Red
            'DERIVED_RULE': '#33FF57',  # Green
            'EXCEPTION': '#3357FF',  # Blue
            'DEFINITION': '#FFFF33',  # Yellow
            'EXAMPLE': '#FF33FF',  # Magenta
            'TABLE': '#33FFFF',  # Cyan
            'UNKNOWN': '#AAAAAA'  # Grey
        }
        
        # Add nodes to the network
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            
            # Get rule text (truncated for tooltip)
            rule = next((r for r in self.rules if r['id'] == node_id), None)
            tooltip = rule['title'] if rule else node_data.get('label', '')
            if rule and rule.get('text'):
                tooltip += ": " + rule['text'][:100] + "..."
            
            # Determine node color based on type
            node_type = node_data.get('type', 'UNKNOWN')
            color = color_map.get(node_type, '#AAAAAA')
            
            # Add node to network
            net.add_node(
                node_id, 
                label=node_data.get('label', node_id),
                title=tooltip,
                color=color,
                size=10
            )
        
        # Add edges to the network
        for source, target, edge_data in self.G.edges(data=True):
            # Determine edge label based on relationship type
            edge_type = edge_data.get('type', '')
            term = edge_data.get('term', '')
            
            if edge_type == 'REFERENCES':
                label = 'references'
            elif edge_type == 'USES_TERM':
                label = f'uses "{term}"'
            else:
                label = edge_type.lower().replace('_', ' ')
            
            # Add edge to network
            net.add_edge(source, target, label=label, arrows='to')
        
        # Set physics options for better layout
        net.set_options('''
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            }
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": true
          }
        }
        ''')
        
        # Save the network
        net.save_graph(output_path)
        print(f"Interactive graph saved to {output_path}")
    
    def visualize_query_subgraph(self, rule_ids, output_path="query_subgraph.html"):
        """Create a visualization of a subgraph for specific rules."""
        # Create a subgraph with the specified rules and their neighbors
        subgraph_nodes = set(rule_ids)
        
        # Add direct neighbors
        for rule_id in rule_ids:
            subgraph_nodes.update(self.G.predecessors(rule_id))
            subgraph_nodes.update(self.G.successors(rule_id))
        
        # Create the subgraph
        subgraph = self.G.subgraph(subgraph_nodes)
        
        # Create pyvis network
        net = Network(height="600px", width="100%", directed=True, notebook=False)
        
        # Define node colors by type
        color_map = {
            'CORE_RULE': '#FF5733',  # Red
            'DERIVED_RULE': '#33FF57',  # Green
            'EXCEPTION': '#3357FF',  # Blue
            'DEFINITION': '#FFFF33',  # Yellow
            'EXAMPLE': '#FF33FF',  # Magenta
            'TABLE': '#33FFFF',  # Cyan
            'UNKNOWN': '#AAAAAA'  # Grey
        }
        
        # Add nodes to the network
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            
            # Determine if this is a primary rule (part of the original query)
            is_primary = node_id in rule_ids
            
            # Get rule text (truncated for tooltip)
            rule = next((r for r in self.rules if r['id'] == node_id), None)
            tooltip = rule['title'] if rule else node_data.get('label', '')
            if rule and rule.get('text'):
                tooltip += ": " + rule['text'][:100] + "..."
            
            # Determine node color based on type
            node_type = node_data.get('type', 'UNKNOWN')
            color = color_map.get(node_type, '#AAAAAA')
            
            # Adjust size based on whether it's a primary rule
            size = 15 if is_primary else 10
            
            # Add node to network
            net.add_node(
                node_id, 
                label=node_data.get('label', node_id),
                title=tooltip,
                color=color,
                size=size,
                borderWidth=3 if is_primary else 1
            )
        
        #