import ast
import networkx as nx
from typing import Dict, List, Set, Tuple, Any
import re

class ASTSCMBuilder:
    """
    Builds Structural Causal Models from Abstract Syntax Trees.
    Maps code elements to causal nodes and identifies dependencies.
    """
    
    def __init__(self):
        self.node_types = {
            'function_def': ast.FunctionDef,
            'class_def': ast.ClassDef,
            'for_loop': ast.For,
            'while_loop': ast.While,
            'if_statement': ast.If,
            'assignment': ast.Assign,
            'function_call': ast.Call,
            'import': ast.Import,
            'import_from': ast.ImportFrom,
            'return': ast.Return,
            'variable': ast.Name,
            'literal': (ast.Constant, ast.Num, ast.Str),
            'binary_op': ast.BinOp,
            'comparison': ast.Compare,
            'list_comp': ast.ListComp,
            'generator_exp': ast.GeneratorExp
        }
        
    def parse_code_to_ast(self, code: str) -> ast.AST:
        """Parse code string to AST."""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")
    
    def extract_ast_nodes(self, ast_tree: ast.AST) -> List[Tuple[str, ast.AST, Dict]]:
        """Extract meaningful nodes from AST with metadata."""
        nodes = []
        
        for node in ast.walk(ast_tree):
            node_info = self._classify_node(node)
            if node_info:
                nodes.append((node_info['type'], node, node_info))
        
        return nodes
    
    def _classify_node(self, node: ast.AST) -> Dict[str, Any]:
        """Classify AST node and extract relevant information."""
        if isinstance(node, ast.FunctionDef):
            return {
                'type': 'function_def',
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node)),
                'has_loop': any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node)),
                'has_recursion': self._detect_recursion(node, node.name)
            }
        elif isinstance(node, ast.ClassDef):
            return {
                'type': 'class_def',
                'name': node.name,
                'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            }
        elif isinstance(node, ast.For):
            return {
                'type': 'for_loop',
                'target': self._get_node_text(node.target),
                'iter': self._get_node_text(node.iter),
                'body_size': len(node.body)
            }
        elif isinstance(node, ast.While):
            return {
                'type': 'while_loop',
                'test': self._get_node_text(node.test),
                'body_size': len(node.body)
            }
        elif isinstance(node, ast.If):
            return {
                'type': 'if_statement',
                'test': self._get_node_text(node.test),
                'has_else': len(node.orelse) > 0
            }
        elif isinstance(node, ast.Assign):
            return {
                'type': 'assignment',
                'targets': [self._get_node_text(t) for t in node.targets],
                'value': self._get_node_text(node.value)
            }
        elif isinstance(node, ast.Call):
            return {
                'type': 'function_call',
                'func': self._get_node_text(node.func),
                'args': [self._get_node_text(arg) for arg in node.args]
            }
        elif isinstance(node, ast.Return):
            return {
                'type': 'return',
                'value': self._get_node_text(node.value) if node.value else None
            }
        elif isinstance(node, ast.Name):
            return {
                'type': 'variable',
                'name': node.id,
                'ctx': type(node.ctx).__name__
            }
        elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
            return {
                'type': 'literal',
                'value': getattr(node, 'value', getattr(node, 'n', getattr(node, 's', None)))
            }
        
        return {}
    
    def _get_node_text(self, node: ast.AST) -> str:
        """Extract text representation of AST node."""
        if node is None:
            return ""
        try:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
        except:
            return str(node)
    
    def _detect_recursion(self, func_node: ast.FunctionDef, func_name: str) -> bool:
        """Detect if function contains recursive calls to itself."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == func_name:
                    return True
        return False
    
    def build_scm_from_ast(self, ast_tree: ast.AST) -> nx.DiGraph:
        """Build SCM graph from AST nodes with causal relationships."""
        G = nx.DiGraph()
        nodes = self.extract_ast_nodes(ast_tree)
        
        # Add nodes to graph
        for i, (node_type, node, info) in enumerate(nodes):
            node_id = f"{node_type}_{i}"
            G.add_node(node_id, type=node_type, info=info, ast_node=node)
        
        # Add causal edges based on dependencies
        self._add_causal_edges(G, nodes)
        
        return G
    
    def _add_causal_edges(self, G: nx.DiGraph, nodes: List[Tuple[str, ast.AST, Dict]]):
        """Add causal edges between AST nodes based on dependencies."""
        node_map = {f"{node_type}_{i}": (node_type, node, info) 
                   for i, (node_type, node, info) in enumerate(nodes)}
        
        for node_id, (node_type, node, info) in node_map.items():
            # Function definition dependencies
            if node_type == 'function_def':
                self._add_function_dependencies(G, node_id, node, info, node_map)
            
            # Loop dependencies
            elif node_type in ['for_loop', 'while_loop']:
                self._add_loop_dependencies(G, node_id, node, info, node_map)
            
            # Assignment dependencies
            elif node_type == 'assignment':
                self._add_assignment_dependencies(G, node_id, node, info, node_map)
            
            # Function call dependencies
            elif node_type == 'function_call':
                self._add_call_dependencies(G, node_id, node, info, node_map)
    
    def _add_function_dependencies(self, G: nx.DiGraph, node_id: str, node: ast.AST, 
                                 info: Dict, node_map: Dict):
        """Add dependencies for function definitions."""
        func_name = info['name']
        
        # Find function calls that depend on this function
        for other_id, (other_type, other_node, other_info) in node_map.items():
            if other_type == 'function_call' and other_info['func'] == func_name:
                G.add_edge(node_id, other_id, relation='defines')
            
            # Find variables used in function
            if other_type == 'variable' and other_info['ctx'] == 'Load':
                if other_info['name'] in self._get_variables_in_scope(node):
                    G.add_edge(other_id, node_id, relation='used_by')
    
    def _add_loop_dependencies(self, G: nx.DiGraph, node_id: str, node: ast.AST,
                             info: Dict, node_map: Dict):
        """Add dependencies for loop constructs."""
        # Find variables used in loop condition
        for other_id, (other_type, other_node, other_info) in node_map.items():
            if other_type == 'variable' and other_info['ctx'] == 'Load':
                if other_info['name'] in self._get_variables_in_scope(node):
                    G.add_edge(other_id, node_id, relation='controls')
    
    def _add_assignment_dependencies(self, G: nx.DiGraph, node_id: str, node: ast.AST,
                                   info: Dict, node_map: Dict):
        """Add dependencies for assignments."""
        # Find variables that depend on this assignment
        for target in info['targets']:
            for other_id, (other_type, other_node, other_info) in node_map.items():
                if other_type == 'variable' and other_info['name'] == target:
                    G.add_edge(node_id, other_id, relation='assigns')
    
    def _add_call_dependencies(self, G: nx.DiGraph, node_id: str, node: ast.AST,
                             info: Dict, node_map: Dict):
        """Add dependencies for function calls."""
        # Find function definition this call depends on
        func_name = info['func']
        for other_id, (other_type, other_node, other_info) in node_map.items():
            if other_type == 'function_def' and other_info['name'] == func_name:
                G.add_edge(other_id, node_id, relation='called_by')
    
    def _get_variables_in_scope(self, node: ast.AST) -> Set[str]:
        """Get all variable names used in an AST node."""
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                variables.add(child.id)
        return variables
    
    def get_intervention_points(self, scm_graph: nx.DiGraph) -> List[Dict]:
        """Identify potential intervention points in the SCM."""
        interventions = []
        
        for node_id, node_data in scm_graph.nodes(data=True):
            node_type = node_data['type']
            info = node_data['info']
            
            if node_type == 'for_loop':
                interventions.append({
                    'node_id': node_id,
                    'type': 'loop_to_recursion',
                    'description': f"Convert for loop to recursive function",
                    'original': f"for {info['target']} in {info['iter']}",
                    'intervention': f"recursive function with {info['target']} as parameter"
                })
            
            elif node_type == 'while_loop':
                interventions.append({
                    'node_id': node_id,
                    'type': 'while_to_for',
                    'description': f"Convert while loop to for loop",
                    'original': f"while {info['test']}",
                    'intervention': f"for loop with range or iterator"
                })
            
            elif node_type == 'function_def':
                if info['has_loop']:
                    interventions.append({
                        'node_id': node_id,
                        'type': 'iterative_to_recursive',
                        'description': f"Convert iterative function to recursive",
                        'original': f"iterative {info['name']}",
                        'intervention': f"recursive {info['name']}"
                    })
                elif info['has_recursion']:
                    interventions.append({
                        'node_id': node_id,
                        'type': 'recursive_to_iterative',
                        'description': f"Convert recursive function to iterative",
                        'original': f"recursive {info['name']}",
                        'intervention': f"iterative {info['name']}"
                    })
            
            elif node_type == 'assignment':
                if 'list' in str(info['value']).lower():
                    interventions.append({
                        'node_id': node_id,
                        'type': 'list_to_generator',
                        'description': f"Convert list to generator expression",
                        'original': f"list assignment",
                        'intervention': f"generator expression"
                    })
        
        return interventions
    
    def apply_intervention(self, scm_graph: nx.DiGraph, intervention: Dict) -> nx.DiGraph:
        """Apply an intervention to the SCM graph."""
        modified_graph = scm_graph.copy()
        node_id = intervention['node_id']
        
        if node_id in modified_graph.nodes:
            # Update node information based on intervention type
            node_data = modified_graph.nodes[node_id]
            node_data['intervention_applied'] = intervention
            node_data['modified'] = True
            
            # Add intervention metadata
            modified_graph.nodes[node_id]['intervention_type'] = intervention['type']
            modified_graph.nodes[node_id]['intervention_description'] = intervention['description']
        
        return modified_graph 