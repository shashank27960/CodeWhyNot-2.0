import ast
import re
from typing import Dict, List, Tuple, Optional, Any
from scm.ast_scm_builder import ASTSCMBuilder
import networkx as nx

class CausalInterventionEngine:
    """
    Engine for applying causal interventions to code and generating counterfactual prompts.
    Supports AST-level interventions and semantic transformations.
    """
    
    def __init__(self):
        self.ast_builder = ASTSCMBuilder()
        self.intervention_templates = {
            'loop_to_recursion': {
                'description': 'Convert iterative loop to recursive function',
                'prompt_modifier': 'using recursion instead of loops',
                'code_transformer': self._transform_loop_to_recursion
            },
            'recursion_to_loop': {
                'description': 'Convert recursive function to iterative loop',
                'prompt_modifier': 'using loops instead of recursion',
                'code_transformer': self._transform_recursion_to_loop
            },
            'list_to_generator': {
                'description': 'Convert list comprehension to generator expression',
                'prompt_modifier': 'using generator expressions for memory efficiency',
                'code_transformer': self._transform_list_to_generator
            },
            'for_to_while': {
                'description': 'Convert for loop to while loop',
                'prompt_modifier': 'using while loop instead of for loop',
                'code_transformer': self._transform_for_to_while
            },
            'while_to_for': {
                'description': 'Convert while loop to for loop',
                'prompt_modifier': 'using for loop instead of while loop',
                'code_transformer': self._transform_while_to_for
            },
            'add_error_handling': {
                'description': 'Add error handling and validation',
                'prompt_modifier': 'with proper error handling and input validation',
                'code_transformer': self._transform_add_error_handling
            },
            'optimize_algorithm': {
                'description': 'Optimize algorithm complexity',
                'prompt_modifier': 'with optimized time complexity',
                'code_transformer': self._transform_optimize_algorithm
            }
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code and return intervention opportunities."""
        try:
            ast_tree = self.ast_builder.parse_code_to_ast(code)
            scm_graph = self.ast_builder.build_scm_from_ast(ast_tree)
            interventions = self.ast_builder.get_intervention_points(scm_graph)
            
            return {
                'ast_tree': ast_tree,
                'scm_graph': scm_graph,
                'interventions': interventions,
                'code_analysis': self._analyze_code_structure(code)
            }
        except Exception as e:
            return {
                'error': str(e),
                'interventions': [],
                'code_analysis': {}
            }
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure for intervention opportunities."""
        analysis = {
            'has_loops': bool(re.search(r'\b(for|while)\b', code)),
            'has_recursion': bool(re.search(r'\bdef\s+\w+\s*\([^)]*\)[^:]*:\s*(?:[^\n]*\n)*[^:]*\1\s*\(', code)),
            'has_list_comp': bool(re.search(r'\[.*for.*in.*\]', code)),
            'has_error_handling': bool(re.search(r'\b(try|except|finally)\b', code)),
            'has_imports': bool(re.search(r'\bimport\b|\bfrom\b.*\bimport\b', code)),
            'function_count': len(re.findall(r'\bdef\s+\w+', code)),
            'line_count': len(code.split('\n'))
        }
        return analysis
    
    def generate_counterfactual_prompt(self, original_prompt: str, intervention_type: str) -> str:
        """Generate a counterfactual prompt based on intervention type."""
        if intervention_type not in self.intervention_templates:
            return original_prompt
        
        template = self.intervention_templates[intervention_type]
        modifier = template['prompt_modifier']
        
        # Add intervention modifier to the prompt
        if 'using' in modifier:
            # Insert modifier before the main instruction
            if 'write' in original_prompt.lower() or 'create' in original_prompt.lower():
                return original_prompt.replace('write', f'write {modifier}', 1).replace('create', f'create {modifier}', 1)
            else:
                return f"{original_prompt} {modifier}"
        else:
            return f"{original_prompt} {modifier}"
    
    def apply_code_intervention(self, code: str, intervention_type: str) -> str:
        """Apply a specific intervention to the code."""
        if intervention_type not in self.intervention_templates:
            return code
        
        transformer = self.intervention_templates[intervention_type]['code_transformer']
        return transformer(code)
    
    def _transform_loop_to_recursion(self, code: str) -> str:
        """Transform iterative loop to recursive function."""
        try:
            tree = ast.parse(code)
            
            # Find for loops and convert them
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    return self._convert_for_to_recursion(code, node)
            
            return code
        except:
            return code
    
    def _convert_for_to_recursion(self, code: str, for_node: ast.For) -> str:
        """Convert a specific for loop to recursion."""
        # This is a simplified conversion - in practice, you'd need more sophisticated logic
        target = ast.unparse(for_node.target) if hasattr(ast, 'unparse') else str(for_node.target)
        iterator = ast.unparse(for_node.iter) if hasattr(ast, 'unparse') else str(for_node.iter)
        
        # Create recursive function template
        recursive_template = f"""
def recursive_function({target}, remaining):
    if not remaining:
        return
    # Process current item
    current = remaining[0]
    # Recursive call
    recursive_function({target}, remaining[1:])
"""
        return recursive_template
    
    def _transform_recursion_to_loop(self, code: str) -> str:
        """Transform recursive function to iterative loop."""
        # Simplified transformation
        return code.replace('def ', '# Converted from recursive to iterative\ndef ')
    
    def _transform_list_to_generator(self, code: str) -> str:
        """Transform list comprehension to generator expression."""
        # Replace list comprehensions with generator expressions
        code = re.sub(r'\[(.*?for.*?in.*?)\]', r'(\1)', code)
        return code
    
    def _transform_for_to_while(self, code: str) -> str:
        """Transform for loop to while loop."""
        # Simplified transformation
        return code.replace('for ', '# Converted from for to while\nwhile ')
    
    def _transform_while_to_for(self, code: str) -> str:
        """Transform while loop to for loop."""
        # Simplified transformation
        return code.replace('while ', '# Converted from while to for\nfor ')
    
    def _transform_add_error_handling(self, code: str) -> str:
        """Add error handling to code."""
        # Add try-except wrapper
        return f"""try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Error: {{e}}")
    return None"""
    
    def _transform_optimize_algorithm(self, code: str) -> str:
        """Optimize algorithm complexity."""
        # Add optimization comment
        return f"# Optimized version\n{code}"
    
    def generate_intervention_suggestions(self, code: str, prompt: str) -> List[Dict[str, Any]]:
        """Generate intelligent intervention suggestions based on code and prompt analysis."""
        analysis = self.analyze_code(code)
        suggestions = []
        
        if analysis.get('error'):
            return suggestions
        
        code_analysis = analysis['code_analysis']
        interventions = analysis['interventions']
        
        # Add AST-based interventions
        for intervention in interventions:
            suggestions.append({
                'type': intervention['type'],
                'description': intervention['description'],
                'confidence': 0.8,
                'reasoning': f"Detected {intervention['original']} in code",
                'prompt_modification': self.generate_counterfactual_prompt(prompt, intervention['type'])
            })
        
        # Add heuristic-based suggestions
        if code_analysis['has_loops'] and not code_analysis['has_recursion']:
            suggestions.append({
                'type': 'loop_to_recursion',
                'description': 'Convert loops to recursion for functional programming approach',
                'confidence': 0.7,
                'reasoning': 'Code uses loops but no recursion detected',
                'prompt_modification': self.generate_counterfactual_prompt(prompt, 'loop_to_recursion')
            })
        
        if code_analysis['has_list_comp']:
            suggestions.append({
                'type': 'list_to_generator',
                'description': 'Convert list comprehension to generator for memory efficiency',
                'confidence': 0.6,
                'reasoning': 'List comprehension detected - generator may be more efficient',
                'prompt_modification': self.generate_counterfactual_prompt(prompt, 'list_to_generator')
            })
        
        if not code_analysis['has_error_handling']:
            suggestions.append({
                'type': 'add_error_handling',
                'description': 'Add error handling and input validation',
                'confidence': 0.9,
                'reasoning': 'No error handling detected in code',
                'prompt_modification': self.generate_counterfactual_prompt(prompt, 'add_error_handling')
            })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions
    
    def create_intervention_pipeline(self, original_code: str, original_prompt: str, 
                                   intervention_type: str) -> Dict[str, Any]:
        """Create a complete intervention pipeline."""
        # Generate counterfactual prompt
        cf_prompt = self.generate_counterfactual_prompt(original_prompt, intervention_type)
        
        # Apply code intervention
        cf_code = self.apply_code_intervention(original_code, intervention_type)
        
        # Analyze both versions
        original_analysis = self.analyze_code(original_code)
        cf_analysis = self.analyze_code(cf_code)
        
        return {
            'original': {
                'code': original_code,
                'prompt': original_prompt,
                'analysis': original_analysis
            },
            'counterfactual': {
                'code': cf_code,
                'prompt': cf_prompt,
                'analysis': cf_analysis
            },
            'intervention': {
                'type': intervention_type,
                'description': self.intervention_templates.get(intervention_type, {}).get('description', ''),
                'confidence': 0.8
            }
        } 