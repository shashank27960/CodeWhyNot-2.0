import ast
import re
from typing import Dict, List, Tuple, Optional, Any
from ast_diff.fidelity_scorer import FidelityScorer
from evaluation.code_executor import CodeExecutor
import numpy as np

class MetricsCalculator:
    """
    Comprehensive metrics calculator for evaluating code quality, correctness,
    and causal intervention effectiveness.
    """
    
    def __init__(self):
        self.code_executor = CodeExecutor()
        self.fidelity_scorer = None
    
    def calculate_comprehensive_metrics(self, original_code: str, counterfactual_code: str,
                                      original_prompt: str, counterfactual_prompt: str,
                                      intervention_type: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for code comparison."""
        
        # Initialize fidelity scorer
        self.fidelity_scorer = FidelityScorer(original_code, counterfactual_code)
        
        # Calculate all metrics
        metrics = {
            'structural_metrics': self._calculate_structural_metrics(original_code, counterfactual_code),
            'functional_metrics': self._calculate_functional_metrics(original_code, counterfactual_code),
            'semantic_metrics': self._calculate_semantic_metrics(original_code, counterfactual_code),
            'causal_metrics': self._calculate_causal_metrics(original_prompt, counterfactual_prompt, intervention_type),
            'quality_metrics': self._calculate_quality_metrics(original_code, counterfactual_code)
        }
        
        # Calculate overall scores
        metrics['overall_scores'] = self._calculate_overall_scores(metrics)
        
        return metrics
    
    def _calculate_structural_metrics(self, original_code: str, counterfactual_code: str) -> Dict[str, Any]:
        """Calculate structural similarity metrics."""
        try:
            # AST-based metrics
            ast_distance = self.fidelity_scorer.ast_edit_distance()
            logic_shift = self.fidelity_scorer.logic_shift()
            
            # Parse ASTs for detailed analysis
            orig_ast = ast.parse(original_code) if self.fidelity_scorer.ast1 else None
            cf_ast = ast.parse(counterfactual_code) if self.fidelity_scorer.ast2 else None
            
            # Count different node types
            orig_node_counts = self._count_ast_nodes(orig_ast) if orig_ast else {}
            cf_node_counts = self._count_ast_nodes(cf_ast) if cf_ast else {}
            
            # Calculate structural similarity
            structural_similarity = self._calculate_node_similarity(orig_node_counts, cf_node_counts)
            
            return {
                'ast_edit_distance': ast_distance,
                'logic_shift_detected': logic_shift,
                'structural_similarity': structural_similarity,
                'original_node_counts': orig_node_counts,
                'counterfactual_node_counts': cf_node_counts,
                'syntax_correct_original': self.fidelity_scorer.is_syntax_correct(original_code),
                'syntax_correct_counterfactual': self.fidelity_scorer.is_syntax_correct(counterfactual_code)
            }
        except Exception as e:
            return {
                'error': str(e),
                'ast_edit_distance': 1.0,
                'logic_shift_detected': False,
                'structural_similarity': 0.0
            }
    
    def _calculate_functional_metrics(self, original_code: str, counterfactual_code: str) -> Dict[str, Any]:
        """Calculate functional correctness metrics."""
        try:
            # Execute both codes
            orig_result = self.code_executor.execute_code(original_code)
            cf_result = self.code_executor.execute_code(counterfactual_code)
            
            # Create test cases
            test_cases = self.code_executor.create_test_cases(original_code)
            
            # Run tests on both versions
            orig_test_result = self.code_executor.execute_code(original_code, test_cases)
            cf_test_result = self.code_executor.execute_code(counterfactual_code, test_cases)
            
            # Calculate pass rates
            orig_pass_rate = self.code_executor.calculate_pass_at_k(orig_test_result['test_results'])
            cf_pass_rate = self.code_executor.calculate_pass_at_k(cf_test_result['test_results'])
            
            # Calculate execution metrics
            orig_metrics = self.code_executor.get_execution_metrics(orig_result)
            cf_metrics = self.code_executor.get_execution_metrics(cf_result)
            
            return {
                'original_execution': orig_metrics,
                'counterfactual_execution': cf_metrics,
                'original_pass_rate': orig_pass_rate,
                'counterfactual_pass_rate': cf_pass_rate,
                'functional_regression': max(0, orig_pass_rate - cf_pass_rate),
                'functional_improvement': max(0, cf_pass_rate - orig_pass_rate),
                'execution_time_ratio': cf_metrics['execution_time'] / max(orig_metrics['execution_time'], 0.001),
                'test_results_original': orig_test_result['test_results'],
                'test_results_counterfactual': cf_test_result['test_results']
            }
        except Exception as e:
            return {
                'error': str(e),
                'original_pass_rate': 0.0,
                'counterfactual_pass_rate': 0.0,
                'functional_regression': 0.0,
                'functional_improvement': 0.0
            }
    
    def _calculate_semantic_metrics(self, original_code: str, counterfactual_code: str) -> Dict[str, Any]:
        """Calculate semantic similarity metrics."""
        try:
            # Extract function signatures and behavior
            orig_functions = self._extract_functions(original_code)
            cf_functions = self._extract_functions(counterfactual_code)
            
            # Calculate semantic similarity
            semantic_similarity = self._calculate_function_similarity(orig_functions, cf_functions)
            
            # Analyze algorithmic complexity
            orig_complexity = self._analyze_complexity(original_code)
            cf_complexity = self._analyze_complexity(counterfactual_code)
            
            # Calculate code style metrics
            orig_style = self._analyze_code_style(original_code)
            cf_style = self._analyze_code_style(counterfactual_code)
            
            return {
                'semantic_similarity': semantic_similarity,
                'original_complexity': orig_complexity,
                'counterfactual_complexity': cf_complexity,
                'complexity_change': self._compare_complexity(orig_complexity, cf_complexity),
                'original_style': orig_style,
                'counterfactual_style': cf_style,
                'style_similarity': self._calculate_style_similarity(orig_style, cf_style)
            }
        except Exception as e:
            return {
                'error': str(e),
                'semantic_similarity': 0.0,
                'complexity_change': 'unknown'
            }
    
    def _calculate_causal_metrics(self, original_prompt: str, counterfactual_prompt: str, 
                                intervention_type: str) -> Dict[str, Any]:
        """Calculate causal intervention effectiveness metrics."""
        try:
            # Calculate prompt similarity
            prompt_similarity = self._calculate_prompt_similarity(original_prompt, counterfactual_prompt)
            
            # Analyze intervention effectiveness
            intervention_effectiveness = self._analyze_intervention_effectiveness(
                original_prompt, counterfactual_prompt, intervention_type
            )
            
            # Calculate causal influence
            causal_influence = self._calculate_causal_influence(original_prompt, counterfactual_prompt)
            
            return {
                'prompt_similarity': prompt_similarity,
                'intervention_effectiveness': intervention_effectiveness,
                'causal_influence': causal_influence,
                'intervention_type': intervention_type,
                'prompt_change_magnitude': len(counterfactual_prompt) - len(original_prompt)
            }
        except Exception as e:
            return {
                'error': str(e),
                'prompt_similarity': 0.0,
                'intervention_effectiveness': 0.0,
                'causal_influence': 0.0
            }
    
    def _calculate_quality_metrics(self, original_code: str, counterfactual_code: str) -> Dict[str, Any]:
        """Calculate code quality metrics."""
        try:
            # Calculate readability metrics
            orig_readability = self._calculate_readability(original_code)
            cf_readability = self._calculate_readability(counterfactual_code)
            
            # Calculate maintainability metrics
            orig_maintainability = self._calculate_maintainability(original_code)
            cf_maintainability = self._calculate_maintainability(counterfactual_code)
            
            # Calculate efficiency metrics
            orig_efficiency = self._calculate_efficiency(original_code)
            cf_efficiency = self._calculate_efficiency(counterfactual_code)
            
            return {
                'original_readability': orig_readability,
                'counterfactual_readability': cf_readability,
                'readability_change': cf_readability - orig_readability,
                'original_maintainability': orig_maintainability,
                'counterfactual_maintainability': cf_maintainability,
                'maintainability_change': cf_maintainability - orig_maintainability,
                'original_efficiency': orig_efficiency,
                'counterfactual_efficiency': cf_efficiency,
                'efficiency_change': cf_efficiency - orig_efficiency
            }
        except Exception as e:
            return {
                'error': str(e),
                'readability_change': 0.0,
                'maintainability_change': 0.0,
                'efficiency_change': 0.0
            }
    
    def _calculate_overall_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall scores from individual metrics."""
        try:
            # Structural score (30% weight)
            structural_score = 1.0 - metrics['structural_metrics'].get('ast_edit_distance', 1.0)
            
            # Functional score (40% weight)
            functional_score = metrics['functional_metrics'].get('counterfactual_pass_rate', 0.0)
            
            # Semantic score (20% weight)
            semantic_score = metrics['semantic_metrics'].get('semantic_similarity', 0.0)
            
            # Causal score (10% weight)
            causal_score = metrics['causal_metrics'].get('intervention_effectiveness', 0.0)
            
            # Calculate weighted overall score
            overall_score = (
                0.3 * structural_score +
                0.4 * functional_score +
                0.2 * semantic_score +
                0.1 * causal_score
            )
            
            # Calculate fidelity score (from FidelityScorer)
            fidelity_score = self.fidelity_scorer.score() if self.fidelity_scorer else 0.0
            
            return {
                'overall_score': overall_score,
                'fidelity_score': fidelity_score,
                'structural_score': structural_score,
                'functional_score': functional_score,
                'semantic_score': semantic_score,
                'causal_score': causal_score
            }
        except Exception as e:
            return {
                'overall_score': 0.0,
                'fidelity_score': 0.0,
                'error': str(e)
            }
    
    # Helper methods for detailed analysis
    def _count_ast_nodes(self, ast_tree: ast.AST) -> Dict[str, int]:
        """Count different types of AST nodes."""
        counts = {}
        for node in ast.walk(ast_tree):
            node_type = type(node).__name__
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _calculate_node_similarity(self, counts1: Dict[str, int], counts2: Dict[str, int]) -> float:
        """Calculate similarity between node count distributions."""
        all_types = set(counts1.keys()) | set(counts2.keys())
        if not all_types:
            return 1.0
        
        total_diff = 0
        total_count = 0
        
        for node_type in all_types:
            count1 = counts1.get(node_type, 0)
            count2 = counts2.get(node_type, 0)
            total_diff += abs(count1 - count2)
            total_count += max(count1, count2)
        
        if total_count == 0:
            return 1.0
        
        return 1.0 - (total_diff / total_count)
    
    def _extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function information from code."""
        functions = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node)),
                        'line_count': len(node.body)
                    })
        except:
            pass
        return functions
    
    def _calculate_function_similarity(self, funcs1: List[Dict], funcs2: List[Dict]) -> float:
        """Calculate similarity between function sets."""
        if not funcs1 and not funcs2:
            return 1.0
        if not funcs1 or not funcs2:
            return 0.0
        
        # Simple similarity based on function names and argument counts
        names1 = {f['name'] for f in funcs1}
        names2 = {f['name'] for f in funcs2}
        
        intersection = len(names1 & names2)
        union = len(names1 | names2)
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity."""
        try:
            tree = ast.parse(code)
            complexity = {
                'cyclomatic': 1,  # Base complexity
                'nesting_depth': 0,
                'function_count': 0,
                'loop_count': 0,
                'conditional_count': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity['cyclomatic'] += 1
                if isinstance(node, ast.FunctionDef):
                    complexity['function_count'] += 1
                if isinstance(node, (ast.For, ast.While)):
                    complexity['loop_count'] += 1
                if isinstance(node, ast.If):
                    complexity['conditional_count'] += 1
            
            return complexity
        except:
            return {'cyclomatic': 1, 'nesting_depth': 0, 'function_count': 0, 'loop_count': 0, 'conditional_count': 0}
    
    def _compare_complexity(self, comp1: Dict, comp2: Dict) -> str:
        """Compare complexity between two code versions."""
        if comp1['cyclomatic'] < comp2['cyclomatic']:
            return 'increased'
        elif comp1['cyclomatic'] > comp2['cyclomatic']:
            return 'decreased'
        else:
            return 'unchanged'
    
    def _analyze_code_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style metrics."""
        lines = code.split('\n')
        return {
            'line_count': len(lines),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'comment_ratio': len([l for l in lines if l.strip().startswith('#')]) / len(lines) if lines else 0,
            'blank_line_ratio': len([l for l in lines if not l.strip()]) / len(lines) if lines else 0
        }
    
    def _calculate_style_similarity(self, style1: Dict, style2: Dict) -> float:
        """Calculate style similarity between two code versions."""
        # Normalize and compare style metrics
        metrics = ['avg_line_length', 'comment_ratio', 'blank_line_ratio']
        similarities = []
        
        for metric in metrics:
            val1 = style1.get(metric, 0)
            val2 = style2.get(metric, 0)
            max_val = max(val1, val2)
            if max_val > 0:
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between prompts."""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_intervention_effectiveness(self, original_prompt: str, counterfactual_prompt: str, 
                                         intervention_type: str) -> float:
        """Analyze how effective the intervention was."""
        # Simple heuristic: more different prompts should indicate more effective interventions
        prompt_similarity = self._calculate_prompt_similarity(original_prompt, counterfactual_prompt)
        return 1.0 - prompt_similarity  # Higher difference = more effective intervention
    
    def _calculate_causal_influence(self, original_prompt: str, counterfactual_prompt: str) -> float:
        """Calculate the magnitude of causal influence."""
        # Measure how much the prompt changed
        return abs(len(counterfactual_prompt) - len(original_prompt)) / max(len(original_prompt), 1)
    
    def _calculate_readability(self, code: str) -> float:
        """Calculate code readability score."""
        try:
            lines = code.split('\n')
            if not lines:
                return 0.0
            
            # Factors: line length, comment ratio, function complexity
            avg_line_length = np.mean([len(line) for line in lines])
            comment_ratio = len([l for l in lines if l.strip().startswith('#')]) / len(lines)
            
            # Normalize scores (lower line length = better, higher comment ratio = better)
            line_score = max(0, 1 - avg_line_length / 100)
            comment_score = min(1, comment_ratio * 10)  # Cap at 10% comments
            
            return (line_score + comment_score) / 2
        except:
            return 0.5
    
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate code maintainability score."""
        try:
            complexity = self._analyze_complexity(code)
            style = self._analyze_code_style(code)
            
            # Factors: cyclomatic complexity, function count, line count
            complexity_score = max(0, 1 - complexity['cyclomatic'] / 10)
            function_score = min(1, complexity['function_count'] / 5)  # Prefer more functions
            size_score = max(0, 1 - style['line_count'] / 100)
            
            return (complexity_score + function_score + size_score) / 3
        except:
            return 0.5
    
    def _calculate_efficiency(self, code: str) -> float:
        """Calculate code efficiency score."""
        try:
            complexity = self._analyze_complexity(code)
            
            # Factors: fewer loops, fewer conditionals, fewer functions (simpler code)
            loop_score = max(0, 1 - complexity['loop_count'] / 5)
            conditional_score = max(0, 1 - complexity['conditional_count'] / 10)
            function_score = max(0, 1 - complexity['function_count'] / 3)
            
            return (loop_score + conditional_score + function_score) / 3
        except:
            return 0.5 