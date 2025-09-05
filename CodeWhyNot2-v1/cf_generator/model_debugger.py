import ast
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import difflib

class ModelBehaviorType(Enum):
    """Types of model behavior patterns."""
    CONSISTENT = "consistent"
    VARIABLE = "variable"
    UNPREDICTABLE = "unpredictable"
    BIASED = "biased"

@dataclass
class PromptAnalysis:
    """Analysis of prompt engineering effectiveness."""
    prompt_clarity: float
    prompt_specificity: float
    prompt_complexity: float
    expected_response_type: str
    potential_ambiguities: List[str]
    optimization_suggestions: List[str]

@dataclass
class ModelBehaviorAnalysis:
    """Analysis of model behavior patterns."""
    behavior_type: ModelBehaviorType
    consistency_score: float
    bias_indicators: List[str]
    response_patterns: Dict[str, Any]
    prompt_sensitivity: Dict[str, float]
    reliability_metrics: Dict[str, float]

@dataclass
class CausalInfluenceAnalysis:
    """Analysis of causal influence in model responses."""
    prompt_component_weights: Dict[str, float]
    intervention_effectiveness: Dict[str, float]
    causal_paths: List[Dict[str, Any]]
    influence_attribution: Dict[str, float]

class ModelDebugger:
    """
    Provides deep analysis of LLM behavior for researchers and maintainers.
    Focuses on understanding model decisions, prompt engineering effects,
    and causal relationships in code generation.
    """
    
    def __init__(self):
        self.behavior_patterns = {
            'consistency': {
                'indicators': ['similar_responses', 'stable_patterns', 'predictable_outputs'],
                'thresholds': {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            },
            'bias': {
                'indicators': ['preferred_patterns', 'avoided_constructs', 'stereotypical_responses'],
                'thresholds': {'high': 0.7, 'medium': 0.5, 'low': 0.3}
            },
            'sensitivity': {
                'indicators': ['prompt_variation_impact', 'keyword_sensitivity', 'context_dependency'],
                'thresholds': {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            }
        }
    
    def analyze_prompt_engineering(self, original_prompt: str, counterfactual_prompt: str,
                                 original_response: str, counterfactual_response: str) -> PromptAnalysis:
        """Analyze prompt engineering effectiveness and provide optimization suggestions."""
        
        # Analyze prompt characteristics
        prompt_clarity = self._calculate_prompt_clarity(original_prompt)
        prompt_specificity = self._calculate_prompt_specificity(original_prompt)
        prompt_complexity = self._calculate_prompt_complexity(original_prompt)
        
        # Determine expected response type
        expected_response_type = self._determine_expected_response_type(original_prompt)
        
        # Identify potential ambiguities
        potential_ambiguities = self._identify_prompt_ambiguities(original_prompt)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_prompt_optimizations(
            original_prompt, counterfactual_prompt, original_response, counterfactual_response
        )
        
        return PromptAnalysis(
            prompt_clarity=prompt_clarity,
            prompt_specificity=prompt_specificity,
            prompt_complexity=prompt_complexity,
            expected_response_type=expected_response_type,
            potential_ambiguities=potential_ambiguities,
            optimization_suggestions=optimization_suggestions
        )
    
    def analyze_model_behavior(self, prompt_variations: List[str], 
                             responses: List[str], context: str = "") -> ModelBehaviorAnalysis:
        """Analyze model behavior patterns across different prompts."""
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(responses)
        
        # Identify bias indicators
        bias_indicators = self._identify_bias_indicators(prompt_variations, responses)
        
        # Analyze response patterns
        response_patterns = self._analyze_response_patterns(responses)
        
        # Calculate prompt sensitivity
        prompt_sensitivity = self._calculate_prompt_sensitivity(prompt_variations, responses)
        
        # Calculate reliability metrics
        reliability_metrics = self._calculate_reliability_metrics(responses)
        
        # Determine behavior type
        behavior_type = self._determine_behavior_type(consistency_score, bias_indicators, prompt_sensitivity)
        
        return ModelBehaviorAnalysis(
            behavior_type=behavior_type,
            consistency_score=consistency_score,
            bias_indicators=bias_indicators,
            response_patterns=response_patterns,
            prompt_sensitivity=prompt_sensitivity,
            reliability_metrics=reliability_metrics
        )
    
    def analyze_causal_influence(self, original_prompt: str, counterfactual_prompt: str,
                               original_response: str, counterfactual_response: str,
                               intervention_type: str) -> CausalInfluenceAnalysis:
        """Analyze causal influence of prompt components on model responses."""
        
        # Analyze prompt component weights
        prompt_component_weights = self._analyze_prompt_component_weights(
            original_prompt, counterfactual_prompt
        )
        
        # Calculate intervention effectiveness
        intervention_effectiveness = self._calculate_intervention_effectiveness(
            original_response, counterfactual_response, intervention_type
        )
        
        # Identify causal paths
        causal_paths = self._identify_causal_paths(
            original_prompt, counterfactual_prompt, original_response, counterfactual_response
        )
        
        # Calculate influence attribution
        influence_attribution = self._calculate_influence_attribution(
            prompt_component_weights, intervention_effectiveness, causal_paths
        )
        
        return CausalInfluenceAnalysis(
            prompt_component_weights=prompt_component_weights,
            intervention_effectiveness=intervention_effectiveness,
            causal_paths=causal_paths,
            influence_attribution=influence_attribution
        )
    
    def generate_debugging_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive debugging insights for model maintainers."""
        
        insights = {
            'model_stability': self._assess_model_stability(analysis_results),
            'prompt_engineering_insights': self._generate_prompt_insights(analysis_results),
            'bias_analysis': self._analyze_model_bias(analysis_results),
            'performance_optimization': self._suggest_performance_optimizations(analysis_results),
            'reliability_improvements': self._suggest_reliability_improvements(analysis_results)
        }
        
        return insights
    
    def _calculate_prompt_clarity(self, prompt: str) -> float:
        """Calculate prompt clarity score."""
        # Factors: sentence structure, vocabulary complexity, ambiguity
        sentences = prompt.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Shorter sentences are generally clearer
        clarity_score = max(0, 1 - avg_sentence_length / 20)
        
        # Check for ambiguous words
        ambiguous_words = ['it', 'this', 'that', 'these', 'those']
        ambiguity_count = sum(1 for word in ambiguous_words if word in prompt.lower())
        clarity_score -= ambiguity_count * 0.1
        
        return max(0, clarity_score)
    
    def _calculate_prompt_specificity(self, prompt: str) -> float:
        """Calculate prompt specificity score."""
        # Factors: concrete vs abstract language, specific requirements
        concrete_indicators = ['function', 'class', 'loop', 'recursion', 'algorithm', 'data structure']
        abstract_indicators = ['good', 'efficient', 'clean', 'proper', 'correct']
        
        concrete_count = sum(1 for indicator in concrete_indicators if indicator in prompt.lower())
        abstract_count = sum(1 for indicator in abstract_indicators if indicator in prompt.lower())
        
        specificity_score = concrete_count / (concrete_count + abstract_count + 1)
        return min(1.0, specificity_score)
    
    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score."""
        # Factors: length, vocabulary diversity, syntactic complexity
        words = prompt.split()
        unique_words = set(words)
        
        # Vocabulary diversity
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # Length complexity
        length_complexity = min(1.0, len(words) / 50)
        
        # Syntactic complexity (simplified)
        syntactic_complexity = len([c for c in prompt if c in ',;:()']) / len(prompt) if prompt else 0
        
        return (lexical_diversity + length_complexity + syntactic_complexity) / 3
    
    def _determine_expected_response_type(self, prompt: str) -> str:
        """Determine the expected type of response from the prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['function', 'def', 'method']):
            return 'function_definition'
        elif any(word in prompt_lower for word in ['class', 'object']):
            return 'class_definition'
        elif any(word in prompt_lower for word in ['algorithm', 'approach', 'solution']):
            return 'algorithm_implementation'
        elif any(word in prompt_lower for word in ['fix', 'bug', 'error']):
            return 'bug_fix'
        elif any(word in prompt_lower for word in ['optimize', 'improve', 'enhance']):
            return 'optimization'
        else:
            return 'general_code'
    
    def _identify_prompt_ambiguities(self, prompt: str) -> List[str]:
        """Identify potential ambiguities in the prompt."""
        ambiguities = []
        
        # Check for vague terms
        vague_terms = ['good', 'efficient', 'clean', 'proper', 'correct', 'better']
        for term in vague_terms:
            if term in prompt.lower():
                ambiguities.append(f"Vague term '{term}' - consider being more specific")
        
        # Check for pronouns without clear antecedents
        pronouns = ['it', 'this', 'that', 'these', 'those']
        for pronoun in pronouns:
            if pronoun in prompt.lower():
                ambiguities.append(f"Pronoun '{pronoun}' may be ambiguous")
        
        # Check for missing context
        if len(prompt.split()) < 10:
            ambiguities.append("Prompt may be too brief - consider adding more context")
        
        return ambiguities
    
    def _generate_prompt_optimizations(self, original_prompt: str, counterfactual_prompt: str,
                                     original_response: str, counterfactual_response: str) -> List[str]:
        """Generate prompt optimization suggestions."""
        suggestions = []
        
        # Compare prompt effectiveness
        original_quality = self._assess_response_quality(original_response)
        counterfactual_quality = self._assess_response_quality(counterfactual_response)
        
        if counterfactual_quality > original_quality:
            suggestions.append("Counterfactual prompt produced better results - consider incorporating its elements")
        
        # Analyze prompt differences
        prompt_diff = self._analyze_prompt_differences(original_prompt, counterfactual_prompt)
        
        for difference in prompt_diff:
            if difference['impact'] == 'positive':
                suggestions.append(f"Consider adding: {difference['element']}")
            elif difference['impact'] == 'negative':
                suggestions.append(f"Consider removing: {difference['element']}")
        
        # General optimization suggestions
        if len(original_prompt.split()) < 15:
            suggestions.append("Add more specific requirements to improve response quality")
        
        if not any(word in original_prompt.lower() for word in ['function', 'class', 'algorithm']):
            suggestions.append("Specify the expected code structure (function, class, etc.)")
        
        return suggestions
    
    def _calculate_consistency_score(self, responses: List[str]) -> float:
        """Calculate consistency score across multiple responses."""
        if len(responses) < 2:
            return 1.0
        
        # Calculate similarity between all pairs of responses
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_response_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _identify_bias_indicators(self, prompts: List[str], responses: List[str]) -> List[str]:
        """Identify potential bias indicators in model responses."""
        bias_indicators = []
        
        # Check for preferred patterns
        pattern_counts = {}
        for response in responses:
            patterns = self._extract_code_patterns(response)
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Identify overused patterns
        total_responses = len(responses)
        for pattern, count in pattern_counts.items():
            if count / total_responses > 0.7:  # More than 70% of responses use this pattern
                bias_indicators.append(f"Overused pattern: {pattern}")
        
        # Check for avoided constructs
        avoided_constructs = ['lambda', 'generator', 'decorator', 'context manager']
        for construct in avoided_constructs:
            construct_count = sum(1 for response in responses if construct in response)
            if construct_count / total_responses < 0.1:  # Less than 10% use this construct
                bias_indicators.append(f"Avoided construct: {construct}")
        
        return bias_indicators
    
    def _analyze_response_patterns(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze patterns in model responses."""
        patterns = {
            'common_structures': self._find_common_structures(responses),
            'code_style_preferences': self._analyze_code_style_preferences(responses),
            'algorithm_preferences': self._analyze_algorithm_preferences(responses),
            'complexity_distribution': self._analyze_complexity_distribution(responses)
        }
        
        return patterns
    
    def _calculate_prompt_sensitivity(self, prompts: List[str], responses: List[str]) -> Dict[str, float]:
        """Calculate model sensitivity to different prompt aspects."""
        sensitivity_metrics = {
            'keyword_sensitivity': self._calculate_keyword_sensitivity(prompts, responses),
            'length_sensitivity': self._calculate_length_sensitivity(prompts, responses),
            'structure_sensitivity': self._calculate_structure_sensitivity(prompts, responses),
            'context_sensitivity': self._calculate_context_sensitivity(prompts, responses)
        }
        
        return sensitivity_metrics
    
    def _calculate_reliability_metrics(self, responses: List[str]) -> Dict[str, float]:
        """Calculate reliability metrics for model responses."""
        metrics = {
            'syntax_correctness': self._calculate_syntax_correctness(responses),
            'completeness': self._calculate_completeness(responses),
            'consistency': self._calculate_consistency_score(responses),
            'predictability': self._calculate_predictability(responses)
        }
        
        return metrics
    
    def _determine_behavior_type(self, consistency_score: float, bias_indicators: List[str],
                               prompt_sensitivity: Dict[str, float]) -> ModelBehaviorType:
        """Determine the overall behavior type of the model."""
        if consistency_score > 0.8 and len(bias_indicators) < 2:
            return ModelBehaviorType.CONSISTENT
        elif consistency_score > 0.6:
            return ModelBehaviorType.VARIABLE
        elif any(sensitivity > 0.7 for sensitivity in prompt_sensitivity.values()):
            return ModelBehaviorType.UNPREDICTABLE
        else:
            return ModelBehaviorType.BIASED
    
    def _analyze_prompt_component_weights(self, original_prompt: str, counterfactual_prompt: str) -> Dict[str, float]:
        """Analyze the relative importance of different prompt components."""
        components = {
            'task_description': 0.3,
            'constraints': 0.25,
            'examples': 0.2,
            'style_guidance': 0.15,
            'context': 0.1
        }
        
        # Adjust weights based on prompt analysis
        if 'example' in counterfactual_prompt.lower():
            components['examples'] += 0.1
        if 'constraint' in counterfactual_prompt.lower():
            components['constraints'] += 0.1
        
        return components
    
    def _calculate_intervention_effectiveness(self, original_response: str, counterfactual_response: str,
                                           intervention_type: str) -> Dict[str, float]:
        """Calculate the effectiveness of different types of interventions."""
        effectiveness = {
            'structural_change': self._calculate_structural_change_effectiveness(original_response, counterfactual_response),
            'algorithmic_change': self._calculate_algorithmic_change_effectiveness(original_response, counterfactual_response),
            'style_change': self._calculate_style_change_effectiveness(original_response, counterfactual_response),
            'complexity_change': self._calculate_complexity_change_effectiveness(original_response, counterfactual_response)
        }
        
        return effectiveness
    
    def _identify_causal_paths(self, original_prompt: str, counterfactual_prompt: str,
                             original_response: str, counterfactual_response: str) -> List[Dict[str, Any]]:
        """Identify causal paths from prompt changes to response changes."""
        causal_paths = []
        
        # Analyze prompt differences
        prompt_diff = self._analyze_prompt_differences(original_prompt, counterfactual_prompt)
        
        # Analyze response differences
        response_diff = self._analyze_response_differences(original_response, counterfactual_response)
        
        # Map prompt changes to response changes
        for prompt_change in prompt_diff:
            for response_change in response_diff:
                if self._is_causally_related(prompt_change, response_change):
                    causal_paths.append({
                        'prompt_change': prompt_change,
                        'response_change': response_change,
                        'confidence': self._calculate_causal_confidence(prompt_change, response_change)
                    })
        
        return causal_paths
    
    def _calculate_influence_attribution(self, prompt_weights: Dict[str, float],
                                       intervention_effectiveness: Dict[str, float],
                                       causal_paths: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate influence attribution for different prompt components."""
        attribution = {}
        
        # Base attribution from prompt weights
        for component, weight in prompt_weights.items():
            attribution[component] = weight
        
        # Adjust based on intervention effectiveness
        for intervention_type, effectiveness in intervention_effectiveness.items():
            if intervention_type in attribution:
                attribution[intervention_type] *= (1 + effectiveness)
        
        # Normalize
        total = sum(attribution.values())
        if total > 0:
            attribution = {k: v / total for k, v in attribution.items()}
        
        return attribution
    
    def _assess_model_stability(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model stability based on analysis results."""
        stability_metrics = {
            'consistency': analysis_results.get('consistency_score', 0),
            'reliability': analysis_results.get('reliability_metrics', {}).get('consistency', 0),
            'predictability': analysis_results.get('reliability_metrics', {}).get('predictability', 0)
        }
        
        overall_stability = sum(stability_metrics.values()) / len(stability_metrics)
        
        return {
            'overall_stability': overall_stability,
            'stability_level': 'high' if overall_stability > 0.8 else 'medium' if overall_stability > 0.6 else 'low',
            'metrics': stability_metrics,
            'recommendations': self._generate_stability_recommendations(stability_metrics)
        }
    
    def _generate_prompt_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about prompt engineering effectiveness."""
        prompt_analysis = analysis_results.get('prompt_analysis', {})
        
        insights = {
            'clarity_score': prompt_analysis.get('prompt_clarity', 0),
            'specificity_score': prompt_analysis.get('prompt_specificity', 0),
            'complexity_score': prompt_analysis.get('prompt_complexity', 0),
            'optimization_opportunities': prompt_analysis.get('optimization_suggestions', []),
            'best_practices': self._generate_prompt_best_practices(prompt_analysis)
        }
        
        return insights
    
    def _analyze_model_bias(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model bias patterns."""
        behavior_analysis = analysis_results.get('behavior_analysis', {})
        
        bias_analysis = {
            'bias_indicators': behavior_analysis.get('bias_indicators', []),
            'response_patterns': behavior_analysis.get('response_patterns', {}),
            'bias_severity': self._calculate_bias_severity(behavior_analysis.get('bias_indicators', [])),
            'mitigation_strategies': self._generate_bias_mitigation_strategies(behavior_analysis)
        }
        
        return bias_analysis
    
    def _suggest_performance_optimizations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Suggest performance optimizations for the model."""
        suggestions = []
        
        # Analyze response patterns for performance issues
        response_patterns = analysis_results.get('behavior_analysis', {}).get('response_patterns', {})
        
        if 'complexity_distribution' in response_patterns:
            complexity_dist = response_patterns['complexity_distribution']
            if complexity_dist.get('high_complexity_ratio', 0) > 0.5:
                suggestions.append("Consider optimizing prompts to reduce response complexity")
        
        # Check for consistency issues
        consistency_score = analysis_results.get('behavior_analysis', {}).get('consistency_score', 0)
        if consistency_score < 0.6:
            suggestions.append("Improve prompt consistency to reduce response variability")
        
        return suggestions
    
    def _suggest_reliability_improvements(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Suggest reliability improvements for the model."""
        suggestions = []
        
        reliability_metrics = analysis_results.get('behavior_analysis', {}).get('reliability_metrics', {})
        
        if reliability_metrics.get('syntax_correctness', 0) < 0.9:
            suggestions.append("Add syntax validation to improve code correctness")
        
        if reliability_metrics.get('completeness', 0) < 0.8:
            suggestions.append("Enhance prompts to ensure complete responses")
        
        if reliability_metrics.get('predictability', 0) < 0.7:
            suggestions.append("Standardize prompt structure for more predictable responses")
        
        return suggestions
    
    # Helper methods for detailed analysis
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses."""
        # Use difflib for sequence similarity
        return difflib.SequenceMatcher(None, response1, response2).ratio()
    
    def _extract_code_patterns(self, code: str) -> List[str]:
        """Extract common code patterns from a response."""
        patterns = []
        
        if 'def ' in code:
            patterns.append('function_definition')
        if 'class ' in code:
            patterns.append('class_definition')
        if 'for ' in code:
            patterns.append('for_loop')
        if 'while ' in code:
            patterns.append('while_loop')
        if 'if ' in code:
            patterns.append('conditional')
        if 'try:' in code:
            patterns.append('error_handling')
        
        return patterns
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess the quality of a model response."""
        quality_score = 0.5  # Base score
        
        # Check for syntax correctness
        try:
            ast.parse(response)
            quality_score += 0.2
        except:
            pass
        
        # Check for completeness
        if 'def ' in response or 'class ' in response:
            quality_score += 0.2
        
        # Check for documentation
        if '#' in response or '"""' in response:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _analyze_prompt_differences(self, prompt1: str, prompt2: str) -> List[Dict[str, Any]]:
        """Analyze differences between two prompts."""
        # Simplified analysis - in practice, this would be more sophisticated
        differences = []
        
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        added_words = words2 - words1
        removed_words = words1 - words2
        
        for word in added_words:
            differences.append({
                'element': word,
                'type': 'added',
                'impact': 'positive'  # Simplified assumption
            })
        
        for word in removed_words:
            differences.append({
                'element': word,
                'type': 'removed',
                'impact': 'negative'  # Simplified assumption
            })
        
        return differences
    
    def _analyze_response_differences(self, response1: str, response2: str) -> List[Dict[str, Any]]:
        """Analyze differences between two responses."""
        # Simplified analysis
        differences = []
        
        if len(response1) != len(response2):
            differences.append({
                'type': 'length_change',
                'magnitude': abs(len(response1) - len(response2))
            })
        
        return differences
    
    def _is_causally_related(self, prompt_change: Dict[str, Any], response_change: Dict[str, Any]) -> bool:
        """Determine if a prompt change is causally related to a response change."""
        # Simplified causal relationship detection
        return True  # In practice, this would use more sophisticated analysis
    
    def _calculate_causal_confidence(self, prompt_change: Dict[str, Any], response_change: Dict[str, Any]) -> float:
        """Calculate confidence in a causal relationship."""
        # Simplified confidence calculation
        return 0.7  # In practice, this would be based on statistical analysis
    
    def _find_common_structures(self, responses: List[str]) -> Dict[str, int]:
        """Find common code structures across responses."""
        structure_counts = {}
        
        for response in responses:
            patterns = self._extract_code_patterns(response)
            for pattern in patterns:
                structure_counts[pattern] = structure_counts.get(pattern, 0) + 1
        
        return structure_counts
    
    def _analyze_code_style_preferences(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze code style preferences in responses."""
        style_analysis = {
            'indentation_style': 'consistent',  # Simplified
            'naming_conventions': 'snake_case',  # Simplified
            'comment_style': 'minimal'  # Simplified
        }
        
        return style_analysis
    
    def _analyze_algorithm_preferences(self, responses: List[str]) -> Dict[str, int]:
        """Analyze algorithm preferences in responses."""
        algorithm_counts = {
            'iterative': 0,
            'recursive': 0,
            'functional': 0,
            'object_oriented': 0
        }
        
        for response in responses:
            if 'for ' in response or 'while ' in response:
                algorithm_counts['iterative'] += 1
            if 'def ' in response and 'return ' in response:
                algorithm_counts['functional'] += 1
            if 'class ' in response:
                algorithm_counts['object_oriented'] += 1
        
        return algorithm_counts
    
    def _analyze_complexity_distribution(self, responses: List[str]) -> Dict[str, float]:
        """Analyze complexity distribution across responses."""
        complexities = []
        
        for response in responses:
            try:
                ast_tree = ast.parse(response)
                complexity = len([n for n in ast.walk(ast_tree) if isinstance(n, (ast.If, ast.For, ast.While))])
                complexities.append(complexity)
            except:
                complexities.append(0)
        
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            high_complexity_ratio = sum(1 for c in complexities if c > 5) / len(complexities)
        else:
            avg_complexity = 0
            high_complexity_ratio = 0
        
        return {
            'average_complexity': avg_complexity,
            'high_complexity_ratio': high_complexity_ratio
        }
    
    def _calculate_keyword_sensitivity(self, prompts: List[str], responses: List[str]) -> float:
        """Calculate model sensitivity to keywords."""
        # Simplified calculation
        return 0.6
    
    def _calculate_length_sensitivity(self, prompts: List[str], responses: List[str]) -> float:
        """Calculate model sensitivity to prompt length."""
        # Simplified calculation
        return 0.5
    
    def _calculate_structure_sensitivity(self, prompts: List[str], responses: List[str]) -> float:
        """Calculate model sensitivity to prompt structure."""
        # Simplified calculation
        return 0.4
    
    def _calculate_context_sensitivity(self, prompts: List[str], responses: List[str]) -> float:
        """Calculate model sensitivity to context."""
        # Simplified calculation
        return 0.7
    
    def _calculate_syntax_correctness(self, responses: List[str]) -> float:
        """Calculate syntax correctness rate."""
        correct_count = 0
        
        for response in responses:
            try:
                ast.parse(response)
                correct_count += 1
            except:
                pass
        
        return correct_count / len(responses) if responses else 0
    
    def _calculate_completeness(self, responses: List[str]) -> float:
        """Calculate response completeness."""
        complete_count = 0
        
        for response in responses:
            if 'def ' in response or 'class ' in response:
                complete_count += 1
        
        return complete_count / len(responses) if responses else 0
    
    def _calculate_predictability(self, responses: List[str]) -> float:
        """Calculate response predictability."""
        # Simplified calculation based on consistency
        return self._calculate_consistency_score(responses)
    
    def _calculate_structural_change_effectiveness(self, original: str, counterfactual: str) -> float:
        """Calculate effectiveness of structural changes."""
        # Simplified calculation
        return 0.6
    
    def _calculate_algorithmic_change_effectiveness(self, original: str, counterfactual: str) -> float:
        """Calculate effectiveness of algorithmic changes."""
        # Simplified calculation
        return 0.7
    
    def _calculate_style_change_effectiveness(self, original: str, counterfactual: str) -> float:
        """Calculate effectiveness of style changes."""
        # Simplified calculation
        return 0.4
    
    def _calculate_complexity_change_effectiveness(self, original: str, counterfactual: str) -> float:
        """Calculate effectiveness of complexity changes."""
        # Simplified calculation
        return 0.5
    
    def _calculate_bias_severity(self, bias_indicators: List[str]) -> str:
        """Calculate the severity of model bias."""
        if len(bias_indicators) > 5:
            return 'high'
        elif len(bias_indicators) > 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_bias_mitigation_strategies(self, behavior_analysis: Dict[str, Any]) -> List[str]:
        """Generate strategies to mitigate model bias."""
        strategies = []
        
        bias_indicators = behavior_analysis.get('bias_indicators', [])
        
        if any('overused' in indicator for indicator in bias_indicators):
            strategies.append("Diversify training data to reduce pattern overuse")
        
        if any('avoided' in indicator for indicator in bias_indicators):
            strategies.append("Include examples of avoided constructs in training")
        
        return strategies
    
    def _generate_stability_recommendations(self, stability_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving model stability."""
        recommendations = []
        
        if stability_metrics.get('consistency', 0) < 0.7:
            recommendations.append("Standardize prompt templates for better consistency")
        
        if stability_metrics.get('predictability', 0) < 0.6:
            recommendations.append("Reduce prompt variability to improve predictability")
        
        return recommendations
    
    def _generate_prompt_best_practices(self, prompt_analysis: Dict[str, Any]) -> List[str]:
        """Generate best practices for prompt engineering."""
        practices = []
        
        if prompt_analysis.get('prompt_clarity', 0) < 0.7:
            practices.append("Use clear, concise language in prompts")
        
        if prompt_analysis.get('prompt_specificity', 0) < 0.6:
            practices.append("Include specific requirements and constraints")
        
        return practices 