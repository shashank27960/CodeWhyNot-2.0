import ast
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import difflib

@dataclass
class CodeExplanation:
    """Represents a step-by-step explanation of code changes."""
    step_number: int
    title: str
    description: str
    code_snippet: str
    reasoning: str
    impact: str
    difficulty_level: str  # 'beginner', 'intermediate', 'advanced'

@dataclass
class BugFixExplanation:
    """Represents an explanation of how to fix a specific bug."""
    bug_type: str
    bug_description: str
    original_code: str
    fixed_code: str
    fix_explanation: str
    why_it_works: str
    common_mistakes: List[str]

class EducationalExplainer:
    """
    Provides educational explanations for code generation and debugging.
    Designed for students learning programming and understanding LLM decisions.
    """
    
    def __init__(self):
        self.bug_patterns = {
            'infinite_loop': {
                'pattern': r'while\s+True:',
                'description': 'Infinite loop without proper exit condition',
                'fix': 'Add a break condition or modify the loop condition'
            },
            'off_by_one': {
                'pattern': r'range\([^)]*\)',
                'description': 'Potential off-by-one error in range',
                'fix': 'Check if range bounds are correct'
            },
            'undefined_variable': {
                'pattern': r'\b[a-zA-Z_]\w*\b(?!\s*[=\(])',
                'description': 'Variable used before definition',
                'fix': 'Define the variable before using it'
            },
            'missing_return': {
                'pattern': r'def\s+\w+\s*\([^)]*\)[^:]*:\s*(?:[^\n]*\n)*[^:]*\b(?!return\b)',
                'description': 'Function may not return a value',
                'fix': 'Add return statement or ensure all paths return'
            }
        }
    
    def explain_code_generation(self, original_prompt: str, generated_code: str, 
                              intervention_type: str = None) -> List[CodeExplanation]:
        """Generate step-by-step explanation of how the code was generated."""
        explanations = []
        
        # Step 1: Understanding the prompt
        explanations.append(CodeExplanation(
            step_number=1,
            title="Understanding the Request",
            description="Analyzing what the user wants to accomplish",
            code_snippet=original_prompt,
            reasoning="The LLM first parses the natural language request to understand the programming task",
            impact="This determines the overall approach and algorithm choice",
            difficulty_level="beginner"
        ))
        
        # Step 2: Algorithm selection
        algorithm_choice = self._identify_algorithm_choice(generated_code)
        explanations.append(CodeExplanation(
            step_number=2,
            title="Algorithm Selection",
            description=f"Choosing {algorithm_choice} approach",
            code_snippet=self._extract_algorithm_snippet(generated_code),
            reasoning=f"The LLM selected {algorithm_choice} because it's appropriate for this task",
            impact="This affects performance, readability, and complexity",
            difficulty_level="intermediate"
        ))
        
        # Step 3: Code structure
        structure_analysis = self._analyze_code_structure(generated_code)
        explanations.append(CodeExplanation(
            step_number=3,
            title="Code Structure Design",
            description=f"Organizing code with {structure_analysis['structure_type']}",
            code_snippet=structure_analysis['example_snippet'],
            reasoning=structure_analysis['reasoning'],
            impact="Good structure makes code readable and maintainable",
            difficulty_level="intermediate"
        ))
        
        # Step 4: Implementation details
        implementation_details = self._analyze_implementation_details(generated_code)
        explanations.append(CodeExplanation(
            step_number=4,
            title="Implementation Details",
            description="Adding specific implementation logic",
            code_snippet=implementation_details['snippet'],
            reasoning=implementation_details['reasoning'],
            impact="These details ensure the code works correctly",
            difficulty_level="advanced"
        ))
        
        # Step 5: Intervention explanation (if applicable)
        if intervention_type:
            intervention_explanation = self._explain_intervention(intervention_type, original_prompt)
            explanations.append(CodeExplanation(
                step_number=5,
                title="Applied Intervention",
                description=f"Modified approach using {intervention_type}",
                code_snippet=intervention_explanation['snippet'],
                reasoning=intervention_explanation['reasoning'],
                impact=intervention_explanation['impact'],
                difficulty_level="advanced"
            ))
        
        return explanations
    
    def explain_bug_fixes(self, buggy_code: str, fixed_code: str) -> List[BugFixExplanation]:
        """Explain how to fix bugs in the code."""
        bug_fixes = []
        
        # Analyze differences
        diff = list(difflib.unified_diff(
            buggy_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile='buggy_code.py',
            tofile='fixed_code.py'
        ))
        
        # Identify bug patterns
        for bug_type, pattern_info in self.bug_patterns.items():
            if re.search(pattern_info['pattern'], buggy_code):
                bug_fixes.append(BugFixExplanation(
                    bug_type=bug_type,
                    bug_description=pattern_info['description'],
                    original_code=self._extract_buggy_snippet(buggy_code, bug_type),
                    fixed_code=self._extract_fixed_snippet(fixed_code, bug_type),
                    fix_explanation=pattern_info['fix'],
                    why_it_works=self._explain_why_fix_works(bug_type),
                    common_mistakes=self._get_common_mistakes(bug_type)
                ))
        
        return bug_fixes
    
    def explain_intervention_impact(self, original_code: str, counterfactual_code: str,
                                  intervention_type: str) -> Dict[str, Any]:
        """Explain the impact of a causal intervention."""
        impact_analysis = {
            'structural_changes': self._analyze_structural_changes(original_code, counterfactual_code),
            'performance_impact': self._analyze_performance_impact(original_code, counterfactual_code),
            'readability_impact': self._analyze_readability_impact(original_code, counterfactual_code),
            'maintainability_impact': self._analyze_maintainability_impact(original_code, counterfactual_code),
            'learning_value': self._assess_learning_value(intervention_type)
        }
        
        return impact_analysis
    
    def generate_learning_objectives(self, code: str, intervention_type: str = None) -> List[str]:
        """Generate learning objectives for the code."""
        objectives = []
        
        # Basic programming concepts
        if 'def ' in code:
            objectives.append("Understand function definition and scope")
        if 'for ' in code or 'while ' in code:
            objectives.append("Learn about loops and iteration")
        if 'if ' in code:
            objectives.append("Understand conditional statements")
        if 'return ' in code:
            objectives.append("Learn about function return values")
        
        # Advanced concepts
        if 'try:' in code:
            objectives.append("Understand error handling and exceptions")
        if 'lambda ' in code:
            objectives.append("Learn about lambda functions")
        if 'list(' in code and 'for ' in code:
            objectives.append("Understand list comprehensions")
        
        # Intervention-specific objectives
        if intervention_type == 'loop_to_recursion':
            objectives.append("Compare iterative vs recursive approaches")
        elif intervention_type == 'add_error_handling':
            objectives.append("Learn defensive programming practices")
        elif intervention_type == 'optimize_algorithm':
            objectives.append("Understand algorithm complexity and optimization")
        
        return objectives
    
    def _identify_algorithm_choice(self, code: str) -> str:
        """Identify the main algorithm approach used in the code."""
        if 'def ' in code and 'return ' in code:
            return "functional"
        elif 'for ' in code or 'while ' in code:
            return "iterative"
        elif 'if ' in code and 'else' in code:
            return "conditional"
        else:
            return "procedural"
    
    def _extract_algorithm_snippet(self, code: str) -> str:
        """Extract a representative snippet showing the algorithm choice."""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ['def ', 'for ', 'while ', 'if ']):
                return '\n'.join(lines[max(0, i-1):min(len(lines), i+2)])
        return code[:100] + "..." if len(code) > 100 else code
    
    def _analyze_code_structure(self, code: str) -> Dict[str, str]:
        """Analyze the structure of the code."""
        if 'class ' in code:
            return {
                'structure_type': 'object-oriented',
                'example_snippet': 'class Example:\n    def __init__(self):\n        pass',
                'reasoning': 'Using classes for better organization and encapsulation'
            }
        elif 'def ' in code:
            return {
                'structure_type': 'functional',
                'example_snippet': 'def function_name():\n    return result',
                'reasoning': 'Using functions to break down the problem into smaller parts'
            }
        else:
            return {
                'structure_type': 'procedural',
                'example_snippet': 'variable = value\nresult = calculation',
                'reasoning': 'Using simple procedural approach for straightforward tasks'
            }
    
    def _analyze_implementation_details(self, code: str) -> Dict[str, str]:
        """Analyze specific implementation details."""
        # Find the most important line
        lines = code.split('\n')
        important_line = ""
        for line in lines:
            if any(keyword in line for keyword in ['return ', 'print(', 'result']):
                important_line = line.strip()
                break
        
        return {
            'snippet': important_line or lines[0] if lines else "",
            'reasoning': 'This line contains the core logic that produces the desired output'
        }
    
    def _explain_intervention(self, intervention_type: str, original_prompt: str) -> Dict[str, str]:
        """Explain why a specific intervention was applied."""
        explanations = {
            'loop_to_recursion': {
                'snippet': 'def recursive_function(n):\n    if n <= 1:\n        return 1\n    return n * recursive_function(n-1)',
                'reasoning': 'Converting loops to recursion can make the code more elegant and easier to understand for certain problems',
                'impact': 'May improve readability but could impact performance for large inputs'
            },
            'add_error_handling': {
                'snippet': 'try:\n    result = calculation()\nexcept ValueError:\n    print("Invalid input")',
                'reasoning': 'Adding error handling makes the code more robust and user-friendly',
                'impact': 'Improves reliability and user experience'
            },
            'optimize_algorithm': {
                'snippet': '# Using more efficient algorithm\nresult = optimized_calculation()',
                'reasoning': 'Optimizing the algorithm improves performance and scalability',
                'impact': 'Better performance, especially for larger inputs'
            }
        }
        
        return explanations.get(intervention_type, {
            'snippet': 'Modified code',
            'reasoning': 'Applied intervention to improve the code',
            'impact': 'Enhanced code quality'
        })
    
    def _extract_buggy_snippet(self, code: str, bug_type: str) -> str:
        """Extract the buggy part of the code."""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if re.search(self.bug_patterns[bug_type]['pattern'], line):
                return '\n'.join(lines[max(0, i-1):min(len(lines), i+2)])
        return code[:100] + "..." if len(code) > 100 else code
    
    def _extract_fixed_snippet(self, code: str, bug_type: str) -> str:
        """Extract the fixed version of the code."""
        # This would need more sophisticated logic to show the actual fix
        return "Fixed version of the code"
    
    def _explain_why_fix_works(self, bug_type: str) -> str:
        """Explain why a particular fix works."""
        explanations = {
            'infinite_loop': 'Adding a proper exit condition prevents the loop from running forever',
            'off_by_one': 'Correcting the range bounds ensures we process the right number of elements',
            'undefined_variable': 'Defining variables before use ensures they have valid values',
            'missing_return': 'Adding return statements ensures the function produces the expected output'
        }
        return explanations.get(bug_type, 'The fix addresses the specific issue in the code')
    
    def _get_common_mistakes(self, bug_type: str) -> List[str]:
        """Get common mistakes related to this bug type."""
        mistakes = {
            'infinite_loop': [
                'Forgetting to update loop variables',
                'Using wrong loop conditions',
                'Missing break statements'
            ],
            'off_by_one': [
                'Using <= instead of < in range',
                'Starting from 1 instead of 0',
                'Incorrect boundary conditions'
            ],
            'undefined_variable': [
                'Using variables before assignment',
                'Misspelling variable names',
                'Using wrong scope'
            ],
            'missing_return': [
                'Forgetting return statements',
                'Returning wrong values',
                'Missing return in some code paths'
            ]
        }
        return mistakes.get(bug_type, ['Common programming mistakes'])
    
    def _analyze_structural_changes(self, original: str, counterfactual: str) -> Dict[str, Any]:
        """Analyze structural changes between original and counterfactual code."""
        return {
            'lines_added': len(counterfactual.split('\n')) - len(original.split('\n')),
            'complexity_change': 'increased' if len(counterfactual) > len(original) else 'decreased',
            'structure_type': 'similar' if len(counterfactual.split('\n')) == len(original.split('\n')) else 'different'
        }
    
    def _analyze_performance_impact(self, original: str, counterfactual: str) -> Dict[str, str]:
        """Analyze performance impact of changes."""
        # Simplified analysis - in practice, this would be more sophisticated
        return {
            'time_complexity': 'similar',
            'space_complexity': 'similar',
            'overall_impact': 'minimal'
        }
    
    def _analyze_readability_impact(self, original: str, counterfactual: str) -> Dict[str, str]:
        """Analyze readability impact of changes."""
        return {
            'clarity': 'improved' if len(counterfactual) > len(original) else 'similar',
            'comments': 'adequate',
            'naming': 'clear'
        }
    
    def _analyze_maintainability_impact(self, original: str, counterfactual: str) -> Dict[str, str]:
        """Analyze maintainability impact of changes."""
        return {
            'modularity': 'good',
            'documentation': 'adequate',
            'testability': 'improved'
        }
    
    def _assess_learning_value(self, intervention_type: str) -> Dict[str, str]:
        """Assess the learning value of an intervention."""
        learning_values = {
            'loop_to_recursion': 'high',
            'add_error_handling': 'medium',
            'optimize_algorithm': 'high',
            'recursion_to_loop': 'medium'
        }
        
        return {
            'concept_difficulty': learning_values.get(intervention_type, 'medium'),
            'practical_value': 'high',
            'transferability': 'good'
        } 