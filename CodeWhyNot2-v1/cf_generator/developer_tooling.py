import ast
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import difflib

class ConfidenceLevel(Enum):
    """Confidence levels for intervention suggestions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class InterventionSuggestion:
    """Represents a ranked intervention suggestion for developers."""
    intervention_type: str
    description: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    reasoning: str
    code_patch: str
    expected_impact: Dict[str, Any]
    risk_assessment: str
    implementation_difficulty: str

@dataclass
class CodeAnalysis:
    """Represents comprehensive code analysis for developers."""
    code_quality_score: float
    maintainability_index: float
    complexity_metrics: Dict[str, Any]
    potential_issues: List[str]
    optimization_opportunities: List[str]
    best_practices_violations: List[str]

class DeveloperTooling:
    """
    Provides IDE-like features for developers including:
    - Ranked intervention suggestions with confidence scores
    - Code quality analysis
    - Risk assessment
    - Implementation guidance
    """
    
    def __init__(self):
        self.intervention_templates = {
            'performance_optimization': {
                'description': 'Optimize code performance',
                'confidence_factors': ['algorithm_complexity', 'execution_time', 'memory_usage'],
                'risk_level': 'low',
                'difficulty': 'medium'
            },
            'code_refactoring': {
                'description': 'Improve code structure and readability',
                'confidence_factors': ['code_complexity', 'maintainability', 'readability'],
                'risk_level': 'medium',
                'difficulty': 'medium'
            },
            'bug_fix': {
                'description': 'Fix identified bugs',
                'confidence_factors': ['bug_severity', 'fix_complexity', 'test_coverage'],
                'risk_level': 'high',
                'difficulty': 'variable'
            },
            'security_improvement': {
                'description': 'Enhance code security',
                'confidence_factors': ['security_vulnerability', 'impact_severity', 'fix_complexity'],
                'risk_level': 'high',
                'difficulty': 'high'
            },
            'error_handling': {
                'description': 'Add comprehensive error handling',
                'confidence_factors': ['error_frequency', 'user_impact', 'implementation_complexity'],
                'risk_level': 'low',
                'difficulty': 'low'
            }
        }
    
    def analyze_code_for_developers(self, code: str, context: str = "") -> CodeAnalysis:
        """Perform comprehensive code analysis for developers."""
        try:
            ast_tree = ast.parse(code)
            
            analysis = CodeAnalysis(
                code_quality_score=self._calculate_code_quality_score(code, ast_tree),
                maintainability_index=self._calculate_maintainability_index(code, ast_tree),
                complexity_metrics=self._analyze_complexity(code, ast_tree),
                potential_issues=self._identify_potential_issues(code, ast_tree),
                optimization_opportunities=self._find_optimization_opportunities(code, ast_tree),
                best_practices_violations=self._check_best_practices(code, ast_tree)
            )
            
            return analysis
        except Exception as e:
            return CodeAnalysis(
                code_quality_score=0.0,
                maintainability_index=0.0,
                complexity_metrics={},
                potential_issues=[f"Parse error: {str(e)}"],
                optimization_opportunities=[],
                best_practices_violations=[]
            )
    
    def generate_ranked_interventions(self, code: str, analysis: CodeAnalysis, 
                                    context: str = "") -> List[InterventionSuggestion]:
        """Generate ranked intervention suggestions with confidence scores."""
        suggestions = []
        
        # Performance optimization suggestions
        if analysis.complexity_metrics.get('cyclomatic_complexity', 0) > 10:
            suggestions.append(self._create_performance_suggestion(code, analysis))
        
        # Code refactoring suggestions
        if analysis.maintainability_index < 70:
            suggestions.append(self._create_refactoring_suggestion(code, analysis))
        
        # Bug fix suggestions
        for issue in analysis.potential_issues:
            if 'bug' in issue.lower() or 'error' in issue.lower():
                suggestions.append(self._create_bug_fix_suggestion(code, issue))
        
        # Security improvement suggestions
        if any('security' in issue.lower() for issue in analysis.potential_issues):
            suggestions.append(self._create_security_suggestion(code, analysis))
        
        # Error handling suggestions
        if not self._has_comprehensive_error_handling(code):
            suggestions.append(self._create_error_handling_suggestion(code, analysis))
        
        # Sort by confidence score
        suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return suggestions
    
    def calculate_confidence_score(self, intervention_type: str, code: str, 
                                 analysis: Optional[CodeAnalysis]) -> Tuple[float, ConfidenceLevel]:
        """Calculate confidence score for an intervention."""
        base_score = 0.5
        
        # Handle case where analysis is None
        if analysis is None:
            # For bug_fix, we can still provide a reasonable score
            if intervention_type == 'bug_fix':
                base_score += 0.2  # Moderate confidence for bug fixes without analysis
            elif intervention_type == 'error_handling':
                if not self._has_comprehensive_error_handling(code):
                    base_score += 0.3
        else:
            # Adjust based on intervention type
            if intervention_type == 'performance_optimization':
                complexity = analysis.complexity_metrics.get('cyclomatic_complexity', 0)
                if complexity > 15:
                    base_score += 0.3
                elif complexity > 10:
                    base_score += 0.2
            
            elif intervention_type == 'code_refactoring':
                maintainability = analysis.maintainability_index
                if maintainability < 50:
                    base_score += 0.4
                elif maintainability < 70:
                    base_score += 0.2
            
            elif intervention_type == 'bug_fix':
                issue_count = len([i for i in analysis.potential_issues if 'bug' in i.lower()])
                base_score += min(0.3, issue_count * 0.1)
            
            elif intervention_type == 'security_improvement':
                security_issues = len([i for i in analysis.potential_issues if 'security' in i.lower()])
                base_score += min(0.4, security_issues * 0.2)
            
            elif intervention_type == 'error_handling':
                if not self._has_comprehensive_error_handling(code):
                    base_score += 0.3
        
        # Determine confidence level
        if base_score >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif base_score >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return min(1.0, base_score), confidence_level
    
    def generate_code_patch(self, code: str, intervention_type: str, 
                          suggestion: InterventionSuggestion) -> str:
        """Generate a code patch for the suggested intervention."""
        if intervention_type == 'performance_optimization':
            return self._generate_performance_patch(code)
        elif intervention_type == 'code_refactoring':
            return self._generate_refactoring_patch(code)
        elif intervention_type == 'bug_fix':
            return self._generate_bug_fix_patch(code, suggestion)
        elif intervention_type == 'security_improvement':
            return self._generate_security_patch(code)
        elif intervention_type == 'error_handling':
            return self._generate_error_handling_patch(code)
        else:
            return "# No patch available for this intervention type"
    
    def assess_implementation_risk(self, intervention_type: str, code: str) -> Dict[str, Any]:
        """Assess the risk of implementing an intervention."""
        risk_factors = {
            'code_complexity': self._assess_complexity_risk(code),
            'test_coverage': self._assess_test_coverage_risk(code),
            'dependencies': self._assess_dependency_risk(code),
            'business_criticality': self._assess_business_risk(code)
        }
        
        overall_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            'overall_risk': overall_risk,
            'risk_level': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low',
            'risk_factors': risk_factors,
            'mitigation_strategies': self._suggest_risk_mitigation(intervention_type, risk_factors)
        }
    
    def _calculate_code_quality_score(self, code: str, ast_tree: ast.AST) -> float:
        """Calculate overall code quality score."""
        scores = []
        
        # Complexity score (lower is better)
        complexity = self._calculate_complexity_score(ast_tree)
        scores.append(max(0, 1 - complexity / 20))
        
        # Readability score
        readability = self._calculate_readability_score(code)
        scores.append(readability)
        
        # Maintainability score
        maintainability = self._calculate_maintainability_score(code, ast_tree)
        scores.append(maintainability)
        
        # Documentation score
        documentation = self._calculate_documentation_score(code)
        scores.append(documentation)
        
        return sum(scores) / len(scores)
    
    def _calculate_maintainability_index(self, code: str, ast_tree: ast.AST) -> float:
        """Calculate maintainability index."""
        # Simplified maintainability calculation
        lines = len(code.split('\n'))
        functions = len([node for node in ast.walk(ast_tree) if isinstance(node, ast.FunctionDef)])
        complexity = len([node for node in ast.walk(ast_tree) if isinstance(node, (ast.If, ast.For, ast.While))])
        
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
        # Simplified version
        mi = 171 - 0.23 * complexity - 16.2 * (lines / 100)
        return max(0, min(100, mi))
    
    def _analyze_complexity(self, code: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        cyclomatic = 1  # Base complexity
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                cyclomatic += 1
            elif isinstance(node, ast.BoolOp):
                cyclomatic += len(node.values) - 1
        
        return {
            'cyclomatic_complexity': cyclomatic,
            'function_count': len([n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)]),
            'class_count': len([n for n in ast.walk(ast_tree) if isinstance(n, ast.ClassDef)]),
            'nesting_depth': self._calculate_max_nesting_depth(ast_tree)
        }
    
    def _identify_potential_issues(self, code: str, ast_tree: ast.AST) -> List[str]:
        """Identify potential issues in the code."""
        issues = []
        
        # Check for common issues
        if len(code.split('\n')) > 100:
            issues.append("Code is very long - consider breaking into smaller functions")
        
        complexity = len([n for n in ast.walk(ast_tree) if isinstance(n, (ast.If, ast.For, ast.While))])
        if complexity > 10:
            issues.append("High cyclomatic complexity - consider refactoring")
        
        if not re.search(r'def\s+\w+', code):
            issues.append("No functions defined - consider modularizing code")
        
        if re.search(r'print\s*\(', code) and 'logging' not in code:
            issues.append("Using print statements - consider proper logging")
        
        return issues
    
    def _find_optimization_opportunities(self, code: str, ast_tree: ast.AST) -> List[str]:
        """Find optimization opportunities."""
        opportunities = []
        
        # Check for inefficient patterns
        if re.search(r'for.*in.*range.*len', code):
            opportunities.append("Consider using enumerate() instead of range(len())")
        
        if re.search(r'list\(.*for.*in', code):
            opportunities.append("Consider using list comprehension for better performance")
        
        if re.search(r'if.*in.*list', code):
            opportunities.append("Consider using set for faster membership testing")
        
        return opportunities
    
    def _check_best_practices(self, code: str, ast_tree: ast.AST) -> List[str]:
        """Check for best practices violations."""
        violations = []
        
        # Check naming conventions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef) and not node.name.islower():
                violations.append(f"Function '{node.name}' should use snake_case")
            elif isinstance(node, ast.ClassDef) and not node.name[0].isupper():
                violations.append(f"Class '{node.name}' should use PascalCase")
        
        # Check for magic numbers
        if re.search(r'\b\d{3,}\b', code):
            violations.append("Consider using named constants instead of magic numbers")
        
        return violations
    
    def _create_performance_suggestion(self, code: str, analysis: CodeAnalysis) -> InterventionSuggestion:
        """Create a performance optimization suggestion."""
        confidence_score, confidence_level = self.calculate_confidence_score('performance_optimization', code, analysis)
        
        return InterventionSuggestion(
            intervention_type='performance_optimization',
            description='Optimize code performance by reducing complexity and improving algorithms',
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            reasoning=f"High cyclomatic complexity ({analysis.complexity_metrics.get('cyclomatic_complexity', 0)}) indicates performance optimization opportunities",
            code_patch=self._generate_performance_patch(code),
            expected_impact={'performance': 'high', 'readability': 'medium', 'maintainability': 'medium'},
            risk_assessment='low',
            implementation_difficulty='medium'
        )
    
    def _create_refactoring_suggestion(self, code: str, analysis: CodeAnalysis) -> InterventionSuggestion:
        """Create a code refactoring suggestion."""
        confidence_score, confidence_level = self.calculate_confidence_score('code_refactoring', code, analysis)
        
        return InterventionSuggestion(
            intervention_type='code_refactoring',
            description='Refactor code to improve maintainability and readability',
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            reasoning=f"Low maintainability index ({analysis.maintainability_index:.1f}) suggests refactoring is needed",
            code_patch=self._generate_refactoring_patch(code),
            expected_impact={'performance': 'low', 'readability': 'high', 'maintainability': 'high'},
            risk_assessment='medium',
            implementation_difficulty='medium'
        )
    
    def _create_bug_fix_suggestion(self, code: str, issue: str) -> InterventionSuggestion:
        """Create a bug fix suggestion."""
        confidence_score, confidence_level = self.calculate_confidence_score('bug_fix', code, None)
        
        return InterventionSuggestion(
            intervention_type='bug_fix',
            description=f'Fix identified issue: {issue}',
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            reasoning=f"Potential bug detected: {issue}",
            code_patch=self._generate_bug_fix_patch(code, None),
            expected_impact={'reliability': 'high', 'performance': 'variable', 'maintainability': 'low'},
            risk_assessment='high',
            implementation_difficulty='variable'
        )
    
    def _create_security_suggestion(self, code: str, analysis: CodeAnalysis) -> InterventionSuggestion:
        """Create a security improvement suggestion."""
        confidence_score, confidence_level = self.calculate_confidence_score('security_improvement', code, analysis)
        
        return InterventionSuggestion(
            intervention_type='security_improvement',
            description='Improve code security by addressing identified vulnerabilities',
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            reasoning="Security vulnerabilities detected in code analysis",
            code_patch=self._generate_security_patch(code),
            expected_impact={'security': 'high', 'reliability': 'high', 'performance': 'low'},
            risk_assessment='high',
            implementation_difficulty='high'
        )
    
    def _create_error_handling_suggestion(self, code: str, analysis: CodeAnalysis) -> InterventionSuggestion:
        """Create an error handling suggestion."""
        confidence_score, confidence_level = self.calculate_confidence_score('error_handling', code, analysis)
        
        return InterventionSuggestion(
            intervention_type='error_handling',
            description='Add comprehensive error handling to improve code robustness',
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            reasoning="Code lacks comprehensive error handling",
            code_patch=self._generate_error_handling_patch(code),
            expected_impact={'reliability': 'high', 'user_experience': 'high', 'maintainability': 'medium'},
            risk_assessment='low',
            implementation_difficulty='low'
        )
    
    def _has_comprehensive_error_handling(self, code: str) -> bool:
        """Check if code has comprehensive error handling."""
        return bool(re.search(r'try\s*:', code) and re.search(r'except\s*:', code))
    
    def _generate_performance_patch(self, code: str) -> str:
        """Generate a performance optimization patch."""
        return """# Performance optimization suggestions:
# 1. Use list comprehensions instead of loops where possible
# 2. Consider using sets for membership testing
# 3. Avoid repeated calculations in loops
# 4. Use built-in functions when available

# Example optimization:
# Before: result = []
#         for i in range(len(items)):
#             if condition(items[i]):
#                 result.append(transform(items[i]))
# After:  result = [transform(item) for item in items if condition(item)]
"""
    
    def _generate_refactoring_patch(self, code: str) -> str:
        """Generate a refactoring patch."""
        return """# Refactoring suggestions:
# 1. Extract long functions into smaller, focused functions
# 2. Use meaningful variable and function names
# 3. Reduce nesting levels
# 4. Add type hints for better code clarity

# Example refactoring:
# def long_function():
#     # Break into smaller functions
#     result = process_data()
#     return format_result(result)
"""
    
    def _generate_bug_fix_patch(self, code: str, suggestion: Optional[InterventionSuggestion] = None) -> str:
        """Generate a bug fix patch."""
        return """# Bug fix suggestions:
# 1. Add proper input validation
# 2. Handle edge cases
# 3. Fix off-by-one errors
# 4. Add proper error handling

# Example bug fix:
# Before: result = items[index]  # May raise IndexError
# After:  result = items[index] if 0 <= index < len(items) else None
"""
    
    def _generate_security_patch(self, code: str) -> str:
        """Generate a security patch."""
        return """# Security improvement suggestions:
# 1. Validate all user inputs
# 2. Use parameterized queries for database operations
# 3. Avoid eval() and exec() with user input
# 4. Implement proper authentication and authorization

# Example security fix:
# Before: result = eval(user_input)  # Dangerous!
# After:  result = safe_evaluation(user_input)  # Use safe alternatives
"""
    
    def _generate_error_handling_patch(self, code: str) -> str:
        """Generate an error handling patch."""
        return """# Error handling suggestions:
# 1. Add try-except blocks around risky operations
# 2. Provide meaningful error messages
# 3. Log errors for debugging
# 4. Handle specific exception types

# Example error handling:
# try:
#     result = risky_operation()
# except ValueError as e:
#     logger.error(f"Invalid input: {e}")
#     return None
# except Exception as e:
#     logger.error(f"Unexpected error: {e}")
#     raise
"""
    
    def _calculate_complexity_score(self, ast_tree: ast.AST) -> float:
        """Calculate complexity score."""
        complexity = 1
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def _calculate_readability_score(self, code: str) -> float:
        """Calculate readability score."""
        lines = code.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        return max(0, 1 - avg_line_length / 100)
    
    def _calculate_maintainability_score(self, code: str, ast_tree: ast.AST) -> float:
        """Calculate maintainability score."""
        functions = len([n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)])
        lines = len(code.split('\n'))
        return max(0, 1 - (lines / 50) + (functions * 0.1))
    
    def _calculate_documentation_score(self, code: str) -> float:
        """Calculate documentation score."""
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        total_lines = len(code.split('\n'))
        return min(1.0, comment_lines / max(total_lines, 1) * 5)
    
    def _calculate_max_nesting_depth(self, ast_tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(node, ast.FunctionDef):
                current_depth = 0
        
        return max_depth
    
    def _assess_complexity_risk(self, code: str) -> float:
        """Assess risk based on code complexity."""
        try:
            ast_tree = ast.parse(code)
            complexity = self._calculate_complexity_score(ast_tree)
            return min(1.0, complexity / 20)
        except:
            return 0.5
    
    def _assess_test_coverage_risk(self, code: str) -> float:
        """Assess risk based on test coverage."""
        # Simplified assessment - in practice, this would check actual test coverage
        return 0.3
    
    def _assess_dependency_risk(self, code: str) -> float:
        """Assess risk based on dependencies."""
        # Simplified assessment
        return 0.2
    
    def _assess_business_risk(self, code: str) -> float:
        """Assess risk based on business criticality."""
        # Simplified assessment
        return 0.4
    
    def _suggest_risk_mitigation(self, intervention_type: str, risk_factors: Dict[str, float]) -> List[str]:
        """Suggest risk mitigation strategies."""
        strategies = []
        
        if risk_factors['code_complexity'] > 0.7:
            strategies.append("Break down complex changes into smaller, incremental updates")
        
        if risk_factors['test_coverage'] > 0.5:
            strategies.append("Increase test coverage before implementing changes")
        
        if risk_factors['dependencies'] > 0.5:
            strategies.append("Review and test dependent systems")
        
        if risk_factors['business_criticality'] > 0.7:
            strategies.append("Implement changes in a staging environment first")
        
        return strategies 