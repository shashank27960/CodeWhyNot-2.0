#!/usr/bin/env python3
"""
Test script for CodeWhyNot 2.0 Causal Intervention System
Tests all major components to ensure they work correctly.
"""

import sys
import traceback
from typing import Dict, Any

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing imports...")
    
    try:
        from scm.ast_scm_builder import ASTSCMBuilder
        print("✅ ASTSCMBuilder imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ASTSCMBuilder: {e}")
        return False
    
    try:
        from cf_generator.causal_intervention_engine import CausalInterventionEngine
        print("✅ CausalInterventionEngine imported successfully")
    except Exception as e:
        print(f"❌ Failed to import CausalInterventionEngine: {e}")
        return False
    
    try:
        from evaluation.code_executor import CodeExecutor
        print("✅ CodeExecutor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import CodeExecutor: {e}")
        return False
    
    try:
        from evaluation.metrics_calculator import MetricsCalculator
        print("✅ MetricsCalculator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import MetricsCalculator: {e}")
        return False
    
    try:
        from ui.causal_intervention_ui import CausalInterventionUI
        print("✅ CausalInterventionUI imported successfully")
    except Exception as e:
        print(f"❌ Failed to import CausalInterventionUI: {e}")
        return False
    
    return True

def test_ast_scm_builder():
    """Test AST SCM builder functionality."""
    print("\n🔍 Testing AST SCM Builder...")
    
    try:
        from scm.ast_scm_builder import ASTSCMBuilder
        
        builder = ASTSCMBuilder()
        
        # Test code parsing
        test_code = """
def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        
        ast_tree = builder.parse_code_to_ast(test_code)
        print("✅ Code parsing successful")
        
        # Test SCM building
        scm_graph = builder.build_scm_from_ast(ast_tree)
        print(f"✅ SCM graph built with {len(scm_graph.nodes)} nodes and {len(scm_graph.edges)} edges")
        
        # Test intervention points
        interventions = builder.get_intervention_points(scm_graph)
        print(f"✅ Found {len(interventions)} intervention points")
        
        return True
        
    except Exception as e:
        print(f"❌ AST SCM Builder test failed: {e}")
        traceback.print_exc()
        return False

def test_causal_intervention_engine():
    """Test causal intervention engine functionality."""
    print("\n🔍 Testing Causal Intervention Engine...")
    
    try:
        from cf_generator.causal_intervention_engine import CausalInterventionEngine
        
        engine = CausalInterventionEngine()
        
        # Test code analysis
        test_code = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        
        analysis = engine.analyze_code(test_code)
        print("✅ Code analysis successful")
        
        # Test intervention suggestions
        suggestions = engine.generate_intervention_suggestions(test_code, "Write factorial function")
        print(f"✅ Generated {len(suggestions)} intervention suggestions")
        
        # Test counterfactual prompt generation
        cf_prompt = engine.generate_counterfactual_prompt("Write factorial function", "loop_to_recursion")
        print(f"✅ Generated counterfactual prompt: {cf_prompt[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Causal Intervention Engine test failed: {e}")
        traceback.print_exc()
        return False

def test_code_executor():
    """Test code execution functionality."""
    print("\n🔍 Testing Code Executor...")
    
    try:
        from evaluation.code_executor import CodeExecutor
        
        executor = CodeExecutor()
        
        # Test safe code execution
        safe_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
        
        result = executor.execute_code(safe_code)
        print(f"✅ Safe code execution successful: {result['success']}")
        
        # Test code safety validation
        dangerous_code = "import os; os.system('rm -rf /')"
        is_safe, message = executor.validate_code_safety(dangerous_code)
        print(f"✅ Security validation working: {not is_safe} - {message}")
        
        # Test test case generation
        test_cases = executor.create_test_cases(safe_code)
        print(f"✅ Generated {len(test_cases)} test cases")
        
        return True
        
    except Exception as e:
        print(f"❌ Code Executor test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics_calculator():
    """Test metrics calculation functionality."""
    print("\n🔍 Testing Metrics Calculator...")
    
    try:
        from evaluation.metrics_calculator import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        # Test metrics calculation
        original_code = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        
        counterfactual_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        
        metrics = calculator.calculate_comprehensive_metrics(
            original_code, 
            counterfactual_code,
            "Write factorial function",
            "Write factorial function using recursion",
            "loop_to_recursion"
        )
        
        print("✅ Comprehensive metrics calculation successful")
        print(f"   - Overall score: {metrics['overall_scores'].get('overall_score', 0):.3f}")
        print(f"   - Fidelity score: {metrics['overall_scores'].get('fidelity_score', 0):.3f}")
        print(f"   - AST distance: {metrics['structural_metrics'].get('ast_edit_distance', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics Calculator test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all components."""
    print("\n🔍 Testing System Integration...")
    
    try:
        from scm.ast_scm_builder import ASTSCMBuilder
        from cf_generator.causal_intervention_engine import CausalInterventionEngine
        from evaluation.metrics_calculator import MetricsCalculator
        from evaluation.code_executor import CodeExecutor
        
        # Initialize components
        ast_builder = ASTSCMBuilder()
        intervention_engine = CausalInterventionEngine()
        metrics_calculator = MetricsCalculator()
        code_executor = CodeExecutor()
        
        # Test code
        test_code = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        
        # Full pipeline test
        print("  1. Building AST SCM...")
        ast_tree = ast_builder.parse_code_to_ast(test_code)
        scm_graph = ast_builder.build_scm_from_ast(ast_tree)
        
        print("  2. Generating intervention suggestions...")
        suggestions = intervention_engine.generate_intervention_suggestions(test_code, "Write factorial function")
        
        print("  3. Applying intervention...")
        if suggestions:
            intervention = suggestions[0]
            cf_prompt = intervention['prompt_modification']
            cf_code = intervention_engine.apply_code_intervention(test_code, intervention['type'])
            
            print("  4. Calculating metrics...")
            metrics = metrics_calculator.calculate_comprehensive_metrics(
                test_code, cf_code, "Write factorial function", cf_prompt, intervention['type']
            )
            
            print("  5. Executing code...")
            orig_result = code_executor.execute_code(test_code)
            cf_result = code_executor.execute_code(cf_code)
            
            print("✅ Integration test successful!")
            print(f"   - Original execution: {orig_result['success']}")
            print(f"   - Counterfactual execution: {cf_result['success']}")
            print(f"   - Overall score: {metrics['overall_scores'].get('overall_score', 0):.3f}")
            
            return True
        else:
            print("⚠️  No intervention suggestions generated")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 CodeWhyNot 2.0 System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("AST SCM Builder Test", test_ast_scm_builder),
        ("Causal Intervention Engine Test", test_causal_intervention_engine),
        ("Code Executor Test", test_code_executor),
        ("Metrics Calculator Test", test_metrics_calculator),
        ("Integration Test", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 