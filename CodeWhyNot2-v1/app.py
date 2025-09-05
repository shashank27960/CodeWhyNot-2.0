import streamlit as st
from scm_engine import SCMEngine
from gumbel_cf import gumbel_counterfactual
from codegen.code_llama import CodeLlamaGenerator
from ast_diff.ast_diff_engine import ASTDiffEngine
from ast_diff.fidelity_scorer import FidelityScorer
from ui.visualization import render_causal_tree, render_causal_change_tree, render_scm_graph_editor
from ui.causal_intervention_ui import CausalInterventionUI
from cf_generator.causal_intervention_engine import CausalInterventionEngine
from cf_generator.educational_explainer import EducationalExplainer
from cf_generator.developer_tooling import DeveloperTooling
from cf_generator.model_debugger import ModelDebugger
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.code_executor import CodeExecutor, execute_with_coverage_and_analysis
from scm.ast_scm_builder import ASTSCMBuilder
import networkx as nx
import ast
import astpretty
from cf_generator.ml_counterfactual_ollama import OllamaCounterfactualPromptGenerator
from cf_generator.concept_extractor_ml import extract_concepts_ml
from cf_generator.intervention_suggester_ml import suggest_interventions_ml
from cf_generator.gumbel_cf import GumbelCounterfactualGenerator
import matplotlib.pyplot as plt

st.set_page_config(page_title="CodeWhyNot 2.0", layout="wide")
st.title("CodeWhyNot 2.0: Causal Intervention System for Explainable Code Generation")

# Initialize enhanced components
causal_ui = CausalInterventionUI()
causal_engine = CausalInterventionEngine()
educational_explainer = EducationalExplainer()
developer_tooling = DeveloperTooling()
model_debugger = ModelDebugger()
metrics_calculator = MetricsCalculator()
code_executor = CodeExecutor()
ast_scm_builder = ASTSCMBuilder()

# AST tree visualization utility

def ast_to_nx(ast_node, G=None, parent=None):
    if G is None:
        G = nx.DiGraph()
    node_id = id(ast_node)
    G.add_node(node_id, label=type(ast_node).__name__)
    if parent is not None:
        G.add_edge(parent, node_id)
    for child in ast.iter_child_nodes(ast_node):
        ast_to_nx(child, G, node_id)
    return G

def render_ast_tree(ast_node, title="AST Tree", highlight_nodes=None):
    G = ast_to_nx(ast_node)
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    node_colors = []
    for n in G.nodes:
        if highlight_nodes and n in highlight_nodes:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')
    plt.figure(figsize=(8, 4))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, font_size=8, node_color=node_colors)
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.close()

# Sidebar: Inputs
with st.sidebar:
    st.header("🔧 Causal Intervention Setup")
    
    # Input mode selection
    input_mode = st.radio("Input Mode", ["Text Prompt", "Code + Prompt", "Code Analysis"], key="input_mode")
    
    if input_mode == "Text Prompt":
        prompt = st.text_area("Enter your code prompt", "Write factorial using a loop")
        user_code = None
    elif input_mode == "Code + Prompt":
        prompt = st.text_area("Enter your code prompt", "Write factorial using a loop")
        user_code = st.text_area("Enter your code (optional)", "", height=150)
    else:  # Code Analysis
        prompt = st.text_area("Describe what the code should do", "Calculate factorial")
        user_code = st.text_area("Enter your code for analysis", "", height=200)
    
    # Enhanced SCM graph construction
    if st.button("🔍 Analyze & Build Causal Graph"):
        if user_code:
            # Build AST-based SCM
            try:
                ast_tree = ast_scm_builder.parse_code_to_ast(user_code)
                scm_graph = ast_scm_builder.build_scm_from_ast(ast_tree)
                st.session_state['ast_scm_graph'] = scm_graph
                
                # Get intervention suggestions
                suggestions = causal_engine.generate_intervention_suggestions(user_code, prompt)
                st.session_state['intervention_suggestions'] = suggestions
                
                st.success(f"✅ Built AST-based SCM with {len(scm_graph.nodes)} nodes and {len(scm_graph.edges)} edges")
            except Exception as e:
                st.error(f"❌ Error building AST SCM: {e}")
        else:
            # Fallback to concept-based SCM
            ml_concepts_raw = extract_concepts_ml(prompt)
            # Parse the LLM output string into a list of concepts
            def parse_concepts(concepts_raw):
                import re
                lines = re.split(r'[\n,•\-*]+', concepts_raw)
                return [line.strip() for line in lines if line.strip()]
            ml_concepts = parse_concepts(ml_concepts_raw)
            st.session_state['ml_concepts'] = ml_concepts
            G = nx.DiGraph()
            for concept in ml_concepts:
                G.add_node(concept, label=concept)
            st.session_state['scm_graph'] = G
            st.success(f"✅ Built concept-based SCM with {len(ml_concepts)} concepts")
    
    # Display extracted concepts
    ml_concepts = st.session_state.get('ml_concepts', [])
    if ml_concepts:
        st.markdown("**📋 Extracted Concepts:**")
        st.write(ml_concepts)
    # Interactive SCM graph editor
    scm_graph = st.session_state.get('scm_graph', None)
    if scm_graph is not None:
        st.subheader("SCM Graph Editor (Interactive)")
        # Node addition
        new_node = st.text_input("Add Node (Concept)", "", key="add_node_input")
        if st.button("Add Node", key="add_node_btn") and new_node:
            scm_graph.add_node(new_node, label=new_node)
            st.session_state['scm_graph'] = scm_graph
        # Node removal
        node_to_remove = st.selectbox("Remove Node", list(scm_graph.nodes) if scm_graph.nodes else [""], key="remove_node_select")
        if st.button("Remove Node", key="remove_node_btn") and node_to_remove:
            scm_graph.remove_node(node_to_remove)
            st.session_state['scm_graph'] = scm_graph
        # Edge addition
        nodes = list(scm_graph.nodes)
        src = st.selectbox("Edge Source", nodes if nodes else [""], key="edge_src_select")
        tgt = st.selectbox("Edge Target", nodes if nodes else [""], key="edge_tgt_select")
        if st.button("Add Edge", key="add_edge_btn") and src and tgt and src != tgt:
            scm_graph.add_edge(src, tgt)
            st.session_state['scm_graph'] = scm_graph
        # Edge removal
        edges = list(scm_graph.edges)
        edge_to_remove = st.selectbox("Remove Edge", edges if edges else [("", "")], key="remove_edge_select")
        if (
            st.button("Remove Edge", key="remove_edge_btn")
            and edge_to_remove
            and edge_to_remove in scm_graph.edges
            and all(edge_to_remove)
        ):
            scm_graph.remove_edge(*edge_to_remove)
            st.session_state['scm_graph'] = scm_graph
        # Node renaming
        node_to_rename = st.selectbox("Rename Node", nodes if nodes else [""], key="rename_node_select")
        new_name = st.text_input("New Name", "", key="rename_node_input")
        if (
            st.button("Rename Node", key="rename_node_btn")
            and node_to_rename
            and node_to_rename in scm_graph.nodes
            and new_name
        ):
            nx.relabel_nodes(scm_graph, {node_to_rename: new_name}, copy=False)
            st.session_state['scm_graph'] = scm_graph
        # Show the graph
        # render_scm_graph_editor(scm_graph)  # Comment out to avoid duplicate element error
    # Enhanced intervention selection
    st.subheader("🎯 Intervention Selection")
    
    # Check for intelligent suggestions
    intervention_suggestions = st.session_state.get('intervention_suggestions', [])
    
    if intervention_suggestions:
        st.markdown("**💡 Intelligent Intervention Suggestions:**")
        selected_suggestion = st.selectbox(
            "Choose an intervention",
            options=intervention_suggestions,
            format_func=lambda x: f"{x['type']} ({x['confidence']:.2f})",
            key="suggestion_select"
        )
        
        if selected_suggestion:
            st.info(f"**Description:** {selected_suggestion['description']}")
            st.markdown(f"**Reasoning:** {selected_suggestion['reasoning']}")
            cf_prompt = selected_suggestion['prompt_modification']
            intervention_type = selected_suggestion['type']
            concept = "AST-based"
    else:
        # Fallback to original modes
        cf_mode = st.radio("Counterfactual Generation Mode", ["Manual SCM", "ML-based (LLM)", "Gumbel-based"], key="cf_mode_radio")
        scm = SCMEngine()
        cf_prompt = None
        concept = None
        intervention_type = None
        seed = st.session_state.get('seed', 42)  # Ensure seed is defined
        ast_diff_engine = globals().get('ast_diff_engine', ASTDiffEngine())  # Ensure ast_diff_engine is defined
        if cf_mode == "Manual SCM":
            concept = st.selectbox("Concept to intervene", scm.get_concepts())
            # Contextual suggestions for interventions
            context_interventions = {
                'loop': ['Convert to recursion', 'Optimize', 'Add error handling'],
                'recursion': ['Convert to loop', 'Optimize', 'Add memoization'],
                'input': ['Add input validation', 'Add error handling'],
                'function': ['Refactor', 'Add type hints', 'Add documentation'],
                'variable': ['Rename', 'Change type', 'Add initialization'],
            }
            suggestions = context_interventions.get(concept.lower(), ['Refactor', 'Optimize', 'Add error handling'])
            intervention = st.selectbox("Intervention", suggestions)
            templates = {
                'Convert to recursion': 'Rewrite the loop as a recursive function.',
                'Optimize': 'Improve the performance of this code segment.',
                'Add error handling': 'Add try/except blocks to handle possible errors.',
                'Add input validation': 'Check user input for validity before processing.',
                'Refactor': 'Restructure the code for better readability and maintainability.',
                'Add type hints': 'Add type annotations to function signatures.',
                'Add documentation': 'Add docstrings and comments to explain the code.',
                'Rename': 'Rename the variable/function for clarity.',
                'Change type': 'Change the variable type as needed.',
                'Add initialization': 'Initialize the variable before use.',
                'Add memoization': 'Cache results of recursive calls to improve efficiency.'
            }
            st.markdown(f"**Template/Example:** {templates.get(intervention, 'No template available.')}")
            st.info(f"Selected node: {concept} will be affected by: {intervention}")
            if 'intervention_history' not in st.session_state:
                st.session_state['intervention_history'] = []
            if st.button("Apply Intervention"):
                st.session_state['intervention_history'].append((concept, intervention))
                st.success(f"Applied intervention: {intervention} on {concept}")
                # --- New: Run codegen, AST diff, metrics, and explanations ---
                cf_prompt = scm.generate_counterfactual_prompt(prompt, concept, intervention)
                codegen_func = lambda p: codegen.generate_code(p)
                orig_code, cf_code = gumbel_counterfactual(prompt, cf_prompt, codegen_func, seed=seed)
                # AST diff
                ast_diff = ast_diff_engine.compute_ast_diff(orig_code, cf_code)
                # Metrics
                test_cases = code_executor.create_test_cases(orig_code)
                orig_execution = code_executor.execute_code(orig_code, test_cases=test_cases)
                cf_execution = code_executor.execute_code(cf_code, test_cases=test_cases)
                metrics = metrics_calculator.calculate_comprehensive_metrics(
                    orig_code, cf_code, prompt, cf_prompt, intervention
                )
                # Log intervention and link to code/AST changes
                if 'intervention_log' not in st.session_state:
                    st.session_state['intervention_log'] = []
                st.session_state['intervention_log'].append({
                    'concept': concept,
                    'intervention': intervention,
                    'cf_prompt': cf_prompt,
                    'orig_code': orig_code,
                    'cf_code': cf_code,
                    'ast_diff': ast_diff,
                    'metrics': metrics
                })
                # UI: Visualize AST diff and show metrics/explanations
                st.subheader("AST Difference Visualization")
                st.write(ast_diff)
                st.subheader("Metrics after Intervention")
                st.write(metrics['overall_scores'])
                st.write(metrics['structural_metrics'])
                st.write(metrics['functional_metrics'])
                st.write(metrics['causal_metrics'])
                st.subheader("Educational Feedback")
                st.info(f"Intervention '{intervention}' on '{concept}' led to the following code/AST changes:")
                st.code(cf_code, language='python')
                st.markdown(f"**Explanation:** {templates.get(intervention, 'No template available.')}")
                if metrics['overall_scores'].get('fidelity_score', 0) > 0.8:
                    st.success("The counterfactual code maintains high fidelity to the original intent and passes all tests.")
                else:
                    st.warning("The counterfactual code differs significantly or fails some tests. Review the intervention.")
            if st.button("Undo Last Intervention") and st.session_state['intervention_history']:
                last = st.session_state['intervention_history'].pop()
                st.warning(f"Undid intervention: {last[1]} on {last[0]}")
            st.markdown("**Intervention History:**")
            for idx, (c, i) in enumerate(st.session_state['intervention_history'], 1):
                st.write(f"{idx}. {i} on {c}")
            cf_prompt = scm.generate_counterfactual_prompt(prompt, concept, intervention)
            intervention_type = intervention
        elif cf_mode == "ML-based (LLM)":
            n_cf = st.slider("Number of counterfactuals", 1, 5, 3, key="ml_n_cf_slider")
            if st.button("Suggest Counterfactual Prompts", key="ml_suggest_btn"):
                cf_gen = OllamaCounterfactualPromptGenerator()
                cf_prompts = cf_gen.generate_counterfactuals(prompt, n=n_cf)
                st.session_state['cf_prompts'] = cf_prompts
            cf_prompts = st.session_state.get('cf_prompts', [])
            if cf_prompts:
                cf_prompt = st.selectbox("Select Counterfactual Prompt", cf_prompts, key="ml_cf_prompt_select")
            else:
                cf_prompt = None
            concept = "ML-based"
            intervention_type = "LLM-generated"
        elif cf_mode == "Gumbel-based":
            n_gumbel = st.slider("Number of Gumbel counterfactuals", 1, 5, 3, key="gumbel_n_cf_slider")
            if st.button("Generate Gumbel Counterfactuals", key="gumbel_generate_btn"):
                gumbel_gen = GumbelCounterfactualGenerator()
                gumbel_prompts = []
                scm_graph = st.session_state.get('scm_graph', None)
                if scm_graph is not None:
                    for _ in range(n_gumbel):
                        gumbel_prompts.append(gumbel_gen.generate(scm_graph, use_gumbel=True))
                st.session_state['gumbel_prompts'] = gumbel_prompts
            gumbel_prompts = st.session_state.get('gumbel_prompts', [])
            if gumbel_prompts:
                cf_prompt = st.selectbox("Select Gumbel Counterfactual Prompt", gumbel_prompts, key="gumbel_cf_prompt_select")
            else:
                cf_prompt = None
            concept = "Gumbel-based"
            intervention_type = "Gumbel-generated"
    seed = st.number_input("Random Seed", min_value=0, max_value=2**32-1, value=42)
    backend = st.selectbox("Code Generation Backend", ["Ollama (CodeLlama)", "HuggingFace Transformers"])
    quantized = False
    if backend == "HuggingFace Transformers":
        quantized = st.checkbox("Use 4-bit Quantized Model (bitsandbytes)", value=False, help="Requires supported model and bitsandbytes installed.")
    generate = st.button("Generate Counterfactual")

# Backend setup
backend_key = 'ollama' if backend.startswith('Ollama') else 'huggingface'
codegen = CodeLlamaGenerator(backend=backend_key, quantized=quantized)
ast_diff_engine = ASTDiffEngine()

# Main logic
if generate and prompt and cf_prompt:
    try:
        # Clear any previous test case state to avoid stale test cases
        st.session_state.pop('cf_prompts', None)
        st.session_state.pop('gumbel_prompts', None)
        st.session_state.pop('intervention_data', None)

        # Enhanced code generation with comprehensive analysis
        def codegen_func(p):
            return codegen.generate_code(p)
        
        # Generate original and counterfactual code
        orig_code, cf_code = gumbel_counterfactual(prompt, cf_prompt, codegen_func, seed=seed)
        
        # Comprehensive metrics calculation
        comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(
            orig_code, cf_code, prompt, cf_prompt, intervention_type or "manual"
        )
        
        # --- Define test_cases for white-box testing using LLM ---
        import ast
        orig_valid = True
        cf_valid = True
        try:
            ast.parse(orig_code)
        except Exception as e:
            orig_valid = False
            orig_parse_error = str(e)
        try:
            ast.parse(cf_code)
        except Exception as e:
            cf_valid = False
            cf_parse_error = str(e)

        orig_test_cases = code_executor.create_test_cases(orig_code, use_llm=True) if orig_valid else []
        cf_test_cases = code_executor.create_test_cases(cf_code, use_llm=True) if cf_valid else []

        # --- Execute code for functional testing (needed for get_execution_metrics) ---
        orig_execution = code_executor.execute_code(orig_code, test_cases=orig_test_cases) if orig_valid else []
        cf_execution = code_executor.execute_code(cf_code, test_cases=cf_test_cases) if cf_valid else []
        # --- White-box testing: coverage and static analysis ---
        orig_whitebox = execute_with_coverage_and_analysis(orig_code, orig_test_cases) if orig_valid else {'test_results': [], 'coverage_percent': 0, 'complexity': 0, 'static_warnings': [orig_parse_error]}
        cf_whitebox = execute_with_coverage_and_analysis(cf_code, cf_test_cases) if cf_valid else {'test_results': [], 'coverage_percent': 0, 'complexity': 0, 'static_warnings': [cf_parse_error]}
        
        # Build AST-based SCM if user code provided
        ast_scm_graph = None
        if user_code:
            try:
                ast_tree = ast_scm_builder.parse_code_to_ast(user_code)
                ast_scm_graph = ast_scm_builder.build_scm_from_ast(ast_tree)
            except:
                pass
        
        # Enhanced analysis with new components
        # Educational analysis
        educational_explanations = educational_explainer.explain_code_generation(
            prompt, orig_code, intervention_type or "manual"
        )
        learning_objectives = educational_explainer.generate_learning_objectives(orig_code, intervention_type or "manual")
        
        # Developer tooling analysis
        code_analysis = developer_tooling.analyze_code_for_developers(orig_code, prompt)
        ranked_interventions = developer_tooling.generate_ranked_interventions(orig_code, code_analysis, prompt)
        
        # Model debugging analysis
        prompt_analysis = model_debugger.analyze_prompt_engineering(
            prompt, cf_prompt, orig_code, cf_code
        )
        model_behavior = model_debugger.analyze_model_behavior([prompt, cf_prompt], [orig_code, cf_code])
        causal_influence = model_debugger.analyze_causal_influence(
            prompt, cf_prompt, orig_code, cf_code, intervention_type or "manual"
        )
        
        # Prepare intervention data for UI
        intervention_data = {
            'original_code': orig_code,
            'counterfactual_code': cf_code,
            'original_prompt': prompt,
            'counterfactual_prompt': cf_prompt,
            'intervention_type': intervention_type,
            'intervention_description': selected_suggestion.get('description', 'Manual intervention') if 'selected_suggestion' in locals() and selected_suggestion else 'Manual intervention',
            'fidelity': comprehensive_metrics['overall_scores'].get('fidelity_score', 0),
            'ast_distance': comprehensive_metrics['structural_metrics'].get('ast_edit_distance', 0),
            'pass_rate': comprehensive_metrics['functional_metrics'].get('counterfactual_pass_rate', 0),
            'overall_scores': comprehensive_metrics['overall_scores'],
            'detailed_metrics': comprehensive_metrics,
            'ast_analysis': {
                'original_node_counts': comprehensive_metrics['structural_metrics'].get('original_node_counts', {}),
                'counterfactual_node_counts': comprehensive_metrics['structural_metrics'].get('counterfactual_node_counts', {})
            },
            'complexity_analysis': {
                'original': comprehensive_metrics['semantic_metrics'].get('original_complexity', {}),
                'counterfactual': comprehensive_metrics['semantic_metrics'].get('counterfactual_complexity', {})
            },
            'style_analysis': {
                'original': comprehensive_metrics['semantic_metrics'].get('original_style', {}),
                'counterfactual': comprehensive_metrics['semantic_metrics'].get('counterfactual_style', {})
            },
            'causal_metrics': comprehensive_metrics['causal_metrics'],
            'scm_graph': ast_scm_graph,
            'intervention_suggestions': intervention_suggestions,
            'execution_results': {
                'original': code_executor.get_execution_metrics(orig_execution),
                'counterfactual': code_executor.get_execution_metrics(cf_execution)
            },
            'whitebox_testing': {
                'original': orig_whitebox,
                'counterfactual': cf_whitebox
            },
            # New educational features
            'educational_explanations': educational_explanations,
            'learning_objectives': learning_objectives,
            # New developer tooling features
            'code_analysis': code_analysis,
            'ranked_interventions': ranked_interventions,
            # New model debugging features
            'prompt_analysis': prompt_analysis,
            'model_behavior': model_behavior,
            'causal_influence': causal_influence
        }
        
        # Store results in session state
        st.session_state['intervention_data'] = intervention_data
        
        st.success("✅ Causal intervention analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        st.exception(e)

# Display results using enhanced UI
if 'intervention_data' in st.session_state:
    data = st.session_state['intervention_data']
    # --- Code Preview Tabs ---
    st.subheader("📝 Code Preview")
    tab1, tab2 = st.tabs(["Original Code", "Counterfactual Code"])
    with tab1:
        st.code(data['original_code'], language='python')
    with tab2:
        st.code(data['counterfactual_code'], language='python')
    # --- Fidelity Score Progress Bar ---
    st.subheader("Fidelity Score")
    fidelity = data.get('fidelity', 0)
    st.progress(fidelity if fidelity <= 1 else 1.0, text=f"Fidelity Score: {fidelity:.2f}")
    # --- Split-view AST Diff ---
    st.subheader("AST Diff (Side-by-Side)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original AST**")
        try:
            orig_ast = ast.parse(data['original_code'])
            render_ast_tree(orig_ast, title="Original AST")
        except:
            st.write("(Invalid or empty code)")
    with col2:
        st.markdown("**Counterfactual AST**")
        try:
            cf_ast = ast.parse(data['counterfactual_code'])
            render_ast_tree(cf_ast, title="Counterfactual AST")
        except:
            st.write("(Invalid or empty code)")

    # --- Causal Relationship Graph (Improved) ---
    st.subheader("🌳 Causal Relationship Graph")
    ast_scm_graph = data.get('scm_graph', None)
    ml_concepts = data.get('ml_concepts', [])
    import networkx as nx
    import matplotlib.pyplot as plt
    def draw_graph(G, title="Causal Graph"):
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        if not labels:
            labels = {n: n for n in G.nodes}
        plt.figure(figsize=(8, 4))
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=700, node_color='lightgreen', font_size=10, edge_color='gray')
        plt.title(title)
        st.pyplot(plt.gcf())
        plt.close()
    if ast_scm_graph is not None and len(ast_scm_graph.nodes) > 0:
        st.info(f"Showing AST-based causal graph with {len(ast_scm_graph.nodes)} nodes and {len(ast_scm_graph.edges)} edges.")
        draw_graph(ast_scm_graph, title="AST-based Causal Graph")
    elif ml_concepts:
        st.info(f"No AST-based graph, but showing concept graph with {len(ml_concepts)} concepts.")
        G = nx.DiGraph()
        for concept in ml_concepts:
            G.add_node(concept, label=concept)
        draw_graph(G, title="Concept Graph (No Edges)")
    else:
        st.warning("No causal graph data available. Provide user code to generate AST-based causal relationships.")
    # --- Available Interventions Section (Improved) ---
    st.subheader("🎯 Available Interventions")
    intervention_suggestions = data.get('intervention_suggestions', [])
    if intervention_suggestions:
        # Group by target concept/node if possible
        from collections import defaultdict
        grouped = defaultdict(list)
        for suggestion in intervention_suggestions:
            target = suggestion.get('target', 'General')
            grouped[target].append(suggestion)
        # Summary
        st.markdown(f"**Summary:** {sum(len(v) for v in grouped.values())} interventions available for {len(grouped)} concept(s)/node(s)")
        for target, suggestions in grouped.items():
            st.markdown(f"---\n**Target:** `{target}` ({len(suggestions)} intervention{'s' if len(suggestions) > 1 else ''})")
            for idx, s in enumerate(suggestions, 1):
                st.markdown(f"**{idx}. Type:** `{s.get('type', 'N/A')}` | **Confidence:** {s.get('confidence', 0):.2f}")
                st.markdown(f"- **Description:** {s.get('description', 'N/A')}")
                st.markdown(f"- **Reasoning:** {s.get('reasoning', 'N/A')}")
                st.markdown(f"- **Prompt Modification:** `{s.get('prompt_modification', 'N/A')}`")
                st.markdown("---")
    else:
        st.info("No intelligent interventions found for this code. Try manual or ML-based intervention modes.")
        # Optionally, show generic interventions
        generic = [
            {"type": "Refactor", "description": "Restructure the code for better readability and maintainability."},
            {"type": "Optimize", "description": "Improve the performance of this code segment."},
            {"type": "Add Error Handling", "description": "Add try/except blocks to handle possible errors."},
            {"type": "Add Input Validation", "description": "Check user input for validity before processing."},
        ]
        st.markdown("**Generic Interventions:**")
        for g in generic:
            st.markdown(f"- **Type:** `{g['type']}` | **Description:** {g['description']}")
    # --- Existing dashboard and metrics ---
    causal_ui.render_intervention_dashboard(data)
    # --- White-Box Testing Section ---
    st.subheader("🧪 White-Box Testing")
    
    # --- Test Case Generation Info ---
    st.markdown("**🔧 Test Case Generation Information**")
    test_cases_info = data.get('whitebox_testing', {}).get('original', {}).get('test_results', [])
    if test_cases_info:
        # Count test case sources
        sources = {}
        for tc in test_cases_info:
            if hasattr(tc, 'get') and isinstance(tc, dict):
                source = tc.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
        
        if sources:
            st.info(f"**Test Cases Generated:** {sum(sources.values())} total")
            for source, count in sources.items():
                st.write(f"• {source}: {count} test cases")
    
    # --- Testability badge for counterfactual code ---
    cf_tests = data['whitebox_testing']['counterfactual']['test_results']
    orig_tests = data['whitebox_testing']['original']['test_results']
    # Show parse errors if code is invalid
    if not orig_tests and 'static_warnings' in data['whitebox_testing']['original']:
        for w in data['whitebox_testing']['original']['static_warnings']:
            st.error(f"Original code parse error: {w}")
    if not cf_tests and 'static_warnings' in data['whitebox_testing']['counterfactual']:
        for w in data['whitebox_testing']['counterfactual']['static_warnings']:
            st.error(f"Counterfactual code parse error: {w}")
    # Show warning if test cases are empty
    if not orig_tests:
        st.warning("No test cases generated for original code.")
    if not cf_tests:
        st.warning("No test cases generated for counterfactual code.")
    # Show warning if test cases differ
    if len(orig_tests) != len(cf_tests) or any(o.get('test') != c.get('test') for o, c in zip(orig_tests, cf_tests)):
        st.warning("Test cases for original and counterfactual code differ. Results may not be directly comparable.")
    cf_passed = [t for t in cf_tests if t['passed']]
    orig_passed = [t for t in orig_tests if t['passed']]
    if all(t['passed'] for t in cf_tests) and cf_tests:
        st.success("Counterfactual code is TESTABLE: All tests passed ✅")
    elif any(t['passed'] for t in cf_tests):
        st.warning("Counterfactual code is PARTIALLY TESTABLE: Some tests passed ⚠️")
    else:
        st.error("Counterfactual code is NOT TESTABLE: All tests failed ❌")
    # --- Test cases table with source and details ---
    import pandas as pd
    test_rows = []
    for i, (orig, cf) in enumerate(zip(orig_tests, cf_tests)):
        # Extract test case information
        orig_test_info = orig.get('test', '')
        cf_test_info = cf.get('test', '')
        
        # Try to extract test case type and description if available
        orig_type = "Unknown"
        cf_type = "Unknown"
        orig_desc = ""
        cf_desc = ""
        
        # Look for test case metadata in the test string or error
        if hasattr(orig, 'get') and isinstance(orig, dict):
            orig_type = orig.get('type', 'Unknown')
            orig_desc = orig.get('description', '')
        if hasattr(cf, 'get') and isinstance(cf, dict):
            cf_type = cf.get('type', 'Unknown')
            cf_desc = cf.get('description', '')
        
        test_rows.append({
            "Test Case": f"{orig_test_info[:50]}{'...' if len(orig_test_info) > 50 else ''}",
            "Type": orig_type,
            "Description": orig_desc[:30] + "..." if len(orig_desc) > 30 else orig_desc,
            "Original": "✅" if orig['passed'] else "❌",
            "Counterfactual": "✅" if cf['passed'] else "❌",
            "Original Error": orig.get('error', '')[:50] + "..." if len(orig.get('error', '')) > 50 else orig.get('error', ''),
            "Counterfactual Error": cf.get('error', '')[:50] + "..." if len(cf.get('error', '')) > 50 else cf.get('error', ''),
        })
    df = pd.DataFrame(test_rows)
    st.dataframe(df, hide_index=True)
    
    # --- Test case statistics ---
    st.markdown("**📊 Test Case Statistics**")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_tests = len(orig_tests)
        st.metric("Total Test Cases", total_tests)
    with col2:
        orig_pass_rate = (len([t for t in orig_tests if t['passed']]) / total_tests * 100) if total_tests > 0 else 0
        st.metric("Original Pass Rate", f"{orig_pass_rate:.1f}%")
    with col3:
        cf_pass_rate = (len([t for t in cf_tests if t['passed']]) / total_tests * 100) if total_tests > 0 else 0
        st.metric("Counterfactual Pass Rate", f"{cf_pass_rate:.1f}%")
    
    # --- Test case type breakdown ---
    st.markdown("**🔍 Test Case Type Breakdown**")
    test_types = {}
    for tc in orig_tests:
        if hasattr(tc, 'get') and isinstance(tc, dict):
            tc_type = tc.get('type', 'Unknown')
            test_types[tc_type] = test_types.get(tc_type, 0) + 1
    
    if test_types:
        type_data = pd.DataFrame([
            {"Type": tc_type, "Count": count} 
            for tc_type, count in test_types.items()
        ])
        st.bar_chart(type_data.set_index("Type"))
    
    # --- Expanders for detailed test results ---
    st.markdown("**📋 Detailed Test Results**")
    for i, (orig, cf) in enumerate(zip(orig_tests, cf_tests)):
        # Get test case metadata
        orig_type = orig.get('type', 'Unknown') if hasattr(orig, 'get') and isinstance(orig, dict) else 'Unknown'
        cf_type = cf.get('type', 'Unknown') if hasattr(cf, 'get') and isinstance(cf, dict) else 'Unknown'
        orig_desc = orig.get('description', '') if hasattr(orig, 'get') and isinstance(orig, dict) else ''
        cf_desc = cf.get('description', '') if hasattr(cf, 'get') and isinstance(cf, dict) else ''
        
        with st.expander(f"Test {i+1}: {'✅' if cf['passed'] else '❌'} (Counterfactual) - {cf_type}"):
            st.code(cf['test'])
            if cf_desc:
                st.info(f"**Description:** {cf_desc}")
            if not cf['passed']:
                st.error(cf.get('error', ''))
        
        with st.expander(f"Test {i+1}: {'✅' if orig['passed'] else '❌'} (Original) - {orig_type}"):
            st.code(orig['test'])
            if orig_desc:
                st.info(f"**Description:** {orig_desc}")
            if not orig['passed']:
                st.error(orig.get('error', ''))
    # --- Coverage and complexity ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Code**")
        st.progress(data['whitebox_testing']['original']['coverage_percent'] / 100, text=f"Coverage: {data['whitebox_testing']['original']['coverage_percent']:.1f}%")
        st.write(f"Cyclomatic Complexity: {data['whitebox_testing']['original']['complexity']}")
        if data['whitebox_testing']['original']['static_warnings']:
            st.warning("\n".join(data['whitebox_testing']['original']['static_warnings']))
    with col2:
        st.markdown("**Counterfactual Code**")
        st.progress(data['whitebox_testing']['counterfactual']['coverage_percent'] / 100, text=f"Coverage: {data['whitebox_testing']['counterfactual']['coverage_percent']:.1f}%")
        st.write(f"Cyclomatic Complexity: {data['whitebox_testing']['counterfactual']['complexity']}")
        if data['whitebox_testing']['counterfactual']['static_warnings']:
            st.warning("\n".join(data['whitebox_testing']['counterfactual']['static_warnings']))
    st.subheader("⚡ Execution Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Code Execution**")
        causal_ui.render_execution_results(data['execution_results']['original'])
    with col2:
        st.markdown("**Counterfactual Code Execution**")
        causal_ui.render_execution_results(data['execution_results']['counterfactual'])
    if data.get('intervention_suggestions'):
        st.subheader("💡 Additional Intervention Suggestions")
        causal_ui.render_intervention_suggestions(data['intervention_suggestions'])

else:
    st.info("🚀 **Welcome to CodeWhyNot 2.0: Causal Intervention System**")
    
    st.markdown("""
    ### How to use this system:
    
    1. **📝 Input Mode**: Choose how you want to provide input:
       - **Text Prompt**: Just describe what you want to code
       - **Code + Prompt**: Provide both code and description
       - **Code Analysis**: Analyze existing code for improvements
    
    2. **🔍 Analysis**: Click "Analyze & Build Causal Graph" to:
       - Parse your code into AST (Abstract Syntax Tree)
       - Build causal relationships between code elements
       - Generate intelligent intervention suggestions
    
    3. **🎯 Intervention**: Choose from suggested interventions or use manual modes:
       - **AST-based**: Intelligent suggestions based on code structure
       - **ML-based**: LLM-generated alternatives
       - **Manual SCM**: Traditional concept-based interventions
    
    4. **⚡ Generate**: Click "Generate Counterfactual" to:
       - Generate original and counterfactual code
       - Calculate comprehensive metrics
       - Execute and test both versions
       - Provide detailed analysis and visualization
    
    ### Key Features:
    - **🔬 AST-level Analysis**: Understand code structure at the syntax tree level
    - **🎯 Intelligent Interventions**: AI-powered suggestions for code improvements
    - **📊 Comprehensive Metrics**: Structural, functional, semantic, and causal analysis
    - **⚡ Safe Execution**: Sandboxed code execution with functional testing
    - **🌳 Causal Visualization**: Interactive graphs showing code relationships
    - **📈 Quality Assessment**: Readability, maintainability, and efficiency metrics
    """)

# SCM Graph Editor UI (in main page or sidebar) - only show when no results are displayed
if 'orig_code' not in st.session_state:
    st.subheader("SCM Graph Editor (Experimental)")
    scm_graph = st.session_state.get('scm_graph', None)
    if scm_graph is None:
        scm_graph = SCMEngine().graph.copy()
        st.session_state['scm_graph'] = scm_graph
    agraph_result = render_scm_graph_editor(scm_graph)
    
    # Process agraph_result to update the NetworkX graph (full CRUD)
    if agraph_result is not None and isinstance(agraph_result, dict):
        nodes = agraph_result.get('nodes', [])
        edges = agraph_result.get('edges', [])
        if isinstance(nodes, list) and isinstance(edges, list):
            new_graph = nx.DiGraph()
            for node in nodes:
                if isinstance(node, dict):
                    node_id = node.get('id')
                    label = node.get('label', node_id)
                    new_graph.add_node(node_id, label=label)
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source')
                    target = edge.get('target')
                    new_graph.add_edge(source, target)
            st.session_state['scm_graph'] = new_graph 