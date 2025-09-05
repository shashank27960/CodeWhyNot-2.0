import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config

class CausalInterventionUI:
    """
    Enhanced UI components for the causal intervention system.
    Provides interactive visualizations and analysis tools.
    """
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def render_intervention_dashboard(self, intervention_data: Dict[str, Any]):
        """Render the main intervention dashboard."""
        st.header("🔬 Causal Intervention Analysis Dashboard")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Overview", "🔍 Code Analysis", "📈 Metrics", "🌳 Causal Graph", "🎯 Interventions", 
            "🎓 Educational", "🛠️ Developer Tools", "🔬 Model Debug"
        ])
        
        with tab1:
            self._render_overview_tab(intervention_data)
        
        with tab2:
            self._render_code_analysis_tab(intervention_data)
        
        with tab3:
            self._render_metrics_tab(intervention_data)
        
        with tab4:
            self._render_causal_graph_tab(intervention_data)
        
        with tab5:
            self._render_interventions_tab(intervention_data)
        
        with tab6:
            self._render_educational_tab(intervention_data)
        
        with tab7:
            self._render_developer_tools_tab(intervention_data)
        
        with tab8:
            self._render_model_debug_tab(intervention_data)
    
    def _render_overview_tab(self, data: Dict[str, Any]):
        """Render the overview tab with key metrics."""
        st.subheader("📊 Intervention Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._metric_card(
                "Fidelity Score",
                f"{data.get('fidelity', 0):.2f}",
                "Code similarity after intervention"
            )
        
        with col2:
            self._metric_card(
                "AST Distance",
                f"{data.get('ast_distance', 0):.2f}",
                "Structural difference"
            )
        
        with col3:
            self._metric_card(
                "Pass Rate",
                f"{data.get('pass_rate', 0):.1%}",
                "Functional correctness"
            )
        
        with col4:
            self._metric_card(
                "Intervention Type",
                data.get('intervention_type', 'Unknown'),
                "Applied intervention"
            )
        
        # Intervention summary
        st.subheader("🎯 Intervention Summary")
        if 'intervention_description' in data:
            st.info(f"**Applied Intervention:** {data['intervention_description']}")
        
        # Code comparison
        st.subheader("📝 Code Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Code**")
            st.code(data.get('original_code', ''), language='python')
        
        with col2:
            st.markdown("**Counterfactual Code**")
            st.code(data.get('counterfactual_code', ''), language='python')
    
    def _render_code_analysis_tab(self, data: Dict[str, Any]):
        """Render the code analysis tab."""
        st.subheader("🔍 Code Structure Analysis")
        
        # AST visualization
        if 'ast_analysis' in data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original AST Structure**")
                self._render_ast_heatmap(data['ast_analysis'].get('original_node_counts', {}))
            
            with col2:
                st.markdown("**Counterfactual AST Structure**")
                self._render_ast_heatmap(data['ast_analysis'].get('counterfactual_node_counts', {}))
        
        # Complexity analysis
        if 'complexity_analysis' in data:
            st.subheader("📊 Complexity Analysis")
            self._render_complexity_comparison(data['complexity_analysis'])
        
        # Code style analysis
        if 'style_analysis' in data:
            st.subheader("🎨 Code Style Analysis")
            self._render_style_comparison(data['style_analysis'])
    
    def _render_metrics_tab(self, data: Dict[str, Any]):
        """Render the metrics tab."""
        st.subheader("📈 Comprehensive Metrics")
        
        # Overall scores
        if 'overall_scores' in data:
            st.subheader("🏆 Overall Scores")
            scores = data['overall_scores']
            
            # Create radar chart
            self._render_radar_chart(scores)
            
            # Score breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Score Breakdown**")
                for metric, score in scores.items():
                    if metric != 'overall_score':
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{score:.2f}",
                            delta=None
                        )
            
            with col2:
                st.markdown("**Overall Assessment**")
                overall_score = scores.get('overall_score', 0)
                if overall_score >= 0.8:
                    st.success("🟢 Excellent intervention quality")
                elif overall_score >= 0.6:
                    st.warning("🟡 Good intervention quality")
                else:
                    st.error("🔴 Poor intervention quality")
        
        # Detailed metrics
        if 'detailed_metrics' in data:
            st.subheader("📋 Detailed Metrics")
            self._render_detailed_metrics(data['detailed_metrics'])
    
    def _render_causal_graph_tab(self, data: Dict[str, Any]):
        """Render the causal graph tab."""
        st.subheader("🌳 Causal Relationship Graph")
        
        if 'scm_graph' in data and data['scm_graph'] is not None:
            # Interactive causal graph
            self._render_interactive_causal_graph(data['scm_graph'])
        else:
            st.info("No causal graph data available. Provide user code to generate AST-based causal relationships.")
        
        # Causal influence analysis
        if 'causal_metrics' in data:
            st.subheader("🔗 Causal Influence Analysis")
            self._render_causal_influence_analysis(data['causal_metrics'])
    
    def _render_interventions_tab(self, data: Dict[str, Any]):
        """Render the interventions tab."""
        st.subheader("🎯 Available Interventions")
        
        if 'intervention_suggestions' in data:
            suggestions = data['intervention_suggestions']
            
            for i, suggestion in enumerate(suggestions):
                with st.expander(f"Intervention {i+1}: {suggestion['type']}"):
                    st.markdown(f"**Description:** {suggestion['description']}")
                    st.markdown(f"**Confidence:** {suggestion['confidence']:.2f}")
                    st.markdown(f"**Reasoning:** {suggestion['reasoning']}")
                    
                    if st.button(f"Apply Intervention {i+1}", key=f"apply_{i}"):
                        st.session_state['selected_intervention'] = suggestion
    
    def _metric_card(self, title: str, value: str, description: str):
        """Render a metric card."""
        st.metric(
            label=title,
            value=value,
            help=description
        )
    
    def _render_ast_heatmap(self, node_counts: Dict[str, int]):
        """Render AST node count heatmap."""
        if not node_counts:
            st.info("No AST data available")
            return
        
        # Create heatmap - fix for pandas/matplotlib compatibility
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Reshape data for heatmap without pivot_table
        node_types = list(node_counts.keys())
        counts = list(node_counts.values())
        
        # Create a 2D array for heatmap
        heatmap_data = np.array(counts).reshape(1, -1)
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='d', 
                   cmap='YlOrRd', 
                   ax=ax,
                   xticklabels=node_types,
                   yticklabels=['Count'])
        
        plt.title('AST Node Distribution')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def _render_complexity_comparison(self, complexity_data: Dict[str, Any]):
        """Render complexity comparison chart."""
        if 'original' not in complexity_data or 'counterfactual' not in complexity_data:
            st.info("Complexity data not available")
            return
        
        orig = complexity_data['original']
        cf = complexity_data['counterfactual']
        
        # Create comparison chart
        metrics = ['cyclomatic', 'function_count', 'loop_count', 'conditional_count']
        labels = ['Cyclomatic', 'Functions', 'Loops', 'Conditionals']
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([i - width/2 for i in x], [orig.get(m, 0) for m in metrics], 
               width, label='Original', alpha=0.8)
        ax.bar([i + width/2 for i in x], [cf.get(m, 0) for m in metrics], 
               width, label='Counterfactual', alpha=0.8)
        
        ax.set_xlabel('Complexity Metrics')
        ax.set_ylabel('Count')
        ax.set_title('Code Complexity Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    def _render_style_comparison(self, style_data: Dict[str, Any]):
        """Render code style comparison."""
        if 'original' not in style_data or 'counterfactual' not in style_data:
            st.info("Style data not available")
            return
        
        orig = style_data['original']
        cf = style_data['counterfactual']
        
        # Create style comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Code Style**")
            st.metric("Line Count", orig.get('line_count', 0))
            st.metric("Avg Line Length", f"{orig.get('avg_line_length', 0):.1f}")
            st.metric("Comment Ratio", f"{orig.get('comment_ratio', 0):.1%}")
        
        with col2:
            st.markdown("**Counterfactual Code Style**")
            st.metric("Line Count", cf.get('line_count', 0))
            st.metric("Avg Line Length", f"{cf.get('avg_line_length', 0):.1f}")
            st.metric("Comment Ratio", f"{cf.get('comment_ratio', 0):.1%}")
    
    def _render_radar_chart(self, scores: Dict[str, float]):
        """Render radar chart for overall scores."""
        # Filter out overall_score for radar chart
        metrics = {k: v for k, v in scores.items() if k != 'overall_score'}
        
        if not metrics:
            st.info("No score data available")
            return
        
        # Create radar chart using plotly
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Scores',
            line_color='rgb(32, 201, 151)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Overall Score Breakdown"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_metrics(self, metrics: Dict[str, Any]):
        """Render detailed metrics breakdown."""
        # Create expandable sections for each metric type
        metric_types = ['structural', 'functional', 'semantic', 'causal', 'quality']
        
        for metric_type in metric_types:
            if f'{metric_type}_metrics' in metrics:
                with st.expander(f"{metric_type.title()} Metrics"):
                    metric_data = metrics[f'{metric_type}_metrics']
                    self._render_metric_table(metric_data)
    
    def _render_metric_table(self, metric_data: Dict[str, Any]):
        """Render metrics as a table."""
        # Convert metrics to table format
        table_data = []
        for key, value in metric_data.items():
            if isinstance(value, (int, float)):
                table_data.append([key.replace('_', ' ').title(), f"{value:.3f}"])
            else:
                table_data.append([key.replace('_', ' ').title(), str(value)])
        
        if table_data:
            # Use pandas DataFrame without explicit columns to avoid type issues
            df = pd.DataFrame(table_data)
            df.columns = ['Metric', 'Value']
            st.table(df)
    
    def _render_interactive_causal_graph(self, graph_data: Dict[str, Any]):
        """Render interactive causal graph."""
        if graph_data is None:
            st.info("Graph data not available")
            return
            
        if 'nodes' not in graph_data or 'edges' not in graph_data:
            st.info("Graph data not available")
            return
        
        # Convert to agraph format
        nodes = [Node(id=str(n['id']), label=str(n['label'])) for n in graph_data['nodes']]
        edges = [Edge(source=str(e['source']), target=str(e['target'])) for e in graph_data['edges']]
        
        config = Config(
            width=800,
            height=600,
            directed=True,
            physics=True,
            hierarchical=False,
            node={'labelProperty': 'label'},
            link={'labelProperty': 'label', 'renderLabel': False}
        )
        
        agraph(nodes=nodes, edges=edges, config=config)
    
    def _render_causal_influence_analysis(self, causal_data: Dict[str, Any]):
        """Render causal influence analysis."""
        if not causal_data:
            st.info("Causal influence data not available")
            return
        
        # Create influence visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Causal Influence Metrics**")
            for key, value in causal_data.items():
                if isinstance(value, (int, float)):
                    st.metric(
                        label=key.replace('_', ' ').title(),
                        value=f"{value:.3f}"
                    )
        
        with col2:
            st.markdown("**Influence Analysis**")
            if 'intervention_effectiveness' in causal_data:
                effectiveness = causal_data['intervention_effectiveness']
                if effectiveness > 0.7:
                    st.success("🟢 High intervention effectiveness")
                elif effectiveness > 0.4:
                    st.warning("🟡 Moderate intervention effectiveness")
                else:
                    st.error("🔴 Low intervention effectiveness")
    
    def render_intervention_suggestions(self, suggestions: List[Dict[str, Any]]):
        """Render intervention suggestions with confidence scores."""
        st.subheader("💡 Intelligent Intervention Suggestions")
        
        if not suggestions:
            st.info("No intervention suggestions available")
            return
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, suggestion in enumerate(suggestions):
            confidence = suggestion.get('confidence', 0)
            
            # Color code based on confidence
            if confidence > 0.8:
                st.success(f"**Suggestion {i+1}: {suggestion['type']}**")
            elif confidence > 0.6:
                st.warning(f"**Suggestion {i+1}: {suggestion['type']}**")
            else:
                st.info(f"**Suggestion {i+1}: {suggestion['type']}**")
            
            st.markdown(f"**Description:** {suggestion.get('description', 'N/A')}")
            st.markdown(f"**Confidence:** {confidence:.2f}")
            st.markdown(f"**Reasoning:** {suggestion.get('reasoning', 'N/A')}")
            
            if st.button(f"Apply Suggestion {i+1}", key=f"suggest_{i}"):
                st.session_state['selected_suggestion'] = suggestion
            
            st.divider()
    
    def render_execution_results(self, execution_data: Dict[str, Any]):
        """Render code execution results."""
        st.subheader("⚡ Execution Results")
        
        if not execution_data:
            st.info("No execution data available")
            return
        
        # Execution status
        success = execution_data.get('success', False)
        if success:
            st.success("✅ Code executed successfully")
        else:
            st.error(f"❌ Execution failed: {execution_data.get('error', 'Unknown error')}")
        
        # Execution metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Execution Time",
                f"{execution_data.get('execution_time', 0):.3f}s"
            )
        
        with col2:
            st.metric(
                "Pass Rate",
                f"{execution_data.get('pass_rate', 0):.1%}"
            )
        
        with col3:
            st.metric(
                "Test Count",
                execution_data.get('test_count', 0)
            )
        
        # Test results
        if 'test_results' in execution_data:
            st.subheader("🧪 Test Results")
            test_results = execution_data['test_results']
            
            for i, result in enumerate(test_results):
                status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
                st.markdown(f"**Test {i+1}:** {status}")
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                
                if 'actual' in result and 'expected' in result:
                    st.markdown(f"Expected: `{result['expected']}`, Got: `{result['actual']}`")
    
    def _render_educational_tab(self, data: Dict[str, Any]):
        """Render the educational tab with learning-focused features."""
        st.subheader("🎓 Educational Analysis")
        
        # Learning objectives
        if 'learning_objectives' in data:
            st.subheader("📚 Learning Objectives")
            objectives = data['learning_objectives']
            for i, objective in enumerate(objectives):
                st.markdown(f"**{i+1}.** {objective}")
        
        # Step-by-step explanations
        if 'educational_explanations' in data:
            st.subheader("🔍 Step-by-Step Code Generation Explanation")
            explanations = data['educational_explanations']
            
            for explanation in explanations:
                with st.expander(f"Step {explanation.step_number}: {explanation.title}"):
                    st.markdown(f"**Description:** {explanation.description}")
                    st.markdown(f"**Reasoning:** {explanation.reasoning}")
                    st.markdown(f"**Impact:** {explanation.impact}")
                    st.markdown(f"**Difficulty Level:** {explanation.difficulty_level.title()}")
                    
                    if explanation.code_snippet:
                        st.markdown("**Code Snippet:**")
                        st.code(explanation.code_snippet, language='python')
    
    def _render_developer_tools_tab(self, data: Dict[str, Any]):
        """Render the developer tools tab with IDE-like features."""
        st.subheader("🛠️ Developer Tools Analysis")
        
        # Code quality analysis
        if 'code_analysis' in data:
            st.subheader("📊 Code Quality Analysis")
            analysis = data['code_analysis']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Code Quality Score",
                    f"{analysis.code_quality_score:.2f}"
                )
            
            with col2:
                st.metric(
                    "Maintainability Index",
                    f"{analysis.maintainability_index:.1f}"
                )
            
            with col3:
                st.metric(
                    "Cyclomatic Complexity",
                    analysis.complexity_metrics.get('cyclomatic_complexity', 0)
                )
            
            # Potential issues
            if analysis.potential_issues:
                st.subheader("⚠️ Potential Issues")
                for issue in analysis.potential_issues:
                    st.warning(f"• {issue}")
            
            # Optimization opportunities
            if analysis.optimization_opportunities:
                st.subheader("💡 Optimization Opportunities")
                for opportunity in analysis.optimization_opportunities:
                    st.info(f"• {opportunity}")
        
        # Ranked interventions
        if 'ranked_interventions' in data:
            st.subheader("🎯 Ranked Intervention Suggestions")
            interventions = data['ranked_interventions']
            
            for i, intervention in enumerate(interventions):
                confidence_color = {
                    'high': 'success',
                    'medium': 'warning',
                    'low': 'info'
                }.get(intervention.confidence_level.value, 'info')
                
                with st.expander(f"#{i+1} {intervention.intervention_type.title()} (Confidence: {intervention.confidence_score:.2f})"):
                    st.markdown(f"**Description:** {intervention.description}")
                    st.markdown(f"**Reasoning:** {intervention.reasoning}")
                    st.markdown(f"**Risk Assessment:** {intervention.risk_assessment}")
                    st.markdown(f"**Implementation Difficulty:** {intervention.implementation_difficulty}")
                    
                    # Expected impact
                    st.markdown("**Expected Impact:**")
                    for aspect, impact in intervention.expected_impact.items():
                        st.markdown(f"• {aspect.title()}: {impact}")
                    
                    # Code patch
                    if intervention.code_patch:
                        st.markdown("**Suggested Code Patch:**")
                        st.code(intervention.code_patch, language='python')
    
    def _render_model_debug_tab(self, data: Dict[str, Any]):
        """Render the model debugging tab with LLM analysis."""
        st.subheader("🔬 Model Behavior Analysis")
        
        # Prompt analysis
        if 'prompt_analysis' in data:
            st.subheader("📝 Prompt Engineering Analysis")
            prompt_analysis = data['prompt_analysis']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Prompt Clarity",
                    f"{prompt_analysis.prompt_clarity:.2f}"
                )
            
            with col2:
                st.metric(
                    "Prompt Specificity",
                    f"{prompt_analysis.prompt_specificity:.2f}"
                )
            
            with col3:
                st.metric(
                    "Prompt Complexity",
                    f"{prompt_analysis.prompt_complexity:.2f}"
                )
            
            # Optimization suggestions
            if prompt_analysis.optimization_suggestions:
                st.subheader("💡 Prompt Optimization Suggestions")
                for suggestion in prompt_analysis.optimization_suggestions:
                    st.info(f"• {suggestion}")
        
        # Model behavior analysis
        if 'model_behavior' in data:
            st.subheader("🤖 Model Behavior Patterns")
            behavior = data['model_behavior']
            
            st.markdown(f"**Behavior Type:** {behavior.behavior_type.value.title()}")
            st.markdown(f"**Consistency Score:** {behavior.consistency_score:.2f}")
            
            # Bias indicators
            if behavior.bias_indicators:
                st.subheader("⚠️ Bias Indicators")
                for indicator in behavior.bias_indicators:
                    st.warning(f"• {indicator}")
            
            # Response patterns
            if behavior.response_patterns:
                st.subheader("📊 Response Patterns")
                patterns = behavior.response_patterns
                
                if 'common_structures' in patterns:
                    st.markdown("**Common Code Structures:**")
                    for structure, count in patterns['common_structures'].items():
                        st.markdown(f"• {structure}: {count}")
        
        # Causal influence analysis
        if 'causal_influence' in data:
            st.subheader("🔗 Causal Influence Analysis")
            causal = data['causal_influence']
            
            # Prompt component weights
            if causal.prompt_component_weights:
                st.markdown("**Prompt Component Weights:**")
                for component, weight in causal.prompt_component_weights.items():
                    st.markdown(f"• {component.replace('_', ' ').title()}: {weight:.2f}")
            
            # Intervention effectiveness
            if causal.intervention_effectiveness:
                st.markdown("**Intervention Effectiveness:**")
                for intervention, effectiveness in causal.intervention_effectiveness.items():
                    st.markdown(f"• {intervention.replace('_', ' ').title()}: {effectiveness:.2f}") 