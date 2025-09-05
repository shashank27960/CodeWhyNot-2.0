# CodeWhyNot 2.0 Development Prompt for Cursor AI
# ===============================================

"""
INSTRUCTIONS:
1. Build a Streamlit-based application implementing causal counterfactuals for code generation
2. Use architecture: SCM Editor → Gumbel Engine → Code Llama → AST Diff → Metric Dashboard
3. Strictly follow the PRD phases and technical specifications below
"""

# TECHNICAL STACK
# ---------------
REQUIREMENTS = {
    "python": "3.10",
    "libraries": [
        "streamlit==1.32.0", "networkx==3.2", "transformers==4.40.0",
        "torch==2.2.0", "zss==1.2.0", "tree_sitter==0.20.2",
        "optuna==3.6.0", "code-diff==0.1.0"
    ],
    "quantized_model": "TheBloke/CodeLlama-7B-Instruct-GGUF",
    "gpu_config": {"precision": "fp16", "device_map": "auto"}
}

# CORE MODULES (P0 Features)
# ---------------------------
MODULES = [
    {
        "name": "scm_editor",
        "description": "Structural Causal Model prompt editor with intervention templates",
        "functions": [
            "parse_prompt(natural_language) -> SCM_graph",
            "apply_intervention(scm_graph, intervention_type) -> modified_scm"
        ],
        "tech": "NetworkX + SpaCy dependency parsing",
        "params": {
            "intervention_types": ["loop→recursion", "add_import", "change_data_structure"],
            "node_types": ["NL_token", "code_concept", "io_example"]
        }
    },
    {
        "name": "gumbel_cf_generator",
        "description": "Counterfactual prompt generator using Gumbel-Max SCM",
        "functions": [
            "sample_counterfactual(original_prompt, scm_graph, temperature=0.7) -> cf_prompt",
            "store_rng_states() -> rng_sequence"
        ],
        "tech": "PyTorch GumbelSoftmax + custom sampler",
        "algorithm": """
            # Adapted from 'Counterfactual Token Generation' Algorithm 1
            for token_idx in generation_sequence:
                torch.manual_seed(master_seed + token_idx)
                logits = model_output.logits[0, -1]
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                sampled_token = argmax(logits + gumbel_noise)
        """
    },
    {
        "name": "ast_diff_engine",
        "description": "Semantic code difference analyzer",
        "functions": [
            "compute_ast_distance(code1, code2, lang='python') -> zss_distance",
            "visualize_diff(ast1, ast2) -> diff_svg"
        ],
        "tech": "Tree-sitter + zss Zhang-Shasha algorithm",
        "metrics": {
            "fidelity_score": "0.6*ast_distance + 0.3*test_pass + 0.1*(1 - entropy_diff)"
        }
    }
]

# UI/UX SPECIFICATIONS
# --------------------
STREAMLIT_UI = {
    "layout": [
        {"panel": "left_column", "components": [
            "prompt_input(text_area)",
            "intervention_selector(dropdown)",
            "cf_generate_button"
        ]},
        {"panel": "main_area", "components": [
            "code_preview(tabs: original/counterfactual)",
            "ast_diff_visualizer(split_view)",
            "fidelity_score_card(progress_bar)"
        ]},
        {"panel": "sidebar", "components": [
            "causal_tree_visualizer(interactive_network)",
            "metric_history(line_chart)"
        ]}
    ],
    "design_rules": [
        "Use @st.cache_data for all model inferences",
        "Real-time AST diff rendering with code-diff library",
        "NetworkX causal trees as interactive Plotly graphs"
    ]
}

# OPTIMIZATION CONSTRAINTS
# ------------------------
CONSTRAINTS = {
    "latency": "<3s generation time",
    "memory": "<6GB VRAM usage",
    "actions": [
        "Quantize CodeLlama to 4-bit using bitsandbytes",
        "Implement streaming token generation",
        "Cache AST parse trees using LRU strategy"
    ]
}

# PAPER ALIGNMENT HOOKS
# ---------------------
RESEARCH_HOOKS = [
    {
        "novelty": "Hybrid fidelity metric combining AST/functional/entropy measures",
        "evaluation": "Compare against HumanEval+ and mMBPP+ datasets"
    },
    {
        "novelty": "Multi-agent intervention search (Optuna + GradSearch)",
        "evaluation": "Benchmark against MEMIT and steering techniques"
    }
]

# DEVELOPMENT ROADMAP
# -------------------
PHASED_PLAN = """
Week 1-2: SCM Editor + Gumbel Core
   - Implement NetworkX prompt decomposition
   - Build Gumbel sampler with RNG state tracking
   
Week 3-4: AST Diff + Streamlit UI
   - Integrate Tree-sitter parsers for Python
   - Develop side-by-side diff visualizer
   
Week 5-6: Fidelity Metrics + Quantization
   - Create hybrid scoring formula
   - Optimize CodeLlama with GGUF quantization
   
Week 7-8: Multi-Agent Search
   - Implement genetic algorithm for prompt variants
   - Add gradient-based intervention suggestions
"""

# INSTRUCTION TO CURSOR AI
# ------------------------
"""
ACTION PLAN:
1. Initialize project with `requirements.txt` from TECHNICAL STACK
2. Create module files: scm_editor.py, gumbel_engine.py, ast_diff.py
3. Build Streamlit app according to UI/UX SPECIFICATIONS
4. Implement CONSTRAINTS optimizations
5. Add RESEARCH_HOOKS instrumentation for paper metrics
6. Follow PHASED_PLAN for incremental development

VALIDATION:
- Unit tests for all core functions
- Latency benchmarks using %timeit
- Fidelity score calibration on mMBPP+
"""