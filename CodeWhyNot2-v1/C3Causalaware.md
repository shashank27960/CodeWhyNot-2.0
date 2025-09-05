Of course. This is the perfect time to create a detailed, week-by-week roadmap. This plan is designed to be ambitious yet achievable for a 4-week internship, focusing specifically on building and validating the novel features we've discussed.

This roadmap translates our analysis of the research landscape into a concrete implementation plan.

---

### **Detailed 4-Week Roadmap: Building "Causal Code Counterfactuals (C3)"**

This roadmap is structured to ensure that each week concludes with a significant, demonstrable milestone, directly building towards the final project and paper.

---

### **Week 1: Foundational Scaffolding & Baseline Generation**

**Goal:** Establish a stable development environment and a working, non-causal baseline. This week is about building the chassis of the car before installing the engine.

| Task ID | Task Description                                                                 | Implementation Details & Focus on Novelty                                                                                                                                                                                                                                  |
| :------ | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1.1**   | **Environment & Project Setup**                                                  | **Setup:** Create the project structure (`C3_Project/`, `c3_core/`, etc.) in a Git repository.  **Environment:** Create a `requirements.txt` file and set up a Python virtual environment or configure a Google Colab notebook with GPU runtime. **Deliverable:** A runnable, empty project. |
| **1.2**   | **Build the UI Shell**                                                           | **Framework:** Use Colab's `ipywidgets` or Streamlit. **Components:** Create the basic interactive elements: a dropdown to select a problem (`factorial`, `fibonacci`), a button ("Generate"), and placeholder areas for two code outputs and metrics. **Focus:** This is the *scaffold* for displaying the novelty later. |
| **1.3**   | **Implement Standard (Non-Causal) Code Generation**                              | **Library:** Use the `transformers` library to load the `deepseek-coder-1.3b-instruct` model. **Function:** Create a simple function that takes a prompt string and returns the generated code using the standard `model.generate()` method. **Focus:** This step **intentionally ignores causality**. The goal is to prove you can get *any* code out of the model. This provides a crucial, non-novel baseline to compare against. |
| **1.4**   | **Initial Integration**                                                          | **Action:** Wire the UI button to call the standard generation function. When the user clicks "Generate," the factual prompt should be sent to the model, and the result should appear in the first code output area. **Deliverable:** A simple app that generates one piece of code. |

**Weekly Milestone:** A working application that can generate code for a given prompt, demonstrating a stable baseline. The "counterfactual" side is still empty.

---

### **Week 2: The Causal Core - Implementing the Noise Replay Mechanism**

**Goal:** Implement the project's central technical novelty: the ability to generate a true counterfactual by replaying the sampling noise.

| Task ID | Task Description                                                              | Implementation Details & Focus on Novelty                                                                                                                                                                                                                                                          |
| :------ | :------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2.1**   | **Refactor Generation to be Token-by-Token**                                    | **Action:** Replace the high-level `model.generate()` call with a `for` loop that generates the output one token at a time. Inside the loop, you will manually get the logits for the next token from the model. **Focus:** This is a prerequisite for fine-grained control over the generation process. |
| **2.2**   | **Implement the Seed Replay Algorithm (CORE NOVELTY)**                          | **Function:** Create a new generator function, `_generate_single_version(prompt, master_seed)`. **The Key Line:** Inside the token-by-token generation loop, just before you sample the next token from the logits, you must call **`torch.manual_seed(master_seed + step_index)`**. **Focus:** This is the single most important implementation detail for your paper's "causal-aware" claim. It ensures that the pseudo-random choice at step `i` is identical for both the factual and counterfactual runs. |
| **2.3**   | **Build the Causal Pair Generator**                                             | **Function:** Create a wrapper function `generate_causal_pair(factual_prompt, cf_prompt, master_seed)`. **Logic:** This function will call your new `_generate_single_version()` twice: once with the factual prompt and once with the counterfactual prompt, but critically, **passing the same `master_seed` to both calls**. |
| **2.4**   | **Integrate into the UI**                                                       | **Action:** Update the UI button's logic to call `generate_causal_pair()`. Display the first returned code snippet in the factual panel and the second in the counterfactual panel. **Test:** Verify that changing the prompt results in different code, but using the same prompt and seed twice produces identical code. |

**Weekly Milestone:** A functional application that generates a factual/counterfactual code pair using the seed replay mechanism. This completes the implementation of the project's primary technical novelty.

---

### **Week 3: The Measurement Engine - Domain-Specific Evaluation**

**Goal:** Implement the project's second pillar of novelty: evaluating the generated code using metrics that are meaningful for software engineering.

| Task ID | Task Description                                                              | Implementation Details & Focus on Novelty                                                                                                                                                                                                                                                                         |
| :------ | :------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3.1**   | **Implement AST Edit Distance (NOVELTY 2A)**                                  | **Library:** Use Python's built-in `ast` library to parse code strings into Abstract Syntax Trees. Use the `zss` library to compute the Zhang-Shasha tree edit distance. **Function:** Create `calculate_ast_distance(code1, code2)`. **Focus:** This metric quantifies the *structural* difference between the two programs, which is far more meaningful than a simple text diff. This directly aligns with the research trend of using domain-specific metrics for Code XAI. |
| **3.2**   | **Implement Functional Correctness Testing (NOVELTY 2B)**                       | **Setup:** In your `prompts.py` or configuration cell, add a `unit_test` string containing `assert` statements for each problem. **Function:** Create `run_functional_test(generated_code, unit_test_code)`. Use a `try...except` block around an `exec()` call to run the generated code and the tests together. Catch `AssertionError` and other exceptions. **Focus:** This evaluates the *logical correctness* of the generated code, providing a crucial, real-world measure of the intervention's impact. |
| **3.3**   | **Full Integration and Display**                                                | **Action:** In your main application logic, after generating the causal pair, call both `calculate_ast_distance()` and `run_functional_test()` on the outputs. **UI Update:** Display the returned scores and Pass/Fail statuses clearly in the metrics section of the UI. Use color and icons (✅/❌) to make the results instantly understandable. |

**Weekly Milestone:** A complete, interactive tool that not only generates causal pairs but also automatically evaluates them and presents the structural and functional analysis to the user.

---

### **Week 4: Experimentation, Analysis, and Publication**

**Goal:** Use the completed tool to generate the data for your paper, analyze the results, and write the paper draft. This is the payoff week.

| Task ID | Task Description                                                              | Implementation Details & Focus on Novelty                                                                                                                                                                                                                                                                                             |
| :------ | :------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **4.1**   | **Design and Run the Experiment**                                               | **Protocol:** Define a clear experimental protocol. For example: "For each of the 3 problems (factorial, fibonacci, sum), run the generation with 10 different master seeds (e.g., seeds 1 through 10)." This will give you 30 data points. **Automation:** Write a simple script that automates this process, calling your functions and logging the results (prompts, codes, AST distance, correctness status) to a CSV file or a list of dictionaries. |
| **4.2**   | **Analyze and Visualize Results**                                               | **Tools:** Use `pandas` to load your results into a DataFrame. **Analysis:** Calculate key statistics: mean/std-dev of AST distance per problem, and the success rate (%) of the counterfactual generations. **Visualization:** Use `matplotlib` or `seaborn` to create plots for your paper. A bar chart showing the "Average AST Distance by Problem" would be highly effective. |
| **4.3**   | **Write the Paper Draft**                                                       | **Structure:** Follow the academic structure we defined: Abstract, Introduction, Related Work, **Methodology** (describe your seed replay and AST/functional metrics), **Results** (present your charts and analysis), and **Conclusion/Future Work**. **Focus on Novelty:** Explicitly state your contributions: 1) the interactive tool itself, 2) the application of causal-aware generation to code, and 3) the use of domain-specific metrics for evaluation. |
| **4.4**   | **Final Presentation & Code Cleanup**                                           | **Action:** Prepare a short presentation (5-10 slides) summarizing the project for your mentor/professor. Ensure your code is clean, well-commented, and your Colab notebook is runnable from top to bottom. |

**Weekly Milestone:** A complete project repository with a runnable tool, a dataset of experimental results, a data analysis script, and a submitted draft of the research paper.