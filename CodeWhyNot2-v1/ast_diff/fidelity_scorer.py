import ast

class FidelityScorer:
    def __init__(self, code1: str, code2: str):
        self.code1 = code1
        self.code2 = code2
        self.ast1 = self._parse_code(code1)
        self.ast2 = self._parse_code(code2)

    def _parse_code(self, code: str):
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def is_syntax_correct(self, code: str) -> bool:
        return self._parse_code(code) is not None

    def _ast_node_count(self, node):
        if node is None:
            return 0
        count = 1
        for child in ast.iter_child_nodes(node):
            count += self._ast_node_count(child)
        return count

    def _ast_diff_count(self, node1, node2):
        if node1 is None or node2 is None:
            return max(self._ast_node_count(node1), self._ast_node_count(node2))
        if type(node1) != type(node2):
            return 1 + max(self._ast_node_count(node1), self._ast_node_count(node2))
        diff = 0
        children1 = list(ast.iter_child_nodes(node1))
        children2 = list(ast.iter_child_nodes(node2))
        for c1, c2 in zip(children1, children2):
            diff += self._ast_diff_count(c1, c2)
        # Account for extra children in either node
        diff += abs(len(children1) - len(children2))
        return diff

    def ast_edit_distance(self) -> float:
        total_nodes = max(self._ast_node_count(self.ast1), self._ast_node_count(self.ast2))
        if total_nodes == 0:
            return 1.0  # Completely different or both empty
        diff = self._ast_diff_count(self.ast1, self.ast2)
        return diff / total_nodes

    def _has_loop(self, node):
        if node is None:
            return False
        if isinstance(node, (ast.For, ast.While)):
            return True
        return any(self._has_loop(child) for child in ast.iter_child_nodes(node))

    def _has_recursion(self, node, func_name=None):
        if node is None:
            return False
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if func_name and node.func.id == func_name:
                return True
        return any(self._has_recursion(child, func_name) for child in ast.iter_child_nodes(node))

    def logic_shift(self) -> bool:
        # Returns True if one code uses loop and the other uses recursion
        loop1 = self._has_loop(self.ast1)
        loop2 = self._has_loop(self.ast2)
        rec1 = self._has_recursion(self.ast1)
        rec2 = self._has_recursion(self.ast2)
        return (loop1 != loop2) or (rec1 != rec2)

    def score(self) -> float:
        # Weights can be tuned
        w_ast = 0.7
        w_logic = 0.2
        w_syntax = 0.1

        ast_dist = self.ast_edit_distance()
        logic = 1.0 if self.logic_shift() else 0.0
        syntax1 = 1.0 if self.is_syntax_correct(self.code1) else 0.0
        syntax2 = 1.0 if self.is_syntax_correct(self.code2) else 0.0
        syntax = 1.0 - 0.5 * (1 - syntax1) - 0.5 * (1 - syntax2)  # 1 if both correct, 0.5 if one, 0 if none

        # Lower distance and no logic shift = higher fidelity
        fidelity = 1.0 - (w_ast * ast_dist + w_logic * logic + w_syntax * (1 - syntax))
        return max(0.0, min(1.0, fidelity))

if __name__ == "__main__":
    code1 = "def fact(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result"
    code2 = "def fact(n):\n    if n == 0:\n        return 1\n    else:\n        return n * fact(n-1)"
    scorer = FidelityScorer(code1, code2)
    print("AST Edit Distance:", scorer.ast_edit_distance())
    print("Logic Shift:", scorer.logic_shift())
    print("Syntax Correctness:", scorer.is_syntax_correct(code1), scorer.is_syntax_correct(code2))
    print("Fidelity Score:", scorer.score()) 