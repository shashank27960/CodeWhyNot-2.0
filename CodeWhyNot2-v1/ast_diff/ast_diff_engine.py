import ast
from apted import APTED, Config
from apted.helpers import Tree

class ASTNodeAdapter:
    """Adapter to convert Python AST to APTED-compatible Tree."""
    def __init__(self, node):
        self.node = node
        self.children = [ASTNodeAdapter(child) for child in ast.iter_child_nodes(node)]
        self.label = type(node).__name__

    def get_label(self):
        return self.label

    def get_children(self):
        return self.children

    def to_apted(self):
        return Tree(self.get_label(), [child.to_apted() for child in self.get_children()])

class ASTDiffEngine:
    """
    Computes AST-level diffs and fidelity metrics between code snippets using APTED.
    """
    def __init__(self):
        pass

    def is_syntax_correct(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def diff(self, code1, code2):
        """Return a semantic diff and fidelity metrics between two code snippets using APTED."""
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            adapter1 = ASTNodeAdapter(tree1)
            adapter2 = ASTNodeAdapter(tree2)
            apted_tree1 = adapter1.to_apted()
            apted_tree2 = adapter2.to_apted()
            class LabelConfig(Config):
                def rename(self, node1, node2):
                    return 0 if node1.label == node2.label else 1
            apted = APTED(apted_tree1, apted_tree2, LabelConfig())
            edit_distance = apted.compute_edit_distance()
            max_nodes = max(len(list(ast.walk(tree1))), len(list(ast.walk(tree2))))
            fidelity = 1.0 - (edit_distance / max_nodes) if max_nodes > 0 else 1.0
            diff_str = f"APTED Tree Edit Distance: {edit_distance}\nNodes in code1: {len(list(ast.walk(tree1)))}\nNodes in code2: {len(list(ast.walk(tree2)))}"
        except Exception as e:
            diff_str = f"[AST diff error: {e}]"
            fidelity = 0.0
        # Syntax correctness
        syntax1 = self.is_syntax_correct(code1)
        syntax2 = self.is_syntax_correct(code2)
        if not (syntax1 and syntax2):
            fidelity = 0.0
        return diff_str, round(fidelity, 2) 