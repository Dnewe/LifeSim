import ast
import operator as op
from typing import Dict

# Allowed operators
ops = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

ops_cmp = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
}

def eval_expr(expr: str, variables: Dict[str, float|str]):
    """
    Safely evaluate a math expression.
    expr: string (e.g. "0.5*morphology**-0.5*(1+0.05*(physiology-1)) + speed_knob")
    variables: dict {"morphology":1.2, "physiology":0.8, "sex:"male" ...}
    """
    
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.BinOp):  
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            right = _eval(node.comparators[0])
            return ops_cmp[type(node.ops[0])](left, right)
        if isinstance(node, ast.IfExp):
            return _eval(node.body) if _eval(node.test) else _eval(node.orelse)
        raise TypeError(node)
    
    node = ast.parse(expr, mode='eval').body
    return _eval(node)