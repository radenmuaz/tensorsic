# SIC Interaction Rules
def reconnect_ports(node1, node2, graph):
    # Reconnect auxiliary ports of annihilated nodes
    for i in range(1, 3):  # Auxiliary ports only
        if node1.ports[i] and node2.ports[i]:
            n1, p1 = node1.ports[i]
            n2, p2 = node2.ports[i]
            graph.connect(n1, p1, n2, p2)

def rearrange_connections(node1, node2, graph):
    # Swap auxiliary connections if valid
    aux1 = node1.ports[1]
    aux2 = node2.ports[2]

    if aux1 and aux2:
        aux1_node, aux1_port = aux1
        aux2_node, aux2_port = aux2
        graph.connect(aux1_node, aux1_port, node2, 1)
        graph.connect(aux2_node, aux2_port, node1, 2)

def apply_rules(node1, node2, graph):
    # Annihilation Rule: Nodes of the same type annihilate
    if node1.node_type == node2.node_type:
        reconnect_ports(node1, node2, graph)
        graph.remove_node(node1)
        graph.remove_node(node2)
        return True

    # Commutation Rule: Nodes of different types rearrange connections
    if {node1.node_type, node2.node_type} == {'gamma', 'delta'}:
        if node1.ports[1] and node2.ports[2]:  # Check if ports are valid
            rearrange_connections(node1, node2, graph)
            return True

    return False

def simulate(graph):
    while True:
        changes = False
        for node1 in graph.nodes[:]:
            for i in range(3):
                if node1.ports[i]:
                    node2, port = node1.ports[i]
                    if apply_rules(node1, node2, graph):
                        changes = True
                        break
            if changes:
                break
        if not changes:
            break

import re

class Node:
    def __init__(self, node_type, id):
        self.node_type = node_type  # 'gamma', 'delta', 'epsilon'
        self.id = id                # Unique identifier
        self.ports = [None, None, None]  # [principal, aux1, aux2]
        self.var_name = None   # for delta/epsilon

    def __repr__(self):
        return f"{self.node_type}{self.id}({self.var_name})"

class Graph:
    def __init__(self):
        self.nodes = []
        self.next_id = 0

    def create_node(self, node_type):
        n = Node(node_type, self.next_id)
        self.next_id += 1
        self.nodes.append(n)
        return n

    def connect(self, node1, port1, node2, port2):
        node1.ports[port1] = (node2, port2)
        node2.ports[port2] = (node1, port1)

    def remove_node(self, node):
        for i in range(3):
            if node.ports[i]:
                nbr, p = node.ports[i]
                nbr.ports[p] = None
        self.nodes.remove(node)

# ————————————————————————————————
# TOKENIZER + PARSER (unchanged)
# ————————————————————————————————
def tokenize(expr_str):
    token_spec = [
        ('LAMBDA', r'[λ\\]'),
        ('DOT',    r'\.'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('VAR',    r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('SPACE',  r'\s+'),
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pat})' for name,pat in token_spec)
    for m in re.finditer(tok_regex, expr_str):
        kind = m.lastgroup
        val  = m.group()
        if kind != 'SPACE':
            yield (kind, val)

def parse_lambda_expr(expr_str):
    tokens = list(tokenize(expr_str))
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else ('EOF','')
    def consume(expected_kind=None):
        nonlocal pos
        if pos >= len(tokens):
            raise SyntaxError("Unexpected end")
        kind, val = tokens[pos]
        if expected_kind and kind != expected_kind:
            raise SyntaxError(f"Expected {expected_kind}, got {kind}")
        pos += 1
        return val

    def parse_var():
        kind, val = peek()
        if kind == 'VAR':
            return ('var', consume('VAR'))
        else:
            raise SyntaxError("Expected VAR")

    def parse_atom():
        kind, val = peek()
        if kind == 'LAMBDA':
            return parse_lambda()
        elif kind == 'LPAREN':
            consume('LPAREN')
            e = parse_expr()
            consume('RPAREN')
            return e
        elif kind == 'VAR':
            return parse_var()
        else:
            raise SyntaxError(f"Unexpected token {kind}")

    def parse_lambda():
        consume('LAMBDA')
        v = consume('VAR')
        consume('DOT')
        body = parse_expr()
        return ('lam', v, body)

    def parse_expr():
        left = parse_atom()
        while True:
            kind, _ = peek()
            if kind in ('VAR','LPAREN','LAMBDA'):
                right = parse_atom()
                left = ('app', left, right)
            else:
                break
        return left

    result = parse_expr()
    if pos != len(tokens):
        raise SyntaxError("Trailing tokens")
    return result

def lambda_expr_to_str(expr):
    t = expr[0]
    if t == 'var':
        return expr[1]
    if t == 'lam':
        return f"λ{expr[1]}.{lambda_expr_to_str(expr[2])}"
    if t == 'app':
        left = lambda_expr_to_str(expr[1])
        right = lambda_expr_to_str(expr[2])
        if expr[1][0] == 'lam':
            left = f"({left})"
        if expr[2][0] in ('app','lam'):
            right = f"({right})"
        return f"{left} {right}"
    return '?'

# ————————————————————————————————
# UPDATED lambda_to_sic
# ————————————————————————————————
def lambda_to_sic(expr, graph, env=None):
    """
    For each bound variable “x”, we create exactly one Delta(x) linked to the binderGamma.
    BUT for each *occurrence* of x in the body, we create a brand‐new Epsilon(x) and do NOT try to re‐use Delta(x).
    """
    if env is None:
        env = {}

    t = expr[0]
    if t == 'var':
        name = expr[1]
        # Always create a fresh epsilon for every occurrence.
        ep = graph.create_node('epsilon')
        ep.var_name = name
        return ep

    if t == 'lam':
        var_name = expr[1]
        body = expr[2]

        g = graph.create_node('gamma')
        d = graph.create_node('delta')
        d.var_name = var_name

        # Record binder delta in env so that nested lambdas see it (though we never re‐use delta for occurrences)
        new_env = env.copy()
        new_env[var_name] = d

        body_node = lambda_to_sic(body, graph, new_env)

        # Connect the binder gamma → delta on (port1, port0), and the gamma → body on (port2, port0)
        graph.connect(g, 1, d, 0)
        graph.connect(g, 2, body_node, 0)

        return g

    if t == 'app':
        g = graph.create_node('gamma')
        f_node = lambda_to_sic(expr[1], graph, env)
        a_node = lambda_to_sic(expr[2], graph, env)
        graph.connect(g, 1, f_node, 0)
        graph.connect(g, 2, a_node, 0)
        return g

    raise ValueError("Unknown node type in lambda AST")

# ————————————————————————————————
# UPDATED sic_to_lambda
# ————————————————————————————————
def sic_to_lambda(node, visited_ids=None):
    """
    Reconstructs a lambda AST from the SIC graph, under the assumption that:
      • Every real λ‐binder is one gamma whose port 1 goes to a Delta(d) 
        such that delta.ports[0] == (that gamma, 1).
      • All other “var occurrences” are stand‐alone Epsilon(var_name).
      • So, whenever we see gamma.port1 → a delta whose port0 is back here, that gamma is a λ‐abstraction.
      • Otherwise, gamma is a (func arg) application.
    """
    if visited_ids is None:
        visited_ids = set()
    if node.id in visited_ids:
        return ('var', '?')  # break cycles
    visited_ids.add(node.id)

    if node.node_type == 'epsilon':
        # Epsilon always corresponds to a “(var name)”
        return ('var', node.var_name or 'free_var')

    if node.node_type == 'delta':
        # A delta unconnected to a gamma (as port0) cannot happen under our new scheme,
        # but if we do see one, treat it just as a var too.
        return ('var', node.var_name or '?')

    if node.node_type == 'gamma':
        # Look at its two auxiliary ports:
        #   ports[1] == (n1, p1), ports[2] == (n2, p2).
        n1, p1 = node.ports[1]
        n2, p2 = node.ports[2]

        # If n1 is a Delta AND that delta’s port0 points back to (this gamma, 1),
        # then we know “gamma” is a λ‐abstraction for var = n1.var_name.
        if n1.node_type == 'delta' and n1.ports[0] == (node, 1):
            var = n1.var_name
            body = sic_to_lambda(n2, visited_ids)
            return ('lam', var, body)

        # Otherwise, it is an application: ( func_expr ) ( arg_expr )
        func = sic_to_lambda(n1, visited_ids)
        arg  = sic_to_lambda(n2, visited_ids)
        return ('app', func, arg)

    # Fallback (shouldn’t happen):
    return ('var', '?')

# ————————————————————————————————
# TEST ON “λf.λx.f (f (f x))”
# ————————————————————————————————
if __name__ == "__main__":
    # expr_str = "λf.λx.f (f (f x))"
    # print("Input: ", expr_str)
    # parsed = parse_lambda_expr(expr_str)
    # print("Parsed AST: ", parsed)

    # g = Graph()
    # root = lambda_to_sic(parsed, g)
    # back_expr = sic_to_lambda(root)
    # back_str = lambda_expr_to_str(back_expr)
    # print("Back-translated Lambda:", back_str)
    # exit()
    increment = "λn.λf.λx.f (n f x)"
    one       = "λf.λx.f x"
    two       = "λf.λx.f (f x)"

    # (2) Parse all three into our AST form:
    parsed_inc = parse_lambda_expr(increment)
    parsed_one = parse_lambda_expr(one)
    parsed_two = parse_lambda_expr(two)
    parsed_incr_one = parse_lambda_expr(f"{increment} {one}")

    g = Graph()
    n = lambda_to_sic(parsed_incr_one, g)
    # n = lambda_to_sic(parsed_one, g)
    simulate(g)

    result_ast = sic_to_lambda(n)

    result_str = lambda_expr_to_str(result_ast)

    print("Expected two   =", two)
    print("Result of (increment one) →", result_str)

    assert result_str == two, f"FAIL: got {result_str} instead of {two}"
    print("✔︎  (increment one)  ≡ two")

