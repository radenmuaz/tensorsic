import re

class Node:
    def __init__(self, node_type, id):
        self.node_type = node_type  # 'gamma', 'delta', 'epsilon'
        self.id = id
        self.ports = [None, None, None]  # (Node, port_index)
        self.var_name = None  # For epsilon or delta, store variable name

    def __repr__(self):
        conns = []
        for i, c in enumerate(self.ports):
            if c is None:
                conns.append(f"port{i}->None")
            else:
                n, p = c
                conns.append(f"port{i}->{n.node_type}{n.id}.port{p}")
        return f"{self.node_type}{self.id}({', '.join(conns)})"

class Graph:
    def __init__(self):
        self.nodes = []
        self.next_id = 0

    def create_node(self, node_type):
        node = Node(node_type, self.next_id)
        self.next_id += 1
        self.nodes.append(node)
        return node

    def connect(self, node1, port1, node2, port2):
        node1.ports[port1] = (node2, port2)
        node2.ports[port2] = (node1, port1)

def tokenize(expr_str):
    token_spec = [
        ('LAMBDA', r'[λ\\]'),
        ('DOT', r'\.'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('VAR', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('SPACE', r'\s+'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_spec)
    for mo in re.finditer(tok_regex, expr_str):
        kind = mo.lastgroup
        val = mo.group()
        if kind != 'SPACE':
            yield (kind, val)

def parse_lambda_expr(expr_str):
    tokens = list(tokenize(expr_str))
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else ('EOF', '')

    def consume(expected_kind=None):
        nonlocal pos
        if pos >= len(tokens):
            raise SyntaxError("Unexpected end of input")
        kind, val = tokens[pos]
        if expected_kind and kind != expected_kind:
            raise SyntaxError(f"Expected {expected_kind}, got {kind}")
        pos += 1
        return val

    def parse_var():
        kind, val = peek()
        if kind == 'VAR':
            consume('VAR')
            return ('var', val)
        else:
            raise SyntaxError("Expected variable")

    def parse_atom():
        kind, val = peek()
        if kind == 'LAMBDA':
            return parse_lambda()
        elif kind == 'LPAREN':
            consume('LPAREN')
            expr = parse_expr()
            consume('RPAREN')
            return expr
        elif kind == 'VAR':
            return parse_var()
        else:
            raise SyntaxError(f"Unexpected token: {kind}")

    def parse_lambda():
        consume('LAMBDA')
        var_name = consume('VAR')
        consume('DOT')
        body = parse_expr()
        return ('lambda', var_name, body)

    def parse_expr():
        left = parse_atom()
        while True:
            kind, _ = peek()
            if kind in ('VAR', 'LPAREN', 'LAMBDA'):
                right = parse_atom()
                left = ('app', left, right)
            else:
                break
        return left

    result = parse_expr()
    if pos != len(tokens):
        raise SyntaxError("Unexpected tokens at end")
    return result

def lambda_to_sic(expr, graph, env=None):
    if env is None:
        env = {}

    t = expr[0]

    if t == 'var':
        name = expr[1]
        if name in env:
            # Bound var: return delta node stored in env
            return env[name]
        else:
            # Free var: epsilon node with var_name
            ep = graph.create_node('epsilon')
            ep.var_name = name
            return ep

    if t == 'lambda':
        var_name = expr[1]
        body = expr[2]

        g = graph.create_node('gamma')
        d = graph.create_node('delta')
        d.var_name = var_name  # store var name on delta to identify var later

        new_env = env.copy()
        new_env[var_name] = d

        body_node = lambda_to_sic(body, graph, new_env)

        graph.connect(g, 1, d, 0)    # gamma port1 <-> delta port0
        graph.connect(g, 2, body_node, 0)  # gamma port2 <-> body_node port0

        return g

    if t == 'app':
        g = graph.create_node('gamma')
        func_node = lambda_to_sic(expr[1], graph, env)
        arg_node = lambda_to_sic(expr[2], graph, env)
        graph.connect(g, 1, func_node, 0)
        graph.connect(g, 2, arg_node, 0)
        return g

def sic_to_lambda(node, visited=None):
    if visited is None:
        visited = set()

    # Mark only gamma nodes as visited to avoid infinite loops
    if node.node_type == 'gamma':
        if node.id in visited:
            return ('var', '?')
        visited.add(node.id)

    if node.node_type == 'epsilon':
        return ('var', node.var_name if node.var_name else 'free_var')

    if node.node_type == 'delta':
        # delta node with var_name is a bound var
        if node.var_name:
            return ('var', node.var_name)
        else:
            # Defensive fallback: try to get from connected epsilon
            conn = node.ports[0]
            if conn:
                connected, _ = conn
                if connected.node_type == 'epsilon':
                    return ('var', connected.var_name)
        return ('var', '?')

    if node.node_type == 'gamma':
        # Distinguish lambda abstraction vs application
        port1 = node.ports[1]
        port2 = node.ports[2]
        if port1 and port2:
            n1, _ = port1
            n2, _ = port2

            # Lambda if port1 connects to delta node (bound var), else application
            if n1.node_type == 'delta':
                var_expr = sic_to_lambda(n1, visited)
                if var_expr[0] == 'var':
                    var_name = var_expr[1]
                    body_expr = sic_to_lambda(n2, visited)
                    return ('lambda', var_name, body_expr)

            # Otherwise application
            func_expr = sic_to_lambda(n1, visited)
            arg_expr = sic_to_lambda(n2, visited)
            return ('app', func_expr, arg_expr)

        return ('var', '?')

    return ('var', '?')

def lambda_expr_to_str(expr, parent=None):
    kind = expr[0]
    if kind == 'var':
        return expr[1]
    if kind == 'lambda':
        body_str = lambda_expr_to_str(expr[2], parent='lambda')
        return f"λ{expr[1]}.{body_str}"
    if kind == 'app':
        func, arg = expr[1], expr[2]
        func_str = lambda_expr_to_str(func, parent='app_func')
        arg_str = lambda_expr_to_str(arg, parent='app_arg')
        # Parens rules:
        if func[0] == 'lambda':
            func_str = f"({func_str})"
        if arg[0] in ('app', 'lambda'):
            arg_str = f"({arg_str})"
        return f"{func_str} {arg_str}"
    return '?'

def print_graph(graph):
    print("Graph:")
    for node in graph.nodes:
        print(node)

if __name__ == "__main__":
    tests = [
        "λx.x y",
        "(λx.x) y z",
        "λx.λy.x y",
        "a b c",
        "λx.(x) y",
        "λx.x",
        "((λx.(x)) y)",
    ]
    for expr_str in tests:
        print(f"\nInput: {expr_str}")
        parsed = parse_lambda_expr(expr_str)
        print("Parsed AST:", parsed)

        g = Graph()
        root = lambda_to_sic(parsed, g)

        print(f"Generated graph ({len(g.nodes)} nodes):")
        print_graph(g)

        back = sic_to_lambda(root)
        back_str = lambda_expr_to_str(back)
        print("Back-translated lambda:")
        print(back_str)
