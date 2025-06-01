import re

class Node:
    def __init__(self, node_type, id):
        self.node_type = node_type  # 'gamma', 'delta', 'epsilon'
        self.id = id               # Unique identifier for the node
        self.ports = [None, None, None]  # Connections: Principal, Auxiliary 1, Auxiliary 2
        self.var_name = None  # Store variable name (used in delta and epsilon nodes)

    def __repr__(self):
        return f"{self.node_type}{self.id}({self.var_name})"

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

    def remove_node(self, node):
        for i in range(3):
            if node.ports[i]:
                neighbor, port = node.ports[i]
                neighbor.ports[port] = None
        self.nodes.remove(node)

# SIC Interaction Rules
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

# Simulation: Apply Rules Until No More Changes
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

# Parser for Lambda Calculus
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

def unparse_lambda_expr(expr):
    """
    Convert a parsed lambda expression back to a string.
    expr can be:
      - ('var', name)
      - ('lambda', var, body)
      - ('app', func, arg)
    """

    def needs_parens(subexpr):
        # Determine if subexpr should be parenthesized
        return isinstance(subexpr, tuple) and subexpr[0] == 'app'

    if isinstance(expr, tuple):
        tag = expr[0]
        if tag == 'var':
            return expr[1]
        elif tag == 'lambda':
            var = expr[1]
            body = unparse_lambda(expr[2])
            return f"λ{var}.{body}"
        elif tag == 'app':
            func = expr[1]
            arg = expr[2]
            func_str = unparse_lambda(func)
            arg_str = unparse_lambda(arg)
            if needs_parens(func):
                func_str = f"({func_str})"
            if needs_parens(arg):
                arg_str = f"({arg_str})"
            return f"{func_str} {arg_str}"
    else:
        # For plain variables or unexpected cases
        return str(expr)


# Lambda to SIC Translation
def lambda_to_sic(expr, graph, env=None):
    if env is None:
        env = {}

    t = expr[0]

    if t == 'var':
        name = expr[1]
        if name in env:
            return env[name]
        else:
            ep = graph.create_node('epsilon')
            ep.var_name = name
            return ep

    if t == 'lambda':
        var_name = expr[1]
        body = expr[2]

        g = graph.create_node('gamma')
        d = graph.create_node('delta')
        d.var_name = var_name

        new_env = env.copy()
        new_env[var_name] = d

        body_node = lambda_to_sic(body, graph, new_env)

        graph.connect(g, 1, d, 0)
        graph.connect(g, 2, body_node, 0)

        return g

    if t == 'app':
        g = graph.create_node('gamma')
        func_node = lambda_to_sic(expr[1], graph, env)
        arg_node = lambda_to_sic(expr[2], graph, env)
        graph.connect(g, 1, func_node, 0)
        graph.connect(g, 2, arg_node, 0)
        return g

# SIC to Lambda Back-Translation
def sic_to_lambda(node, visited=None):
    if visited is None:
        visited = set()

    if node.node_type == 'gamma':
        if node.id in visited:
            return ('var', '?')
        visited.add(node.id)

    if node.node_type == 'epsilon':
        return ('var', node.var_name or 'free_var')

    if node.node_type == 'delta':
        return ('var', node.var_name or '?')

    if node.node_type == 'gamma':
        n1, _ = node.ports[1]
        n2, _ = node.ports[2]
        if n1.node_type == 'delta':
            var = sic_to_lambda(n1, visited)
            body = sic_to_lambda(n2, visited)
            return ('lambda', var[1], body)
        func = sic_to_lambda(n1, visited)
        arg = sic_to_lambda(n2, visited)
        return ('app', func, arg)

def lambda_expr_to_str(expr):
    t = expr[0]
    if t == 'var':
        return expr[1]
    if t == 'lambda':
        return f"λ{expr[1]}.{lambda_expr_to_str(expr[2])}"
    if t == 'app':
        left = lambda_expr_to_str(expr[1])
        right = lambda_expr_to_str(expr[2])
        if expr[1][0] == 'lambda':
            left = f"({left})"
        if expr[2][0] == 'app' or expr[2][0] == 'lambda':
            right = f"({right})"
        return f"{left} {right}"
    return '?'

    # Define Unary Numbers as Strings
def test_unary_numbers():
    zero = "λf.λx.x"  # 0
    one = "λf.λx.f x"  # 1
    two = "λf.λx.f (f x)"  # 2
    three = "λf.λx.f (f (f x))"  # 3
    increment = "λn.λf.λx.f (n f x)"  # Increment: n + 1
    add = "λn.λm.λf.λx.n f (m f x)"  # Add: n + m
    print("Testing Unary Numbers and Operations...")
    # 1 + 1 = 2
    one_plus_one = f"({add}) ({one}) ({one})"  # Add 1 and 1
    g = lambda_to_sic(parse_lambda_expr(one_plus_one))
    simulate(g)
    print("1 + 1 =", unparse_lambda_expr(sic_to_lambda(g)))
    # # 2 + 3 = 5
    two_plus_three = f"({add}) ({two}) ({three})"  # Add 2 and 3
    g = lambda_to_sic(parse_lambda_expr(two_plus_three))
    simulate(g)
    print("2 + 3 =", unparse_lambda_expr(sic_to_lambda(g)))

    # Increment 2 = 3
    inc_two = f"({increment}) ({two})"  # Increment 2
    g = lambda_to_sic(parse_lambda_expr(inc_two))
    simulate(g)
    print("Increment 2 =", unparse_lambda_expr(sic_to_lambda(g)))

    # # Run Tests
    # test_unary_numbers()

def test_parsing_and_simulation():
    zero_str = "λf.λx.x"  
    one_str = "λf.λx.f x"  
    two_str = "λf.λx.f (f x)"  
    three_str = "λf.λx.f (f (f x))"  

    increment_str = "λn.λf.λx.f (n f x)"  
    add_str = "λn.λm.λf.λx.n f (m f x)"  

    # 2. Expected token structures (AST tuples) for checking parser correctness
    zero_ast = ('lambda', 'f', ('lambda', 'x', ('var', 'x')))
    one_ast = ('lambda', 'f', ('lambda', 'x', ('app', ('var', 'f'), ('var', 'x'))))
    two_ast = ('lambda', 'f', ('lambda', 'x', ('app', ('var', 'f'), ('app', ('var', 'f'), ('var', 'x')))))
    three_ast = ('lambda', 'f', ('lambda', 'x', ('app', ('var', 'f'), ('app', ('var', 'f'), ('app', ('var', 'f'), ('var', 'x'))))))

    increment_ast = ('lambda', 'n', ('lambda', 'f', ('lambda', 'x', ('app', ('var', 'f'), ('app', ('app', ('var', 'n'), ('var', 'f')), ('var', 'x'))))))
    add_ast = ('lambda', 'n', ('lambda', 'm', ('lambda', 'f', ('lambda', 'x', ('app', ('app', ('var', 'n'), ('var', 'f')), ('app', ('app', ('var', 'm'), ('var', 'f')), ('var', 'x')))))))

    test_cases = [
        (zero_str, zero_ast, "zero"),
        (one_str, one_ast, "one"),
        (two_str, two_ast, "two"),
        (three_str, three_ast, "three"),
        (increment_str, increment_ast, "increment"),
        (add_str, add_ast, "add"),
    ]

    for s, expected_ast, name in test_cases:
        print(f"\nTesting parsing for {name}:")
        ast = parse_lambda_expr(s)
        print("Parsed AST:", ast)
        assert ast == expected_ast, f"Parser output does not match expected AST for {name}"

        g = lambda_to_sic(ast)
        simulate(g)
        back_ast = sic_to_lambda(g)
        # print("Back-translated lambda:", unparse_lambda(back_ast))
# test_parsing_and_simulation()


if __name__ == "__main__":
    expr_str = "λx.x (y z)"

    print(f"Input: {expr_str}")
    parsed = parse_lambda_expr(expr_str)
    print("Parsed:", parsed)

    g = Graph()
    root = lambda_to_sic(parsed, g)
    simulate(g)

    back_expr = sic_to_lambda(root)
    back_str = lambda_expr_to_str(back_expr)
    print("Back-translated Lambda:", back_str)
