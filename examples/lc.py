import re

# ─────────────────────────────────────────────────────────────────────────────
# 1) LEXER / PARSER
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_LAMBDA     = 'LAMBDA'
TOKEN_DOT        = 'DOT'
TOKEN_LPAREN     = 'LPAREN'
TOKEN_RPAREN     = 'RPAREN'
TOKEN_IDENTIFIER = 'IDENTIFIER'
TOKEN_EOF        = 'EOF'

def tokenize(expr):
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c == 'λ':
            tokens.append((TOKEN_LAMBDA, c))
            i += 1
        elif c == '.':
            tokens.append((TOKEN_DOT, c))
            i += 1
        elif c == '(':
            tokens.append((TOKEN_LPAREN, c))
            i += 1
        elif c == ')':
            tokens.append((TOKEN_RPAREN, c))
            i += 1
        else:
            match = re.match(r"[a-zA-Z_]\w*", expr[i:])
            if match:
                ident = match.group(0)
                tokens.append((TOKEN_IDENTIFIER, ident))
                i += len(ident)
            else:
                raise SyntaxError(f"Unexpected character: {c}")
    tokens.append((TOKEN_EOF, None))
    return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current = tokens[0]

    def eat(self, token_type):
        if self.current[0] == token_type:
            self.pos += 1
            self.current = self.tokens[self.pos]
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current[0]}")

    def parse(self):
        expr = self.parse_expr()
        if self.current[0] != TOKEN_EOF:
            raise SyntaxError("Unexpected token after expression")
        return expr

    def parse_expr(self):
        term = self.parse_term()
        while self.current[0] in (TOKEN_LAMBDA, TOKEN_IDENTIFIER, TOKEN_LPAREN):
            right = self.parse_term()
            term = ('app', term, right)
        return term

    def parse_term(self):
        if self.current[0] == TOKEN_LAMBDA:
            return self.parse_abstraction()
        elif self.current[0] == TOKEN_IDENTIFIER:
            name = self.current[1]
            self.eat(TOKEN_IDENTIFIER)
            return ('var', name)
        elif self.current[0] == TOKEN_LPAREN:
            self.eat(TOKEN_LPAREN)
            expr = self.parse_expr()
            self.eat(TOKEN_RPAREN)
            return expr
        else:
            raise SyntaxError(f"Unexpected token: {self.current[0]}")

    def parse_abstraction(self):
        self.eat(TOKEN_LAMBDA)
        if self.current[0] != TOKEN_IDENTIFIER:
            raise SyntaxError("Expected identifier after λ")
        var = self.current[1]
        self.eat(TOKEN_IDENTIFIER)
        self.eat(TOKEN_DOT)
        body = self.parse_expr()
        return ('lam', var, body)

def parse_lambda(expr):
    return Parser(tokenize(expr)).parse()

# ─────────────────────────────────────────────────────────────────────────────
# 2) COUNT OCCURRENCES (for computing how many times a var appears)
# ─────────────────────────────────────────────────────────────────────────────

def count_occurrences(var, expr):
    kind = expr[0]
    if kind == 'var':
        return 1 if expr[1] == var else 0
    elif kind == 'lam':
        param, body = expr[1], expr[2]
        if param == var:
            return 0
        return count_occurrences(var, body)
    elif kind == 'app':
        return count_occurrences(var, expr[1]) + count_occurrences(var, expr[2])
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# 3) SIC‐GRAPH CLASSES
# ─────────────────────────────────────────────────────────────────────────────

_node_id_counter = 0
def next_node_id():
    global _node_id_counter
    _node_id_counter += 1
    return str(_node_id_counter)

class Port:
    def __init__(self, node, name):
        self.node = node
        self.name = name   # 'principal', 'aux1', or 'aux2'
        self.connection = None

    def connect(self, other):
        # If either port is already connected, disconnect it first:
        if self.connection:
            self.disconnect()
        if other.connection:
            other.disconnect()
        self.connection = other
        other.connection = self

    def disconnect(self):
        if self.connection:
            other = self.connection
            self.connection = None
            other.connection = None

    def is_active(self):
        return (self.name == 'principal'
                and self.connection
                and self.connection.name == 'principal')

    def __repr__(self):
        if self.connection:
            return f"{self.node.id}.{self.name}→{self.connection.node.id}.{self.connection.name}"
        else:
            return f"{self.node.id}.{self.name}→None"

class Node:
    def __init__(self, node_type, subtype=None, var_name=None):
        self.id = next_node_id()
        self.node_type = node_type    # 'GAMMA', 'DELTA', 'EPSILON', 'VAR'
        self.subtype = subtype        # for 'GAMMA': 'LAMBDA' or 'APPLY'
        self.var_name = var_name      # for VarNode or Γ-LAMBDA
        self.principal = Port(self, 'principal')
        if node_type == 'GAMMA':
            self.aux1 = Port(self, 'aux1')
            self.aux2 = Port(self, 'aux2')
        elif node_type == 'DELTA':
            self.aux1 = Port(self, 'aux1')
            self.aux2 = Port(self, 'aux2')
        else:
            self.aux1 = None
            self.aux2 = None

    def ports(self):
        ps = [self.principal]
        if self.aux1: ps.append(self.aux1)
        if self.aux2: ps.append(self.aux2)
        return ps

    def __repr__(self):
        if self.node_type == 'GAMMA':
            return f"{self.id}:Γ-{self.subtype}-{self.var_name or ''}"
        if self.node_type == 'DELTA':
            return f"{self.id}:Δ"
        if self.node_type == 'EPSILON':
            return f"{self.id}:ε"
        return f"{self.id}:VAR-{self.var_name}"

class VarNode(Node):
    def __init__(self, var_name, bound):
        super().__init__('VAR', var_name=var_name)
        self.bound = bound  # True if bound under some λ, False if free

    def __repr__(self):
        tag = 'B' if self.bound else 'F'
        return f"{self.id}:Var-{tag}-{self.var_name}"

# ─────────────────────────────────────────────────────────────────────────────
# 4) BUILD Δ‐TREE FOR DUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def build_delta_tree(k, graph):
    """
    Build a binary Δ‐tree that yields exactly k leaf ports.
    If k == 1: return (None, None). If k == 2: create one Δ with two leaves.
    Otherwise split into left/right subtrees.
    Returns (delta_root, [leaf_port1, leaf_port2, ..., leaf_portk]).
    """
    if k == 1:
        return (None, None)
    if k == 2:
        d = Node('DELTA')
        graph.append(d)
        return (d, [d.aux1, d.aux2])

    left = k // 2
    right = k - left
    root = Node('DELTA')
    graph.append(root)
    lroot, lleaves = build_delta_tree(left, graph)
    rroot, rleaves = build_delta_tree(right, graph)

    if lroot:
        root.aux1.connect(lroot.principal)
    if rroot:
        root.aux2.connect(rroot.principal)

    leaves = []
    if lroot:
        leaves.extend(lleaves)
    else:
        leaves.append(root.aux1)
    if rroot:
        leaves.extend(rleaves)
    else:
        leaves.append(root.aux2)

    return (root, leaves)

# ─────────────────────────────────────────────────────────────────────────────
# 5) BUILD SIC GRAPH (AST → [Nodes], root_port)
# ─────────────────────────────────────────────────────────────────────────────

def build_sic(expr, graph, env):
    """
    Build the SIC graph for the λ‐expression `expr`.  `env` maps each
    bound variable name to a list of VarNode principals representing its
    occurrences. Free variables (not in `env`) become VarNode(var, False).

    Returns (root_port, occ_map) where:
      - root_port is the Port of this sub‐expression’s “output”
      - occ_map maps var_name → [list of VarNode.principal ports]
    """
    kind = expr[0]

    # (a) VARIABLE
    if kind == 'var':
        name = expr[1]
        bound = (name in env)
        node = VarNode(name, bound)
        graph.append(node)
        return (node.principal, { name: [ node.principal ] })

    # (b) LAMBDA
    if kind == 'lam':
        _, var, body = expr

        # Recurse under a new environment where `var` is bound
        new_env = dict(env)
        new_env[var] = []
        p_body, occ_map = build_sic(body, graph, new_env)

        var_ports = occ_map.get(var, [])
        k = len(var_ports)

        gamma = Node('GAMMA', subtype='LAMBDA', var_name=var)
        graph.append(gamma)

        # (b.1) VAR‐port (aux1)
        if k == 0:
            e = Node('EPSILON')
            graph.append(e)
            gamma.aux1.connect(e.principal)
        elif k == 1:
            # Always share that single occurrence port
            p_occ = var_ports[0]
            gamma.aux1.connect(p_occ)
            gamma.aux2.connect(p_occ)
        else:
            droot, leaves = build_delta_tree(k, graph)
            gamma.aux1.connect(droot.principal)
            for leaf_port, vp in zip(leaves, var_ports):
                leaf_port.connect(vp)

        # (b.2) BODY‐port (aux2)
        if not (k == 1 and gamma.aux2.connection is not None):
            gamma.aux2.connect(p_body)

        new_occ = { v: ports for (v, ports) in occ_map.items() if v != var }
        return (gamma.principal, new_occ)

    # (c) APPLICATION
    if kind == 'app':
        _, f, m = expr
        p_f, occ_f = build_sic(f, graph, env)
        p_m, occ_m = build_sic(m, graph, env)

        gamma = Node('GAMMA', subtype='APPLY')
        graph.append(gamma)

        gamma.principal.connect(p_f)
        gamma.aux1.connect(p_m)

        merged = {}
        for v, ports in occ_f.items():
            merged.setdefault(v, []).extend(ports)
        for v, ports in occ_m.items():
            merged.setdefault(v, []).extend(ports)

        return (gamma.aux2, merged)

    raise RuntimeError("Unknown AST node in build_sic")

def build_graph(ast_expr):
    graph = []
    root_port, _ = build_sic(ast_expr, graph, {})
    return graph, root_port

# ─────────────────────────────────────────────────────────────────────────────
# 6) REDUCTION RULES (Γ–Γ, Δ–Δ, Δ–ε)
# ─────────────────────────────────────────────────────────────────────────────

def reduce_graph(graph):
    def find_active_pairs():
        pairs = []
        for node in list(graph):
            p = node.principal
            q = p.connection
            if q and q.name == 'principal' and p.node != q.node:
                if (q, p) not in pairs:
                    pairs.append((p, q))
        return pairs

    def remove_node(node):
        node.principal.disconnect()
        if node.aux1:
            node.aux1.disconnect()
        if node.aux2:
            node.aux2.disconnect()
        if node in graph:
            graph.remove(node)

    changed = True
    while changed:
        changed = False
        pairs = find_active_pairs()
        if not pairs:
            break

        for (p, q) in pairs:
            n1, n2 = p.node, q.node

            # (1) Γ^LAMBDA – Γ^APPLY annihilation
            if n1.node_type == 'GAMMA' and n2.node_type == 'GAMMA':
                if n1.subtype == 'LAMBDA' and n2.subtype == 'APPLY':
                    lam, app = n1, n2
                elif n1.subtype == 'APPLY' and n2.subtype == 'LAMBDA':
                    lam, app = n2, n1
                else:
                    continue

                varp = lam.aux1.connection
                argp = app.aux1.connection
                bodyp = lam.aux2.connection
                outp  = app.aux2
                varn  = varp.node if varp else None

                # Record successor of APPLY’s aux2
                succ = outp.connection

                # Disconnect principals and aux (except outp)
                lam.principal.disconnect()
                app.principal.disconnect()
                lam.aux1.disconnect()
                app.aux1.disconnect()
                lam.aux2.disconnect()

                # (a) splice argument into var occurrence
                if varp and argp:
                    varp.disconnect()
                    argp.connect(varp)
                    if isinstance(varn, VarNode) and varn.bound:
                        remove_node(varn)

                # (b) splice body into successor
                if bodyp:
                    if succ:
                        bodyp.connect(succ)
                    # else leave as new root

                # Now sever outp → succ
                app.aux2.disconnect()
                remove_node(lam)
                remove_node(app)

                changed = True
                break

            # (2) Δ – Δ annihilation
            if n1.node_type == 'DELTA' and n2.node_type == 'DELTA':
                d1, d2 = n1, n2
                a1, b1 = d1.aux1, d1.aux2
                a2, b2 = d2.aux1, d2.aux2
                d1.principal.disconnect()
                d2.principal.disconnect()
                a1.disconnect(); a2.disconnect(); a1.connect(a2)
                b1.disconnect(); b2.disconnect(); b1.connect(b2)
                remove_node(d1)
                remove_node(d2)
                changed = True
                break

            # (3) Δ – ε erasure
            if ((n1.node_type == 'DELTA' and n2.node_type == 'EPSILON') or
                (n2.node_type == 'DELTA' and n1.node_type == 'EPSILON')):
                d = n1 if n1.node_type == 'DELTA' else n2
                e = n2 if d is n1 else n1
                a, b = d.aux1, d.aux2
                d.principal.disconnect()
                e.principal.disconnect()
                remove_node(d)
                remove_node(e)
                eps1 = Node('EPSILON')
                eps2 = Node('EPSILON')
                graph.append(eps1)
                graph.append(eps2)
                a.disconnect(); a.connect(eps1.principal)
                b.disconnect(); b.connect(eps2.principal)
                changed = True
                break

        # end for pairs

    return

# ─────────────────────────────────────────────────────────────────────────────
# 7) READ‐BACK (SIC Graph → λ‐AST)
# ─────────────────────────────────────────────────────────────────────────────

def find_root_port(graph):
    lambda_ports = []
    used = set()
    for node in graph:
        if node.node_type == 'GAMMA' and node.subtype == 'LAMBDA':
            lambda_ports.append(node.principal)
        if (node.node_type == 'GAMMA' and node.subtype == 'LAMBDA'
            and node.aux2 and node.aux2.connection):
            used.add(node.aux2.connection)
    for p in lambda_ports:
        if p not in used:
            return p

    roots = []
    for n in graph:
        p = n.principal
        if p.connection is None or p.connection.name != 'principal':
            roots.append(p)
    if not roots:
        return None
    for p in roots:
        if isinstance(p.node, VarNode) and not p.node.bound:
            return p
    for p in roots:
        if p.node.node_type == 'GAMMA' and p.node.subtype == 'LAMBDA':
            return p
    return roots[0]

def read_back(port, visited=None):
    if visited is None:
        visited = set()
    node = port.node
    if node in visited:
        return None
    visited.add(node)

    if node.node_type == 'VAR':
        return ('var', node.var_name)

    if node.node_type == 'GAMMA' and node.subtype == 'LAMBDA':
        var = node.var_name
        body_ast = None
        if node.aux2 and node.aux2.connection:
            body_ast = read_back(node.aux2.connection, visited)
        return ('lam', var, body_ast)

    if node.node_type == 'GAMMA' and node.subtype == 'APPLY':
        f_p = node.principal.connection
        a_p = node.aux1.connection
        if not f_p or not a_p:
            return None
        return ('app', read_back(f_p, visited), read_back(a_p, visited))

    return None

# ─────────────────────────────────────────────────────────────────────────────
# 8) TEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

def sic_reduce(expr_str):
    ast = parse_lambda(expr_str)
    graph, root_port = build_graph(ast)
    reduce_graph(graph)
    root = find_root_port(graph)
    return read_back(root)

def ast_to_str(node):
    if node is None:
        return "None"
    if node[0] == 'var':
        return node[1]
    if node[0] == 'lam':
        return f"λ{node[1]}.{ast_to_str(node[2])}"
    if node[0] == 'app':
        return f"({ast_to_str(node[1])} {ast_to_str(node[2])})"
    return "?"

def ast_equal(a, b):
    if a is None or b is None:
        return a == b
    if a[0] != b[0]:
        return False
    if a[0] == 'var':
        return a[1] == b[1]
    if a[0] == 'lam':
        return a[1] == b[1] and ast_equal(a[2], b[2])
    if a[0] == 'app':
        return ast_equal(a[1], b[1]) and ast_equal(a[2], b[2])
    return False

def print_result(name, got, expected):
    gs = ast_to_str(got)
    es = ast_to_str(expected)
    if ast_equal(got, expected):
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}")
        print(f"  → Got     : {gs}")
        print(f"  → Expected: {es}")
    print()

def test_identity_y():
    name = "Identity (λx.x) y"
    got = sic_reduce("(λx. x) y")
    expected = ('var', 'y')
    print_result(name, got, expected)

def test_identity_K_ab():
    name = "K combinator ((λx.λy.x) a) b"
    got = sic_reduce("((λx. λy. x) a) b")
    expected = ('var', 'a')
    print_result(name, got, expected)

def test_true_false_noop():
    name1 = "No-op true"
    name2 = "No-op false"
    true_str  = "λt. λf. t"
    false_str = "λt. λf. f"
    got_t = sic_reduce(true_str)
    got_f = sic_reduce(false_str)
    expected_t = parse_lambda(true_str)
    expected_f = parse_lambda(false_str)
    print_result(name1, got_t, expected_t)
    print_result(name2, got_f, expected_f)

def test_true_and_true():
    name = "true AND true"
    true_str = "λt. λf. t"
    and_str  = "λb. λc. b c (λt. λf. f)"
    expr     = f"(({and_str}) {true_str}) {true_str}"
    got = sic_reduce(expr)
    expected = parse_lambda(true_str)
    print_result(name, got, expected)

if __name__ == "__main__":
    print("== Tests ==\n")
    test_identity_y()
    test_identity_K_ab()
    test_true_false_noop()
    test_true_and_true()
