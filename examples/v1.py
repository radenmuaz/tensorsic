import re
import uuid
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# 1) LEXER / PARSER (unchanged from your starting point)
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
        while True:
            if self.current[0] in (TOKEN_LAMBDA, TOKEN_IDENTIFIER, TOKEN_LPAREN):
                right = self.parse_term()
                term = ('app', term, right)
            else:
                break
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
    tokens = tokenize(expr)
    parser = Parser(tokens)
    return parser.parse()

# ─────────────────────────────────────────────────────────────────────────────
# 2) COUNT FREE/BIND Occurrences (two small helpers)
# ─────────────────────────────────────────────────────────────────────────────

def free_vars(expr):
    kind = expr[0]
    if kind == 'var':
        return {expr[1]}
    elif kind == 'lam':
        param, body = expr[1], expr[2]
        return free_vars(body) - {param}
    elif kind == 'app':
        return free_vars(expr[1]) | free_vars(expr[2])
    return set()

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
# 3) CORE SIC‐GRAPH DATA STRUCTURES
#    - Port:   Has a single “connection” pointer to another Port
#    - Node:   Has 1 principal port + up to 2 aux ports
#    - VarNode: distinguishes bound‐vs‐free by a flag
#    - Gamma, Delta, Epsilon: constructed via Node(…)
# ─────────────────────────────────────────────────────────────────────────────

class Port:
    def __init__(self, node, name):
        self.node = node      # back‐pointer to the parent Node
        self.name = name      # 'principal', 'aux1', or 'aux2'
        self.connection = None

    def connect(self, other):
        # Wire this port ↔ other port
        self.connection = other
        other.connection = self

    def disconnect(self):
        if self.connection:
            other = self.connection
            self.connection = None
            other.connection = None

    def is_active(self):
        # Active iff a principal‐port connected to another principal
        return (self.name == 'principal' 
                and self.connection 
                and self.connection.name == 'principal')

    def __repr__(self):
        return f"Port({self.node}, {self.name})"

class Node:
    def __init__(self, node_type, subtype=None, var_name=None):
        self.id = str(uuid.uuid4())[:8]
        self.node_type = node_type    # 'GAMMA', 'DELTA', 'EPSILON', 'VAR'
        self.subtype = subtype        # for GAMMA: 'LAMBDA' or 'APPLY'
        self.var_name = var_name      # for GAMMA‐LAMBDA (the λ‐binder name)
        # Always have a principal port
        self.principal = Port(self, 'principal')
        # Add aux ports only if needed
        if node_type == 'GAMMA':
            self.aux1 = Port(self, 'aux1')
            self.aux2 = Port(self, 'aux2')
        elif node_type == 'DELTA':
            self.aux1 = Port(self, 'aux1')
            self.aux2 = Port(self, 'aux2')
        else:  # EPSILON or VAR
            self.aux1 = None
            self.aux2 = None

    def __repr__(self):
        if self.node_type == 'GAMMA':
            return f"Node(Γ-{self.subtype}-{self.var_name or ''})"
        elif self.node_type == 'DELTA':
            return "Node(Δ)"
        elif self.node_type == 'EPSILON':
            return "Node(ε)"
        else:  # VAR
            return f"Node(VAR-{self.var_name})"

class VarNode(Node):
    def __init__(self, var_name, bound):
        super().__init__('VAR', var_name=var_name)
        self.bound = bound  # True if bound w.r.t. some λ, False if “free” in the original program

    def __repr__(self):
        return f"VarNode({'B' if self.bound else 'F'}-{self.var_name})"


# ─────────────────────────────────────────────────────────────────────────────
# 4) BUILDING A DELTA‐TREE for “k > 1” copies of a bound variable
#    so we can fan out to exactly k occurrences
# ─────────────────────────────────────────────────────────────────────────────

def build_delta_tree(k, graph):
    """
    If k == 1, no δ‐node is needed: return (None, [“direct leaf”]).
    If k == 2, create a single δ with 2 outputs.
    If k > 2, build a small binary tree of Δ‐nodes so that exactly k "leaf ports" appear.
    Returns (δ_root, leaf_ports_list).
    """
    if k == 1:
        return (None, None)

    if k == 2:
        delta = Node('DELTA')
        graph.append(delta)
        return (delta, [delta.aux1, delta.aux2])

    # Split k into two parts: left_count = k//2, right_count = k - left_count
    left_count = k // 2
    right_count = k - left_count
    root = Node('DELTA')
    graph.append(root)

    # Build subtrees
    left_root, left_leaves = build_delta_tree(left_count, graph)
    right_root, right_leaves = build_delta_tree(right_count, graph)

    # Wire root’s aux‐ports
    if left_root is None:
        # left_count == 1: use root.aux1 directly as a leaf
        pass
    else:
        root.aux1.connect(left_root.principal)

    if right_root is None:
        # right_count == 1
        pass
    else:
        root.aux2.connect(right_root.principal)

    # Collect leaves:
    leaves = []
    if left_root is None:
        # root.aux1 is a leaf‐port
        leaves.append(root.aux1)
    else:
        leaves.extend(left_leaves)

    if right_root is None:
        leaves.append(root.aux2)
    else:
        leaves.extend(right_leaves)

    return (root, leaves)


# ─────────────────────────────────────────────────────────────────────────────
# 5) THE ACTUAL build_sic FUNCTION (AST → SIC GRAPH)
#
#    We carry around an “env = set of currently‐bound λ‐variables.”
#
#    Returns: (root_port, occ_map) where
#       – root_port is the single “output” port for this subtree
#       – occ_map maps each bound‐var‐name → [list of Ports]
#         representing exactly those occurrences (un‐under any inner λ for that name).
# ─────────────────────────────────────────────────────────────────────────────

def build_sic(expr, graph, env):
    kind = expr[0]

    # 5.1) VARIABLE LEAF
    if kind == 'var':
        var_name = expr[1]
        if var_name in env:
            node = VarNode(var_name, bound=True)
        else:
            node = VarNode(var_name, bound=False)
        graph.append(node)
        # This is a single occurrence of var_name
        return node.principal, {var_name: [node.principal]}

    # 5.2) LAMBDA
    elif kind == 'lam':
        _, var, body = expr
        # Build the body first, adding var to the bound‐env
        new_env = set(env)
        new_env.add(var)
        p_body, occ_map_body = build_sic(body, graph, new_env)

        # How many “free occurrences” of var are in body (not shadowed by deeper lambdas)?
        var_occ_ports = occ_map_body.get(var, [])
        k = len(var_occ_ports)

        # Create a λ‐Γ node
        gamma = Node('GAMMA', subtype='LAMBDA', var_name=var)
        graph.append(gamma)

        # 5.2.1) VAR‐PORT (aux1) handling
        if k == 0:
            eps = Node('EPSILON')
            graph.append(eps)
            gamma.aux1.connect(eps.principal)

        elif k == 1:
            gamma.aux1.connect(var_occ_ports[0])

        else:
            # k > 1: build a Δ‐tree so that we have k leaves
            delta_root, leaves = build_delta_tree(k, graph)
            gamma.aux1.connect(delta_root.principal)
            for leaf_port, occ_port in zip(leaves, var_occ_ports):
                leaf_port.connect(occ_port)

        # 5.2.2) BODY‐PORT (aux2)
        gamma.aux2.connect(p_body)

        # Remove var from the occ_map for children before returning upward
        new_occ_map = {v: ps for v, ps in occ_map_body.items() if v != var}
        return gamma.principal, new_occ_map

    # 5.3) APPLICATION
    elif kind == 'app':
        _, f, m = expr

        p_f, occ_f = build_sic(f, graph, env)
        p_m, occ_m = build_sic(m, graph, env)

        gamma = Node('GAMMA', subtype='APPLY')
        graph.append(gamma)

        # Connect principal (Function) ↔ p_f
        gamma.principal.connect(p_f)

        # Connect argument
        gamma.aux1.connect(p_m)

        # Merge occurrence‐maps
        merged = {}
        for v, ports in occ_f.items():
            merged.setdefault(v, []).extend(ports)
        for v, ports in occ_m.items():
            merged.setdefault(v, []).extend(ports)

        # This application’s “result” is gamma.aux2 (dangling)
        return gamma.aux2, merged

    else:
        raise RuntimeError("Unknown AST node type in build_sic")

def build_graph(ast_expr):
    graph = []
    root_port, occ_map = build_sic(ast_expr, graph, set())
    return graph, root_port

# ─────────────────────────────────────────────────────────────────────────────
# 6) SIC REDUCTION:  “FIND ACTIVE PAIRS” + “FIRE THE 6 RULES”
#
#    We only need (ΓΛ‐Γ@), (Δ‐Δ), and (Δ‐ε / ε‐Δ) for ordinary λ‐calculus.
#    (We never introduce ζ, and we never intentionally “unfold” ε‐ε→δ/ζ.)
# ─────────────────────────────────────────────────────────────────────────────

def reduce_graph(graph):
    def find_active_pairs():
        pairs = []
        for node in list(graph):
            p = node.principal
            q = p.connection
            if q and q.name == 'principal' and p.node != q.node:
                # only record once (p,q) or (q,p)
                if (q, p) not in pairs:
                    pairs.append((p, q))
        return pairs

    def remove_node(node):
        # Disconnect all ports, then remove from graph list
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

        for p, q in pairs:
            n1, n2 = p.node, q.node

            # 6.1) ΓΛ – Γ@ annihilation
            if n1.node_type == 'GAMMA' and n2.node_type == 'GAMMA':
                if n1.subtype == 'LAMBDA' and n2.subtype == 'APPLY':
                    lam, app = n1, n2
                elif n1.subtype == 'APPLY' and n2.subtype == 'LAMBDA':
                    lam, app = n2, n1
                else:
                    continue  # not a reducible λ‐@ pair

                # The “var” port on lam is lam.aux1; the “arg” port on app is app.aux1.
                var_branch_port = lam.aux1.connection    # E.g. a BVarNode‐port
                arg_branch_port = app.aux1.connection    # E.g. VarNode (free or BVar) for the argument
                body_branch_port = lam.aux2.connection   # entire subgraph of M
                out_port = app.aux2                      # “where (F M)'s result flows”

                # If there was a bound‐var node in the graph, remove it after splicing
                var_node = var_branch_port.node if var_branch_port else None

                # Disconnect the two principals
                lam.principal.disconnect()
                app.principal.disconnect()
                # Disconnect the aux links
                lam.aux1.disconnect()
                app.aux1.disconnect()
                lam.aux2.disconnect()
                app.aux2.disconnect()

                #  a) Rewire “argument” into “var occurrence”
                if var_branch_port and arg_branch_port:
                    # Connect var_occurrence_port ↔ argument_port
                    arg_branch_port.connect(var_branch_port)

                    # If that var_node was a bound‐var “placeholder,” remove it now
                    if isinstance(var_node, VarNode) and var_node.bound:
                        remove_node(var_node)

                #  b) Rewire “body” into “output”
                if body_branch_port:
                    body_branch_port.connect(out_port)

                # Finally remove the two Γ‐nodes
                remove_node(lam)
                remove_node(app)

                changed = True
                break

            # 6.2) Δ – Δ (annihilation)
            if n1.node_type == 'DELTA' and n2.node_type == 'DELTA':
                d1, d2 = n1, n2
                a1, b1 = d1.aux1, d1.aux2
                a2, b2 = d2.aux1, d2.aux2
                # Disconnect principals
                d1.principal.disconnect()
                d2.principal.disconnect()
                # Splice: a1↔a2, b1↔b2
                a1.disconnect(); a2.disconnect(); a1.connect(a2)
                b1.disconnect(); b2.disconnect(); b1.connect(b2)
                remove_node(d1)
                remove_node(d2)
                changed = True
                break

            # 6.3) Δ – ε  (or ε – Δ) (erasure)
            if ((n1.node_type == 'DELTA' and n2.node_type == 'EPSILON') or
                (n2.node_type == 'DELTA' and n1.node_type == 'EPSILON')):
                d = n1 if n1.node_type == 'DELTA' else n2
                e = n2 if d is n1 else n1
                a, b = d.aux1, d.aux2
                # Disconnect principals
                d.principal.disconnect()
                e.principal.disconnect()
                # Remove Δ & ε
                remove_node(d)
                remove_node(e)
                # For each of Δ’s two auxiliary ports, attach a fresh ε
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
# 7) READ‐BACK a LAMBDA‐AST from a (fully‐reduced) SIC graph
#
#    We look for a single “root” principal that is not connected to any other principal.
#    Then:
#      – If it’s a VarNode(bound=False), return ('var', name).
#      – If it’s a Γ‐LAMBDA, reconstruct ('lam', var, body).
#      – Otherwise fail / return None.
# ─────────────────────────────────────────────────────────────────────────────

def find_root_port(graph):
    # Collect all “dangling principals” (i.e. principal ports not connected to another principal).
    roots = []
    for node in graph:
        p = node.principal
        if p.connection is None or p.connection.name != 'principal':
            roots.append(p)
    if not roots:
        return None
    # Prefer a free VarNode if present
    for p in roots:
        if isinstance(p.node, VarNode) and not p.node.bound:
            return p
    # Otherwise prefer a gamma‐lambda
    for p in roots:
        if p.node.node_type == 'GAMMA' and p.node.subtype == 'LAMBDA':
            return p
    # Else just take the first
    return roots[0]

def read_back(port, visited=None):
    if visited is None:
        visited = set()
    node = port.node
    if node in visited:
        return None
    visited.add(node)

    if node.node_type == 'VAR':
        # Whether bound or not, we emit ('var', var_name)
        return ('var', node.var_name)

    if node.node_type == 'GAMMA' and node.subtype == 'LAMBDA':
        var = node.var_name
        if node.aux2.connection:
            body_ast = read_back(node.aux2.connection, visited)
        else:
            body_ast = None
        return ('lam', var, body_ast)

    if node.node_type == 'GAMMA' and node.subtype == 'APPLY':
        # In a “fully‐reduced” graph we actually shouldn’t see any Γ@ left,
        # but if we do, read it as ('app', …).
        f_port = node.principal.connection
        a_port = node.aux1.connection
        if not f_port or not a_port:
            return None
        func_ast = read_back(f_port, visited)
        arg_ast  = read_back(a_port, visited)
        return ('app', func_ast, arg_ast)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# 8) A VERY SMALL TEST SUITE
#    We confirm that our SIC reduction agrees with β‐reduction on two simple examples:
#
#      A)  (λx. x) y   ⇒  y
#      B)  ((λx. λy. x) a) b  ⇒  a
#
#    If you want more “church‐numeral” testing, you can extend this in the same style.
# ─────────────────────────────────────────────────────────────────────────────

def test_sic_identity():
    expr = "(λx. x) y"
    ast = parse_lambda(expr)
    graph, _ = build_graph(ast)
    reduce_graph(graph)
    root_port = find_root_port(graph)
    final_ast = read_back(root_port)
    assert final_ast == ('var', 'y'), f"Expected ('var','y'), got {final_ast}"
    print("✔ SIC identity test passed: (λx.x) y  ⇒  y")

def test_sic_simple_app():
    # ((λx. λy. x) a) b  ⇒  a
    expr = "((λx. λy. x) a) b"
    ast = parse_lambda(expr)
    graph, _ = build_graph(ast)
    reduce_graph(graph)
    root_port = find_root_port(graph)
    final_ast = read_back(root_port)
    assert final_ast == ('var', 'a'), f"Expected ('var','a'), got {final_ast}"
    print("✔ SIC simple‐app test passed: ((λx.λy.x) a) b  ⇒  a")

def test_sic_simple():
    test_sic_identity()
    test_sic_simple_app()

if __name__ == "__main__":
    test_sic_simple()
