#!/usr/bin/env python3
import re

# ——————————————————————————————————————————————
# NODE + GRAPH CLASSES
# ——————————————————————————————————————————————
class Node:
    def __init__(self, node_type, id):
        self.node_type = node_type   # 'gamma', 'delta', or 'epsilon'
        self.id = id                 # unique integer
        # ports: [principal, aux1, aux2]
        #   • γ(gamma):   ports[0]=principal, ports[1]=“left” child, ports[2]=“right” child
        #   • δ(delta):   ports[0]=principal, ports[1]&ports[2]=its two auxiliary‐slots
        #   • ε(epsilon): ports[0]=principal (if hooked up), no aux1/aux2
        self.ports = [None, None, None]
        self.var_name = None        # used for delta/epsilon to record “which variable”

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
        """
        Connect node1.ports[port1] ↔ node2.ports[port2].
        """
        node1.ports[port1] = (node2, port2)
        node2.ports[port2] = (node1, port1)

    def remove_node(self, node):
        """
        Unhook all of node’s ports and remove it from self.nodes.
        """
        for i in range(3):
            if node.ports[i]:
                nbr, p = node.ports[i]
                nbr.ports[p] = None
        self.nodes.remove(node)


# ——————————————————————————————————————————————
# LAMBDA PARSER + PRINTER (unchanged)
# ——————————————————————————————————————————————
def tokenize(expr_str):
    token_spec = [
        ('LAMBDA', r'[λ\\]'),
        ('DOT',    r'\.'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('VAR',    r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('SPACE',  r'\s+'),
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pat})' for name, pat in token_spec)
    for m in re.finditer(tok_regex, expr_str):
        kind = m.lastgroup
        val = m.group()
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
            return ('var', consume('VAR'))
        else:
            raise SyntaxError("Expected variable")

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
            if kind in ('VAR', 'LPAREN', 'LAMBDA'):
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
        if expr[2][0] in ('app', 'lam'):
            right = f"({right})"
        return f"{left} {right}"
    return '?'


# ——————————————————————————————————————————————
# LAMBDA → SIC TRANSLATION (with ERASER insertion)
# ——————————————————————————————————————————————
def lambda_to_sic(expr, graph, env=None):
    """
    Translate a λ‐calculus AST into a SIC graph, inserting “eraser” ε‐nodes whenever
    a δ ends up with fewer than two real occurrences of its variable.

    env: maps var_name → [delta_node, next_aux_port_to_use]
         next_aux_port_to_use ∈ {1, 2, 3}.  It starts at 1, is bumped to 2 after
         the first real occurrence, to 3 after the second.

    Cases:
      • ('var', x): if x ∈ env, attach a fresh ε(x) to δ(x) at the next port.
                     Otherwise create a free ε(x).
      • ('lam', x, M): create γ and δ, set env[x]=[δ,1], recurse on M,
                       then insert 0/1/2 eraser‐ε's so δ has exactly two children.
      • ('app', M, N): create γ, hook γ.p1 ← SIC(M).p0, γ.p2 ← SIC(N).p0.
    """
    if env is None:
        env = {}

    t = expr[0]
    if t == 'var':
        name = expr[1]
        if name in env:
            d, next_port = env[name]
            if next_port > 2:
                raise ValueError(f"Too many occurrences of '{name}'. A single δ only has two ports.")
            ep = graph.create_node('epsilon')
            ep.var_name = name
            graph.connect(d, next_port, ep, 0)
            env[name][1] = next_port + 1
            return ep
        else:
            # free variable → ε(name) alone
            ep = graph.create_node('epsilon')
            ep.var_name = name
            return ep

    if t == 'lam':
        var_name = expr[1]
        body = expr[2]

        g = graph.create_node('gamma')
        d = graph.create_node('delta')
        d.var_name = var_name

        # Copy outer environment so outer binders stay intact
        new_env = {v: [env[v][0], env[v][1]] for v in env}
        new_env[var_name] = [d, 1]

        body_node = lambda_to_sic(body, graph, new_env)

        # γ.p1 ↔ δ.p0  and  γ.p2 ↔ body_node.p0
        graph.connect(g, 1, d, 0)
        graph.connect(g, 2, body_node, 0)

        used_up_to = new_env[var_name][1]
        if used_up_to == 1:
            # zero real occurrences → insert TWO eraser‐ε's
            e1 = graph.create_node('epsilon'); e1.var_name = var_name
            e2 = graph.create_node('epsilon'); e2.var_name = var_name
            graph.connect(d, 1, e1, 0)
            graph.connect(d, 2, e2, 0)
        elif used_up_to == 2:
            # exactly one real occurrence → insert ONE eraser at δ.p2
            e2 = graph.create_node('epsilon'); e2.var_name = var_name
            graph.connect(d, 2, e2, 0)
        # if used_up_to == 3, two real occurrences → no extra ε needed

        return g

    if t == 'app':
        g = graph.create_node('gamma')
        f_node = lambda_to_sic(expr[1], graph, env)
        a_node = lambda_to_sic(expr[2], graph, env)
        graph.connect(g, 1, f_node, 0)
        graph.connect(g, 2, a_node, 0)
        return g

    raise ValueError("Unknown AST node")


# ——————————————————————————————————————————————
# AUXILIARY REWIRING HELPERS
# ——————————————————————————————————————————————
def reconnect_auxiliary_ports(n1, n2, graph):
    """
    When two nodes of the same type (δ–δ or γ–γ) meet on their principal ports,
    we reconnect their auxiliaries in cross: p1↔p2 and p2↔p1.
    """
    for aux_idx in (1, 2):
        if n1.ports[aux_idx] and n2.ports[aux_idx]:
            c1, p1 = n1.ports[aux_idx]
            c2, p2 = n2.ports[aux_idx]
            graph.connect(c1, p2, c2, p1)


# ——————————————————————————————————————————————
# SIC REDUCTION: SIX SYMMETRIC RULES (with ERASER‐ε handling)
# ——————————————————————————————————————————————
def apply_rules(n1, n2, graph):
    """
    Attempt one of the six SIC rules on the connected pair (n1, n2).
    Return True if a rule was applied, False otherwise.

    1) ε – ε annihilation (principal 0↔0).
    2) δ – ε annihilation (principal 0↔0).
    3) δ – δ annihilation (principal 0↔0): cross-wire auxiliaries.
    4) γ – γ annihilation (principal 0↔0): cross-wire auxiliaries.
    5) δ – γ commutation when δ.p0 ↔ γ.p1.
    6) γ – δ commutation when γ.p1 ↔ δ.p0.
    """
    # 1) ε – ε ANNIHILATION
    if n1.node_type == 'epsilon' and n2.node_type == 'epsilon':
        if (n1.ports[0] and n1.ports[0][0] is n2 and n1.ports[0][1] == 0):
            reconnect_auxiliary_ports(n1, n2, graph)
            graph.remove_node(n1)
            graph.remove_node(n2)
            return True
        if (n2.ports[0] and n2.ports[0][0] is n1 and n2.ports[0][1] == 0):
            reconnect_auxiliary_ports(n1, n2, graph)
            graph.remove_node(n1)
            graph.remove_node(n2)
            return True

    # 2) δ – ε ANNIHILATION
    if n1.node_type == 'delta' and n2.node_type == 'epsilon':
        if (n1.ports[0] and n1.ports[0][0] is n2 and n1.ports[0][1] == 0):
            # Unhook δ’s auxiliaries
            for i in (1, 2):
                if n1.ports[i]:
                    child, child_port = n1.ports[i]
                    child.ports[child_port] = None
            graph.remove_node(n2)
            graph.remove_node(n1)
            return True
    if n2.node_type == 'delta' and n1.node_type == 'epsilon':
        if (n2.ports[0] and n2.ports[0][0] is n1 and n2.ports[0][1] == 0):
            for i in (1, 2):
                if n2.ports[i]:
                    child, child_port = n2.ports[i]
                    child.ports[child_port] = None
            graph.remove_node(n1)
            graph.remove_node(n2)
            return True

    # 3) δ – δ ANNIHILATION
    if n1.node_type == 'delta' and n2.node_type == 'delta':
        if (n1.ports[0] and n1.ports[0][0] is n2 and n1.ports[0][1] == 0) or \
           (n2.ports[0] and n2.ports[0][0] is n1 and n2.ports[0][1] == 0):
            reconnect_auxiliary_ports(n1, n2, graph)
            graph.remove_node(n1)
            graph.remove_node(n2)
            return True

    # 4) γ – γ ANNIHILATION
    if n1.node_type == 'gamma' and n2.node_type == 'gamma':
        if (n1.ports[0] and n1.ports[0][0] is n2 and n1.ports[0][1] == 0) or \
           (n2.ports[0] and n2.ports[0][0] is n1 and n2.ports[0][1] == 0):
            reconnect_auxiliary_ports(n1, n2, graph)
            graph.remove_node(n1)
            graph.remove_node(n2)
            return True

    # 5) δ – γ COMMUTATION (δ.p0 ↔ γ.p1)
    if n1.node_type == 'delta' and n2.node_type == 'gamma':
        if (n1.ports[0] and n1.ports[0][0] is n2 and n1.ports[0][1] == 1 and n2.ports[0]):
            old_d = n1
            parentP, parentPort = n2.ports[0]
            Ainfo = n1.ports[1]
            Binfo = n1.ports[2]
            Rinfo = n2.ports[2]

            # Unhook both δ and γ
            for v in (n1, n2):
                for pidx in (0, 1, 2):
                    if v.ports[pidx]:
                        nbr, nbrport = v.ports[pidx]
                        nbr.ports[nbrport] = None
            graph.remove_node(n1)
            graph.remove_node(n2)

            # Create new δ' and two γ's
            new_delta = graph.create_node('delta')
            new_delta.var_name = old_d.var_name
            gamma1 = graph.create_node('gamma')
            gamma2 = graph.create_node('gamma')

            # Splice new_delta into the old γ’s parent
            parentP.ports[parentPort] = (new_delta, 0)
            new_delta.ports[0] = (parentP, parentPort)

            # δ'.p1 ↔ γ1.p0, δ'.p2 ↔ γ2.p0
            graph.connect(new_delta, 1, gamma1, 0)
            graph.connect(new_delta, 2, gamma2, 0)

            # Hook γ1.p1 ← Ainfo
            if Ainfo:
                a_node, a_port = Ainfo
                a_node.ports[a_port] = (gamma1, 1)
                gamma1.ports[1] = (a_node, a_port)

            # Hook γ2.p1 ← Binfo
            if Binfo:
                b_node, b_port = Binfo
                b_node.ports[b_port] = (gamma2, 1)
                gamma2.ports[1] = (b_node, b_port)

            # Hook γ2.p2 ← Rinfo
            if Rinfo:
                r_node, r_port = Rinfo
                r_node.ports[r_port] = (gamma2, 2)
                gamma2.ports[2] = (r_node, r_port)

            return True

    # 6) γ – δ COMMUTATION (γ.p1 ↔ δ.p0)
    if n1.node_type == 'gamma' and n2.node_type == 'delta':
        if (n2.ports[0] and n2.ports[0][0] is n1 and n2.ports[0][1] == 1 and n1.ports[0]):
            old_d = n2
            parentP, parentPort = n1.ports[0]
            Ainfo = n2.ports[1]
            Binfo = n2.ports[2]
            Rinfo = n1.ports[2]

            # Unhook both γ and δ
            for v in (n1, n2):
                for pidx in (0, 1, 2):
                    if v.ports[pidx]:
                        nbr, nbrport = v.ports[pidx]
                        nbr.ports[nbrport] = None
            graph.remove_node(n1)
            graph.remove_node(n2)

            # Create δ' and two γ's
            new_delta = graph.create_node('delta')
            new_delta.var_name = old_d.var_name
            gamma1 = graph.create_node('gamma')
            gamma2 = graph.create_node('gamma')

            parentP.ports[parentPort] = (new_delta, 0)
            new_delta.ports[0] = (parentP, parentPort)

            graph.connect(new_delta, 1, gamma1, 0)
            graph.connect(new_delta, 2, gamma2, 0)

            if Ainfo:
                a_node, a_port = Ainfo
                a_node.ports[a_port] = (gamma1, 1)
                gamma1.ports[1] = (a_node, a_port)

            if Binfo:
                b_node, b_port = Binfo
                b_node.ports[b_port] = (gamma2, 1)
                gamma2.ports[1] = (b_node, b_port)

            if Rinfo:
                r_node, r_port = Rinfo
                r_node.ports[r_port] = (gamma2, 2)
                gamma2.ports[2] = (r_node, r_port)

            return True

    # No rule matched
    return False


def simulate(graph):
    """
    Repeatedly scan all nodes for any connected pair (n, m) where apply_rules(n, m)
    returns True. After a successful rule, restart scanning from scratch. Stop when
    no more rules apply.
    """
    while True:
        changed = False
        for n in list(graph.nodes):
            for port_idx in range(3):
                if n.ports[port_idx]:
                    m, _ = n.ports[port_idx]
                    if apply_rules(n, m, graph):
                        changed = True
                        break
            if changed:
                break
        if not changed:
            return


# ——————————————————————————————————————————————
# SIC → LAMBDA RECONSTRUCTION (unchanged)
# ——————————————————————————————————————————————
def sic_to_lambda(node, visited_ids=None):
    """
    From a reduced SIC graph, rebuild a λ‐term:
      • ε(x) ↦ ('var', x).
      • If a γ whose p1 points at δ = d, and that δ’s p0 points back to that same γ at port=1,
        then treat this γ as (λ d.var_name . body), where body = sic_to_lambda(γ’s p2 child).
      • Otherwise γ is (func arg), taken from ports 1 and 2 recursively.
      • A stray δ is treated as ('var', δ.var_name).
    """
    if visited_ids is None:
        visited_ids = set()
    if node.id in visited_ids:
        return ('var', '?')
    visited_ids.add(node.id)

    if node.node_type == 'epsilon':
        return ('var', node.var_name or 'free_var')

    if node.node_type == 'delta':
        return ('var', node.var_name or '?')

    if node.node_type == 'gamma':
        left_edge = node.ports[1]
        right_edge = node.ports[2]

        # λ‐abstraction pattern: γ.p1 ↔ δ = d, and d.p0 ↔ γ at port=1
        if left_edge and left_edge[0].node_type == 'delta':
            d = left_edge[0]
            if d.ports[0] == (node, 1):
                var = d.var_name
                body = sic_to_lambda(right_edge[0], visited_ids) if right_edge else ('var', '?')
                return ('lam', var, body)

        # Otherwise γ is application
        func_sub = sic_to_lambda(left_edge[0], visited_ids) if left_edge else ('var', '?')
        arg_sub = sic_to_lambda(right_edge[0], visited_ids) if right_edge else ('var', '?')
        return ('app', func_sub, arg_sub)

    return ('var', '?')


# ——————————————————————————————————————————————
# TEST SUITE: ensure “(increment one) ≡ two” and “(increment zero) ≡ one”
# ——————————————————————————————————————————————
if __name__ == "__main__":
    zero      = "λf.λx.x"
    one       = "λf.λx.f x"
    two       = "λf.λx.f (f x)"
    increment = "λn.λf.λx.f (n f x)"

    # 1) Test (increment one) → two
    parsed_inc      = parse_lambda_expr(increment)
    parsed_one      = parse_lambda_expr(one)
    parsed_incr_one = parse_lambda_expr(f"({increment}) ({one})")

    g1 = Graph()
    root1 = lambda_to_sic(parsed_incr_one, g1)
    simulate(g1)
    result_ast1 = sic_to_lambda(root1)
    result_str1 = lambda_expr_to_str(result_ast1)
    print("Expected two   =", two)
    print("Result of (increment one) →", result_str1)
    assert result_str1 == two, f"FAIL: got `{result_str1}` instead of `{two}`"
    print("✔︎  (increment one)  ≡ two\n")

    # 2) Test (increment zero) → one
    parsed_incr_zero = parse_lambda_expr(f"({increment}) ({zero})")
    g2 = Graph()
    root2 = lambda_to_sic(parsed_incr_zero, g2)
    simulate(g2)
    result_ast2 = sic_to_lambda(root2)
    result_str2 = lambda_expr_to_str(result_ast2)
    print("Expected one   =", one)
    print("Result of (increment zero) →", result_str2)
    assert result_str2 == one, f"FAIL: got `{result_str2}` instead of `{one}`"
    print("✔︎  (increment zero)  ≡ one")
