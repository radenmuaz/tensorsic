#!/usr/bin/env python3
import re
import string

# ——————————————————————————————————————————————
# TOKENIZER + PARSER (unchanged)
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
# FREE VARIABLES, ALPHA-RENAMING, SUBSTITUTION
# ——————————————————————————————————————————————
def free_vars(expr):
    t = expr[0]
    if t == 'var':
        return {expr[1]}
    if t == 'lam':
        var = expr[1]
        body = expr[2]
        return free_vars(body) - {var}
    if t == 'app':
        return free_vars(expr[1]) | free_vars(expr[2])
    return set()

def fresh_var(used):
    for c in string.ascii_lowercase:
        if c not in used:
            return c
    i = 1
    while True:
        for c in string.ascii_lowercase:
            name = f"{c}{i}"
            if name not in used:
                return name
        i += 1

def alpha_rename(expr, old, new):
    t = expr[0]
    if t == 'var':
        return ('var', new) if expr[1] == old else expr
    if t == 'lam':
        v, body = expr[1], expr[2]
        if v == old:
            return expr
        else:
            return ('lam', v, alpha_rename(body, old, new))
    if t == 'app':
        return ('app', alpha_rename(expr[1], old, new), alpha_rename(expr[2], old, new))
    return expr

def substitute(expr, var, val):
    t = expr[0]
    if t == 'var':
        return val if expr[1] == var else expr
    if t == 'lam':
        v, body = expr[1], expr[2]
        if v == var:
            return expr
        if v not in free_vars(val):
            return ('lam', v, substitute(body, var, val))
        else:
            used = free_vars(body) | free_vars(val) | {var}
            new_v = fresh_var(used)
            body_renamed = alpha_rename(body, v, new_v)
            return ('lam', new_v, substitute(body_renamed, var, val))
    if t == 'app':
        return ('app', substitute(expr[1], var, val), substitute(expr[2], var, val))
    return expr


# ——————————————————————————————————————————————
# NORMAL-ORDER β-REDUCTION
# ——————————————————————————————————————————————
def beta_reduce_step(expr):
    t = expr[0]
    if t == 'app':
        left, right = expr[1], expr[2]
        if left[0] == 'lam':
            v, M = left[1], left[2]
            return substitute(M, v, right), True
        else:
            left_red, changed = beta_reduce_step(left)
            if changed:
                return ('app', left_red, right), True
            right_red, changed2 = beta_reduce_step(right)
            if changed2:
                return ('app', left, right_red), True
            return expr, False

    if t == 'lam':
        v, body = expr[1], expr[2]
        body_red, changed = beta_reduce_step(body)
        if changed:
            return ('lam', v, body_red), True
        return expr, False

    return expr, False

def beta_normalize(expr):
    current = expr
    while True:
        next_expr, changed = beta_reduce_step(current)
        if not changed:
            return current
        current = next_expr


# ——————————————————————————————————————————————
# TEST MAIN: “(increment one) ≡ two”
# ——————————————————————————————————————————————
if __name__ == "__main__":
    increment = "λn.λf.λx.f (n f x)"
    one       = "λf.λx.f x"
    two       = "λf.λx.f (f x)"

    parsed_inc      = parse_lambda_expr(increment)
    parsed_one      = parse_lambda_expr(one)
    parsed_incr_one = parse_lambda_expr(f"({increment}) ({one})")

    reduced_ast = beta_normalize(parsed_incr_one)
    result_str = lambda_expr_to_str(reduced_ast)

    print("Expected two   =", two)
    print("Result of (increment one) →", result_str)
    assert result_str == two, f"FAIL: got `{result_str}` instead of `{two}`"
    print("✔︎  (increment one)  ≡ two")
