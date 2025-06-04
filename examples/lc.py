import re

# Token types
TOKEN_LAMBDA = 'LAMBDA'
TOKEN_DOT = 'DOT'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_IDENTIFIER = 'IDENTIFIER'
TOKEN_EOF = 'EOF'

# Lexer: convert input string into tokens
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

# Parser: recursive descent
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

# Parse a lambda expression into a tuple-based AST
def parse_lambda(expr):
    tokens = tokenize(expr)
    parser = Parser(tokens)
    return parser.parse()

# Compute free variables in expr
def free_vars(expr):
    kind = expr[0]
    if kind == 'var':
        return {expr[1]}
    elif kind == 'lam':
        param = expr[1]
        body = expr[2]
        return free_vars(body) - {param}
    elif kind == 'app':
        return free_vars(expr[1]) | free_vars(expr[2])
    return set()

# Alpha-renaming: rename all occurrences of old_var to new_var in expr
def rename(expr, old_var, new_var):
    kind = expr[0]
    if kind == 'var':
        return ('var', new_var) if expr[1] == old_var else expr
    elif kind == 'lam':
        param, body = expr[1], expr[2]
        if param == old_var:
            return ('lam', new_var, rename(body, old_var, new_var))
        else:
            return ('lam', param, rename(body, old_var, new_var))
    elif kind == 'app':
        left = rename(expr[1], old_var, new_var)
        right = rename(expr[2], old_var, new_var)
        return ('app', left, right)
    return expr

# Substitute variable var with value in expr, with alpha-renaming to avoid capture
def substitute(expr, var, value):
    kind = expr[0]
    if kind == 'var':
        return value if expr[1] == var else expr
    elif kind == 'lam':
        param, body = expr[1], expr[2]
        if param == var:
            return expr
        if param in free_vars(value):
            # rename bound variable to a fresh name
            new_param = f"{param}_" + str(id(expr))
            body = rename(body, param, new_param)
            param = new_param
        return ('lam', param, substitute(body, var, value))
    elif kind == 'app':
        left = substitute(expr[1], var, value)
        right = substitute(expr[2], var, value)
        return ('app', left, right)
    return expr

# Beta reduction with alpha-rename support
def beta_reduce(expr, max_depth=1000):
    def reduce_once(e):
        kind = e[0]
        if kind == 'app':
            func = reduce_once(e[1])
            arg = reduce_once(e[2])
            if func[0] == 'lam':
                return substitute(func[2], func[1], arg)
            return ('app', func, arg)
        elif kind == 'lam':
            return ('lam', e[1], reduce_once(e[2]))
        else:
            return e

    prev = expr
    for _ in range(max_depth):
        reduced = reduce_once(prev)
        if reduced == prev:
            return reduced
        prev = reduced
    return prev

# Convert AST to lambda string
def ast_to_str(node):
    kind = node[0]
    if kind == 'var':
        return node[1]
    elif kind == 'lam':
        return f"λ{node[1]}. {ast_to_str(node[2])}"
    elif kind == 'app':
        left = ast_to_str(node[1])
        right = ast_to_str(node[2])
        return f"({left} {right})"
    return ''

# Tests for standard Church numerals and identity
def test():
    zero = "λf. λx. x"
    one = "λf. λx. (f x)"
    two = "λf. λx. (f (f x))"

    succ = "λn. λf. λx. (f (n f x))"
    succ_zero_str = f"({succ}) {zero}"
    succ_one_str = f"({succ}) {one}"

    identity = "λx. x"
    identity_zero_str = f"({identity}) {zero}"
    identity_one_str = f"({identity}) {one}"

    parsed_zero = parse_lambda(zero)
    parsed_one = parse_lambda(one)
    parsed_two = parse_lambda(two)

    reduced_zero_succ = beta_reduce(parse_lambda(succ_zero_str))
    reduced_one_succ = beta_reduce(parse_lambda(succ_one_str))
    reduced_identity_zero = beta_reduce(parse_lambda(identity_zero_str))
    reduced_identity_one = beta_reduce(parse_lambda(identity_one_str))

    print("zero:", ast_to_str(parsed_zero))
    print("one:", ast_to_str(parsed_one))
    print("two:", ast_to_str(parsed_two))

    print("succ zero:", ast_to_str(reduced_zero_succ))  # should be one
    print("succ one:", ast_to_str(reduced_one_succ))    # should be two

    print("identity zero:", ast_to_str(reduced_identity_zero))  # should be zero
    print("identity one:", ast_to_str(reduced_identity_one))    # should be one

# Alpha-renaming test cases
def alpha_rename_tests():
    # Test 1: rename variable in simple abstraction
    expr1 = parse_lambda("λx. x")
    renamed1 = rename(expr1, "x", "y")
    assert ast_to_str(renamed1) == "λy. y", f"Expected λy. y, got {ast_to_str(renamed1)}"

    # Test 2: nested abstraction, no conflict
    expr2 = parse_lambda("λx. (λy. x y)")
    renamed2 = rename(expr2, "x", "z")
    assert ast_to_str(renamed2) == "λz. λy. (z y)", f"Expected λz. λy. (z y), got {ast_to_str(renamed2)}"

    # Test 3: substitution requiring alpha-renaming to avoid capture
    # (λx. λy. x) y  ->  λy_fresh. y
    app3 = parse_lambda(f"(λx. λy. x) y")
    reduced3 = beta_reduce(app3)
    reduced3_str = ast_to_str(reduced3)
    assert reduced3_str.startswith("λy_"), f"Expected fresh y_, got {reduced3_str}"
    assert reduced3_str.endswith(". y"), f"Expected body y, got {reduced3_str}"

    # Test 4: (λx. λy. x y) y  ->  λy_fresh. (y y_fresh)
    app4 = parse_lambda(f"(λx. λy. x y) y")
    reduced4 = beta_reduce(app4)
    reduced4_str = ast_to_str(reduced4)
    assert reduced4_str.startswith("λy_"), f"Expected fresh y_, got {reduced4_str}"
    assert "(" in reduced4_str and ")" in reduced4_str, f"Expected application in result, got {reduced4_str}"

    print("All alpha-renaming tests passed.")

if __name__ == "__main__":
    test()
    alpha_rename_tests()