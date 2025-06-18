
def test_unary_numbers():
    zero = "λf.λx.x"  # 0
    one = "λf.λx.f x"  # 1
    two = "λf.λx.f (f x)"  # 2
    three = "λf.λx.f (f (f x))"  # 3
    succ = "λn.λf.λx.f (n f x)"  # Increment: n + 1
    add = "λn.λm.λf.λx.n f (m f x)"  # Add: n + m
    succ_zero = f"({succ}) {zero}"
    succ_one = f"({succ}) {one}"
    succ_two = f"({succ}) {two}"


def test_parsing_and_simulation():
    zero_str = "λf.λx.x" 
    one_str = "λf.λx.f x"
    two_str = "λf.λx.f (f x)"
    three_str = "λf.λx.f (f (f x))"

    increment_str = "λn.λf.λx.f (n f x)"  
    add_str = "λn.λm.λf.λx.n f (m f x)"  

    # 2. Expected token structures (AST tuples) for checking parser correctness
    zero_ast = ('lam', 'f', ('lam', 'x', ('var', 'x')))
    one_ast = ('lam', 'f', ('lam', 'x', ('app', ('var', 'f'), ('var', 'x'))))
    two_ast = ('lam', 'f', ('lam', 'x', ('app', ('var', 'f'), ('app', ('var', 'f'), ('var', 'x')))))
    three_ast = ('lam', 'f', ('lam', 'x', ('app', ('var', 'f'), ('app', ('var', 'f'), ('app', ('var', 'f'), ('var', 'x'))))))

    increment_ast = ('lam', 'n', ('lam', 'f', ('lam', 'x', ('app', ('var', 'f'), ('app', ('app', ('var', 'n'), ('var', 'f')), ('var', 'x'))))))
    add_ast = ('lam', 'n', ('lam', 'm', ('lam', 'f', ('lam', 'x', ('app', ('app', ('var', 'n'), ('var', 'f')), ('app', ('app', ('var', 'm'), ('var', 'f')), ('var', 'x')))))))

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
        print("Back-translated lambda:", unparse_lambda_expr(back_ast))