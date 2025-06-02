# Booleans
true  = "λt.λf.t"
false = "λt.λf.f"

# Boolean operations
not_expr = "λa.a false true"
and_expr = "λa.λb.a b false"
or_expr  = "λa.λb.a true b"
xor_expr = "λa.λb.a (not_expr b) b"

# Pair (Church pair) constructor and selectors
pair = "λx.λy.λf.f x y"
first = "λp.p (λx.λy.x)"
second = "λp.p (λx.λy.y)"

# 8-bit boolean number encoded as nested pairs:
# e.g. bit1(bit2(...(bit8 nil)...))
# nil can be the empty pair: pair false false

nil = f"{pair} {false} {false}"

def make_8bit(b1,b2,b3,b4,b5,b6,b7,b8):
    # nest pairs right-associative: pair b1 (pair b2 (pair b3 ...))
    return (
        f"{pair} {b1} ("
        f"{pair} {b2} ("
        f"{pair} {b3} ("
        f"{pair} {b4} ("
        f"{pair} {b5} ("
        f"{pair} {b6} ("
        f"{pair} {b7} ("
        f"{pair} {b8} {nil}"
        f")))))))"
    )

# Example 8-bit numbers
zero_8bit = make_8bit(false,false,false,false,false,false,false,false)
one_8bit = make_8bit(false,false,false,false,false,false,false,true)
three_8bit = make_8bit(false,false,false,false,false,false,true,true)

# Bitwise operations on pairs:
# bitwise_op8 = λn.λm.
#   pair (bitwise_op (first n) (first m))
#        (bitwise_op8 (second n) (second m))
# with base case when second is nil

bitwise_op_8 = lambda bitwise_op: (
    f"Y (λf.λn.λm."
    f"  if_expr (is_nil n)"
    f"    nil"
    f"    ("
    f"      {pair} "
    f"        ({bitwise_op} ({first} n) ({first} m)) "
    f"        (f ({second} n) ({second} m))"
    f"    )"
    f")"
)

# Need is_nil for pairs (true if pair == nil)
is_nil = (
    f"λl.l (λx.λr.false) true"
)

# Define first and second functions (already above as strings)
# Need Y combinator and if_expr from your original definitions for full recursion and conditionals

Y = "λf.(λx.f (x x)) (λx.f (x x))"
if_expr = "λb.λx.λy.b x y"

# Combine full bitwise ops:
and_8bit = bitwise_op_8("and_expr")
or_8bit = bitwise_op_8("or_expr")
xor_8bit = bitwise_op_8("xor_expr")

# NOT operates on a single 8-bit number:
# bitwise_not8 = λn.
#   pair (not_expr (first n))
#        (bitwise_not8 (second n))
# base case nil -> nil

not_8bit = (
    f"Y (λf.λn."
    f"  if_expr (is_nil n)"
    f"    nil"
    f"    ("
    f"      {pair} "
    f"        (not_expr ({first} n)) "
    f"        (f ({second} n))"
    f"    )"
    f")"
)

# Church numerals for bits count
zero_church = "λf.λx.x"
one_church = "λf.λx.f x"
succ_church = "λn.λf.λx.f (n f x)"
two_church = f"{succ_church} {one_church}"
three_church = f"{succ_church} {two_church}"
four_church = f"{succ_church} {three_church}"
five_church = f"{succ_church} {four_church}"
six_church = f"{succ_church} {five_church}"
seven_church = f"{succ_church} {six_church}"
eight_church = f"{succ_church} {seven_church}"

# Nil list
nil = f"{pair} {false} {false}"

# is_nil helper (returns true if list is nil)
is_nil = "λl.l (λx.λr.false) true"

# first and second bits of pair
# Already defined as first, second

# Full adder bit
majority = (
    "λa.λb.λc."
    f"{or_expr} ({or_expr} ({and_expr} a b) ({and_expr} b c)) ({and_expr} a c)"
)

sum_bit = (
    "λa.λb.λc."
    f"{xor_expr} ({xor_expr} a b) c"
)

full_adder_bit = (
    f"λa.λb.λc_in."
    f"{pair} ({sum_bit} a b c_in) ({majority} a b c_in)"
)

# Recursive 8-bit adder with carry (adder_8bit)
adder_8bit = (
    f"{Y} (λf.λa.λb.λc_in."
    f"  {if_expr} ({is_nil} a)"
    f"    ({pair} {nil} c_in)"
    f"    ("
    f"      (λsum_carry."
    f"       (λs."
    f"        (λc_out."
    f"          {pair} s (f ({second} a) ({second} b) c_out)"
    f"        ) ({second} sum_carry)"
    f"       ) ({first} sum_carry)"
    f"      ) ({full_adder_bit} ({first} a) ({first} b) c_in)"
    f"    )"
    f")"
)

# Bitwise NOT on 8-bit number (not_8bit)
not_8bit = (
    f"{Y} (λf.λn."
    f"  {if_expr} ({is_nil} n)"
    f"    {nil}"
    f"    ({pair} ({not_expr} ({first} n)) (f ({second} n)))"
    f")"
)

# One in 8-bit pair form (7 false bits + true at LSB)
one_8bit = (
    f"{pair} {false} ("
    f"{pair} {false} ("
    f"{pair} {false} ("
    f"{pair} {false} ("
    f"{pair} {false} ("
    f"{pair} {false} ("
    f"{pair} {false} ("
    f"{pair} {true} {nil}"
    f")))))))"
)

# Negate 8-bit two's complement
negate_8bit = f"λx.{adder_8bit} ({not_8bit} x) {one_8bit} {false}"

# Pred and is_zero (for loop counters)
pred = (
    "λn.λf.λx.n (λg.λh.h (g f)) (λu.x) (λu.u)"
)

is_zero = (
    "λn.n (λx.false) true"
)

# lsb (least significant bit extractor)
lsb = (
    f"{Y} (λf.λn."
    f"  {if_expr} ({is_nil} ({second} n))"
    f"    ({first} n)"
    f"    (f ({second} n))"
    f")"
)

# shift_left_8bit (drop MSB, add false at LSB)
shift_left_8bit = (
    f"{Y} (λf.λn."
    f"  {if_expr} ({is_nil} n)"
    f"    {nil}"
    f"    ({pair} ({first} ({second} n)) (f ({second} n)))"
    f")"
)

# shift_right_8bit (arithmetic shift right, preserve MSB)
shift_right_8bit = (
    f"{Y} (λf.λn."
    f"  {if_expr} ({is_nil} n)"
    f"    {nil}"
    f"    ({pair} ({first} n) (f ({second} n)))"
    f")"
)

# multiplier_8bit: multiply two 8-bit signed numbers (two's complement)
multiplier_8bit = (
    f"{Y} (λf.λm.λn."
    f"  (λsign_m.λsign_n.λsign_res."
    f"    (λabs_m.λabs_n."
    f"      (λloop."
    f"        loop {eight_church} abs_m abs_n {nil}"
    f"      ) ("
    f"        {Y} (λloop.λcount.λacc_m.λacc_n.λres."
    f"          {if_expr} ({is_zero} count)"
    f"            res"
    f"            ("
    f"              (λlsb_n.λres1.λacc_m1.λacc_n1."
    f"                loop (pred count) acc_m1 acc_n1 res1"
    f"              ) ({lsb} acc_n)"
    f"                ({if_expr} ({lsb} acc_n) ({adder_8bit} res acc_m {false}) res)"
    f"                ({shift_left_8bit} acc_m)"
    f"                ({shift_right_8bit} acc_n)"
    f"            )"
    f"        )"
    f"      )"
    f"    ) ("
    f"      {if_expr} sign_m ({negate_8bit} m) m"
    f"    ) ("
    f"      {if_expr} sign_n ({negate_8bit} n) n"
    f"    )"
    f"  ) ({first} m) ({first} n) ({xor_expr} ({first} m) ({first} n))"
    f")"
)

# ---- END ----

if __name__ == "__main__":
    # Print examples or save to file as needed
    print("# Basic lambda calculus combinators and 8-bit adder and multiplier expressions")
    print("true =", true)
    print("false =", false)
    print("pair =", pair)
    print("first =", first)
    print("second =", second)
    print("Y =", Y)
    print("if_expr =", if_expr)
    print("and_expr =", and_expr)
    print("or_expr =", or_expr)
    print("not_expr =", not_expr)
    print("xor_expr =", xor_expr)
    print("adder_8bit =", adder_8bit)
    print("multiplier_8bit =", multiplier_8bit)
