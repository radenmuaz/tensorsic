# Boolean
true = "λt.λf.t"
false = "λt.λf.f"
if_expr = "λb.λx.λy.b x y"
and_expr = "λa.λb.a b false"
or_expr = "λa.λb.a true b"
not_expr = "λa.a false true"

xor_expr = "λa.λb.a (not_expr b) b"               # Exclusive OR
nand_expr = "λa.λb.not_expr (and_expr a b)"       # NOT AND
nor_expr = "λa.λb.not_expr (or_expr a b)"         # NOT OR
implication = "λa.λb.or_expr (not_expr a) b"      # Implication (a → b)
equivalence = "λa.λb.and_expr (implication a b) (implication b a)"  # Logical equivalence

# Unary
zero = "λf.λx.x"                           # 0
one = "λf.λx.f x"                         # 1
two = "λf.λx.f (f x)"                     # 2
three = "λf.λx.f (f (f x))"               # 3
four = "λf.λx.f (f (f (f x)))"            # 4
five = "λf.λx.f (f (f (f (f x))))"        # 5
six = "λf.λx.f (f (f (f (f (f x)))))"            # 6
seven = "λf.λx.f (f (f (f (f (f (f x))))))"       # 7
eight = "λf.λx.f (f (f (f (f (f (f (f x)))))))"   # 8
nine = "λf.λx.f (f (f (f (f (f (f (f (f x))))))))" # 9
ten = "λf.λx.f (f (f (f (f (f (f (f (f (f x)))))))))" # 10

succ = "λn.λf.λx.f (n f x)"
add = "λn.λm.λf.λx.n f (m f x)"
multiply = "λn.λm.λf.n (m f)"
power = "λn.λm.m n"
pred = "λn.λf.λx.n (λg.λh.h (g f)) (λu.x) (λu.u)"
subtract = "λn.λm.m pred n"
is_zero = "λn.n (λx.false) true"
less_or_equal = "λm.λn.is_zero (subtract m n)"
Y = "λf.(λx.f (x x)) (λx.f (x x))"
division = (
    "Y (λf.λn.λm."
    "  if (less_or_equal m n)"
    "    (succ (f (subtract n m) m))"
    "    zero)"
)
modulo = (
    "Y (λf.λn.λm."
    "  if (less_or_equal m n)"
    "     (f (subtract n m) m)"
    "     n)"
)
square = "λn.multiply n n"
greater_than = "λm.λn.not_expr (less_or_equal m n)"
max_expr = "λm.λn.if_expr (less_or_equal m n) n m"
min_expr = "λm.λn.if_expr (less_or_equal m n) m n"

# List
nil = "λc.λn.n"
cons = "λx.λl.λc.λn.c x (l c n)"
is_nil = "λl.l (λx.λr.false) true"
head = "λl.l (λx.λr.x) nil"  # Returns nil if list is empty
tail = (
    "λl.λc.λn."
    "l (λx.λg.λh.h (g x)) (λu.n) (λu.u) c n"
)
append = (
    "λx.λl.Y (λg.λl. "
    "  if_expr (is_nil l) "
    "    (cons x nil) "
    "    (cons (head l) (g (tail l)))) l"
)
reverse = (
    "λl.Y (λg.λacc.λl."
    "  if_expr (is_nil l)"
    "    acc"
    "    (g (cons (head l) acc) (tail l))) nil l"
)
concat = (
    "λl1.λl2.Y (λg.λl1. "
    "  if_expr (is_nil l1) "
    "    l2 "
    "    (cons (head l1) (g (tail l1)))) l1"
)
map_expr = (
    "λf.λl.Y (λg.λl. "
    "  if_expr (is_nil l) "
    "    nil "
    "    (cons (f (head l)) (g (tail l)))) l"
)
filter_expr = (
    "λp.λl.Y (λg.λl. "
    "  if_expr (is_nil l) "
    "    nil "
    "    (if_expr (p (head l)) "
    "      (cons (head l) (g (tail l))) "
    "      (g (tail l)))) l"
)
reduce_expr = (
    "λf.λacc.λl.Y (λg.λacc.λl."
    "  if_expr (is_nil l)"
    "    acc"
    "    (g (f acc (head l)) (tail l))) acc l"
)
length = "reduce_expr (λacc.λx.succ acc) zero"


list_abc = "(cons one (cons two (cons three nil)))"
test_is_empty = f"is_nil {list_abc}"  # false
test_head = f"head {list_abc}"         # a
test_tail = f"tail {list_abc}"         # (cons b (cons c nil))

# Create a string from characters (a basic example with 'a', 'b', 'c')
char_a = "λf.λx.f x"                               # 'a'
char_b = "λf.λx.f (f x)"                           # 'b'
char_c = "λf.λx.f (f (f x))"                       # 'c'
string_abc = "(cons char_a (cons char_b (cons char_c nil)))"

# Length of a string (general list length function reused)
length_of_string = "length string_abc"

# Utils
identity = "λx.x"
const = "λx.λy.x"
apply = "λf.λx.f x"
compose = "λf.λg.λx.f (g x)"


# Fixed-Point Combinators
# Y = "λf.(λx.f (x x)) (λx.f (x x))"  # Y Combinator
factorial = (
    "Y (λf.λn.if_expr (is_zero n) one (multiply n (f (pred n))))"
)
fibonacci = (
    "Y (λf.λn.if_expr (less_or_equal n one) n "
    "(add (f (pred n)) (f (pred (pred n)))))"
)
factorial_example = "factorial five"  # Output: 120
fibonacci_example = "fibonacci five"  # Output: 5

T = "λf.(λx.f (λv.x x v)) (λx.f (λv.x x v))"  # Turing Fixed-Point Combinator
factorial_T = "T (λf.λn.if_expr (is_zero n) one (multiply n (f (pred n))))"
fibonacci_T = (
    "T (λf.λn.if_expr (less_or_equal n one) n "
    "(add (f (pred n)) (f (pred (pred n)))))"
)

Z = "λf.(λx.f (λy.(x x) y)) (λx.f (λy.(x x) y))"  # Z Combinator (strict)
factorial_Z = (
    "Z (λf.λn.if_expr (is_zero n) one (multiply n (f (pred n))))"
)
fibonacci_Z = (
    "Z (λf.λn.if_expr (less_or_equal n one) n "
    "(add (f (pred n)) (f (pred (pred n)))))"
)

I = "λx.x"  # Identity Combinator
K = "λx.λy.x"  # Constant Combinator
S = "λx.λy.λz.x z (y z)"  # Starling Combinator

I_example = "I true"  # Output: true
K_example = "K true false"  # Output: true
S_example = "S add one two"  # Output: 3

# Self-Application Combinators
Omega = "(λx.x x) (λx.x x)"  # Non-terminating combinator
Theta = "λf.(λx.f (x x)) (λx.f (x x))"  # Theta Combinator for recursion

factorial_theta = "Theta (λf.λn.if_expr (is_zero n) one (multiply n (f (pred n))))"

# Boolean Logic Combinators
C = "λf.λx.λy.f y x"  # Flip Combinator

# Boolean Logic Examples
C_example = "C subtract three five"  # Output: 2

# Composition Combinators
B = "λf.λg.λx.f (g x)"  # Function composition
Psi = "λf.λg.λx.f x (g x)"  # Apply both arguments to a binary function

# Composition Examples
B_example = "B succ multiply two three"  # Output: 7
Psi_example = "Psi add succ two"  # Output: 5

# Pairing and Projection Combinators
Pair = "λx.λy.λf.f x y"  # Create a pair
Fst = "λp.p (λx.λy.x)"  # First element of a pair
Snd = "λp.p (λx.λy.y)"  # Second element of a pair

# Pairing and Projection Examples
pair_example = "Pair three four"  # Pairing
fst_example = "Fst (Pair three four)"  # Output: three
snd_example = "Snd (Pair three four)"  # Output: four

# Functional Utilities
W = "λf.λx.f x x"  # Double application
Dup = "λx.Pair x x"  # Create a pair of the same value

# Functional Utilities Examples
W_example = "W succ two"  # Output: 4
dup_example = "Dup two"  # Output: Pair two two

# Derived Combinators
A = "λx.λy.x y"  # Applicator
D = "λf.λx.f x x"  # Apply a function to a duplicated input

# Derived Combinator Examples
A_example = "A succ three"  # Output: 4
D_example = "D add three"  # Output: 6
