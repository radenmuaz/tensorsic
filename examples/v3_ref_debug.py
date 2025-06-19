import threading
from collections import deque
import sys

# Implementing the parser and evaluation with debugger integrated, as per the prompt.

# AST definitions
class Term:
    pass

class Var(Term):
    def __init__(self, name: int):
        self.name = name

    def __str__(self):
        return name_str(self.name)
    __repr__ = __str__

class Let(Term):
    def __init__(self, name: int, t1: Term, t2: Term):
        self.name = name
        self.t1 = t1
        self.t2 = t2

    def __str__(self):
        return f"! {name_str(self.name)} = {self.t1}; {self.t2}"
    __repr__ = __str__

class Era(Term):
    def __str__(self):
        return "*"
    __repr__ = __str__

class Sup(Term):
    def __init__(self, label: int, left: Term, right: Term):
        self.label = label
        self.left = left
        self.right = right

    def __str__(self):
        l = self.label
        if l == 0:
            return "{" + f"{self.left},{self.right}" + "}"
        elif l == 1:
            return "<" + f"{self.left},{self.right}" + ">"
        else:
            return f"&{l}{{{self.left},{self.right}}}"
    __repr__ = __str__

class Dup(Term):
    def __init__(self, label: int, x: int, y: int, val: Term, body: Term):
        self.label = label
        self.x = x
        self.y = y
        self.val = val
        self.body = body

    def __str__(self):
        l = self.label
        xs = name_str(self.x)
        ys = name_str(self.y)
        if l == 0:
            return f"! {{{xs},{ys}}} = {self.val}; {self.body}"
        elif l == 1:
            return f"! <{xs},{ys}> = {self.val}; {self.body}"
        else:
            return f"! &{l}{{{xs},{ys}}} = {self.val}; {self.body}"
    __repr__ = __str__

class Lam(Term):
    def __init__(self, label: int, x: int, body: Term):
        self.label = label
        self.x = x
        self.body = body

    def __str__(self):
        l = self.label
        xs = name_str(self.x)
        if l == 0:
            return f"λ{xs}.{self.body}"
        elif l == 1:
            return f"Λ{xs}.{self.body}"
        else:
            return f"&{l} λ{xs}.{self.body}"
    __repr__ = __str__

class App(Term):
    def __init__(self, label: int, func: Term, arg: Term):
        self.label = label
        self.func = func
        self.arg = arg

    def __str__(self):
        l = self.label
        if l == 0:
            return f"({self.func} {self.arg})"
        elif l == 1:
            return f"[{self.func} {self.arg}]"
        else:
            return f"&{l} ({self.func} {self.arg})"
    __repr__ = __str__

# Globals for substitution, fresh names, inters, and debugger
_gSUBST = {}
_gFRESH = 0
_gINTERS = 0
_gSTOP = False

_gSUBST_lock = threading.Lock()
_gFRESH_lock = threading.Lock()
_gINTERS_lock = threading.Lock()
_gSTOP_lock = threading.Lock()

def set_subst(name: int, term: Term):
    with _gSUBST_lock:
        _gSUBST[name] = term

def get_subst(name: int):
    with _gSUBST_lock:
        if name in _gSUBST:
            return _gSUBST.pop(name)
    return None

def fresh():
    global _gFRESH
    with _gFRESH_lock:
        n = _gFRESH
        _gFRESH += 1
    return n

def inc_inters():
    global _gINTERS
    with _gINTERS_lock:
        _gINTERS += 1

def read_inters():
    with _gINTERS_lock:
        return _gINTERS

def name_str(k: int) -> str:
    target = k + 1
    queue = deque([""])
    count = 0
    while queue:
        prefix = queue.popleft()
        if count == target:
            return prefix
        count += 1
        for c in map(chr, range(ord('a'), ord('z')+1)):
            queue.append(c + prefix)
    return f"v{k}"

# Debugger controls
def mark_reduction():
    global _gSTOP
    with _gSTOP_lock:
        _gSTOP = True

def has_reduced():
    with _gSTOP_lock:
        return _gSTOP

def reset_reduction():
    global _gSTOP
    with _gSTOP_lock:
        _gSTOP = False

# Evaluation primitives with debugger marking
def app_era(f: Term, arg: Term):
    if isinstance(f, Era):
        inc_inters()
        mark_reduction()
        return Era()
    else:
        raise RuntimeError("app_era: expected Era as first argument")

def app_lam(f: Term, arg: Term, app_lab: int):
    if isinstance(f, Lam):
        lam_lab = f.label
        nam = f.x
        bod = f.body
        inc_inters()
        mark_reduction()
        if lam_lab == app_lab:
            set_subst(nam, arg)
            return whnf(bod)
        else:
            y = fresh()
            z = fresh()
            set_subst(nam, Lam(app_lab, y, Var(z)))
            inner = App(lam_lab, arg, Var(y))
            newb = App(app_lab, bod, inner)
            return whnf(Lam(lam_lab, z, newb))
    else:
        raise RuntimeError("app_lam: expected Lam as first argument")

def app_sup(f: Term, arg: Term, app_lab: int):
    if isinstance(f, Sup):
        inc_inters()
        mark_reduction()
        lab = f.label
        left = f.left
        right = f.right
        c0 = fresh()
        c1 = fresh()
        a0 = App(app_lab, left, Var(c0))
        a1 = App(app_lab, right, Var(c1))
        return whnf(Dup(lab, c0, c1, arg, Sup(lab, a0, a1)))
    else:
        raise RuntimeError("app_sup: expected Sup as first argument")

def app_dup(term: Term):
    if isinstance(term, App) and isinstance(term.arg, Dup):
        app_lab = term.label
        f = term.func
        dup: Dup = term.arg
        inc_inters()
        mark_reduction()
        return whnf(Dup(dup.label, dup.x, dup.y, dup.val, App(app_lab, f, dup.body)))
    else:
        raise RuntimeError("app_dup: expected App with Dup")

def dup_era(dup_term: Dup, v: Term):
    if isinstance(v, Era):
        inc_inters()
        mark_reduction()
        set_subst(dup_term.x, Era())
        set_subst(dup_term.y, Era())
        return whnf(dup_term.body)
    else:
        raise RuntimeError("dup_era: expected Dup and Era")

def dup_lam(dup_term: Dup, v: Term):
    if isinstance(v, Lam):
        inc_inters()
        mark_reduction()
        lam_lab = v.label
        x = v.x
        f = v.body
        x0 = fresh()
        x1 = fresh()
        f0 = fresh()
        f1 = fresh()
        set_subst(dup_term.x, Lam(lam_lab, x0, Var(f0)))
        set_subst(dup_term.y, Lam(lam_lab, x1, Var(f1)))
        set_subst(x, Sup(dup_term.label, Var(x0), Var(x1)))
        return whnf(Dup(dup_term.label, f0, f1, f, dup_term.body))
    else:
        raise RuntimeError("dup_lam: expected Dup and Lam")

def dup_sup(dup_term: Dup, v: Term):
    if isinstance(v, Sup):
        inc_inters()
        mark_reduction()
        sup_lab = v.label
        a = v.left
        b = v.right
        if dup_term.label == sup_lab:
            set_subst(dup_term.x, a)
            set_subst(dup_term.y, b)
            return whnf(dup_term.body)
        else:
            a0 = fresh()
            a1 = fresh()
            b0 = fresh()
            b1 = fresh()
            set_subst(dup_term.x, Sup(sup_lab, Var(a0), Var(b0)))
            set_subst(dup_term.y, Sup(sup_lab, Var(a1), Var(b1)))
            inner = Dup(dup_term.label, b0, b1, b, dup_term.body)
            return whnf(Dup(dup_term.label, a0, a1, a, inner))
    else:
        raise RuntimeError("dup_sup: expected Dup and Sup")

def dup_dup(dup_term: Dup, v: Term):
    if isinstance(v, Dup):
        inc_inters()
        mark_reduction()
        return whnf(Dup(dup_term.label, dup_term.x, dup_term.y, v.x, Dup(dup_term.label, v.x, v.y, v.val, dup_term.body)))
    else:
        raise RuntimeError("dup_dup: expected Dup with inner Dup")

# Whnf with debug prints and reduction marking
def whnf(term: Term) -> Term:
    if isinstance(term, Var):
        sub = get_subst(term.name)
        if sub is not None:
            mark_reduction()
            return whnf(sub)
        else:
            return term
    elif isinstance(term, Let):
        print("LET")
        v_whnf = whnf(term.t1)
        if has_reduced():
            return Let(term.name, v_whnf, term.t2)
        else:
            set_subst(term.name, v_whnf)
            mark_reduction()
            return whnf(term.t2)
    elif isinstance(term, App):
        f_whnf = whnf(term.func)
        if has_reduced():
            return App(term.label, f_whnf, term.arg)
        else:
            print("APP")
            if isinstance(f_whnf, Lam):
                return app_lam(f_whnf, term.arg, term.label)
            elif isinstance(f_whnf, Sup):
                return app_sup(f_whnf, term.arg, term.label)
            elif isinstance(f_whnf, Era):
                return app_era(f_whnf, term.arg)
            elif isinstance(f_whnf, Dup):
                return app_dup(App(term.label, f_whnf, term.arg))
            else:
                return App(term.label, f_whnf, term.arg)
    elif isinstance(term, Dup):
        print("DUP")
        v_whnf = whnf(term.val)
        if has_reduced():
            return Dup(term.label, term.x, term.y, v_whnf, term.body)
        else:
            if isinstance(v_whnf, Lam):
                return dup_lam(term, v_whnf)
            elif isinstance(v_whnf, Sup):
                return dup_sup(term, v_whnf)
            elif isinstance(v_whnf, Era):
                return dup_era(term, v_whnf)
            elif isinstance(v_whnf, Dup):
                return dup_dup(term, v_whnf)
            else:
                return Dup(term.label, term.x, term.y, v_whnf, term.body)
    else:
        return term

# Step and normal with debugging
def step(term: Term) -> Term:
    reset_reduction()
    term1 = whnf(term)
    if has_reduced():
        return term1
    else:
        reset_reduction()
        if isinstance(term1, Lam):
            body1 = step(term1.body)
            if has_reduced():
                mark_reduction()
                return Lam(term1.label, term1.x, body1)
            else:
                return term1
        elif isinstance(term1, App):
            f1 = step(term1.func)
            if has_reduced():
                mark_reduction()
                return App(term1.label, f1, term1.arg)
            else:
                reset_reduction()
                a1 = step(term1.arg)
                if has_reduced():
                    mark_reduction()
                    return App(term1.label, term1.func, a1)
                else:
                    return term1
        elif isinstance(term1, Sup):
            l1 = step(term1.left)
            if has_reduced():
                mark_reduction()
                return Sup(term1.label, l1, term1.right)
            else:
                reset_reduction()
                r1 = step(term1.right)
                if has_reduced():
                    mark_reduction()
                    return Sup(term1.label, term1.left, r1)
                else:
                    return term1
        elif isinstance(term1, Dup):
            v1 = step(term1.val)
            if has_reduced():
                mark_reduction()
                return Dup(term1.label, term1.x, term1.y, v1, term1.body)
            else:
                reset_reduction()
                b1 = step(term1.body)
                if has_reduced():
                    mark_reduction()
                    return Dup(term1.label, term1.x, term1.y, term1.val, b1)
                else:
                    return term1
        else:
            return term1

def show_subst():
    with _gSUBST_lock:
        if not _gSUBST:
            return ""
        lines = []
        for k, v in _gSUBST.items():
            lines.append(f"{name_str(k)} <- {v}")
        return "\n".join(lines) + "\n"

def normal(term: Term) -> Term:
    subst_str = show_subst()
    if subst_str:
        print(subst_str, end="")
    print(term)
    print("-" * 40)
    term1 = step(term)
    if has_reduced():
        return normal(term1)
    else:
        return term1

# Parser implementation
class ParseError(Exception):
    pass

class Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.len = len(text)
        self.global_map = {}

    def current(self):
        if self.pos < self.len:
            return self.text[self.pos]
        return None

    def eof(self):
        return self.pos >= self.len

    def advance(self, n=1):
        self.pos += n

    def skip_whitespace_and_comments(self):
        while True:
            while not self.eof() and self.current().isspace():
                self.advance()
            if self.pos+1 < self.len and self.text[self.pos:self.pos+2] == "//":
                self.advance(2)
                while not self.eof() and self.current() not in "\r\n":
                    self.advance()
                while not self.eof() and self.current() in "\r\n":
                    self.advance()
                continue
            break

    def match_string(self, s: str) -> bool:
        self.skip_whitespace_and_comments()
        if self.text.startswith(s, self.pos):
            self.advance(len(s))
            return True
        return False

    def expect_string(self, s: str):
        if not self.match_string(s):
            raise ParseError(f"Expected '{s}' at position {self.pos}")

    def parse_natural(self) -> int:
        self.skip_whitespace_and_comments()
        start = self.pos
        while not self.eof() and self.current().isdigit():
            self.advance()
        if start == self.pos:
            raise ParseError(f"Expected natural number at position {self.pos}")
        return int(self.text[start:self.pos])

    def parseVarName(self) -> str:
        self.skip_whitespace_and_comments()
        if not self.eof() and self.current() == '$':
            start = self.pos
            self.advance()
            while not self.eof() and (self.current().isalnum() or self.current() == '_'):
                self.advance()
            return self.text[start:self.pos]
        else:
            start = self.pos
            if not self.eof() and (self.current().isalnum() or self.current() == '_'):
                while not self.eof() and (self.current().isalnum() or self.current() == '_'):
                    self.advance()
                return self.text[start:self.pos]
            else:
                raise ParseError(f"Expected variable name at position {self.pos}")

    def isGlobalName(self, name: str) -> bool:
        return name.startswith('$')

    def getGlobalName(self, gname: str) -> int:
        if gname in self.global_map:
            return self.global_map[gname]
        else:
            n = fresh()
            self.global_map[gname] = n
            return n

    def bindVar(self, name: str, ctx: dict) -> (int, dict):
        if self.isGlobalName(name):
            n = self.getGlobalName(name)
            return n, ctx
        else:
            n = fresh()
            ctx2 = ctx.copy()
            ctx2[name] = n
            return n, ctx2

    def getVar(self, name: str, ctx: dict) -> int:
        if self.isGlobalName(name):
            return self.getGlobalName(name)
        else:
            if name in ctx:
                return ctx[name]
            else:
                raise ParseError(f"Unbound local variable: {name}")

    def parse(self):
        self.skip_whitespace_and_comments()
        term = self.parseTerm({})
        self.skip_whitespace_and_comments()
        if not self.eof():
            raise ParseError(f"Unexpected trailing input at position {self.pos}")
        return term

    def parseTerm(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        for parse_fn in [self.parseApp, self.parseLet, self.parseLam, self.parseSup, self.parseDup]:
            try:
                return parse_fn(ctx)
            except ParseError:
                self.pos = pos0
        return self.parseSimpleTerm(ctx)

    def parseSimpleTerm(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        if self.pos < self.len and self.current() == '*':
            return self.parseEra()
        elif self.pos < self.len and self.current() == '(':
            self.expect_string('(')
            t = self.parseTerm(ctx)
            self.expect_string(')')
            return t
        else:
            return self.parseVar(ctx)

    def parseVar(self, ctx: dict) -> Term:
        name = self.parseVarName()
        n = self.getVar(name, ctx)
        return Var(n)

    def parseEra(self) -> Term:
        self.expect_string('*')
        return Era()

    def parseLam(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("&"):
            try:
                lab = self.parse_natural()
                if not self.match_string("λ"):
                    raise ParseError("Expected λ after label in lambda")
                varname = self.parseVarName()
                n, ctx2 = self.bindVar(varname, ctx)
                self.expect_string(".")
                body = self.parseTerm(ctx2)
                return Lam(lab, n, body)
            except ParseError:
                self.pos = pos0
        if self.match_string("λ"):
            varname = self.parseVarName()
            n, ctx2 = self.bindVar(varname, ctx)
            self.expect_string(".")
            body = self.parseTerm(ctx2)
            return Lam(0, n, body)
        if self.match_string("Λ"):
            varname = self.parseVarName()
            n, ctx2 = self.bindVar(varname, ctx)
            self.expect_string(".")
            body = self.parseTerm(ctx2)
            return Lam(1, n, body)
        raise ParseError("Not a lambda")

    def parseApp(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("&"):
            try:
                lab = self.parse_natural()
                self.expect_string("(")
                f = self.parseTerm(ctx)
                self.skip_whitespace_and_comments()
                a = self.parseTerm(ctx)
                self.expect_string(")")
                return App(lab, f, a)
            except ParseError:
                self.pos = pos0
        if self.match_string("("):
            f = self.parseTerm(ctx)
            self.skip_whitespace_and_comments()
            a = self.parseTerm(ctx)
            self.expect_string(")")
            return App(0, f, a)
        if self.match_string("["):
            f = self.parseTerm(ctx)
            self.skip_whitespace_and_comments()
            a = self.parseTerm(ctx)
            self.expect_string("]")
            return App(1, f, a)
        raise ParseError("Not an application")

    def parseSup(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("&"):
            try:
                lab = self.parse_natural()
                self.expect_string("{")
                a = self.parseTerm(ctx)
                self.expect_string(",")
                b = self.parseTerm(ctx)
                self.expect_string("}")
                return Sup(lab, a, b)
            except ParseError:
                self.pos = pos0
        if self.match_string("{"):
            a = self.parseTerm(ctx)
            self.expect_string(",")
            b = self.parseTerm(ctx)
            self.expect_string("}")
            return Sup(0, a, b)
        if self.match_string("<"):
            a = self.parseTerm(ctx)
            self.expect_string(",")
            b = self.parseTerm(ctx)
            self.expect_string(">")
            return Sup(1, a, b)
        raise ParseError("Not a Sup")

    def parseDup(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("!"):
            pos1 = self.pos
            if self.match_string("&"):
                try:
                    lab = self.parse_natural()
                    if self.match_string("{"):
                        name1 = self.parseVarName()
                        self.expect_string(",")
                        name2 = self.parseVarName()
                        self.expect_string("}")
                        self.expect_string("=")
                        val = self.parseTerm(ctx)
                        self.expect_string(";")
                        n1, ctx1 = self.bindVar(name1, ctx)
                        n2, ctx2 = self.bindVar(name2, ctx1)
                        body = self.parseTerm(ctx2)
                        return Dup(lab, n1, n2, val, body)
                    else:
                        raise ParseError("Expected { after !&label in Dup")
                except ParseError:
                    self.pos = pos0
                    raise
            else:
                self.pos = pos1
                if self.match_string("{"):
                    name1 = self.parseVarName()
                    self.expect_string(",")
                    name2 = self.parseVarName()
                    self.expect_string("}")
                    self.expect_string("=")
                    val = self.parseTerm(ctx)
                    self.expect_string(";")
                    n1, ctx1 = self.bindVar(name1, ctx)
                    n2, ctx2 = self.bindVar(name2, ctx1)
                    body = self.parseTerm(ctx2)
                    return Dup(0, n1, n2, val, body)
                self.pos = pos1
                if self.match_string("<"):
                    name1 = self.parseVarName()
                    self.expect_string(",")
                    name2 = self.parseVarName()
                    self.expect_string(">")
                    self.expect_string("=")
                    val = self.parseTerm(ctx)
                    self.expect_string(";")
                    n1, ctx1 = self.bindVar(name1, ctx)
                    n2, ctx2 = self.bindVar(name2, ctx1)
                    body = self.parseTerm(ctx2)
                    return Dup(1, n1, n2, val, body)
        raise ParseError("Not a Dup")

    def parseLet(self, ctx: dict) -> Term:
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("!"):
            name_token = self.parseVarName()
            self.expect_string("=")
            t1 = self.parseTerm(ctx)
            self.expect_string(";")
            n, ctx2 = self.bindVar(name_token, ctx)
            t2 = self.parseTerm(ctx2)
            return Let(n, t1, t2)
        self.pos = pos0
        raise ParseError("Not a Let")

def parseIC(input_str: str) -> Term:
    parser = Parser(input_str)
    return parser.parse()

def doParseIC(input_str: str) -> Term:
    try:
        return parseIC(input_str)
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        raise

def test_term(input_str: str):
    global _gINTERS, _gSUBST, _gFRESH
    # reset globals
    with _gINTERS_lock:
        _gINTERS = 0
    with _gSUBST_lock:
        _gSUBST.clear()
    with _gFRESH_lock:
        _gFRESH = 0
    term = doParseIC(input_str)
    print("Initial term and reductions:")
    norm = normal(term)
    inters = read_inters()
    print(f"- WORK: {inters}")

# Now run the provided test expression:
expr = "! h = λa.! {b,c} = a; ! {e,f} = λd.(b (c d)); λg.(e (f g)); ((h λi.((i λj.λk.k) λl.λm.l)) λn.λo.n)"
test_term(expr)
