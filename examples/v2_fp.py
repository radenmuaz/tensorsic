# Fully class-free, stateless implementation using tagged tuples for Terms.

import sys
from collections import deque

# Term representation: tagged tuples
# ('Var', name)
# ('Let', name, t1, t2)
# ('Era',)
# ('Sup', label, left, right)
# ('Dup', label, x, y, val, body)
# ('Lam', label, x, body)
# ('App', label, func, arg)

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

# Stateful: state dict with 'subst', 'fresh', 'inters'
def fresh_state(state):
    n = state['fresh']
    new_state = {'subst': state['subst'], 'fresh': n+1, 'inters': state['inters']}
    return n, new_state

def incr_inters(state):
    return {'subst': state['subst'], 'fresh': state['fresh'], 'inters': state['inters']+1}

def set_subst_state(name, term, state):
    new_subst = state['subst'].copy()
    new_subst[name] = term
    return {'subst': new_subst, 'fresh': state['fresh'], 'inters': state['inters']}

def get_subst_state(name, state):
    if name in state['subst']:
        term = state['subst'][name]
        new_subst = state['subst'].copy()
        del new_subst[name]
        new_state = {'subst': new_subst, 'fresh': state['fresh'], 'inters': state['inters']}
        return term, new_state
    else:
        return None, state

# Evaluator primitives: operate on tuple Terms
def app_era(f, arg, state):
    if f[0] == 'Era':
        state2 = incr_inters(state)
        return ('Era',), state2
    else:
        raise RuntimeError("app_era: expected Era")

def app_lam(f, arg, app_lab, state):
    # f = ('Lam', label, x, body)
    lam_lab, nam, bod = f[1], f[2], f[3]
    state1 = incr_inters(state)
    if lam_lab == app_lab:
        state2 = set_subst_state(nam, arg, state1)
        return whnf(bod, state2)
    else:
        y, state2 = fresh_state(state1)
        z, state3 = fresh_state(state2)
        state4 = set_subst_state(nam, ('Lam', app_lab, y, ('Var', z)), state3)
        inner = ('App', lam_lab, arg, ('Var', y))
        newb = ('App', app_lab, bod, inner)
        return whnf(('Lam', lam_lab, z, newb), state4)

def app_sup(f, arg, app_lab, state):
    lab, left, right = f[1], f[2], f[3]
    state1 = incr_inters(state)
    c0, state2 = fresh_state(state1)
    c1, state3 = fresh_state(state2)
    a0 = ('App', app_lab, left, ('Var', c0))
    a1 = ('App', app_lab, right, ('Var', c1))
    return whnf(('Dup', lab, c0, c1, arg, ('Sup', lab, a0, a1)), state3)

def app_dup(term, state):
    # term = ('App', label, func, ('Dup',...))
    dup = term[3]
    state1 = incr_inters(state)
    return whnf(('Dup', dup[1], dup[2], dup[4], ('App', term[1], term[2], dup[5])), state1)

def dup_era(dup_term, v, state):
    state1 = incr_inters(state)
    state2 = set_subst_state(dup_term[2], ('Era',), state1)  # x
    state3 = set_subst_state(dup_term[3], ('Era',), state2)  # y
    return whnf(dup_term[5], state3)  # body

def dup_lam(dup_term, v, state):
    # dup_term = ('Dup', label, x, y, val, body), v = ('Lam', lam_lab, x_var, f)
    state1 = incr_inters(state)
    lam_lab, x_var, f = v[1], v[2], v[3]
    x0, state2 = fresh_state(state1)
    x1, state3 = fresh_state(state2)
    f0, state4 = fresh_state(state3)
    f1, state5 = fresh_state(state4)
    state6 = set_subst_state(dup_term[2], ('Lam', lam_lab, x0, ('Var', f0)), state5)
    state7 = set_subst_state(dup_term[3], ('Lam', lam_lab, x1, ('Var', f1)), state6)
    state8 = set_subst_state(x_var, ('Sup', dup_term[1], ('Var', x0), ('Var', x1)), state7)
    return whnf(('Dup', dup_term[1], f0, f1, f, dup_term[5]), state8)

def dup_sup(dup_term, v, state):
    # dup_term = ('Dup', label, x, y, val, body), v = ('Sup', sup_lab, a, b)
    state1 = incr_inters(state)
    sup_lab, a, b = v[1], v[2], v[3]
    if dup_term[1] == sup_lab:
        state2 = set_subst_state(dup_term[2], a, state1)
        state3 = set_subst_state(dup_term[3], b, state2)
        return whnf(dup_term[5], state3)
    else:
        a0, state2 = fresh_state(state1)
        a1, state3 = fresh_state(state2)
        b0, state4 = fresh_state(state3)
        b1, state5 = fresh_state(state4)
        state6 = set_subst_state(dup_term[2], ('Sup', sup_lab, ('Var', a0), ('Var', b0)), state5)
        state7 = set_subst_state(dup_term[3], ('Sup', sup_lab, ('Var', a1), ('Var', b1)), state6)
        inner = ('Dup', dup_term[1], b0, b1, b, dup_term[5])
        return whnf(('Dup', dup_term[1], a0, a1, a, inner), state7)

def dup_dup(dup_term, v, state):
    state1 = incr_inters(state)
    # mirror pattern: Dup lab x y _ t; inner Dup lab y0 y1 y x
    return whnf(('Dup', dup_term[1], dup_term[2], dup_term[4], v[2], ('Dup', dup_term[1], v[2], v[3], v[4], dup_term[5])), state1)

def whnf(term, state):
    tag = term[0]
    if tag == 'Var':
        sub, state1 = get_subst_state(term[1], state)
        if sub is not None:
            return whnf(sub, state1)
        else:
            return term, state1
    elif tag == 'Let':
        state1 = set_subst_state(term[1], term[2], state)
        return whnf(term[3], state1)
    elif tag == 'App':
        f_term, state1 = whnf(term[2], state)
        if f_term[0] == 'Lam':
            return app_lam(f_term, term[3], term[1], state1)
        elif f_term[0] == 'Sup':
            return app_sup(f_term, term[3], term[1], state1)
        elif f_term[0] == 'Era':
            return app_era(f_term, term[3], state1)
        elif f_term[0] == 'Dup':
            return app_dup(('App', term[1], f_term, term[3]), state1)
        else:
            return ('App', term[1], f_term, term[3]), state1
    elif tag == 'Dup':
        v_term, state1 = whnf(term[4], state)
        if v_term[0] == 'Lam':
            return dup_lam(term, v_term, state1)
        elif v_term[0] == 'Sup':
            return dup_sup(term, v_term, state1)
        elif v_term[0] == 'Era':
            return dup_era(term, v_term, state1)
        elif v_term[0] == 'Dup':
            return dup_dup(term, v_term, state1)
        else:
            return ('Dup', term[1], term[2], term[3], v_term, term[5]), state1
    else:
        # Era or Lam or Sup when not reduced further
        return term, state

def normal(term, state):
    t_whnf, state1 = whnf(term, state)
    tag = t_whnf[0]
    if tag in ('Var', 'Era'):
        return t_whnf, state1
    elif tag == 'Lam':
        body_n, state2 = normal(t_whnf[3], state1)
        return ('Lam', t_whnf[1], t_whnf[2], body_n), state2
    elif tag == 'App':
        fn_n, state2 = normal(t_whnf[2], state1)
        arg_n, state3 = normal(t_whnf[3], state2)
        return ('App', t_whnf[1], fn_n, arg_n), state3
    elif tag == 'Sup':
        left_n, state2 = normal(t_whnf[2], state1)
        right_n, state3 = normal(t_whnf[3], state2)
        return ('Sup', t_whnf[1], left_n, right_n), state3
    elif tag == 'Dup':
        val_n, state2 = normal(t_whnf[4], state1)
        body_n, state3 = normal(t_whnf[5], state2)
        return ('Dup', t_whnf[1], t_whnf[2], t_whnf[3], val_n, body_n), state3
    else:
        return t_whnf, state1

# Parser: returns term and final fresh counter
class ParseError(Exception):
    pass

class Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.len = len(text)
        self.global_map = {}
        self.fresh = 0

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
            raise ParseError(f"Expected '{s}' at pos {self.pos}")

    def parse_natural(self) -> int:
        self.skip_whitespace_and_comments()
        start = self.pos
        while not self.eof() and self.current().isdigit():
            self.advance()
        if start == self.pos:
            raise ParseError(f"Expected number at pos {self.pos}")
        return int(self.text[start:self.pos])

    def parseVarName(self) -> str:
        self.skip_whitespace_and_comments()
        if not self.eof() and self.current() == '$':
            start = self.pos
            self.advance()
            while not self.eof() and (self.current().isalnum() or self.current()=='_'):
                self.advance()
            return self.text[start:self.pos]
        else:
            start = self.pos
            if not self.eof() and (self.current().isalnum() or self.current()=='_'):
                while not self.eof() and (self.current().isalnum() or self.current()=='_'):
                    self.advance()
                return self.text[start:self.pos]
            else:
                raise ParseError(f"Expected var at pos {self.pos}")

    def isGlobalName(self, name: str) -> bool:
        return name.startswith('$')

    def getGlobalName(self, gname: str) -> int:
        if gname in self.global_map:
            return self.global_map[gname]
        else:
            n = self.fresh
            self.fresh += 1
            self.global_map[gname] = n
            return n

    def bindVar(self, name: str, ctx: dict):
        if self.isGlobalName(name):
            n = self.getGlobalName(name)
            return n, ctx
        else:
            n = self.fresh
            self.fresh += 1
            ctx2 = ctx.copy()
            ctx2[name] = n
            return n, ctx2

    def getVar(self, name: str, ctx: dict):
        if self.isGlobalName(name):
            return self.getGlobalName(name)
        else:
            if name in ctx:
                return ctx[name]
            else:
                raise ParseError(f"Unbound var {name}")

    def parse(self):
        self.skip_whitespace_and_comments()
        term = self.parseTerm({})
        self.skip_whitespace_and_comments()
        if not self.eof():
            raise ParseError(f"Trailing input at pos {self.pos}")
        return term, self.fresh

    def parseTerm(self, ctx):
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        for fn in [self.parseApp, self.parseLet, self.parseLam, self.parseSup, self.parseDup]:
            try:
                return fn(ctx)
            except ParseError:
                self.pos = pos0
        return self.parseSimpleTerm(ctx)

    def parseSimpleTerm(self, ctx):
        self.skip_whitespace_and_comments()
        if self.pos<self.len and self.current()=='*':
            return self.parseEra()
        elif self.pos<self.len and self.current()=='(':
            self.expect_string('(')
            t = self.parseTerm(ctx)
            self.expect_string(')')
            return t
        else:
            return self.parseVar(ctx)

    def parseVar(self, ctx):
        name = self.parseVarName()
        n = self.getVar(name, ctx)
        return ('Var', n)

    def parseEra(self):
        self.expect_string('*')
        return ('Era',)

    def parseLam(self, ctx):
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("&"):
            try:
                lab = self.parse_natural()
                if not self.match_string("λ"):
                    raise ParseError("Expected λ")
                varname = self.parseVarName()
                n, ctx2 = self.bindVar(varname, ctx)
                self.expect_string(".")
                body = self.parseTerm(ctx2)
                return ('Lam', lab, n, body)
            except ParseError:
                self.pos = pos0
        if self.match_string("λ"):
            varname = self.parseVarName()
            n, ctx2 = self.bindVar(varname, ctx)
            self.expect_string(".")
            body = self.parseTerm(ctx2)
            return ('Lam', 0, n, body)
        if self.match_string("Λ"):
            varname = self.parseVarName()
            n, ctx2 = self.bindVar(varname, ctx)
            self.expect_string(".")
            body = self.parseTerm(ctx2)
            return ('Lam', 1, n, body)
        raise ParseError("Not lambda")

    def parseApp(self, ctx):
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
                return ('App', lab, f, a)
            except ParseError:
                self.pos = pos0
        if self.match_string("("):
            f = self.parseTerm(ctx)
            self.skip_whitespace_and_comments()
            a = self.parseTerm(ctx)
            self.expect_string(")")
            return ('App', 0, f, a)
        if self.match_string("["):
            f = self.parseTerm(ctx)
            self.skip_whitespace_and_comments()
            a = self.parseTerm(ctx)
            self.expect_string("]")
            return ('App', 1, f, a)
        raise ParseError("Not app")

    def parseSup(self, ctx):
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
                return ('Sup', lab, a, b)
            except ParseError:
                self.pos = pos0
        if self.match_string("{"):
            a = self.parseTerm(ctx)
            self.expect_string(",")
            b = self.parseTerm(ctx)
            self.expect_string("}")
            return ('Sup', 0, a, b)
        if self.match_string("<"):
            a = self.parseTerm(ctx)
            self.expect_string(",")
            b = self.parseTerm(ctx)
            self.expect_string(">")
            return ('Sup', 1, a, b)
        raise ParseError("Not sup")

    def parseDup(self, ctx):
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
                        return ('Dup', lab, n1, n2, val, body)
                    else:
                        raise ParseError("Expected { after !&")
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
                    return ('Dup', 0, n1, n2, val, body)
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
                    return ('Dup', 1, n1, n2, val, body)
        raise ParseError("Not dup")

    def parseLet(self, ctx):
        self.skip_whitespace_and_comments()
        pos0 = self.pos
        if self.match_string("!"):
            name_token = self.parseVarName()
            self.expect_string("=")
            t1 = self.parseTerm(ctx)
            self.expect_string(";")
            n, ctx2 = self.bindVar(name_token, ctx)
            t2 = self.parseTerm(ctx2)
            return ('Let', n, t1, t2)
        self.pos = pos0
        raise ParseError("Not let")

def parseIC(input_str):
    parser = Parser(input_str)
    return parser.parse()

def test_term(input_str):
    term, fresh_after = parseIC(input_str)
    init_state = {'subst': {}, 'fresh': fresh_after, 'inters': 0}
    norm_term, final_state = normal(term, init_state)
    # Convert term to string via name_str recursion
    def term_str(t):
        tag = t[0]
        if tag == 'Var':
            return name_str(t[1])
        if tag == 'Era':
            return "*"
        if tag == 'Lam':
            lab, x, body = t[1], t[2], t[3]
            xs = name_str(x)
            if lab == 0: return f"λ{xs}.{term_str(body)}"
            if lab == 1: return f"Λ{xs}.{term_str(body)}"
            return f"&{lab} λ{xs}.{term_str(body)}"
        if tag == 'App':
            lab, f, a = t[1], t[2], t[3]
            if lab == 0: return f"({term_str(f)} {term_str(a)})"
            if lab == 1: return f"[{term_str(f)} {term_str(a)}]"
            return f"&{lab} ({term_str(f)} {term_str(a)})"
        if tag == 'Sup':
            lab, lft, rgt = t[1], t[2], t[3]
            if lab == 0: return f"{{{term_str(lft)},{term_str(rgt)}}}"
            if lab == 1: return f"<{term_str(lft)},{term_str(rgt)}>"
            return f"&{lab}{{{term_str(lft)},{term_str(rgt)}}}"
        if tag == 'Dup':
            lab, x, y, val, body = t[1], t[2], t[3], t[4], t[5]
            xs, ys = name_str(x), name_str(y)
            if lab == 0: return f"! {{{xs},{ys}}} = {term_str(val)}; {term_str(body)}"
            if lab == 1: return f"! <{xs},{ys}> = {term_str(val)}; {term_str(body)}"
            return f"! &{lab}{{{xs},{ys}}} = {term_str(val)}; {term_str(body)}"
        if tag == 'Let':
            x, v, b = t[1], t[2], t[3]
            return f"! {name_str(x)} = {term_str(v)}; {term_str(b)}"
        return str(t)
    print(term_str(norm_term))
    print(f"- WORK: {final_state['inters']}")

# Run test as before
def test_ic():
    s = """
!F = λf.
  !{f0,f1} = f;
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  !{f0,f1} = λx.(f0 (f1 x));
  λx.(f0 (f1 x));
((F λnx.((nx λt0.λf0.f0) λt1.λf1.t1)) λT.λF.T)
"""
    test_term(s)
    # parse+eval again or capture inters from test_term if returned:
    term, fresh_after = parseIC(s)
    init_state = {'subst': {}, 'fresh': fresh_after, 'inters': 0}
    _, final_state = normal(term, init_state)
    print(f"- WORK: {final_state['inters']}")
    

# Execute test
import sys
sys.setrecursionlimit(10**6)
test_ic()
