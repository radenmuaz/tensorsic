import numpy as np

# --- IC Term classes and parser matching original grammar ---

# Term classes
class Term: pass

class Var(Term):
    def __init__(self, name: int):
        self.name = name
    def __str__(self):
        return f"Var({self.name})"
    __repr__ = __str__

class Let(Term):
    def __init__(self, name: int, t1: Term, t2: Term):
        self.name = name; self.t1 = t1; self.t2 = t2
    def __str__(self):
        return f"Let({self.name}={self.t1};{self.t2})"
    __repr__ = __str__

class Era(Term):
    def __str__(self):
        return "*"
    __repr__ = __str__

class Sup(Term):
    def __init__(self, label: int, left: Term, right: Term):
        self.label = label; self.left = left; self.right = right
    def __str__(self):
        return f"Sup({self.label},{self.left},{self.right})"
    __repr__ = __str__

class Dup(Term):
    def __init__(self, label: int, x: int, y: int, val: Term, body: Term):
        self.label = label; self.x = x; self.y = y; self.val = val; self.body = body
    def __str__(self):
        return f"Dup({self.label},{self.x},{self.y},{self.val},{self.body})"
    __repr__ = __str__

class Lam(Term):
    def __init__(self, label: int, x: int, body: Term):
        self.label = label; self.x = x; self.body = body
    def __str__(self):
        return f"Lam({self.label},{self.x},{self.body})"
    __repr__ = __str__

class App(Term):
    def __init__(self, label: int, func: Term, arg: Term):
        self.label = label; self.func = func; self.arg = arg
    def __str__(self):
        return f"App({self.label},{self.func},{self.arg})"
    __repr__ = __str__

# Parser for original IC syntax
class ParseError(Exception):
    pass

class ParserIC:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.len = len(text)
        self.global_map = {}
        self._fresh = 0

    def fresh(self):
        v = self._fresh
        self._fresh += 1
        return v

    def skip_ws(self):
        while self.pos < self.len and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self):
        self.skip_ws()
        if self.pos < self.len:
            return self.text[self.pos]
        return None

    def consume(self, s: str):
        self.skip_ws()
        if self.text.startswith(s, self.pos):
            self.pos += len(s)
            return True
        return False

    def expect(self, s: str):
        if not self.consume(s):
            raise ParseError(f"Expected '{s}' at pos {self.pos}")

    def parseVarName(self):
        self.skip_ws()
        if self.pos < self.len and (self.text[self.pos].isalpha() or self.text[self.pos] in ['$', '_']):
            start = self.pos
            # allow $ for globals
            if self.text[self.pos] == '$':
                self.pos += 1
            while self.pos < self.len and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                self.pos += 1
            return self.text[start:self.pos]
        else:
            raise ParseError(f"Expected variable name at pos {self.pos}")

    def isGlobalName(self, name: str):
        return name.startswith('$')

    def getGlobal(self, name: str):
        if name in self.global_map:
            return self.global_map[name]
        else:
            n = self.fresh()
            self.global_map[name] = n
            return n

    def parse(self):
        term = self.parseTerm({})
        self.skip_ws()
        if self.pos < self.len:
            raise ParseError(f"Unexpected trailing at pos {self.pos}")
        return term

    def parseTerm(self, ctx):
        self.skip_ws()
        pos0 = self.pos
        for fn in [self.parseApp, self.parseLet, self.parseLam, self.parseSup, self.parseDup]:
            try:
                return fn(ctx)
            except ParseError:
                self.pos = pos0
        return self.parseSimpleTerm(ctx)

    def parseSimpleTerm(self, ctx):
        self.skip_ws()
        c = self.peek()
        if c == '(':
            self.expect('(')
            t = self.parseTerm(ctx)
            self.expect(')')
            return t
        elif c == '*':
            self.pos += 1
            return Era()
        else:
            name = self.parseVarName()
            return Var(self.getVarId(name, ctx))

    def getVarId(self, name, ctx):
        if self.isGlobalName(name):
            return self.getGlobal(name)
        if name in ctx:
            return ctx[name]
        # new local
        vid = self.fresh()
        ctx[name] = vid
        return vid

    def parseLet(self, ctx):
        self.skip_ws()
        pos0 = self.pos
        if self.consume('!'):
            name = self.parseVarName()
            self.expect('=')
            t1 = self.parseTerm(ctx)
            self.expect(';')
            vid = self.getVarId(name, ctx)
            t2 = self.parseTerm(ctx)
            return Let(vid, t1, t2)
        else:
            self.pos = pos0
            raise ParseError()

    def parseLam(self, ctx):
        self.skip_ws()
        pos0 = self.pos
        # label?
        if self.consume('&'):
            # parse numeric label
            num = self.parseNumber()
            if not self.consume('λ'):
                raise ParseError()
            name = self.parseVarName()
            vid = self.getVarId(name, ctx)
            self.expect('.')
            return Lam(num, vid, self.parseTerm(dict(ctx)))
        if self.consume('λ'):
            name = self.parseVarName()
            vid = self.getVarId(name, ctx)
            self.expect('.')
            return Lam(0, vid, self.parseTerm(dict(ctx)))
        if self.consume('Λ'):
            name = self.parseVarName()
            vid = self.getVarId(name, ctx)
            self.expect('.')
            return Lam(1, vid, self.parseTerm(dict(ctx)))
        self.pos = pos0
        raise ParseError()

    def parseApp(self, ctx):
        self.skip_ws()
        pos0 = self.pos
        if self.consume('&'):
            num = self.parseNumber()
            self.expect('(')
            f = self.parseTerm(ctx)
            a = self.parseTerm(ctx)
            self.expect(')')
            return App(num, f, a)
        if self.consume('('):
            f = self.parseTerm(ctx)
            a = self.parseTerm(ctx)
            self.expect(')')
            return App(0, f, a)
        if self.consume('['):
            f = self.parseTerm(ctx)
            a = self.parseTerm(ctx)
            self.expect(']')
            return App(1, f, a)
        self.pos = pos0
        raise ParseError()

    def parseSup(self, ctx):
        self.skip_ws()
        pos0 = self.pos
        if self.consume('&'):
            num = self.parseNumber()
            if not self.consume('{'): raise ParseError()
            a = self.parseTerm(ctx)
            self.expect(',')
            b = self.parseTerm(ctx)
            self.expect('}')
            return Sup(num, a, b)
        if self.consume('{'):
            a = self.parseTerm(ctx); self.expect(','); b = self.parseTerm(ctx); self.expect('}')
            return Sup(0, a, b)
        if self.consume('<'):
            a = self.parseTerm(ctx); self.expect(','); b = self.parseTerm(ctx); self.expect('>')
            return Sup(1, a, b)
        self.pos = pos0
        raise ParseError()

    def parseDup(self, ctx):
        self.skip_ws()
        pos0 = self.pos
        if self.consume('!'):
            if self.consume('&'):
                num = self.parseNumber()
                self.expect('{')
                n1 = self.parseVarName(); self.expect(','); n2 = self.parseVarName()
                self.expect('}'); self.expect('='); val = self.parseTerm(ctx); self.expect(';')
                vid1 = self.getVarId(n1, ctx); vid2 = self.getVarId(n2, ctx)
                body = self.parseTerm(ctx)
                return Dup(num, vid1, vid2, val, body)
            if self.consume('{'):
                n1 = self.parseVarName(); self.expect(','); n2 = self.parseVarName()
                self.expect('}'); self.expect('='); val = self.parseTerm(ctx); self.expect(';')
                vid1 = self.getVarId(n1, ctx); vid2 = self.getVarId(n2, ctx)
                body = self.parseTerm(ctx)
                return Dup(0, vid1, vid2, val, body)
            if self.consume('<'):
                n1 = self.parseVarName(); self.expect(','); n2 = self.parseVarName()
                self.expect('>'); self.expect('='); val = self.parseTerm(ctx); self.expect(';')
                vid1 = self.getVarId(n1, ctx); vid2 = self.getVarId(n2, ctx)
                body = self.parseTerm(ctx)
                return Dup(1, vid1, vid2, val, body)
        self.pos = pos0
        raise ParseError()

    def parseNumber(self):
        self.skip_ws()
        start = self.pos
        if self.pos < self.len and self.text[self.pos].isdigit():
            while self.pos < self.len and self.text[self.pos].isdigit():
                self.pos += 1
            return int(self.text[start:self.pos])
        raise ParseError(f"Expected number at pos {self.pos}")

# Term-to-array and array-to-Term for IC Terms
TYPE_VAR = 0; TYPE_LET = 1; TYPE_ERA = 2; TYPE_SUP = 3; TYPE_DUP = 4; TYPE_LAM = 5; TYPE_APP = 6

def term_to_array_ic(term):
    rows = []
    mapping = {}
    def recurse(t):
        if id(t) in mapping:
            return mapping[id(t)]
        idx = len(rows)
        mapping[id(t)] = idx
        if isinstance(t, Var):
            rows.append([TYPE_VAR, -1, -1, t.name, 1, 0, -1, -1, -1])  # extra cols unused
        elif isinstance(t, Let):
            rows.append([TYPE_LET, -1, -1, t.name, 1, 0, -1, -1, -1])
            t1 = recurse(t.t1); t2 = recurse(t.t2)
            rows[idx][1] = t1; rows[idx][2] = t2
        elif isinstance(t, Era):
            rows.append([TYPE_ERA, -1, -1, -1, 1, 0, -1, -1, -1])
        elif isinstance(t, Sup):
            rows.append([TYPE_SUP, -1, -1, -1, 1, 0, t.label, -1, -1])
            l = recurse(t.left); r = recurse(t.right)
            rows[idx][1] = l; rows[idx][2] = r
        elif isinstance(t, Dup):
            rows.append([TYPE_DUP, -1, -1, -1, 1, 0, t.label, t.x, t.y])
            v = recurse(t.val); b = recurse(t.body)
            rows[idx][1] = v; rows[idx][2] = b
        elif isinstance(t, Lam):
            rows.append([TYPE_LAM, -1, -1, t.x, 1, 0, t.label, -1, -1])
            b = recurse(t.body); rows[idx][1] = b
        elif isinstance(t, App):
            rows.append([TYPE_APP, -1, -1, -1, 1, 0, t.label, -1, -1])
            f = recurse(t.func); a = recurse(t.arg)
            rows[idx][1] = f; rows[idx][2] = a
        else:
            raise RuntimeError("Unknown Term")
        return idx
    root = recurse(term)
    arr = np.array(rows, dtype=np.int32)
    return arr, root

def array_to_term_ic(arr, root_idx):
    cache = {}
    def build(i):
        if i in cache:
            return cache[i]
        typ = arr[i,0]
        c1 = arr[i,1]; c2 = arr[i,2]
        var_id = arr[i,3]; active = arr[i,4]
        label = arr[i,6]
        if active == 0:
            t = Era()
        else:
            if typ == TYPE_VAR:
                t = Var(var_id)
            elif typ == TYPE_LET:
                t1 = build(c1); t2 = build(c2)
                t = Let(var_id, t1, t2)
            elif typ == TYPE_ERA:
                t = Era()
            elif typ == TYPE_SUP:
                left = build(c1); right = build(c2)
                t = Sup(label, left, right)
            elif typ == TYPE_DUP:
                x = arr[i,7]; y = arr[i,8]
                val = build(c1); body = build(c2)
                t = Dup(label, x, y, val, body)
            elif typ == TYPE_LAM:
                body = build(c1)
                t = Lam(label, var_id, body)
            elif typ == TYPE_APP:
                func = build(c1); arg = build(c2)
                t = App(label, func, arg)
            else:
                raise RuntimeError("Bad type")
        cache[i] = t
        return t
    return build(root_idx)

# --- Tests: identity and boolean in IC syntax ---
tests = [
    # identity: ((λx.x) *)
    "( (λx.x) * )",
    # boolean: (( (λa.(λb.a)) t) f)
    "( ( (λa.(λb.a)) t) f)"
]

for s in tests:
    print("Input:", s)
    parser = ParserIC(s)
    try:
        term = parser.parse()
    except Exception as e:
        print("Parse error:", e); continue
    arr, root = term_to_array_ic(term)
    recon = array_to_term_ic(arr, root)
    print("Parsed Term:", term)
    print("Array shape:", arr.shape)
    print(arr)
    print("Reconstructed Term:", recon)
    print("---")
