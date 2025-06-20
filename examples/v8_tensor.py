import numpy as np
import sys
from collections import deque

# --- Type tags ---
TYPE_VAR = 0
TYPE_LET = 1
TYPE_ERA = 2
TYPE_SUP = 3
TYPE_DUP = 4
TYPE_LAM = 5
TYPE_APP = 6

# --- Column indices in static array ---
COL_TYPE    = 0
COL_C1      = 1
COL_C2      = 2
COL_VAR     = 3
COL_ACTIVE  = 4
COL_LABEL   = 5
COL_PARENT  = 6
COL_PFIELD  = 7
COL_X       = 8
COL_Y       = 9

NUM_COLS = 10

# --- AST classes ---
class Term: pass
class Var(Term):
    def __init__(self, name:int): self.name=name
    def __repr__(self): return f"Var({self.name})"
class Era(Term):
    def __repr__(self): return "*"
class Let(Term):
    def __init__(self, name:int, t1:Term, t2:Term):
        self.name, self.t1, self.t2 = name, t1, t2
    def __repr__(self): return f"Let({self.name}={self.t1};{self.t2})"
class Sup(Term):
    def __init__(self, label:int, left:Term, right:Term):
        self.label, self.left, self.right = label, left, right
    def __repr__(self): return f"Sup({self.label},{self.left},{self.right})"
class Dup(Term):
    def __init__(self, label:int, x:int, y:int, val:Term, body:Term):
        self.label, self.x, self.y, self.val, self.body = label, x, y, val, body
    def __repr__(self): return f"Dup({self.label},{self.x},{self.y},{self.val},{self.body})"
class Lam(Term):
    def __init__(self, label:int, x:int, body:Term):
        self.label, self.x, self.body = label, x, body
    def __repr__(self): return f"Lam({self.label},{self.x},{self.body})"
class App(Term):
    def __init__(self, label:int, func:Term, arg:Term):
        self.label, self.func, self.arg = label, func, arg
    def __repr__(self): return f"App({self.label},{self.func},{self.arg})"

# --- Parser for IC syntax ---
class ParseError(Exception): pass

class ParserIC:
    def __init__(self, text:str):
        self.text = text; self.pos = 0; self.len = len(text)
        self.global_map = {}; self._fresh = 0

    def fresh(self):
        v = self._fresh; self._fresh += 1; return v

    def skip_ws(self):
        while self.pos < self.len and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self):
        self.skip_ws()
        return self.text[self.pos] if self.pos < self.len else None

    def consume(self, s:str):
        self.skip_ws()
        if self.text.startswith(s, self.pos):
            self.pos += len(s); return True
        return False

    def expect(self, s:str):
        if not self.consume(s):
            raise ParseError(f"Expected '{s}' at {self.pos}")

    def parseVarName(self):
        self.skip_ws()
        if self.pos < self.len and (self.text[self.pos].isalpha() or self.text[self.pos]=='$' or self.text[self.pos]=='_'):
            start = self.pos
            if self.text[self.pos] == '$':
                self.pos += 1
            while self.pos < self.len and (self.text[self.pos].isalnum() or self.text[self.pos]=='_'):
                self.pos += 1
            return self.text[start:self.pos]
        raise ParseError(f"Expected variable at {self.pos}")

    def isGlobal(self, name:str):
        return name.startswith('$')

    def getGlobal(self, name:str):
        if name not in self.global_map:
            self.global_map[name] = self.fresh()
        return self.global_map[name]

    def getVarId(self, name:str, ctx:dict):
        if self.isGlobal(name):
            return self.getGlobal(name)
        if name not in ctx:
            ctx[name] = self.fresh()
        return ctx[name]

    def parse(self):
        term = self.parseTerm({})
        self.skip_ws()
        if self.pos < self.len:
            raise ParseError(f"Trailing at {self.pos}")
        return term

    def parseTerm(self, ctx):
        self.skip_ws(); pos0 = self.pos
        for fn in (self.parseLet, self.parseLam, self.parseDup, self.parseSup, self.parseApp):
            try:
                return fn(ctx)
            except ParseError:
                self.pos = pos0
        return self.parseSimple(ctx)

    def parseSimple(self, ctx):
        c = self.peek()
        if c == '(':
            self.expect('(')
            t = self.parseTerm(ctx)
            self.expect(')')
            return t
        if c == '*':
            self.pos += 1
            return Era()
        name = self.parseVarName()
        vid = self.getVarId(name, ctx)
        return Var(vid)

    def parseLet(self, ctx):
        if self.consume('!'):
            name = self.parseVarName()
            self.expect('=')
            t1 = self.parseTerm(ctx)
            self.expect(';')
            vid = self.getVarId(name, ctx)
            t2 = self.parseTerm(ctx)
            return Let(vid, t1, t2)
        raise ParseError()

    def parseLam(self, ctx):
        pos0 = self.pos
        if self.consume('&'):
            num = ''
            while self.peek() and self.peek().isdigit():
                num += self.text[self.pos]; self.pos += 1
            lab = int(num or 0)
            self.expect('λ')
            name = self.parseVarName()
            vid = self.getVarId(name, ctx)
            self.expect('.')
            body = self.parseTerm(dict(ctx))
            return Lam(lab, vid, body)
        if self.consume('λ'):
            name = self.parseVarName(); vid = self.getVarId(name, ctx)
            self.expect('.')
            return Lam(0, vid, self.parseTerm(dict(ctx)))
        if self.consume('Λ'):
            name = self.parseVarName(); vid = self.getVarId(name, ctx)
            self.expect('.')
            return Lam(1, vid, self.parseTerm(dict(ctx)))
        raise ParseError()

    def parseApp(self, ctx):
        pos0 = self.pos
        if self.consume('&'):
            num = ''
            while self.peek() and self.peek().isdigit():
                num += self.text[self.pos]; self.pos += 1
            lab = int(num or 0)
            self.expect('(')
            f = self.parseTerm(ctx)
            a = self.parseTerm(ctx)
            self.expect(')')
            return App(lab, f, a)
        if self.consume('('):
            f = self.parseTerm(ctx); a = self.parseTerm(ctx); self.expect(')')
            return App(0, f, a)
        if self.consume('['):
            f = self.parseTerm(ctx); a = self.parseTerm(ctx); self.expect(']')
            return App(1, f, a)
        raise ParseError()

    def parseSup(self, ctx):
        pos0 = self.pos
        if self.consume('&'):
            num = ''
            while self.peek() and self.peek().isdigit():
                num += self.text[self.pos]; self.pos += 1
            lab = int(num or 0)
            if self.consume('{'):
                left = self.parseTerm(ctx); self.expect(','); right = self.parseTerm(ctx); self.expect('}')
                return Sup(lab, left, right)
            self.pos = pos0; raise ParseError()
        if self.consume('{'):
            left = self.parseTerm(ctx); self.expect(','); right = self.parseTerm(ctx); self.expect('}')
            return Sup(0, left, right)
        if self.consume('<'):
            left = self.parseTerm(ctx); self.expect(','); right = self.parseTerm(ctx); self.expect('>')
            return Sup(1, left, right)
        raise ParseError()

    def parseDup(self, ctx):
        pos0 = self.pos
        if self.consume('!'):
            lab = 0
            if self.consume('&'):
                num = ''
                while self.peek() and self.peek().isdigit():
                    num += self.text[self.pos]; self.pos += 1
                lab = int(num or 0)
            if self.consume('{'):
                name1 = self.parseVarName(); self.expect(','); name2 = self.parseVarName(); self.expect('}')
            elif self.consume('<'):
                name1 = self.parseVarName(); self.expect(','); name2 = self.parseVarName(); self.expect('>')
                lab = 1 if lab==0 else lab
            else:
                self.pos = pos0; raise ParseError()
            self.expect('=')
            val = self.parseTerm(ctx)
            self.expect(';')
            vid1 = self.getVarId(name1, ctx)
            vid2 = self.getVarId(name2, ctx)
            body = self.parseTerm(ctx)
            return Dup(lab, vid1, vid2, val, body)
        raise ParseError()

# --- Layout & helpers ---
def build_initial_layout(term):
    offsets, parents, order = {}, {}, []
    def dfs(u, parent_off=None, field=None):
        if id(u) not in offsets:
            off = len(order); offsets[id(u)] = off; order.append(u)
            if parent_off is not None:
                parents[off] = (parent_off, field)
            if isinstance(u, Lam):
                dfs(u.body, off, COL_C1)
            elif isinstance(u, App):
                dfs(u.func, off, COL_C1); dfs(u.arg, off, COL_C2)
            elif isinstance(u, Let):
                dfs(u.t1, off, COL_C1); dfs(u.t2, off, COL_C2)
            elif isinstance(u, Sup):
                dfs(u.left, off, COL_C1); dfs(u.right, off, COL_C2)
            elif isinstance(u, Dup):
                dfs(u.val, off, COL_C1); dfs(u.body, off, COL_C2)
        else:
            off = offsets[id(u)]
            if parent_off is not None:
                parents[off] = (parent_off, field)
        return offsets[id(u)]
    dfs(term)
    return offsets, parents, order

def allocate_static_array(N_initial, MAX_DEPTH, K):
    block_size = N_initial + N_initial * K * MAX_DEPTH
    arr = np.zeros((block_size, NUM_COLS), dtype=np.int32)
    arr[:, COL_TYPE]   = -1
    arr[:, COL_ACTIVE] = 0
    arr[:, COL_PARENT] = -1
    arr[:, COL_PFIELD] = 0
    return arr

def init_static_array(term, MAX_DEPTH, K):
    offsets, parents, order = build_initial_layout(term)
    N = len(order)
    nodes = allocate_static_array(N, MAX_DEPTH, K)
    for off, u in enumerate(order):
        if isinstance(u, Var):
            nodes[off, COL_TYPE]   = TYPE_VAR
            nodes[off, COL_VAR ]   = u.name
            nodes[off, COL_ACTIVE] = 1
        elif isinstance(u, Era):
            nodes[off, COL_TYPE]   = TYPE_ERA
            nodes[off, COL_ACTIVE] = 1
        elif isinstance(u, Lam):
            c = offsets[id(u.body)]
            nodes[off, COL_TYPE]   = TYPE_LAM
            nodes[off, COL_C1]     = c
            nodes[off, COL_VAR]    = u.x
            nodes[off, COL_ACTIVE] = 1
            nodes[off, COL_LABEL]  = u.label
            nodes[c, COL_PARENT]   = off; nodes[c, COL_PFIELD] = COL_C1
        elif isinstance(u, App):
            f = offsets[id(u.func)]; a = offsets[id(u.arg)]
            nodes[off, COL_TYPE]   = TYPE_APP
            nodes[off, COL_C1]     = f
            nodes[off, COL_C2]     = a
            nodes[off, COL_ACTIVE] = 1
            nodes[off, COL_LABEL]  = u.label
            nodes[f, COL_PARENT]   = off; nodes[f, COL_PFIELD] = COL_C1
            nodes[a, COL_PARENT]   = off; nodes[a, COL_PFIELD] = COL_C2
        elif isinstance(u, Let):
            t1 = offsets[id(u.t1)]; t2 = offsets[id(u.t2)]
            nodes[off, COL_TYPE]   = TYPE_LET
            nodes[off, COL_C1]     = t1
            nodes[off, COL_C2]     = t2
            nodes[off, COL_VAR]    = u.name
            nodes[off, COL_ACTIVE] = 1
            nodes[t1, COL_PARENT]  = off; nodes[t1, COL_PFIELD] = COL_C1
            nodes[t2, COL_PARENT]  = off; nodes[t2, COL_PFIELD] = COL_C2
        elif isinstance(u, Sup):
            l = offsets[id(u.left)]; r = offsets[id(u.right)]
            nodes[off, COL_TYPE]   = TYPE_SUP
            nodes[off, COL_C1]     = l
            nodes[off, COL_C2]     = r
            nodes[off, COL_ACTIVE] = 1
            nodes[off, COL_LABEL]  = u.label
            nodes[l, COL_PARENT]   = off; nodes[l, COL_PFIELD] = COL_C1
            nodes[r, COL_PARENT]   = off; nodes[r, COL_PFIELD] = COL_C2
        elif isinstance(u, Dup):
            v = offsets[id(u.val)]; b = offsets[id(u.body)]
            nodes[off, COL_TYPE]   = TYPE_DUP
            nodes[off, COL_C1]     = v
            nodes[off, COL_C2]     = b
            nodes[off, COL_ACTIVE] = 1
            nodes[off, COL_LABEL]  = u.label
            nodes[off, COL_X]      = u.x
            nodes[off, COL_Y]      = u.y
            nodes[v, COL_PARENT]   = off; nodes[v, COL_PFIELD] = COL_C1
            nodes[b, COL_PARENT]   = off; nodes[b, COL_PFIELD] = COL_C2
    return nodes, len(order)

def clone_base(i, N, K, d):
    return N + (d * N + i) * K

def collect_subtree(nodes, root_off, max_iter=50):
    bs = nodes.shape[0]
    parent = nodes[:, COL_PARENT]
    mask_sub = np.zeros(bs, bool)
    mask_front = np.zeros(bs, bool)
    mask_sub[root_off] = True
    mask_front[root_off] = True
    for _ in range(max_iter):
        idxs = np.nonzero(mask_front)[0]
        if idxs.size == 0: break
        mask_front[:] = False
        valid = (parent >= 0)
        ii = np.nonzero(valid)[0]
        ps = parent[ii]
        m = mask_sub[ps]
        children = ii[m]
        new = children[~mask_sub[children]]
        if new.size == 0: break
        mask_sub[new] = True
        mask_front[new] = True
    return mask_sub

# --- Reduction rules implementation ---
def app_era(nodes, i, d, N_initial, K):
    f_off = nodes[i, COL_C1]
    a_off = nodes[i, COL_C2]
    app_lab = nodes[i, COL_LABEL]
    
    # Deactivate App, Era, and argument
    nodes[i, COL_ACTIVE] = 0
    nodes[f_off, COL_ACTIVE] = 0
    nodes[a_off, COL_ACTIVE] = 0

    # Create new Era node
    base = clone_base(i, N_initial, K, d)
    new_era_off = base
    nodes[new_era_off, COL_TYPE] = TYPE_ERA
    nodes[new_era_off, COL_ACTIVE] = 1

    # Update parent pointer
    p = nodes[i, COL_PARENT]
    pf = nodes[i, COL_PFIELD]
    if p >= 0:
        nodes[p, pf] = new_era_off
        nodes[new_era_off, COL_PARENT] = p
        nodes[new_era_off, COL_PFIELD] = pf
        return new_era_off
    else:
        return new_era_off

def app_sup(nodes, i, d, N_initial, K):
    f_off = nodes[i, COL_C1]
    a_off = nodes[i, COL_C2]
    app_lab = nodes[i, COL_LABEL]
    lab_sup = nodes[f_off, COL_LABEL]
    l_off = nodes[f_off, COL_C1]
    r_off = nodes[f_off, COL_C2]
    
    # Deactivate App and Sup
    nodes[i, COL_ACTIVE] = 0
    nodes[f_off, COL_ACTIVE] = 0

    # Create new nodes in preallocated block
    base = clone_base(i, N_initial, K, d)
    off_c0 = base
    off_c1 = base + 1
    off_app0 = base + 2
    off_app1 = base + 3
    off_sup = base + 4
    off_dup = base + 5

    # Create variables
    nodes[off_c0, COL_TYPE] = TYPE_VAR
    nodes[off_c0, COL_VAR] = off_c0
    nodes[off_c0, COL_ACTIVE] = 1

    nodes[off_c1, COL_TYPE] = TYPE_VAR
    nodes[off_c1, COL_VAR] = off_c1
    nodes[off_c1, COL_ACTIVE] = 1

    # Create Apps
    nodes[off_app0, COL_TYPE] = TYPE_APP
    nodes[off_app0, COL_LABEL] = app_lab
    nodes[off_app0, COL_C1] = l_off
    nodes[off_app0, COL_C2] = off_c0
    nodes[off_app0, COL_ACTIVE] = 1
    nodes[l_off, COL_PARENT] = off_app0
    nodes[l_off, COL_PFIELD] = COL_C1
    nodes[off_c0, COL_PARENT] = off_app0
    nodes[off_c0, COL_PFIELD] = COL_C2

    nodes[off_app1, COL_TYPE] = TYPE_APP
    nodes[off_app1, COL_LABEL] = app_lab
    nodes[off_app1, COL_C1] = r_off
    nodes[off_app1, COL_C2] = off_c1
    nodes[off_app1, COL_ACTIVE] = 1
    nodes[r_off, COL_PARENT] = off_app1
    nodes[r_off, COL_PFIELD] = COL_C1
    nodes[off_c1, COL_PARENT] = off_app1
    nodes[off_c1, COL_PFIELD] = COL_C2

    # Create Sup
    nodes[off_sup, COL_TYPE] = TYPE_SUP
    nodes[off_sup, COL_LABEL] = lab_sup
    nodes[off_sup, COL_C1] = off_app0
    nodes[off_sup, COL_C2] = off_app1
    nodes[off_sup, COL_ACTIVE] = 1
    nodes[off_app0, COL_PARENT] = off_sup
    nodes[off_app0, COL_PFIELD] = COL_C1
    nodes[off_app1, COL_PARENT] = off_sup
    nodes[off_app1, COL_PFIELD] = COL_C2

    # Create Dup
    nodes[off_dup, COL_TYPE] = TYPE_DUP
    nodes[off_dup, COL_LABEL] = lab_sup
    nodes[off_dup, COL_X] = off_c0
    nodes[off_dup, COL_Y] = off_c1
    nodes[off_dup, COL_C1] = a_off
    nodes[off_dup, COL_C2] = off_sup
    nodes[off_dup, COL_ACTIVE] = 1
    nodes[a_off, COL_PARENT] = off_dup
    nodes[a_off, COL_PFIELD] = COL_C1
    nodes[off_sup, COL_PARENT] = off_dup
    nodes[off_sup, COL_PFIELD] = COL_C2

    # Update parent pointer
    p = nodes[i, COL_PARENT]
    pf = nodes[i, COL_PFIELD]
    if p >= 0:
        nodes[p, pf] = off_dup
        nodes[off_dup, COL_PARENT] = p
        nodes[off_dup, COL_PFIELD] = pf
        return off_dup
    else:
        return off_dup

def dup_era(nodes, i, d, N_initial, K):
    x_id = nodes[i, COL_X]
    y_id = nodes[i, COL_Y]
    v_off = nodes[i, COL_C1]
    body_off = nodes[i, COL_C2]
    
    # Deactivate Dup and Era
    nodes[i, COL_ACTIVE] = 0
    nodes[v_off, COL_ACTIVE] = 0

    base = clone_base(i, N_initial, K, d)
    mapping = {}
    queue = deque([body_off])
    seen = set([body_off])
    idx = 0
    
    while queue:
        u = queue.popleft()
        new_off = base + idx
        mapping[u] = new_off
        idx += 1
        
        # Copy node
        nodes[new_off, :] = nodes[u, :]
        nodes[new_off, COL_ACTIVE] = 1
        
        # Substitute variables with Era
        if nodes[u, COL_TYPE] == TYPE_VAR:
            if nodes[u, COL_VAR] == x_id or nodes[u, COL_VAR] == y_id:
                nodes[new_off, COL_TYPE] = TYPE_ERA
                nodes[new_off, COL_VAR] = -1  # Clear variable ID
        
        # Check children
        if nodes[u, COL_TYPE] in [TYPE_APP, TYPE_SUP, TYPE_DUP, TYPE_LET]:
            for col in [COL_C1, COL_C2]:
                child = nodes[u, col]
                if child >= 0 and child not in seen:
                    seen.add(child)
                    queue.append(child)
        elif nodes[u, COL_TYPE] == TYPE_LAM:
            child = nodes[u, COL_C1]
            if child >= 0 and child not in seen:
                seen.add(child)
                queue.append(child)
    
    new_body_off = mapping[body_off]
    
    # Update parent pointer
    p = nodes[i, COL_PARENT]
    pf = nodes[i, COL_PFIELD]
    if p >= 0:
        nodes[p, pf] = new_body_off
        nodes[new_body_off, COL_PARENT] = p
        nodes[new_body_off, COL_PFIELD] = pf
        return new_body_off
    else:
        return new_body_off

# --- Static runner with all reduction rules ---
def static_run(nodes, N_initial, MAX_DEPTH, K):
    bs = nodes.shape[0]
    root_off = 0
    
    for d in range(MAX_DEPTH):
        types = nodes[:, COL_TYPE]
        active = nodes[:, COL_ACTIVE]
        c1 = nodes[:, COL_C1]
        c2 = nodes[:, COL_C2]
        labels = nodes[:, COL_LABEL]
        
        reduced = False
        
        # Scan for redexes in order
        for i in range(bs):
            if not active[i]:
                continue
                
            # App rules
            if types[i] == TYPE_APP and active[c1[i]]:
                f_type = types[c1[i]]
                app_lab = labels[i]
                
                # App-Lam
                if f_type == TYPE_LAM:
                    lam_lab = labels[c1[i]]
                    if lam_lab == app_lab:
                        # Beta reduction
                        nam = nodes[c1[i], COL_VAR]
                        arg_off = c2[i]
                        bod_off = nodes[c1[i], COL_C1]
                        
                        # Deactivate App and Lam
                        nodes[i, COL_ACTIVE] = 0
                        nodes[c1[i], COL_ACTIVE] = 0
                        
                        # Create substitutions
                        base = clone_base(i, N_initial, K, d)
                        mapping = {}
                        queue = deque([bod_off])
                        seen = set([bod_off])
                        idx = 0
                        
                        while queue:
                            u = queue.popleft()
                            new_off = base + idx
                            mapping[u] = new_off
                            idx += 1
                            
                            # Copy node
                            nodes[new_off, :] = nodes[u, :]
                            nodes[new_off, COL_ACTIVE] = 1
                            
                            # Substitute bound variable
                            if nodes[u, COL_TYPE] == TYPE_VAR and nodes[u, COL_VAR] == nam:
                                nodes[new_off, :] = nodes[arg_off, :]
                                nodes[new_off, COL_ACTIVE] = 1
                            else:
                                # Check children
                                if nodes[u, COL_TYPE] in [TYPE_APP, TYPE_SUP, TYPE_DUP, TYPE_LET]:
                                    for col in [COL_C1, COL_C2]:
                                        child = nodes[u, col]
                                        if child >= 0 and child not in seen:
                                            seen.add(child)
                                            queue.append(child)
                                elif nodes[u, COL_TYPE] == TYPE_LAM:
                                    child = nodes[u, COL_C1]
                                    if child >= 0 and child not in seen:
                                        seen.add(child)
                                        queue.append(child)
                        
                        new_bod_off = mapping[bod_off]
                        
                        # Update parent pointer
                        p = nodes[i, COL_PARENT]
                        pf = nodes[i, COL_PFIELD]
                        if p >= 0:
                            nodes[p, pf] = new_bod_off
                            nodes[new_bod_off, COL_PARENT] = p
                            nodes[new_bod_off, COL_PFIELD] = pf
                            if i == root_off:
                                root_off = new_bod_off
                        else:
                            root_off = new_bod_off
                        reduced = True
                        break
                    else:
                        # Nested-Let
                        nam = nodes[c1[i], COL_VAR]
                        arg_off = c2[i]
                        bod_off = nodes[c1[i], COL_C1]
                        
                        # Deactivate App and Lam
                        nodes[i, COL_ACTIVE] = 0
                        nodes[c1[i], COL_ACTIVE] = 0
                        
                        # Create fresh variables
                        base = clone_base(i, N_initial, K, d)
                        off_y = base
                        off_z = base + 1
                        off_f = base + 2
                        off_x = base + 3
                        off_v = base + 4
                        off_let_f = base + 5
                        off_app_x = base + 6
                        off_let_x = base + 7
                        off_app_v = base + 8
                        off_let_v = base + 9
                        off_lam = base + 10
                        
                        # Create variables
                        nodes[off_y, COL_TYPE] = TYPE_VAR
                        nodes[off_y, COL_VAR] = off_y
                        nodes[off_y, COL_ACTIVE] = 1
                        
                        nodes[off_z, COL_TYPE] = TYPE_VAR
                        nodes[off_z, COL_VAR] = off_z
                        nodes[off_z, COL_ACTIVE] = 1
                        
                        nodes[off_f, COL_TYPE] = TYPE_VAR
                        nodes[off_f, COL_VAR] = off_f
                        nodes[off_f, COL_ACTIVE] = 1
                        
                        nodes[off_x, COL_TYPE] = TYPE_VAR
                        nodes[off_x, COL_VAR] = off_x
                        nodes[off_x, COL_ACTIVE] = 1
                        
                        nodes[off_v, COL_TYPE] = TYPE_VAR
                        nodes[off_v, COL_VAR] = off_v
                        nodes[off_v, COL_ACTIVE] = 1
                        
                        # Create Let f = bod
                        nodes[off_let_f, COL_TYPE] = TYPE_LET
                        nodes[off_let_f, COL_VAR] = off_f
                        nodes[off_let_f, COL_C1] = bod_off
                        nodes[off_let_f, COL_C2] = off_let_x
                        nodes[off_let_f, COL_ACTIVE] = 1
                        nodes[bod_off, COL_PARENT] = off_let_f
                        nodes[bod_off, COL_PFIELD] = COL_C1
                        
                        # Create App x
                        nodes[off_app_x, COL_TYPE] = TYPE_APP
                        nodes[off_app_x, COL_LABEL] = lam_lab
                        nodes[off_app_x, COL_C1] = arg_off
                        nodes[off_app_x, COL_C2] = off_y
                        nodes[off_app_x, COL_ACTIVE] = 1
                        nodes[arg_off, COL_PARENT] = off_app_x
                        nodes[arg_off, COL_PFIELD] = COL_C1
                        nodes[off_y, COL_PARENT] = off_app_x
                        nodes[off_y, COL_PFIELD] = COL_C2
                        
                        # Create Let x
                        nodes[off_let_x, COL_TYPE] = TYPE_LET
                        nodes[off_let_x, COL_VAR] = off_x
                        nodes[off_let_x, COL_C1] = off_app_x
                        nodes[off_let_x, COL_C2] = off_let_v
                        nodes[off_let_x, COL_ACTIVE] = 1
                        nodes[off_app_x, COL_PARENT] = off_let_x
                        nodes[off_app_x, COL_PFIELD] = COL_C1
                        
                        # Create App v
                        nodes[off_app_v, COL_TYPE] = TYPE_APP
                        nodes[off_app_v, COL_LABEL] = app_lab
                        nodes[off_app_v, COL_C1] = off_f
                        nodes[off_app_v, COL_C2] = off_x
                        nodes[off_app_v, COL_ACTIVE] = 1
                        nodes[off_f, COL_PARENT] = off_app_v
                        nodes[off_f, COL_PFIELD] = COL_C1
                        nodes[off_x, COL_PARENT] = off_app_v
                        nodes[off_x, COL_PFIELD] = COL_C2
                        
                        # Create Let v
                        nodes[off_let_v, COL_TYPE] = TYPE_LET
                        nodes[off_let_v, COL_VAR] = off_v
                        nodes[off_let_v, COL_C1] = off_app_v
                        nodes[off_let_v, COL_C2] = off_lam
                        nodes[off_let_v, COL_ACTIVE] = 1
                        nodes[off_app_v, COL_PARENT] = off_let_v
                        nodes[off_app_v, COL_PFIELD] = COL_C1
                        
                        # Create Lam
                        nodes[off_lam, COL_TYPE] = TYPE_LAM
                        nodes[off_lam, COL_LABEL] = lam_lab
                        nodes[off_lam, COL_VAR] = off_z
                        nodes[off_lam, COL_C1] = off_v
                        nodes[off_lam, COL_ACTIVE] = 1
                        nodes[off_v, COL_PARENT] = off_lam
                        nodes[off_v, COL_PFIELD] = COL_C1
                        
                        # Update parent pointer
                        p = nodes[i, COL_PARENT]
                        pf = nodes[i, COL_PFIELD]
                        if p >= 0:
                            nodes[p, pf] = off_let_f
                            nodes[off_let_f, COL_PARENT] = p
                            nodes[off_let_f, COL_PFIELD] = pf
                            if i == root_off:
                                root_off = off_let_f
                        else:
                            root_off = off_let_f
                        reduced = True
                        break
                
                # App-Era
                elif f_type == TYPE_ERA:
                    new_off = app_era(nodes, i, d, N_initial, K)
                    if i == root_off:
                        root_off = new_off
                    reduced = True
                    break
                
                # App-Sup
                elif f_type == TYPE_SUP:
                    new_off = app_sup(nodes, i, d, N_initial, K)
                    if i == root_off:
                        root_off = new_off
                    reduced = True
                    break
            
            # Dup rules
            elif types[i] == TYPE_DUP and active[c1[i]]:
                v_type = types[c1[i]]
                
                # Dup-Era
                if v_type == TYPE_ERA:
                    new_off = dup_era(nodes, i, d, N_initial, K)
                    if i == root_off:
                        root_off = new_off
                    reduced = True
                    break
        
        if not reduced:
            break
    
    return nodes, root_off
def name_str(k: int) -> str:
    if k < 0:
        return "?"
    target = k
    queue = deque([""])
    count = 0
    while queue:
        prefix = queue.popleft()
        if count == target:
            return prefix if prefix else "x0"
        count += 1
        for c in "abcdefghijklmnopqrstuvwxyz":
            queue.append(prefix + c)
    return f"v{k}"
# --- Pretty printer for final result ---
def print_term(nodes, root_off, indent=0):
    if not nodes[root_off, COL_ACTIVE]:
        return "Inactive"
    
    typ = nodes[root_off, COL_TYPE]
    
    if typ == TYPE_VAR:
        return f"Var({nodes[root_off, COL_VAR]})"
    elif typ == TYPE_ERA:
        return "*"
    elif typ == TYPE_LAM:
        x = nodes[root_off, COL_VAR]
        body_off = nodes[root_off, COL_C1]
        body = print_term(nodes, body_off, indent+2)
        return f"λ{x}.{body}"
    elif typ == TYPE_APP:
        func_off = nodes[root_off, COL_C1]
        arg_off = nodes[root_off, COL_C2]
        func = print_term(nodes, func_off, indent+2)
        arg = print_term(nodes, arg_off, indent+2)
        return f"({func} {arg})"
    elif typ == TYPE_SUP:
        left_off = nodes[root_off, COL_C1]
        right_off = nodes[root_off, COL_C2]
        left = print_term(nodes, left_off, indent+2)
        right = print_term(nodes, right_off, indent+2)
        label = nodes[root_off, COL_LABEL]
        if label == 0:
            return f"{{{left},{right}}}"
        elif label == 1:
            return f"<{left},{right}>"
        else:
            return f"&{label}{{{left},{right}}}"
            
    elif typ == TYPE_DUP:
        x = nodes[root_off, COL_X]
        y = nodes[root_off, COL_Y]
        val_off = nodes[root_off, COL_C1]
        body_off = nodes[root_off, COL_C2]
        val = print_term(nodes, val_off, indent+2)
        body = print_term(nodes, body_off, indent+2)
        label = nodes[root_off, COL_LABEL]
        if label == 0:
            return f"! {{{name_str(x)},{name_str(y)}}} = {val}; {body}"
        elif label == 1:
            return f"! <{name_str(x)},{name_str(y)}> = {val}; {body}"
        else:
            return f"! &{label}{{{name_str(x)},{name_str(y)}}} = {val}; {body}"
    elif typ == TYPE_LET:
        x = nodes[root_off, COL_VAR]
        t1_off = nodes[root_off, COL_C1]
        t2_off = nodes[root_off, COL_C2]
        t1 = print_term(nodes, t1_off, indent+2)
        t2 = print_term(nodes, t2_off, indent+2)
        return f"! {x} = {t1}; {t2}"
    else:
        return f"UnknownType{typ}"

# --- Example tests ---
def run_test(expr, MAX_DEPTH=10, K=50):
    parser = ParserIC(expr)
    term = parser.parse()
    print(f"Original: {term}")
    
    nodes, N_initial = init_static_array(term, MAX_DEPTH, K)
    nodes_after, root_off = static_run(nodes, N_initial, MAX_DEPTH, K)
    
    result = print_term(nodes_after, root_off)
    print(f"Result: {result}")
    print("-" * 40)
    return result

if __name__ == "__main__":
    # Test 1: Identity function applied to Era
    expr1 = "( (λx.x) * )"
    print("Test 1: Identity function applied to Era")
    result1 = run_test(expr1)
    
    # Test 2: App-Era: Era applied to some argument
    expr2 = "( * * )"
    print("Test 2: App-Era")
    result2 = run_test(expr2)
    
    # Test 3: App-Sup: Sup applied to an argument
    expr3 = "( {λa.a, λb.b} * )"
    print("Test 3: App-Sup")
    result3 = run_test(expr3)
    
    # Test 4: Dup-Era: Duplication of an Era
    expr4 = "! {x,y} = *; (x y)"
    print("Test 4: Dup-Era")
    result4 = run_test(expr4)
    
    print("Test Summary:")
    print(f"1. Identity: {result1} (Expected: *)")
    print(f"2. App-Era: {result2} (Expected: *)")
    print(f"3. App-Sup: {result3} (Expected: Dup structure)")
    print(f"4. Dup-Era: {result4} (Expected: (* *))")