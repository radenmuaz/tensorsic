# static_ic.py

import numpy as np

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

# --- static_run with general beta and nested-Let for App-Lam ---
def static_run(nodes, N_initial, MAX_DEPTH, K):
    bs = nodes.shape[0]
    root_off = 0
    for d in range(MAX_DEPTH):
        types  = nodes[:, COL_TYPE]
        active = nodes[:, COL_ACTIVE]
        c1     = nodes[:, COL_C1]
        mask_app = (types == TYPE_APP) & (active == 1)
        valid = (c1>=0)&(c1<bs)
        ftype = np.full(bs, -1)
        ii = np.nonzero(valid)[0]
        ftype[ii] = nodes[c1[ii], COL_TYPE]
        mask_al = mask_app & (ftype == TYPE_LAM)
        idxs = np.nonzero(mask_al)[0]
        if idxs.size == 0:
            break
        ro = idxs[0]
        lam_off = nodes[ro, COL_C1]; arg_off = nodes[ro, COL_C2]
        lab_app = nodes[ro, COL_LABEL]; lam_lab = nodes[lam_off, COL_LABEL]
        # deactivate subtree
        mask_sub = collect_subtree(nodes, ro)
        nodes[mask_sub, COL_ACTIVE] = 0
        base = clone_base(ro, N_initial, K, d)
        if lam_lab == lab_app:
            # general beta-substitution clone
            bound_var = nodes[lam_off, COL_VAR]
            # gather subtree under lam_off
            subtree = []
            queue = [lam_off]
            seen = {lam_off}
            while queue:
                u = queue.pop(0)
                subtree.append(u)
                u_type = nodes[u, COL_TYPE]
                if u_type == TYPE_LAM:
                    c = nodes[u, COL_C1]
                    if not (nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==bound_var):
                        if c not in seen:
                            seen.add(c); queue.append(c)
                elif u_type in (TYPE_APP, TYPE_LET, TYPE_SUP, TYPE_DUP):
                    for cf in (COL_C1, COL_C2):
                        c = nodes[u, cf]
                        if c<0: continue
                        if nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==bound_var:
                            continue
                        if c not in seen:
                            seen.add(c); queue.append(c)
            mapping = {}
            cnt = 0
            for u in subtree:
                mapping[u] = base + cnt
                cnt += 1
                if cnt > K:
                    raise RuntimeError("K too small for beta-clone")
            # clone nodes
            for u in subtree:
                new_off = mapping[u]
                u_type = nodes[u, COL_TYPE]
                if u_type == TYPE_VAR:
                    nodes[new_off, COL_TYPE] = TYPE_VAR
                    nodes[new_off, COL_VAR]  = nodes[u, COL_VAR]
                elif u_type == TYPE_ERA:
                    nodes[new_off, COL_TYPE] = TYPE_ERA
                elif u_type == TYPE_LAM:
                    nodes[new_off, COL_TYPE]   = TYPE_LAM
                    nodes[new_off, COL_VAR]    = nodes[u, COL_VAR]
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    c = nodes[u, COL_C1]
                    if nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==bound_var:
                        nodes[new_off, COL_C1] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C1
                    else:
                        cnew = mapping[c]
                        nodes[new_off, COL_C1] = cnew
                        nodes[cnew, COL_PARENT] = new_off
                        nodes[cnew, COL_PFIELD] = COL_C1
                elif u_type == TYPE_APP:
                    nodes[new_off, COL_TYPE]   = TYPE_APP
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    f = nodes[u, COL_C1]; a = nodes[u, COL_C2]
                    if nodes[f, COL_TYPE]==TYPE_VAR and nodes[f, COL_VAR]==bound_var:
                        nodes[new_off, COL_C1] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C1
                    else:
                        fnew = mapping[f]
                        nodes[new_off, COL_C1] = fnew
                        nodes[fnew, COL_PARENT] = new_off
                        nodes[fnew, COL_PFIELD] = COL_C1
                    if nodes[a, COL_TYPE]==TYPE_VAR and nodes[a, COL_VAR]==bound_var:
                        nodes[new_off, COL_C2] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C2
                    else:
                        anew = mapping[a]
                        nodes[new_off, COL_C2] = anew
                        nodes[anew, COL_PARENT] = new_off
                        nodes[anew, COL_PFIELD] = COL_C2
                elif u_type == TYPE_LET:
                    nodes[new_off, COL_TYPE]   = TYPE_LET
                    nodes[new_off, COL_VAR]    = nodes[u, COL_VAR]
                    t1 = nodes[u, COL_C1]; t2 = nodes[u, COL_C2]
                    if nodes[t1, COL_TYPE]==TYPE_VAR and nodes[t1, COL_VAR]==bound_var:
                        nodes[new_off, COL_C1] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C1
                    else:
                        t1n = mapping[t1]
                        nodes[new_off, COL_C1] = t1n
                        nodes[t1n, COL_PARENT] = new_off
                        nodes[t1n, COL_PFIELD] = COL_C1
                    if nodes[t2, COL_TYPE]==TYPE_VAR and nodes[t2, COL_VAR]==bound_var:
                        nodes[new_off, COL_C2] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C2
                    else:
                        t2n = mapping[t2]
                        nodes[new_off, COL_C2] = t2n
                        nodes[t2n, COL_PARENT] = new_off
                        nodes[t2n, COL_PFIELD] = COL_C2
                elif u_type == TYPE_SUP:
                    nodes[new_off, COL_TYPE]   = TYPE_SUP
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    l = nodes[u, COL_C1]; r = nodes[u, COL_C2]
                    if nodes[l, COL_TYPE]==TYPE_VAR and nodes[l, COL_VAR]==bound_var:
                        nodes[new_off, COL_C1] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C1
                    else:
                        ln = mapping[l]
                        nodes[new_off, COL_C1] = ln
                        nodes[ln, COL_PARENT] = new_off
                        nodes[ln, COL_PFIELD] = COL_C1
                    if nodes[r, COL_TYPE]==TYPE_VAR and nodes[r, COL_VAR]==bound_var:
                        nodes[new_off, COL_C2] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C2
                    else:
                        rn = mapping[r]
                        nodes[new_off, COL_C2] = rn
                        nodes[rn, COL_PARENT] = new_off
                        nodes[rn, COL_PFIELD] = COL_C2
                elif u_type == TYPE_DUP:
                    nodes[new_off, COL_TYPE]   = TYPE_DUP
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    nodes[new_off, COL_X]      = nodes[u, COL_X]
                    nodes[new_off, COL_Y]      = nodes[u, COL_Y]
                    v = nodes[u, COL_C1]; b = nodes[u, COL_C2]
                    if nodes[v, COL_TYPE]==TYPE_VAR and nodes[v, COL_VAR]==bound_var:
                        nodes[new_off, COL_C1] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C1
                    else:
                        vn = mapping[v]
                        nodes[new_off, COL_C1] = vn
                        nodes[vn, COL_PARENT] = new_off
                        nodes[vn, COL_PFIELD] = COL_C1
                    if nodes[b, COL_TYPE]==TYPE_VAR and nodes[b, COL_VAR]==bound_var:
                        nodes[new_off, COL_C2] = arg_off
                        nodes[arg_off, COL_PARENT] = new_off
                        nodes[arg_off, COL_PFIELD] = COL_C2
                    else:
                        bn = mapping[b]
                        nodes[new_off, COL_C2] = bn
                        nodes[bn, COL_PARENT] = new_off
                        nodes[bn, COL_PFIELD] = COL_C2
                else:
                    raise RuntimeError("Unknown type in beta-clone")
                nodes[new_off, COL_ACTIVE] = 1
            new_off = mapping[lam_off]
        else:
            # nested Let rewrite
            # assign fresh var_ids = offsets based on base
            off_y         = base + 0
            off_z         = base + 1
            off_fnode     = base + 2
            off_xnode     = base + 3
            off_vnode     = base + 4
            off_Let_f     = base + 5
            off_App_x     = base + 6
            off_Let_x     = base + 7
            off_App_v     = base + 8
            off_Let_v     = base + 9
            off_Lam_inner = base + 10
            if off_Lam_inner >= nodes.shape[0]:
                raise RuntimeError("K too small for nested-Let")
            # Var nodes
            for offn in (off_y, off_z, off_fnode, off_xnode, off_vnode):
                nodes[offn, COL_TYPE]   = TYPE_VAR
                nodes[offn, COL_VAR]    = offn
                nodes[offn, COL_ACTIVE] = 1
            # Let f = bod
            nodes[off_Let_f, COL_TYPE]   = TYPE_LET
            nodes[off_Let_f, COL_VAR]    = off_fnode
            nodes[off_Let_f, COL_ACTIVE] = 1
            bod_off = nodes[lam_off, COL_C1]
            nodes[off_Let_f, COL_C1]     = bod_off
            nodes[bod_off, COL_PARENT]   = off_Let_f; nodes[bod_off, COL_PFIELD] = COL_C1
            # App x = App(lam_lab, arg_off, Var y)
            nodes[off_App_x, COL_TYPE]   = TYPE_APP
            nodes[off_App_x, COL_LABEL]  = lam_lab
            nodes[off_App_x, COL_ACTIVE] = 1
            nodes[off_App_x, COL_C1]     = arg_off
            nodes[arg_off, COL_PARENT]   = off_App_x; nodes[arg_off, COL_PFIELD] = COL_C1
            nodes[off_App_x, COL_C2]     = off_y
            nodes[off_y, COL_PARENT]     = off_App_x; nodes[off_y, COL_PFIELD] = COL_C2
            # Let x
            nodes[off_Let_x, COL_TYPE]   = TYPE_LET
            nodes[off_Let_x, COL_VAR]    = off_xnode
            nodes[off_Let_x, COL_ACTIVE] = 1
            nodes[off_Let_x, COL_C1]     = off_App_x
            nodes[off_App_x, COL_PARENT] = off_Let_x; nodes[off_App_x, COL_PFIELD] = COL_C1
            # link Let f.child2 = Let_x
            nodes[off_Let_f, COL_C2]     = off_Let_x
            nodes[off_Let_x, COL_PARENT] = off_Let_f; nodes[off_Let_x, COL_PFIELD] = COL_C2
            # App v = App(app_lab, Var f, Var x)
            nodes[off_App_v, COL_TYPE]   = TYPE_APP
            nodes[off_App_v, COL_LABEL]  = lab_app
            nodes[off_App_v, COL_ACTIVE] = 1
            nodes[off_App_v, COL_C1]     = off_fnode
            nodes[off_fnode, COL_PARENT] = off_App_v; nodes[off_fnode, COL_PFIELD] = COL_C1
            nodes[off_App_v, COL_C2]     = off_xnode
            nodes[off_xnode, COL_PARENT] = off_App_v; nodes[off_xnode, COL_PFIELD] = COL_C2
            # Let v
            nodes[off_Let_v, COL_TYPE]   = TYPE_LET
            nodes[off_Let_v, COL_VAR]    = off_vnode
            nodes[off_Let_v, COL_ACTIVE] = 1
            nodes[off_Let_v, COL_C1]     = off_App_v
            nodes[off_App_v, COL_PARENT] = off_Let_v; nodes[off_App_v, COL_PFIELD] = COL_C1
            # link Let x.child2 = Let_v
            nodes[off_Let_x, COL_C2]     = off_Let_v
            nodes[off_Let_v, COL_PARENT] = off_Let_x; nodes[off_Let_v, COL_PFIELD] = COL_C2
            # Lam inner
            nodes[off_Lam_inner, COL_TYPE]   = TYPE_LAM
            nodes[off_Lam_inner, COL_LABEL]  = lam_lab
            nodes[off_Lam_inner, COL_VAR]    = off_z
            nodes[off_Lam_inner, COL_ACTIVE] = 1
            nodes[off_Lam_inner, COL_C1]     = off_vnode
            nodes[off_vnode, COL_PARENT]     = off_Lam_inner; nodes[off_vnode, COL_PFIELD] = COL_C1
            # link Let_v.child2 = Lam_inner
            nodes[off_Let_v, COL_C2]         = off_Lam_inner
            nodes[off_Lam_inner, COL_PARENT] = off_Let_v; nodes[off_Lam_inner, COL_PFIELD] = COL_C2
            new_off = off_Let_f
        # update parent pointer
        p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
        if p >= 0:
            nodes[p, pf] = new_off
            nodes[new_off, COL_PARENT] = p
            nodes[new_off, COL_PFIELD] = pf
        else:
            root_off = new_off
    return nodes, root_off

# --- Example tests ---
if __name__ == "__main__":
    MAX_DEPTH = 5
    K = 50  # adjust as needed

    # 1) Identity test: ((λx.x) *)
    expr1 = "( (λx.x) * )"
    parser = ParserIC(expr1)
    term1 = parser.parse()
    nodes1, N1 = init_static_array(term1, MAX_DEPTH, K)
    nodes1_after, root1 = static_run(nodes1, N1, MAX_DEPTH, K)
    print("Identity test active offsets:", np.nonzero(nodes1_after[:, COL_ACTIVE])[0])

    # 2) Simple beta test: ((λa.(λb.a)) t) f
    # Build AST manually
    # var_ids: a=0, b=1, t=2, f=3
    lam_b_a    = Lam(0, 1, Var(0))      # λb.a
    lam_a_body = Lam(0, 0, lam_b_a)     # λa.(λb.a)
    expr2      = App(0, App(0, lam_a_body, Var(2)), Var(3))
    nodes2, N2 = init_static_array(expr2, MAX_DEPTH, K)
    nodes2_after, root2 = static_run(nodes2, N2, MAX_DEPTH, K)
    print("Beta test active offsets:", np.nonzero(nodes2_after[:, COL_ACTIVE])[0], "root:", root2)

    # 3) Nested-Let test: App label mismatch: App(1, Lam(0,a->a), t)
    lam_simple = Lam(0, 0, Var(0))
    expr3      = App(1, lam_simple, Var(2))
    nodes3, N3 = init_static_array(expr3, MAX_DEPTH, K)
    nodes3_after, root3 = static_run(nodes3, N3, MAX_DEPTH, K)
    print("Nested-Let test active offsets:", np.nonzero(nodes3_after[:, COL_ACTIVE])[0], "root:", root3)
    # Print subtree
    def print_subtree(nodes, off, indent=0):
        typ = nodes[off, COL_TYPE]
        if typ==TYPE_VAR:
            print(" "*indent + f"Var({nodes[off, COL_VAR]})")
        elif typ==TYPE_LAM:
            print(" "*indent + f"Lam(label={nodes[off, COL_LABEL]}, var={nodes[off, COL_VAR]})")
            print_subtree(nodes, nodes[off, COL_C1], indent+2)
        elif typ==TYPE_APP:
            print(" "*indent + f"App(label={nodes[off, COL_LABEL]})")
            print_subtree(nodes, nodes[off, COL_C1], indent+2)
            print_subtree(nodes, nodes[off, COL_C2], indent+2)
        elif typ==TYPE_LET:
            print(" "*indent + f"Let(var={nodes[off, COL_VAR]})")
            print_subtree(nodes, nodes[off, COL_C1], indent+2)
            print_subtree(nodes, nodes[off, COL_C2], indent+2)
        else:
            print(" "*indent + f"Type{typ} at {off}")
    print("Nested-Let result subtree:")
    if root3 is not None:
        print_subtree(nodes3_after, root3)
