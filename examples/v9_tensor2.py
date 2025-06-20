# static_ic_engine.py

import numpy as np

# --- Type tags and columns for the static array ---
TYPE_VAR = 0
TYPE_LET = 1
TYPE_ERA = 2
TYPE_SUP = 3
TYPE_DUP = 4
TYPE_LAM = 5
TYPE_APP = 6

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

# --- AST classes for tests and initial layout ---
class Term: pass
class Var(Term):
    def __init__(self, name:int): self.name=name
    def __repr__(self): return f"Var({self.name})"
class Era(Term):
    def __repr__(self): return "*"
class Lam(Term):
    def __init__(self, label:int, x:int, body:Term):
        self.label, self.x, self.body = label, x, body
    def __repr__(self): return f"Lam({self.label},{self.x},{self.body})"
class Sup(Term):
    def __init__(self, label:int, left:Term, right:Term):
        self.label, self.left, self.right = label, left, right
    def __repr__(self): return f"Sup({self.label},{self.left},{self.right})"
class Dup(Term):
    def __init__(self, label:int, x:int, y:int, val:Term, body:Term):
        self.label, self.x, self.y, self.val, self.body = label, x, y, val, body
    def __repr__(self): return f"Dup({self.label},{self.x},{self.y},{self.val},{self.body})"
class Let(Term):
    def __init__(self, name:int, t1:Term, t2:Term):
        self.name, self.t1, self.t2 = name, t1, t2
    def __repr__(self): return f"Let({self.name}={self.t1};{self.t2})"
class App(Term):
    def __init__(self, label:int, func:Term, arg:Term):
        self.label, self.func, self.arg = label, func, arg
    def __repr__(self): return f"App({self.label},{self.func},{self.arg})"

# --- Build layout for initial AST into static array ---
def build_layout(u):
    offsets, order = {}, []
    def dfs(v):
        if id(v) not in offsets:
            off = len(order)
            offsets[id(v)] = off
            order.append(v)
            if isinstance(v, Var) or isinstance(v, Era):
                pass
            elif isinstance(v, Lam):
                dfs(v.body)
            elif isinstance(v, App):
                dfs(v.func); dfs(v.arg)
            elif isinstance(v, Let):
                dfs(v.t1); dfs(v.t2)
            elif isinstance(v, Sup):
                dfs(v.left); dfs(v.right)
            elif isinstance(v, Dup):
                dfs(v.val); dfs(v.body)
        return offsets[id(v)]
    dfs(u)
    return offsets, order

def init_static_array(term, MAX_DEPTH, K):
    offsets, order = build_layout(term)
    N = len(order)
    size = N + N * K * MAX_DEPTH
    nodes = np.zeros((size, NUM_COLS), dtype=np.int32)
    nodes[:, COL_TYPE]   = -1
    nodes[:, COL_ACTIVE] = 0
    nodes[:, COL_PARENT] = -1
    nodes[:, COL_PFIELD] = 0
    for off, u in enumerate(order):
        if isinstance(u, Var):
            nodes[off, COL_TYPE]   = TYPE_VAR
            nodes[off, COL_VAR]    = u.name
            nodes[off, COL_ACTIVE] = 1
        elif isinstance(u, Era):
            nodes[off, COL_TYPE]   = TYPE_ERA
            nodes[off, COL_ACTIVE] = 1
        elif isinstance(u, Lam):
            nodes[off, COL_TYPE]   = TYPE_LAM
            nodes[off, COL_VAR]    = u.x
            nodes[off, COL_LABEL]  = u.label
            nodes[off, COL_ACTIVE] = 1
            c = offsets[id(u.body)]
            nodes[off, COL_C1]     = c
            nodes[c, COL_PARENT]   = off; nodes[c, COL_PFIELD] = COL_C1
        elif isinstance(u, App):
            nodes[off, COL_TYPE]   = TYPE_APP
            nodes[off, COL_LABEL]  = u.label
            nodes[off, COL_ACTIVE] = 1
            f = offsets[id(u.func)]; a = offsets[id(u.arg)]
            nodes[off, COL_C1]     = f
            nodes[off, COL_C2]     = a
            nodes[f, COL_PARENT]   = off; nodes[f, COL_PFIELD] = COL_C1
            nodes[a, COL_PARENT]   = off; nodes[a, COL_PFIELD] = COL_C2
        elif isinstance(u, Let):
            nodes[off, COL_TYPE]   = TYPE_LET
            nodes[off, COL_VAR]    = u.name
            nodes[off, COL_ACTIVE] = 1
            t1 = offsets[id(u.t1)]; t2 = offsets[id(u.t2)]
            nodes[off, COL_C1]     = t1
            nodes[off, COL_C2]     = t2
            nodes[t1, COL_PARENT]  = off; nodes[t1, COL_PFIELD] = COL_C1
            nodes[t2, COL_PARENT]  = off; nodes[t2, COL_PFIELD] = COL_C2
        elif isinstance(u, Sup):
            nodes[off, COL_TYPE]   = TYPE_SUP
            nodes[off, COL_LABEL]  = u.label
            nodes[off, COL_ACTIVE] = 1
            l = offsets[id(u.left)]; r = offsets[id(u.right)]
            nodes[off, COL_C1]     = l
            nodes[off, COL_C2]     = r
            nodes[l, COL_PARENT]   = off; nodes[l, COL_PFIELD] = COL_C1
            nodes[r, COL_PARENT]   = off; nodes[r, COL_PFIELD] = COL_C2
        elif isinstance(u, Dup):
            nodes[off, COL_TYPE]   = TYPE_DUP
            nodes[off, COL_LABEL]  = u.label
            nodes[off, COL_X]      = u.x
            nodes[off, COL_Y]      = u.y
            nodes[off, COL_ACTIVE] = 1
            v = offsets[id(u.val)]; b = offsets[id(u.body)]
            nodes[off, COL_C1]     = v
            nodes[off, COL_C2]     = b
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

# --- static_run implementing IC patterns with static arrays and loops ---
def static_run(nodes, N_initial, MAX_DEPTH, K):
    bs = nodes.shape[0]
    root_off = None
    for d in range(MAX_DEPTH):
        types  = nodes[:, COL_TYPE]
        active = nodes[:, COL_ACTIVE]
        c1 = nodes[:, COL_C1]
        mask_app = (types == TYPE_APP) & (active == 1)
        valid = (c1>=0) & (c1<bs)
        ftype = np.full(bs, -1)
        ii = np.nonzero(valid)[0]
        ftype[ii] = nodes[c1[ii], COL_TYPE]

        # 1) App-Lam equal-label beta
        mask_al = mask_app & (ftype == TYPE_LAM)
        if np.any(mask_al):
            ro = np.nonzero(mask_al)[0][0]
            lam_off = nodes[ro, COL_C1]; arg_off = nodes[ro, COL_C2]
            lab_app = nodes[ro, COL_LABEL]; lam_lab = nodes[lam_off, COL_LABEL]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            if lam_lab == lab_app:
                bound_var = nodes[lam_off, COL_VAR]
                SUB_MAX = K
                subtree = np.full(SUB_MAX, -1, dtype=np.int32)
                visited = np.zeros(bs, bool)
                subtree[0] = lam_off; visited[lam_off] = True
                for idx in range(SUB_MAX):
                    u = subtree[idx]
                    if u < 0: continue
                    u_type = nodes[u, COL_TYPE]
                    if u_type == TYPE_LAM:
                        c = nodes[u, COL_C1]
                        if not (nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==bound_var):
                            if not visited[c]:
                                for j in range(SUB_MAX):
                                    if subtree[j] < 0:
                                        subtree[j] = c; break
                                visited[c] = True
                    elif u_type in (TYPE_APP, TYPE_LET, TYPE_SUP, TYPE_DUP):
                        for cf in (COL_C1, COL_C2):
                            c = nodes[u, cf]
                            if c<0: continue
                            if nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==bound_var:
                                continue
                            if not visited[c]:
                                for j in range(SUB_MAX):
                                    if subtree[j] < 0:
                                        subtree[j] = c; break
                                visited[c] = True
                mapping_arr = np.full(bs, -1, dtype=np.int32)
                for idx in range(SUB_MAX):
                    u = subtree[idx]
                    if u < 0: break
                    mapping_arr[u] = base + idx
                for idx in range(SUB_MAX):
                    u = subtree[idx]
                    if u < 0: break
                    new_off = mapping_arr[u]
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
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C1
                        else:
                            cnew = mapping_arr[c]
                            nodes[new_off, COL_C1] = cnew
                            nodes[cnew, COL_PARENT] = new_off; nodes[cnew, COL_PFIELD] = COL_C1
                    elif u_type == TYPE_APP:
                        nodes[new_off, COL_TYPE]   = TYPE_APP
                        nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                        f = nodes[u, COL_C1]; a = nodes[u, COL_C2]
                        if nodes[f, COL_TYPE]==TYPE_VAR and nodes[f, COL_VAR]==bound_var:
                            nodes[new_off, COL_C1] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C1
                        else:
                            fnew = mapping_arr[f]
                            nodes[new_off, COL_C1] = fnew
                            nodes[fnew, COL_PARENT] = new_off; nodes[fnew, COL_PFIELD] = COL_C1
                        if nodes[a, COL_TYPE]==TYPE_VAR and nodes[a, COL_VAR]==bound_var:
                            nodes[new_off, COL_C2] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C2
                        else:
                            anew = mapping_arr[a]
                            nodes[new_off, COL_C2] = anew
                            nodes[anew, COL_PARENT] = new_off; nodes[anew, COL_PFIELD] = COL_C2
                    elif u_type == TYPE_LET:
                        nodes[new_off, COL_TYPE]   = TYPE_LET
                        nodes[new_off, COL_VAR]    = nodes[u, COL_VAR]
                        t1 = nodes[u, COL_C1]; t2 = nodes[u, COL_C2]
                        if nodes[t1, COL_TYPE]==TYPE_VAR and nodes[t1, COL_VAR]==bound_var:
                            nodes[new_off, COL_C1] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C1
                        else:
                            t1n = mapping_arr[t1]
                            nodes[new_off, COL_C1] = t1n
                            nodes[t1n, COL_PARENT] = new_off; nodes[t1n, COL_PFIELD] = COL_C1
                        if nodes[t2, COL_TYPE]==TYPE_VAR and nodes[t2, COL_VAR]==bound_var:
                            nodes[new_off, COL_C2] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C2
                        else:
                            t2n = mapping_arr[t2]
                            nodes[new_off, COL_C2] = t2n
                            nodes[t2n, COL_PARENT] = new_off; nodes[t2n, COL_PFIELD] = COL_C2
                    elif u_type == TYPE_SUP:
                        nodes[new_off, COL_TYPE]   = TYPE_SUP
                        nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                        l = nodes[u, COL_C1]; r = nodes[u, COL_C2]
                        if nodes[l, COL_TYPE]==TYPE_VAR and nodes[l, COL_VAR]==bound_var:
                            nodes[new_off, COL_C1] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C1
                        else:
                            ln = mapping_arr[l]
                            nodes[new_off, COL_C1] = ln
                            nodes[ln, COL_PARENT] = new_off; nodes[ln, COL_PFIELD] = COL_C1
                        if nodes[r, COL_TYPE]==TYPE_VAR and nodes[r, COL_VAR]==bound_var:
                            nodes[new_off, COL_C2] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C2
                        else:
                            rn = mapping_arr[r]
                            nodes[new_off, COL_C2] = rn
                            nodes[rn, COL_PARENT] = new_off; nodes[rn, COL_PFIELD] = COL_C2
                    elif u_type == TYPE_DUP:
                        nodes[new_off, COL_TYPE]   = TYPE_DUP
                        nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                        nodes[new_off, COL_X]      = nodes[u, COL_X]
                        nodes[new_off, COL_Y]      = nodes[u, COL_Y]
                        v = nodes[u, COL_C1]; b = nodes[u, COL_C2]
                        if nodes[v, COL_TYPE]==TYPE_VAR and nodes[v, COL_VAR]==bound_var:
                            nodes[new_off, COL_C1] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C1
                        else:
                            vn = mapping_arr[v]
                            nodes[new_off, COL_C1] = vn
                            nodes[vn, COL_PARENT] = new_off; nodes[vn, COL_PFIELD] = COL_C1
                        if nodes[b, COL_TYPE]==TYPE_VAR and nodes[b, COL_VAR]==bound_var:
                            nodes[new_off, COL_C2] = arg_off
                            nodes[arg_off, COL_PARENT] = new_off; nodes[arg_off, COL_PFIELD] = COL_C2
                        else:
                            bn = mapping_arr[b]
                            nodes[new_off, COL_C2] = bn
                            nodes[bn, COL_PARENT] = new_off; nodes[bn, COL_PFIELD] = COL_C2
                    nodes[new_off, COL_ACTIVE] = 1
                new_off = mapping_arr[lam_off]
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = new_off
                nodes[new_off, COL_PARENT] = p; nodes[new_off, COL_PFIELD] = pf
            else:
                root_off = new_off
            continue

        # 2) App-Sup
        mask_as = mask_app & (ftype == TYPE_SUP)
        if np.any(mask_as):
            ro = np.nonzero(mask_as)[0][0]
            sup_off = nodes[ro, COL_C1]; arg_off = nodes[ro, COL_C2]
            app_lab = nodes[ro, COL_LABEL]; sup_lab = nodes[sup_off, COL_LABEL]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            # fresh c0,c1
            off_c0 = base+0; off_c1 = base+1
            nodes[off_c0, COL_TYPE]   = TYPE_VAR; nodes[off_c0, COL_VAR]=off_c0; nodes[off_c0, COL_ACTIVE]=1
            nodes[off_c1, COL_TYPE]   = TYPE_VAR; nodes[off_c1, COL_VAR]=off_c1; nodes[off_c1, COL_ACTIVE]=1
            # App left
            off_app0 = base+2
            left = nodes[sup_off, COL_C1]
            nodes[off_app0, COL_TYPE]=TYPE_APP; nodes[off_app0, COL_LABEL]=app_lab; nodes[off_app0, COL_ACTIVE]=1
            nodes[off_app0, COL_C1]=left; nodes[left, COL_PARENT]=off_app0; nodes[left, COL_PFIELD]=COL_C1
            nodes[off_app0, COL_C2]=off_c0; nodes[off_c0, COL_PARENT]=off_app0; nodes[off_c0, COL_PFIELD]=COL_C2
            # App right
            off_app1 = base+3
            right = nodes[sup_off, COL_C2]
            nodes[off_app1, COL_TYPE]=TYPE_APP; nodes[off_app1, COL_LABEL]=app_lab; nodes[off_app1, COL_ACTIVE]=1
            nodes[off_app1, COL_C1]=right; nodes[right, COL_PARENT]=off_app1; nodes[right, COL_PFIELD]=COL_C1
            nodes[off_app1, COL_C2]=off_c1; nodes[off_c1, COL_PARENT]=off_app1; nodes[off_c1, COL_PFIELD]=COL_C2
            # Sup combining
            off_sup_new = base+4
            nodes[off_sup_new, COL_TYPE]=TYPE_SUP; nodes[off_sup_new, COL_LABEL]=sup_lab; nodes[off_sup_new, COL_ACTIVE]=1
            nodes[off_sup_new, COL_C1]=off_app0; nodes[off_app0, COL_PARENT]=off_sup_new; nodes[off_app0, COL_PFIELD]=COL_C1
            nodes[off_sup_new, COL_C2]=off_app1; nodes[off_app1, COL_PARENT]=off_sup_new; nodes[off_app1, COL_PFIELD]=COL_C2
            # Dup node
            off_dup = base+5
            nodes[off_dup, COL_TYPE]=TYPE_DUP; nodes[off_dup, COL_LABEL]=sup_lab; nodes[off_dup, COL_X]=off_c0; nodes[off_dup, COL_Y]=off_c1; nodes[off_dup, COL_ACTIVE]=1
            nodes[off_dup, COL_C1]=arg_off; nodes[arg_off, COL_PARENT]=off_dup; nodes[arg_off, COL_PFIELD]=COL_C1
            nodes[off_dup, COL_C2]=off_sup_new; nodes[off_sup_new, COL_PARENT]=off_dup; nodes[off_sup_new, COL_PFIELD]=COL_C2
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = off_dup
                nodes[off_dup, COL_PARENT] = p; nodes[off_dup, COL_PFIELD] = pf
            else:
                root_off = off_dup
            continue

        # 3) App-Era: ( * arg ) → *
        mask_era = mask_app & (ftype == TYPE_ERA)
        if np.any(mask_era):
            ro = np.nonzero(mask_era)[0][0]
            lam_off = nodes[ro, COL_C1]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            nodes[lam_off, COL_ACTIVE] = 1
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = lam_off
                nodes[lam_off, COL_PARENT] = p; nodes[lam_off, COL_PFIELD] = pf
            else:
                root_off = lam_off
            continue

        # 4) Let reduction: Let x=v; body → substitute v into body (simple Var/Era)
        mask_let = (types == TYPE_LET) & (active == 1)
        if np.any(mask_let):
            ro = np.nonzero(mask_let)[0][0]
            var_id = nodes[ro, COL_VAR]
            v_off = nodes[ro, COL_C1]
            body_off = nodes[ro, COL_C2]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            SUB_MAX = K
            subtree = np.full(SUB_MAX, -1, dtype=np.int32)
            visited = np.zeros(bs, bool)
            subtree[0] = body_off; visited[body_off] = True
            for idx in range(SUB_MAX):
                u = subtree[idx]
                if u<0: continue
                u_type = nodes[u, COL_TYPE]
                if u_type == TYPE_LAM:
                    c = nodes[u, COL_C1]
                    if not (nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==var_id):
                        if not visited[c]:
                            for j in range(SUB_MAX):
                                if subtree[j]<0:
                                    subtree[j]=c; break
                            visited[c]=True
                elif u_type in (TYPE_APP, TYPE_LET, TYPE_SUP, TYPE_DUP):
                    for cf in (COL_C1, COL_C2):
                        c = nodes[u, cf]
                        if c<0: continue
                        if nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==var_id:
                            continue
                        if not visited[c]:
                            for j in range(SUB_MAX):
                                if subtree[j]<0:
                                    subtree[j]=c; break
                            visited[c]=True
            mapping_arr = np.full(bs, -1, dtype=np.int32)
            for idx in range(SUB_MAX):
                u = subtree[idx]
                if u<0: break
                mapping_arr[u] = base + idx
            for idx in range(SUB_MAX):
                u = subtree[idx]
                if u<0: break
                new_off = mapping_arr[u]
                u_type = nodes[u, COL_TYPE]
                if u_type == TYPE_VAR:
                    if nodes[u, COL_VAR] == var_id:
                        v_type = nodes[v_off, COL_TYPE]
                        nodes[new_off, COL_TYPE] = v_type
                        if v_type == TYPE_VAR:
                            nodes[new_off, COL_VAR] = nodes[v_off, COL_VAR]
                        elif v_type == TYPE_ERA:
                            pass
                        else:
                            raise NotImplementedError("Let-substitute non-simple subtree")
                    else:
                        nodes[new_off, COL_TYPE] = TYPE_VAR
                        nodes[new_off, COL_VAR]  = nodes[u, COL_VAR]
                elif u_type == TYPE_ERA:
                    nodes[new_off, COL_TYPE] = TYPE_ERA
                elif u_type == TYPE_LAM:
                    nodes[new_off, COL_TYPE]   = TYPE_LAM
                    nodes[new_off, COL_VAR]    = nodes[u, COL_VAR]
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    c = nodes[u, COL_C1]
                    if nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR]==var_id:
                        nodes[new_off, COL_C1] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C1
                    else:
                        cnew = mapping_arr[c]
                        nodes[new_off, COL_C1] = cnew
                        nodes[cnew, COL_PARENT] = new_off; nodes[cnew, COL_PFIELD] = COL_C1
                elif u_type == TYPE_APP:
                    nodes[new_off, COL_TYPE]   = TYPE_APP
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    f = nodes[u, COL_C1]; a = nodes[u, COL_C2]
                    if nodes[f, COL_TYPE]==TYPE_VAR and nodes[f, COL_VAR]==var_id:
                        nodes[new_off, COL_C1] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C1
                    else:
                        fnew = mapping_arr[f]
                        nodes[new_off, COL_C1] = fnew
                        nodes[fnew, COL_PARENT] = new_off; nodes[fnew, COL_PFIELD] = COL_C1
                    if nodes[a, COL_TYPE]==TYPE_VAR and nodes[a, COL_VAR]==var_id:
                        nodes[new_off, COL_C2] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C2
                    else:
                        anew = mapping_arr[a]
                        nodes[new_off, COL_C2] = anew
                        nodes[anew, COL_PARENT] = new_off; nodes[anew, COL_PFIELD] = COL_C2
                elif u_type == TYPE_SUP:
                    nodes[new_off, COL_TYPE]   = TYPE_SUP
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    l = nodes[u, COL_C1]; r = nodes[u, COL_C2]
                    if nodes[l, COL_TYPE]==TYPE_VAR and nodes[l, COL_VAR]==var_id:
                        nodes[new_off, COL_C1] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C1
                    else:
                        ln = mapping_arr[l]
                        nodes[new_off, COL_C1] = ln
                        nodes[ln, COL_PARENT] = new_off; nodes[ln, COL_PFIELD] = COL_C1
                    if nodes[r, COL_TYPE]==TYPE_VAR and nodes[r, COL_VAR]==var_id:
                        nodes[new_off, COL_C2] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C2
                    else:
                        rn = mapping_arr[r]
                        nodes[new_off, COL_C2] = rn
                        nodes[rn, COL_PARENT] = new_off; nodes[rn, COL_PFIELD] = COL_C2
                elif u_type == TYPE_DUP:
                    nodes[new_off, COL_TYPE]   = TYPE_DUP
                    nodes[new_off, COL_LABEL]  = nodes[u, COL_LABEL]
                    nodes[new_off, COL_X]      = nodes[u, COL_X]
                    nodes[new_off, COL_Y]      = nodes[u, COL_Y]
                    v2 = nodes[u, COL_C1]; b2 = nodes[u, COL_C2]
                    if nodes[v2, COL_TYPE]==TYPE_VAR and nodes[v2, COL_VAR]==var_id:
                        nodes[new_off, COL_C1] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C1
                    else:
                        vn = mapping_arr[v2]
                        nodes[new_off, COL_C1] = vn
                        nodes[vn, COL_PARENT] = new_off; nodes[vn, COL_PFIELD] = COL_C1
                    if nodes[b2, COL_TYPE]==TYPE_VAR and nodes[b2, COL_VAR]==var_id:
                        nodes[new_off, COL_C2] = v_off
                        nodes[v_off, COL_PARENT] = new_off; nodes[v_off, COL_PFIELD] = COL_C2
                    else:
                        bn = mapping_arr[b2]
                        nodes[new_off, COL_C2] = bn
                        nodes[bn, COL_PARENT] = new_off; nodes[bn, COL_PFIELD] = COL_C2
                nodes[new_off, COL_ACTIVE] = 1
            new_off = mapping_arr[body_off]
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = new_off
                nodes[new_off, COL_PARENT] = p; nodes[new_off, COL_PFIELD] = pf
            else:
                root_off = new_off
            continue

        # 5) App-Dup: App(f, Dup(...)) → Dup(..., App(f, body))
        c2 = nodes[:, COL_C2]
        mask_ad = (types == TYPE_APP) & (active == 1) & (c2>=0) & (c2<bs) & (nodes[c2, COL_TYPE] == TYPE_DUP)
        if np.any(mask_ad):
            ro = np.nonzero(mask_ad)[0][0]
            f_off = nodes[ro, COL_C1]
            dup_off = nodes[ro, COL_C2]
            lab_app = nodes[ro, COL_LABEL]
            dup_lab = nodes[dup_off, COL_LABEL]
            x_id = nodes[dup_off, COL_X]
            y_id = nodes[dup_off, COL_Y]
            val_off = nodes[dup_off, COL_C1]
            body_off = nodes[dup_off, COL_C2]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            off_app_new = base+0
            nodes[off_app_new, COL_TYPE]   = TYPE_APP
            nodes[off_app_new, COL_LABEL]  = lab_app
            nodes[off_app_new, COL_ACTIVE] = 1
            nodes[off_app_new, COL_C1]     = f_off
            nodes[f_off, COL_PARENT]      = off_app_new; nodes[f_off, COL_PFIELD] = COL_C1
            nodes[off_app_new, COL_C2]     = body_off
            nodes[body_off, COL_PARENT]   = off_app_new; nodes[body_off, COL_PFIELD] = COL_C2
            off_dup_new = base+1
            nodes[off_dup_new, COL_TYPE]   = TYPE_DUP
            nodes[off_dup_new, COL_LABEL]  = dup_lab
            nodes[off_dup_new, COL_X]      = x_id
            nodes[off_dup_new, COL_Y]      = y_id
            nodes[off_dup_new, COL_ACTIVE] = 1
            nodes[off_dup_new, COL_C1]     = val_off
            nodes[val_off, COL_PARENT]     = off_dup_new; nodes[val_off, COL_PFIELD] = COL_C1
            nodes[off_dup_new, COL_C2]     = off_app_new
            nodes[off_app_new, COL_PARENT] = off_dup_new; nodes[off_app_new, COL_PFIELD] = COL_C2
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = off_dup_new
                nodes[off_dup_new, COL_PARENT] = p; nodes[off_dup_new, COL_PFIELD] = pf
            else:
                root_off = off_dup_new
            continue

        # 6) Dup-Era: Dup(..., val=Era, body) → body
        mask_dup_era = (types == TYPE_DUP) & (active == 1) & (nodes[:, COL_C1]>=0) & (nodes[:, COL_C1]<bs) & (nodes[nodes[:, COL_C1], COL_TYPE] == TYPE_ERA)
        if np.any(mask_dup_era):
            ro = np.nonzero(mask_dup_era)[0][0]
            x_id = nodes[ro, COL_X]; y_id = nodes[ro, COL_Y]
            body_off = nodes[ro, COL_C2]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            # clone body subtree substituting x,y→Era
            SUB_MAX = K
            subtree = np.full(SUB_MAX, -1, dtype=np.int32)
            visited = np.zeros(bs, bool)
            subtree[0] = body_off; visited[body_off] = True
            for idx in range(SUB_MAX):
                u = subtree[idx]
                if u<0: continue
                u_type = nodes[u, COL_TYPE]
                if u_type == TYPE_LAM:
                    c = nodes[u, COL_C1]
                    if not (nodes[c, COL_TYPE]==TYPE_VAR and (nodes[c, COL_VAR] in (x_id,y_id))):
                        if not visited[c]:
                            for j in range(SUB_MAX):
                                if subtree[j]<0:
                                    subtree[j]=c; break
                            visited[c]=True
                elif u_type in (TYPE_APP, TYPE_LET, TYPE_SUP, TYPE_DUP):
                    for cf in (COL_C1, COL_C2):
                        c = nodes[u, cf]
                        if c<0: continue
                        if nodes[c, COL_TYPE]==TYPE_VAR and (nodes[c, COL_VAR] in (x_id,y_id)):
                            continue
                        if not visited[c]:
                            for j in range(SUB_MAX):
                                if subtree[j]<0:
                                    subtree[j]=c; break
                            visited[c]=True
            mapping_arr = np.full(bs, -1, dtype=np.int32)
            # create one Era node at base
            era_off = base
            nodes[era_off, COL_TYPE] = TYPE_ERA
            nodes[era_off, COL_ACTIVE] = 1
            for idx in range(SUB_MAX):
                u = subtree[idx]
                if u<0: break
                mapping_arr[u] = base + 1 + idx
            for idx in range(SUB_MAX):
                u = subtree[idx]
                if u<0: break
                new_off = mapping_arr[u]
                u_type = nodes[u, COL_TYPE]
                if u_type == TYPE_VAR:
                    if nodes[u, COL_VAR] in (x_id,y_id):
                        nodes[new_off, COL_TYPE] = TYPE_ERA
                        nodes[new_off, COL_ACTIVE] = 1
                    else:
                        nodes[new_off, COL_TYPE] = TYPE_VAR
                        nodes[new_off, COL_VAR]  = nodes[u, COL_VAR]
                        nodes[new_off, COL_ACTIVE] = 1
                elif u_type == TYPE_ERA:
                    nodes[new_off, COL_TYPE] = TYPE_ERA
                    nodes[new_off, COL_ACTIVE] = 1
                else:
                    nodes[new_off, COL_TYPE]  = u_type
                    nodes[new_off, COL_LABEL] = nodes[u, COL_LABEL]
                    c1_u = nodes[u, COL_C1]; c2_u = nodes[u, COL_C2]
                    if c1_u>=0:
                        if nodes[c1_u, COL_TYPE]==TYPE_VAR and nodes[c1_u, COL_VAR] in (x_id,y_id):
                            nodes[new_off, COL_C1] = era_off
                            nodes[era_off, COL_PARENT] = new_off; nodes[era_off, COL_PFIELD] = COL_C1
                        else:
                            c1n = mapping_arr[c1_u]
                            nodes[new_off, COL_C1] = c1n
                            nodes[c1n, COL_PARENT] = new_off; nodes[c1n, COL_PFIELD] = COL_C1
                    if c2_u>=0:
                        if nodes[c2_u, COL_TYPE]==TYPE_VAR and nodes[c2_u, COL_VAR] in (x_id,y_id):
                            nodes[new_off, COL_C2] = era_off
                            nodes[era_off, COL_PARENT] = new_off; nodes[era_off, COL_PFIELD] = COL_C2
                        else:
                            c2n = mapping_arr[c2_u]
                            nodes[new_off, COL_C2] = c2n
                            nodes[c2n, COL_PARENT] = new_off; nodes[c2n, COL_PFIELD] = COL_C2
                    nodes[new_off, COL_ACTIVE] = 1
            new_off = mapping_arr[body_off]
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = new_off
                nodes[new_off, COL_PARENT] = p; nodes[new_off, COL_PFIELD] = pf
            else:
                root_off = new_off
            continue

        # 7) Dup-Lam
        mask_dup_lam = (types == TYPE_DUP) & (active == 1) & (nodes[:, COL_C1]>=0) & (nodes[:, COL_C1]<bs) & (nodes[nodes[:, COL_C1], COL_TYPE] == TYPE_LAM)
        if np.any(mask_dup_lam):
            ro = np.nonzero(mask_dup_lam)[0][0]
            dup_lab = nodes[ro, COL_LABEL]
            x_bind = nodes[ro, COL_X]; y_bind = nodes[ro, COL_Y]
            lam_off = nodes[ro, COL_C1]; body_off = nodes[ro, COL_C2]
            lam_lab = nodes[lam_off, COL_LABEL]
            inner_var = nodes[lam_off, COL_VAR]
            val_body_off = nodes[lam_off, COL_C1]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            # fresh x0,x1,f0,f1 at base..base+3
            off_x0 = base; off_x1 = base+1; off_f0 = base+2; off_f1 = base+3
            for off_var in (off_x0, off_x1, off_f0, off_f1):
                nodes[off_var, COL_TYPE] = TYPE_VAR
                nodes[off_var, COL_VAR]  = off_var
                nodes[off_var, COL_ACTIVE] = 1
            # Lam nodes for x0->Var(f0), x1->Var(f1)
            # r = Lam(lam_lab, x0, Var(f0)); s = Lam(lam_lab, x1, Var(f1))
            off_r = base+4; off_body_r = base+6
            nodes[off_r, COL_TYPE]  = TYPE_LAM; nodes[off_r, COL_LABEL]=lam_lab; nodes[off_r, COL_VAR]=off_x0; nodes[off_r, COL_ACTIVE]=1
            nodes[off_body_r, COL_TYPE]  = TYPE_VAR; nodes[off_body_r, COL_VAR]=off_f0; nodes[off_body_r, COL_ACTIVE]=1
            nodes[off_r, COL_C1] = off_body_r; nodes[off_body_r, COL_PARENT]=off_r; nodes[off_body_r, COL_PFIELD]=COL_C1
            off_s = base+5; off_body_s = base+7
            nodes[off_s, COL_TYPE]  = TYPE_LAM; nodes[off_s, COL_LABEL]=lam_lab; nodes[off_s, COL_VAR]=off_x1; nodes[off_s, COL_ACTIVE]=1
            nodes[off_body_s, COL_TYPE]  = TYPE_VAR; nodes[off_body_s, COL_VAR]=off_f1; nodes[off_body_s, COL_ACTIVE]=1
            nodes[off_s, COL_C1] = off_body_s; nodes[off_body_s, COL_PARENT]=off_s; nodes[off_body_s, COL_PFIELD]=COL_C1
            # Sup for x_bind: Sup(dup_lab, Var(x0), Var(x1))
            off_sup = base+8
            nodes[off_sup, COL_TYPE]=TYPE_SUP; nodes[off_sup, COL_LABEL]=dup_lab; nodes[off_sup, COL_ACTIVE]=1
            nodes[off_sup, COL_C1]=off_x0; nodes[off_x0, COL_PARENT]=off_sup; nodes[off_x0, COL_PFIELD]=COL_C1
            nodes[off_sup, COL_C2]=off_x1; nodes[off_x1, COL_PARENT]=off_sup; nodes[off_x1, COL_PFIELD]=COL_C2
            # New Dup: Dup(dup_lab, f0, f1, val_body_off, body_off)
            off_dup_new = base+9
            nodes[off_dup_new, COL_TYPE]=TYPE_DUP; nodes[off_dup_new, COL_LABEL]=dup_lab
            nodes[off_dup_new, COL_X]=off_f0; nodes[off_dup_new, COL_Y]=off_f1; nodes[off_dup_new, COL_ACTIVE]=1
            nodes[off_dup_new, COL_C1]=val_body_off; nodes[val_body_off, COL_PARENT]=off_dup_new; nodes[val_body_off, COL_PFIELD]=COL_C1
            nodes[off_dup_new, COL_C2]=body_off; nodes[body_off, COL_PARENT]=off_dup_new; nodes[body_off, COL_PFIELD]=COL_C2
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = off_dup_new
                nodes[off_dup_new, COL_PARENT]=p; nodes[off_dup_new, COL_PFIELD]=pf
            else:
                root_off = off_dup_new
            continue

        # 8) Dup-Sup
        mask_dup_sup = (types == TYPE_DUP) & (active == 1) & (nodes[:, COL_C1]>=0) & (nodes[:, COL_C1]<bs) & (nodes[nodes[:, COL_C1], COL_TYPE] == TYPE_SUP)
        if np.any(mask_dup_sup):
            ro = np.nonzero(mask_dup_sup)[0][0]
            dup_lab = nodes[ro, COL_LABEL]
            x_bind = nodes[ro, COL_X]; y_bind = nodes[ro, COL_Y]
            sup_off = nodes[ro, COL_C1]; body_off = nodes[ro, COL_C2]
            sup_lab = nodes[sup_off, COL_LABEL]
            left = nodes[sup_off, COL_C1]; right = nodes[sup_off, COL_C2]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            if dup_lab == sup_lab:
                # substitute in body
                SUB_MAX = K
                subtree = np.full(SUB_MAX, -1, dtype=np.int32)
                visited = np.zeros(bs, bool)
                subtree[0] = body_off; visited[body_off] = True
                for idx in range(SUB_MAX):
                    u = subtree[idx]
                    if u<0: continue
                    u_type = nodes[u, COL_TYPE]
                    if u_type == TYPE_LAM:
                        c = nodes[u, COL_C1]
                        if not (nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR] in (x_bind,y_bind)):
                            if not visited[c]:
                                for j in range(SUB_MAX):
                                    if subtree[j]<0:
                                        subtree[j]=c; break
                                visited[c]=True
                    elif u_type in (TYPE_APP, TYPE_LET, TYPE_SUP, TYPE_DUP):
                        for cf in (COL_C1, COL_C2):
                            c = nodes[u, cf]
                            if c<0: continue
                            if nodes[c, COL_TYPE]==TYPE_VAR and nodes[c, COL_VAR] in (x_bind,y_bind):
                                continue
                            if not visited[c]:
                                for j in range(SUB_MAX):
                                    if subtree[j]<0:
                                        subtree[j]=c; break
                                visited[c]=True
                mapping_arr = np.full(bs, -1, dtype=np.int32)
                for idx in range(SUB_MAX):
                    u = subtree[idx]
                    if u<0: break
                    mapping_arr[u] = base + idx
                for idx in range(SUB_MAX):
                    u = subtree[idx]
                    if u<0: break
                    new_off = mapping_arr[u]
                    u_type = nodes[u, COL_TYPE]
                    if u_type == TYPE_VAR:
                        if nodes[u, COL_VAR] == x_bind:
                            nodes[new_off, COL_TYPE] = TYPE_VAR
                            nodes[new_off, COL_VAR]  = left
                        elif nodes[u, COL_VAR] == y_bind:
                            nodes[new_off, COL_TYPE] = TYPE_VAR
                            nodes[new_off, COL_VAR]  = right
                        else:
                            nodes[new_off, COL_TYPE] = TYPE_VAR
                            nodes[new_off, COL_VAR]  = nodes[u, COL_VAR]
                        nodes[new_off, COL_ACTIVE] = 1
                    elif u_type == TYPE_ERA:
                        nodes[new_off, COL_TYPE] = TYPE_ERA; nodes[new_off, COL_ACTIVE] = 1
                    else:
                        nodes[new_off, COL_TYPE]  = u_type
                        nodes[new_off, COL_LABEL] = nodes[u, COL_LABEL]
                        c1_u = nodes[u, COL_C1]; c2_u = nodes[u, COL_C2]
                        if c1_u>=0:
                            if nodes[c1_u, COL_TYPE]==TYPE_VAR and nodes[c1_u, COL_VAR]==x_bind:
                                nodes[new_off, COL_C1] = left
                                nodes[left, COL_PARENT] = new_off; nodes[left, COL_PFIELD] = COL_C1
                            elif nodes[c1_u, COL_TYPE]==TYPE_VAR and nodes[c1_u, COL_VAR]==y_bind:
                                nodes[new_off, COL_C1] = right
                                nodes[right, COL_PARENT] = new_off; nodes[right, COL_PFIELD] = COL_C1
                            else:
                                c1n = mapping_arr[c1_u]
                                nodes[new_off, COL_C1] = c1n
                                nodes[c1n, COL_PARENT] = new_off; nodes[c1n, COL_PFIELD] = COL_C1
                        if c2_u>=0:
                            if nodes[c2_u, COL_TYPE]==TYPE_VAR and nodes[c2_u, COL_VAR]==x_bind:
                                nodes[new_off, COL_C2] = left
                                nodes[left, COL_PARENT] = new_off; nodes[left, COL_PFIELD] = COL_C2
                            elif nodes[c2_u, COL_TYPE]==TYPE_VAR and nodes[c2_u, COL_VAR]==y_bind:
                                nodes[new_off, COL_C2] = right
                                nodes[right, COL_PARENT] = new_off; nodes[right, COL_PFIELD] = COL_C2
                            else:
                                c2n = mapping_arr[c2_u]
                                nodes[new_off, COL_C2] = c2n
                                nodes[c2n, COL_PARENT] = new_off; nodes[c2n, COL_PFIELD] = COL_C2
                        nodes[new_off, COL_ACTIVE] = 1
                new_off = mapping_arr[body_off]
            else:
                # dup_lab != sup_lab
                off_a0 = base; off_a1 = base+1; off_b0 = base+2; off_b1 = base+3
                for off_var in (off_a0, off_a1, off_b0, off_b1):
                    nodes[off_var, COL_TYPE] = TYPE_VAR
                    nodes[off_var, COL_VAR]  = off_var
                    nodes[off_var, COL_ACTIVE] = 1
                # Sup for x_bind
                off_sup_x = base+4
                nodes[off_sup_x, COL_TYPE]   = TYPE_SUP
                nodes[off_sup_x, COL_LABEL]  = sup_lab
                nodes[off_sup_x, COL_ACTIVE] = 1
                nodes[off_sup_x, COL_C1] = off_a0
                nodes[off_a0, COL_PARENT] = off_sup_x; nodes[off_a0, COL_PFIELD] = COL_C1
                nodes[off_sup_x, COL_C2] = off_a1
                nodes[off_a1, COL_PARENT] = off_sup_x; nodes[off_a1, COL_PFIELD] = COL_C2
                # nested Dup for right branch
                off_inner = base+5
                nodes[off_inner, COL_TYPE]  = TYPE_DUP
                nodes[off_inner, COL_LABEL] = dup_lab
                nodes[off_inner, COL_X]     = off_b0
                nodes[off_inner, COL_Y]     = off_b1
                nodes[off_inner, COL_ACTIVE] = 1
                nodes[off_inner, COL_C1]    = right
                nodes[right, COL_PARENT]    = off_inner; nodes[right, COL_PFIELD] = COL_C1
                nodes[off_inner, COL_C2]    = body_off
                nodes[body_off, COL_PARENT] = off_inner; nodes[body_off, COL_PFIELD] = COL_C2
                # outer Dup
                off_dup_new = base+6
                nodes[off_dup_new, COL_TYPE]   = TYPE_DUP
                nodes[off_dup_new, COL_LABEL]  = dup_lab
                nodes[off_dup_new, COL_X]      = off_a0
                nodes[off_dup_new, COL_Y]      = off_a1
                nodes[off_dup_new, COL_ACTIVE] = 1
                nodes[off_dup_new, COL_C1]     = left
                nodes[left, COL_PARENT]        = off_dup_new; nodes[left, COL_PFIELD] = COL_C1
                nodes[off_dup_new, COL_C2]     = off_inner
                nodes[off_inner, COL_PARENT]   = off_dup_new; nodes[off_inner, COL_PFIELD] = COL_C2
                new_off = off_dup_new
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = new_off
                nodes[new_off, COL_PARENT] = p; nodes[new_off, COL_PFIELD] = pf
            else:
                root_off = new_off
            continue

        # 9) Dup-Dup
        mask_dup_dup = (types == TYPE_DUP) & (active == 1) & (nodes[:, COL_C1]>=0) & (nodes[:, COL_C1]<bs) & (nodes[nodes[:, COL_C1], COL_TYPE] == TYPE_DUP)
        if np.any(mask_dup_dup):
            ro = np.nonzero(mask_dup_dup)[0][0]
            dup_lab = nodes[ro, COL_LABEL]
            x_bind = nodes[ro, COL_X]; y_bind = nodes[ro, COL_Y]
            inner = nodes[ro, COL_C1]
            x0 = nodes[inner, COL_X]; y0 = nodes[inner, COL_Y]
            val0 = nodes[inner, COL_C1]; body0 = nodes[inner, COL_C2]
            mask_sub = collect_subtree(nodes, ro)
            nodes[mask_sub, COL_ACTIVE] = 0
            base = clone_base(ro, N_initial, K, d)
            # simplest static nesting
            off_inner_new = base
            nodes[off_inner_new, COL_TYPE]   = TYPE_DUP
            nodes[off_inner_new, COL_LABEL]  = dup_lab
            nodes[off_inner_new, COL_X]      = x0
            nodes[off_inner_new, COL_Y]      = y0
            nodes[off_inner_new, COL_ACTIVE] = 1
            nodes[off_inner_new, COL_C1]     = val0
            nodes[val0, COL_PARENT] = off_inner_new; nodes[val0, COL_PFIELD] = COL_C1
            nodes[off_inner_new, COL_C2]     = body0
            nodes[body0, COL_PARENT] = off_inner_new; nodes[body0, COL_PFIELD] = COL_C2
            off_outer = base+1
            nodes[off_outer, COL_TYPE]   = TYPE_DUP
            nodes[off_outer, COL_LABEL]  = dup_lab
            nodes[off_outer, COL_X]      = x_bind
            nodes[off_outer, COL_Y]      = y_bind
            nodes[off_outer, COL_ACTIVE] = 1
            nodes[off_outer, COL_C1]     = val0
            nodes[val0, COL_PARENT] = off_outer; nodes[val0, COL_PFIELD] = COL_C1
            nodes[off_outer, COL_C2]     = off_inner_new
            nodes[off_inner_new, COL_PARENT] = off_outer; nodes[off_inner_new, COL_PFIELD] = COL_C2
            new_off = off_outer
            p = nodes[ro, COL_PARENT]; pf = nodes[ro, COL_PFIELD]
            if p>=0:
                nodes[p, pf] = new_off
                nodes[new_off, COL_PARENT] = p; nodes[new_off, COL_PFIELD] = pf
            else:
                root_off = new_off
            continue

        # no more patterns apply
        break

    return nodes, root_off

# --- Reconstruction from array back to AST form for testing/inspection ---
def array_to_ast(nodes, off):
    if off is None: return None
    typ = nodes[off, COL_TYPE]
    if typ == TYPE_VAR:
        return Var(nodes[off, COL_VAR])
    if typ == TYPE_ERA:
        return Era()
    if typ == TYPE_LAM:
        return Lam(nodes[off, COL_LABEL], nodes[off, COL_VAR], array_to_ast(nodes, nodes[off, COL_C1]))
    if typ == TYPE_APP:
        return App(nodes[off, COL_LABEL], array_to_ast(nodes, nodes[off, COL_C1]), array_to_ast(nodes, nodes[off, COL_C2]))
    if typ == TYPE_SUP:
        return Sup(nodes[off, COL_LABEL], array_to_ast(nodes, nodes[off, COL_C1]), array_to_ast(nodes, nodes[off, COL_C2]))
    if typ == TYPE_DUP:
        return Dup(nodes[off, COL_LABEL], nodes[off, COL_X], nodes[off, COL_Y],
                   array_to_ast(nodes, nodes[off, COL_C1]),
                   array_to_ast(nodes, nodes[off, COL_C2]))
    if typ == TYPE_LET:
        return None
    return None

def run_static(expr_ast, MAX_DEPTH=5, K=50):
    nodes, N = init_static_array(expr_ast, MAX_DEPTH, K)
    nodes_after, root = static_run(nodes, N, MAX_DEPTH, K)
    return array_to_ast(nodes_after, root)

# --- Unit tests ---
if __name__ == "__main__":
    # App-Era
    expr1 = App(0, Era(), Var(5))
    print("App-Era →", run_static(expr1, MAX_DEPTH=1, K=10))

    # Identity ((λx.x) t) → t
    lam_id = Lam(0, 0, Var(0))
    expr2 = App(0, lam_id, Var(7))
    print("Identity →", run_static(expr2, MAX_DEPTH=1, K=10))

    # Beta ((λa.(λb.a)) t) f → after 2 steps
    lam_b_a    = Lam(0, 1, Var(0))
    lam_a_body = Lam(0, 0, lam_b_a)
    expr3      = App(0, App(0, lam_a_body, Var(2)), Var(3))
    print("Beta →", run_static(expr3, MAX_DEPTH=2, K=50))

    # Let: Let(3,Var(42),Var(3)) → Var(42)
    expr4 = Let(3, Var(42), Var(3))
    print("Let →", run_static(expr4, MAX_DEPTH=1, K=10))

    # App-Sup: App(5, Sup(1,Var(2),Var(3)), Var(7))
    expr5 = App(5, Sup(1, Var(2), Var(3)), Var(7))
    print("App-Sup →", run_static(expr5, MAX_DEPTH=1, K=20))

    # App-Dup: App(5, Var(1), Dup(2,10,11,Var(3),Var(4)))
    expr6 = App(5, Var(1), Dup(2,10,11,Var(3),Var(4)))
    print("App-Dup →", run_static(expr6, MAX_DEPTH=1, K=10))

    # Dup-Era: Dup(2,10,11,*,Var(5)) → Var(5)
    expr7 = Dup(2,10,11, Era(), Var(5))
    print("Dup-Era →", run_static(expr7, MAX_DEPTH=1, K=10))

    # Dup-Lam
    lam = Lam(0, 100, Var(200))
    expr8 = Dup(3,10,11, lam, Var(5))
    print("Dup-Lam →", run_static(expr8, MAX_DEPTH=1, K=20))

    # Dup-Sup equal-label
    sup = Sup(5, Var(2), Var(3))
    body = App(0, Var(10), Var(3))
    expr9 = Dup(5,10,11, sup, body)
    print("Dup-Sup equal-label →", run_static(expr9, MAX_DEPTH=1, K=50))

    # Dup-Sup diff-label
    sup2 = Sup(7, Var(2), Var(3))
    expr10 = Dup(6,10,11, sup2, Var(5))
    print("Dup-Sup diff-label →", run_static(expr10, MAX_DEPTH=1, K=50))

    # Dup-Dup
    inner = Dup(8,20,21, Var(2), Var(3))
    expr11 = Dup(8,10,11, inner, Var(5))
    print("Dup-Dup →", run_static(expr11, MAX_DEPTH=1, K=20))
