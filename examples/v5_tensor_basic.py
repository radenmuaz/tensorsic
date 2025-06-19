import numpy as np

# Redefine fresh and Term classes
_fresh_counter = 0
def fresh():
    global _fresh_counter
    v = _fresh_counter
    _fresh_counter += 1
    return v

class Term: pass
class Var(Term):
    def __init__(self, name: int):
        self.name = name
    def __str__(self):
        return f"Var({self.name})"
    __repr__ = __str__

class Era(Term):
    def __str__(self):
        return "Era"
    __repr__ = __str__

class Lam(Term):
    def __init__(self, x: int, body: Term):
        self.x = x
        self.body = body
    def __str__(self):
        return f"Lam({self.x}, {self.body})"
    __repr__ = __str__

class App(Term):
    def __init__(self, func: Term, arg: Term):
        self.func = func
        self.arg = arg
    def __str__(self):
        return f"App({self.func}, {self.arg})"
    __repr__ = __str__

# Type tags
TYPE_VAR = 0
TYPE_ERA = 1
TYPE_LAM = 2
TYPE_APP = 3

MAX_NODES = 50
def make_empty_array():
    arr = np.zeros((MAX_NODES, 6), dtype=np.int32)
    arr[:,0] = -1; arr[:,4] = 0
    return arr

def find_free_slots(arr):
    return list(np.nonzero(arr[:,4] == 0)[0])

def term_to_static_array(term, arr):
    mapping = {}
    next_free = 0
    def get_slot():
        nonlocal next_free
        while next_free < MAX_NODES and arr[next_free,4] == 1:
            next_free += 1
        if next_free >= MAX_NODES:
            raise RuntimeError("Out of preallocated slots")
        slot = next_free; next_free += 1
        return slot
    def recurse(t):
        if id(t) in mapping:
            return mapping[id(t)]
        slot = get_slot(); mapping[id(t)] = slot
        if isinstance(t, Var):
            arr[slot] = [TYPE_VAR, -1, -1, t.name, 1, 0]
        elif isinstance(t, Era):
            arr[slot] = [TYPE_ERA, -1, -1, -1, 1, 0]
        elif isinstance(t, Lam):
            arr[slot,0] = TYPE_LAM; arr[slot,3] = t.x; arr[slot,4] = 1; arr[slot,5] = 0
            bidx = recurse(t.body); arr[slot,1] = bidx; arr[slot,2] = -1
        elif isinstance(t, App):
            arr[slot,0] = TYPE_APP; arr[slot,3] = -1; arr[slot,4] = 1; arr[slot,5] = 0
            fidx = recurse(t.func); aidx = recurse(t.arg)
            arr[slot,1], arr[slot,2] = fidx, aidx
        else:
            raise RuntimeError(f"Unknown Term: {t}")
        return slot
    root = recurse(term)
    return arr, root

def array_to_term(arr, root_idx):
    cache = {}
    def build(i):
        if i in cache:
            return cache[i]
        typ = arr[i,0]; active = arr[i,4]
        if active==0 or typ<0:
            t = Era()
        else:
            if typ==TYPE_VAR:
                t = Var(arr[i,3])
            elif typ==TYPE_ERA:
                t = Era()
            elif typ==TYPE_LAM:
                body = build(arr[i,1]); t = Lam(arr[i,3], body)
            elif typ==TYPE_APP:
                f = build(arr[i,1]); a = build(arr[i,2]); t = App(f,a)
            else:
                t = Era()
        cache[i] = t; return t
    return build(root_idx)

def find_reducible(arr):
    reducible = []
    for i in range(arr.shape[0]):
        if arr[i,4] != 1: continue
        if arr[i,0] == TYPE_APP:
            f = arr[i,1]
            if f>=0 and arr[f,4]==1 and arr[f,0] == TYPE_LAM:
                reducible.append(i)
    return reducible

def reduction_step_at(arr, redex_idx):
    # Similar to before but at arbitrary index
    lam_idx = arr[redex_idx,1]
    arg_idx = arr[redex_idx,2]
    param = arr[lam_idx,3]
    body_idx = arr[lam_idx,1]
    # deactivate reachable from redex_idx
    to_visit=[redex_idx]; to_deactivate=set()
    while to_visit:
        i=to_visit.pop()
        if i in to_deactivate: continue
        to_deactivate.add(i)
        t=arr[i,0]
        if t==TYPE_LAM:
            c=arr[i,1]
            if c>=0: to_visit.append(c)
        elif t==TYPE_APP:
            c1,c2=arr[i,1],arr[i,2]
            if c1>=0: to_visit.append(c1)
            if c2>=0: to_visit.append(c2)
    for i in to_deactivate: arr[i,4]=0
    clone_map={}; free_slots=find_free_slots(arr); free_iter=iter(free_slots)
    def clone(i):
        if i in clone_map:
            return clone_map[i]
        typ_i=arr[i,0]
        if typ_i==TYPE_VAR:
            vid=arr[i,3]
            if vid==param:
                return clone(arg_idx)
            else:
                try: new_i=next(free_iter)
                except StopIteration: raise RuntimeError("No free slot")
                arr[new_i]=[TYPE_VAR,-1,-1,vid,1,arr[i,5]+1]
                clone_map[i]=new_i; return new_i
        elif typ_i==TYPE_ERA:
            try: new_i=next(free_iter)
            except StopIteration: raise RuntimeError("No free slot")
            arr[new_i]=[TYPE_ERA,-1,-1,-1,1,arr[i,5]+1]
            clone_map[i]=new_i; return new_i
        elif typ_i==TYPE_LAM:
            nested=arr[i,3]
            try: new_i=next(free_iter)
            except StopIteration: raise RuntimeError("No free slot")
            clone_map[i]=new_i
            c=arr[i,1]; cn=clone(c)
            arr[new_i]=[TYPE_LAM,cn,-1,nested,1,arr[i,5]+1]
            return new_i
        elif typ_i==TYPE_APP:
            try: new_i=next(free_iter)
            except StopIteration: raise RuntimeError("No free slot")
            clone_map[i]=new_i
            f=arr[i,1]; a=arr[i,2]
            fn=clone(f); an=clone(a)
            arr[new_i]=[TYPE_APP,fn,an,-1,1,arr[i,5]+1]
            return new_i
        else:
            raise RuntimeError("Unsupported")
    return clone(body_idx)

# Run boolean reduction using scan
arr = make_empty_array(); _fresh_counter=0
# Build True t f
a = fresh(); b = fresh()
true_term = Lam(a, Lam(b, Var(a)))
t = Var(fresh()); f = Var(fresh())
term_bool = App(App(true_term, t), f)
arr, root = term_to_static_array(term_bool, arr)
print("Before:", array_to_term(arr, root))
# Loop until no reducible
while True:
    redexes = find_reducible(arr)
    if not redexes:
        break
    # pick leftmost (smallest index)
    idx = min(redexes)
    root = reduction_step_at(arr, idx)
    # If root was deactivated, but we do not track parent pointers; for this small test,
    # root index may change only if idx == root. For nested, outer App remains active
print("After:", array_to_term(arr, root))

# Identity test
arr = make_empty_array(); _fresh_counter=0
x = fresh()
term_id = App(Lam(x, Var(x)), Era())
arr, root = term_to_static_array(term_id, arr)
print("Identity before:", array_to_term(arr, root))
redexes = find_reducible(arr)
if redexes:
    root = reduction_step_at(arr, redexes[0])
print("Identity after:", array_to_term(arr, root))
