import torch

n1 = 0
n2 = 1

G = 0
DUP = 1
NULL = 2
DORMANT = 3
ERASERS3 = 4

t_n1 = G
t_n2 = DUP

P, A1, A2 = 0, 1, 2

LEVELS = 2
NSUB = sum([2**d for d in range(LEVELS)]) - 1 # minus first node
# P, A1, A2
c_D = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
c_n1 = [[n2, P, 0], [n1, A2, 0], [n2, A1, 0]]
c_n2 = [[n1, P, 0], [n2, A2, 0], [n1, A1, 0]]
x = [[c_n1] + [c_D]*NSUB , c_n2 + [c_D]*NSUB]
x = torch.tensor([c_n1, c_n2]) # [nnode, [P, A1, A2], nsub]
breakpoint()

def step(x):
    for i in range(x.shape[0]):
        node = x[i]
    return x