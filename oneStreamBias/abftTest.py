import torch
from torch import nn

# control file path
PathToControl = "/home/exouser"
LinFP = PathToControl + "/control/IFLinearABFT.txt"
colFP = PathToControl + "/control/abftCOL_FT.txt"
rowFP = PathToControl + "/control/abftROW_FT.txt"
matFP = PathToControl + "/control/IFABFT.txt"
inj = PathToControl + "/control/Injection.txt"
QKV = PathToControl + "/control/QKV.txt"
passChk = PathToControl + "/control/IFPassChk.txt"
batch = PathToControl + "/control/Batch.txt"
together = PathToControl + "/control/together.txt"

num_batch = 8
num_head = 12
m = 72
hidden_size = num_head * 64

X = torch.randn(num_batch, m, hidden_size).to('cuda')
W = nn.Linear(hidden_size,hidden_size,bias=True).to('cuda')

# For GPT2, QKV together
# W = nn.Linear(hidden_size, 3*hidden_size, bias=True).to('cuda')

with open(batch, 'w') as F:
    F.truncate(0)
    F.write(str(X.size()[0]))
    F.write(" ")
    F.write(str(num_head))

# for gpt-2. qkv togerther
with open(together, 'w') as frTo:
    frTo.truncate(0)
    frTo.write("f")

# Call abft-Linear
with open(LinFP, "w") as frLin:
    frLin.truncate(0)
    frLin.write('t')

# do col chk (col-major view)
with open(colFP, "w") as frCol:
    frCol.truncate(0)
    frCol.write('t')

# do row chk (col-major view)
with open(rowFP, "w") as frRow:
    frRow.truncate(0)
    frRow.write('t')

# pass check sum
with open(passChk, 'w') as frPassChk:
    frPassChk.truncate(0)
    frPassChk.write('t')

# do Q, K or V?
with open(QKV, 'w') as frQKV:
    frQKV.truncate(0)
    frQKV.write('q')

q = W(X)

# print(q)