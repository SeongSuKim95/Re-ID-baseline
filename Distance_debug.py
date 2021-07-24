import torch


t1 = torch.tensor([[1,1],[2,2],[3,3]]).float()
t2 = torch.tensor([[1,1],[2,2],[3,3]]).float()

q = torch.cdist(t1,t2,p =2)

a = torch.ones(64,2048)
a[1] = torch.ones(2048)*2
b = a


aa = torch.pow(a,2).sum(1,keepdim=True).expand(64,64)
bb = torch.pow(b,2).sum(1,keepdim=True).expand(64,64).t()

dist = aa+bb

dist = torch.addmm(dist,a,b.t(),beta=1,alpha=-2)
dist = dist.clamp(min=1e-12).sqrt()

dist_test = torch.cdist(a,b,p =2).clamp(min=1e-6)

distance_matrix = dist_test
distance = torch.zeros(16,16)
for i in range (distance.size(0)):
  distance[i] = i
distance[0][1] = 1

c = torch.tensor([1,1,1,1,3,3,3,3,10,10,10,10,8,8,8,8])
N = c.size(0)
cc = c.expand(N,N)
c_pos = cc.eq(cc.t())
c_neq = cc.ne(cc.t())

d = distance[c_pos]
dd = d.contiguous().view(N,-1)

e,f = torch.max(dd,1,keepdim=True)

e = e.squeeze(1)

g = e.new().resize_as_(e).fill_(1)

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    # m, n = x.size(0), y.size(0)
    # ## torch.pow(x,2) --> element wise square
    # ## sum(1,keepdim=True) --> [64,1] feature wise sum
    # ## expand --> [[1],[2],[3]] --> [[1,1,1],[2,2,2],[3,3,3]]

    # xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    # dist = xx + yy
    # ''' [ [feat1 sum(^2) + feat1 sum(^2), feat2 sum(^2) + feat1 sum(^2), ...]
    #       [feat1 sum(^2) + feat2 sum(^2), feat2 sum(^2) + feat1 sum(^2), ...] 
    # '''
    # dist = torch.addmm(dist,a,b.t(),beta=1,alpha=-2)
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = torch.cdist(x,y,p=2).clamp(min=1e-6)
    
    return dist
