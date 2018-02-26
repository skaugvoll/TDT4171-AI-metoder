'''
Smoothing is the process of computing the distribution over past states k, given evidence up to the present t,
that is P(Xk | e1:t), 0 <= k <= t

This computation is split up into 2 parts.

calculate evidence up to k and the evidence from k + 1 to t.

We can calculate the forward message f1:k by filtering forward from 1 to k.

The backward message bk+1:t can be compyted by a RECURSIve process that runs BACKWARD from t.

'''
