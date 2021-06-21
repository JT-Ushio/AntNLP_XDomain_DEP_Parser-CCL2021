import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

	def __init__(self, d_in, d_out, p_drop, alpha):
		super(GraphAttentionLayer, self).__init__()
		self.d_in, self.d_out, self.p_drop = d_in, d_out, p_drop
		self.alpha = alpha

		self.attn = AddtiveAttention(self.d_in, self.d_out)
		self.leaky_relu = nn.LeakyRelu(self.alpha)

	def reset_parameters(self):
		gain = nn.init.calculate_gain('leaky_relu', self.alpha)
		if isinstance(self.attn, 'AddtiveAttention'):
			nn.init.xavier_uniform_(self.W, gain=gain)
			nn.init.xavier_uniform_(self.a1, gain=gain)
			nn.init.xavier_uniform_(self.a2, gain=gain)
		elif isinstance(self.attn, 'DotProductAttention'):
			nn.init.xavier_uniform_(self.W1, gain=gain)
			nn.init.xavier_uniform_(self.W2, gain=gain)

	def forward(self, h, adj):
		h = self.attn(h)
		
		


class AddtiveAttention(nn.Module):

	def __init__(self, d_in, d_out):
		super(AddtiveAttention, self).__init__()
		self.W = nn.Parameter(torch.Tensor(d_in, d_out))
		self.a1 = nn.Parameter(torch.Tensor(d_out, 1))
		self.a2 = nn.Parameter(torch.Tensor(d_out, 1))

	def forward(self, h):
		h = h * self.W 		# (N, d_out)
		h1 = h * self.a1 	# (N, 1)
		h2 = h * self.a2	# (N, 1)

		N = h.size()[0]
		h1 = h1.expand(-1, N) 		# (N, N)
		h2 = h2.t().expand(N, -1)	# (N, N)
		return h1 + h2


class DotProductAttention(nn.Module):

	def __init__(self, d_in, d_out):
		super(DotProductAttention, self).__init__()
		self.W1 = nn.Parameter(torch.Tensor(d_in, d_out))
		self.W2 = nn.Parameter(torch.Tensor(d_in, d_out))

	def forward(self, h):
		h1 = h * self.W1 	# (N, d_out)
		h2 = h * self.W2	# (N, d_out)
		return h1 * h2.t()
