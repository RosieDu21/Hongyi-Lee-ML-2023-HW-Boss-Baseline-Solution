import math
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
	def __init__(self, beta=1.0):
		super(Swish, self).__init__()
		self.beta = float(beta)

	def forward(self, x):
		return x*torch.sigmoid(self.beta*x)


class FeedForward(nn.Module):
	def __init__(self, input_dim, expand_factor=4, dropout=0.1):
		super(FeedForward, self).__init__()
		self.feed_forward = nn.Sequential(
			nn.LayerNorm(input_dim),
			nn.Linear(input_dim, input_dim*expand_factor),
			Swish(),
			nn.Dropout(dropout),
			nn.Linear(input_dim*expand_factor, input_dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.feed_forward(x)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model=256, max_len=10000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model, requires_grad=False)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, length):
		return self.pe[:, :length]


class MultiHeadAttentioWithRPE(nn.Module):
	def __init__(self, d_model, feat_dim=256, nhead=4, batch_first=False, droupout=0.1):
		super(MultiHeadAttentioWithRPE, self).__init__()
		self.sqrt_d_model = math.sqrt(d_model)
		self.nhead = nhead
		self.feat_dim = feat_dim
		self.batch_first = batch_first

		self.layernorm = nn.LayerNorm(d_model)

		self.q_net = nn.Linear(d_model, feat_dim*nhead, bias=False)
		self.k_net = nn.Linear(d_model, feat_dim*nhead, bias=False)
		self.v_net = nn.Linear(d_model, feat_dim*nhead, bias=False)
		self.r_net = nn.Linear(d_model, feat_dim*nhead, bias=False)

		self.pe = PositionalEncoding(d_model)

		self.u_bias = nn.Parameter(torch.Tensor(self.nhead, self.feat_dim))
		self.v_bias = nn.Parameter(torch.Tensor(self.nhead, self.feat_dim))
		nn.init.xavier_uniform_(self.u_bias)
		nn.init.xavier_uniform_(self.v_bias)

		self.out = nn.Linear(feat_dim*nhead, d_model)

		self.dropout = nn.Dropout(droupout)

	def _left_shift(self, x):
		zero = torch.zeros((x.size(0),1,x.size(2),x.size(3)), device=x.device, dtype=x.dtype)
		padded = torch.cat((zero,x),dim=1)
		padded = padded.view(x.size(1)+1,x.size(0),x.size(2),x.size(3))
		padded = padded[1:,:,:,:]
		x = padded.view_as(x)
		x = torch.tril(x, diagonal=x.size(1)-x.size(0))
		return x

	def forward(self, x):
		h = self.layernorm(x)

		if self.batch_first:
			h.permute(1,0,2)

		pe = self.pe(h.size(0)).flip(1)

		q = self.q_net(h).view(h.size(0),h.size(1),self.nhead,self.feat_dim)
		k = self.q_net(h).view(h.size(0),h.size(1),self.nhead,self.feat_dim)
		v = self.v_net(h).view(h.size(0),h.size(1),self.nhead,self.feat_dim)
		pe = self.r_net(pe).view(h.size(0),self.nhead,self.feat_dim)

		content = torch.einsum('ibnf,jbnf->ijbn',(q+self.u_bias,k))
		pos = torch.einsum('ibnf,jnf->ijbn',(q+self.v_bias,pe))
		pos = self._left_shift(pos)

		attn = (content + pos) / self.sqrt_d_model
		attn = F.softmax(attn, dim=1)

		h = torch.einsum('ijbn,jbnf->ibnf',[attn,v])
		h = h.reshape(h.size(0),h.size(1),self.feat_dim*self.nhead)

		h = self.out(h)
		h = self.dropout(h)

		if self.batch_first:
			h.permute(1,0,2)

		return h


class Convolution(nn.Module):
	def __init__(self, d_model, expand_factor=2, kernel_size=3, stride=1, dropout=0.1):
		super(Convolution, self).__init__()
		self.ln = nn.LayerNorm(d_model)
		self.pointwiseconv1d_1 = nn.Conv1d(
			d_model,
			d_model*expand_factor*2,
			kernel_size=1,
			stride=1,
			padding=0
		)
		self.glu = nn.GLU(dim=1)
		self.depthwiseconv1d = nn.Conv1d(
			d_model*expand_factor,
			d_model*expand_factor,
			kernel_size=kernel_size,
			stride=stride,
			padding=(kernel_size-1)//2,
			groups=d_model*expand_factor
		)
		self.bn = nn.BatchNorm1d(d_model * expand_factor)
		self.swish = Swish()
		self.pointwiseconv1d_2 = nn.Conv1d(
			d_model * expand_factor,
			d_model,
			kernel_size=1,
			stride=1,
			padding=0
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.ln(x)
		x = x.permute(0,2,1)
		x = self.pointwiseconv1d_1(x)
		x = self.glu(x)
		x = self.depthwiseconv1d(x)
		x = self.bn(x)
		x = self.swish(x)
		x = self.pointwiseconv1d_2(x)
		x = x.permute(0,2,1)
		x = self.dropout(x)
		return x


class ConformerBlock(nn.Module):
	def __init__(self, d_model=256, nhead=1, expand_factor=4, dropout=0.1, batch_first=False):
		super(ConformerBlock, self).__init__()
		self.batch_first = batch_first
		self.linear1 = FeedForward(d_model,expand_factor,dropout)
		self.attn = MultiHeadAttentioWithRPE(d_model,d_model,nhead,batch_first,dropout)
		self.conv = Convolution(d_model,expand_factor,dropout=dropout)
		self.linear2 = FeedForward(d_model,expand_factor,dropout)
		self.layernorm = nn.LayerNorm(d_model)

	def forward(self, x):
		x = 0.5*self.linear1(x) + x
		x = self.attn(x) + x
		if not self.batch_first:
			x.permute(1,0,2)
		x = self.conv(x) + x
		if not self.batch_first:
			x.permute(1,0,2)
		x = 0.5*self.linear2(x) + x
		return self.layernorm(x)


class Classifier(nn.Module):
	def __init__(self, d_model=256, nhead=4, expand_factor=4, num_layers=8, n_spks=600, dropout=0.3):
		super(Classifier, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(40, d_model),
			nn.Dropout(dropout),
			*[ConformerBlock(d_model,nhead,expand_factor,dropout,True) for _ in range(num_layers)]
		)
		self.w_net = nn.Linear(d_model,1)

		self.out = nn.Linear(d_model,n_spks, bias=False)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		x = self.net(mels)

		w = F.softmax(self.w_net(x).squeeze(-1),dim=1)
		x = torch.einsum('bl,bld->bd',(w,x))

		# x (batch size, d_model)
		x = F.normalize(x,dim=-1)
		self.out.weight.data = F.normalize(self.out.weight.data,dim=-1)

		x = self.out(x)

		return x