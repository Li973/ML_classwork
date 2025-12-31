import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv


def gumbel_softmax_sample(logits, tau=1.0, hard=False, eps=1e-10):
    #[E, n_edge_types]
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y = F.softmax(gumbels, dim=-1)
    if hard:
        index = y.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y.detach() + y
    return y



class CausalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_edge_types):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.node_emb = nn.Linear(input_dim, hidden_dim)

        self.gcn1 = GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim, activation=F.relu)

        self.edge_logits = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_edge_types)
        )

    def forward(self, x, edge_index, tau=0.5, hard=False):
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.node_emb(x)

        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.size(0))
        g = dgl.add_self_loop(g)
        g.ndata['h'] = h
        h = self.gcn1(g, h)
        h = self.gcn2(g, h)

        src, dst = edge_index  #[2, N*N]
        h_pairs = torch.cat([h[src], h[dst]], dim=-1)  #[N^2, 2*hidden]
        logits = self.edge_logits(h_pairs)
        z = gumbel_softmax_sample(logits, tau=tau, hard=hard)  #[N^2, n_edge_types]
        return z, logits


class CausalDecoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_edge_types, output_dim=None):
        super().__init__()
        self.n_edge_types = n_edge_types
        output_dim = output_dim or node_dim
        self.edge_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, node_dim))
            for _ in range(n_edge_types)
        ])

        self.edge_fns[0] = lambda x: torch.zeros_like(x[:, :node_dim])

        self.node_update = nn.Sequential(
            nn.Linear(node_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.residual = True

    def forward(self, x_t, z, edge_index):
        src, dst = edge_index
        agg_msg = torch.zeros_like(x_t)
        for e in range(self.n_edge_types):
            weight_e = z[:, e].unsqueeze(-1)
            msg_input = torch.cat([x_t[src], x_t[dst]], dim=-1)
            msg = self.edge_fns[e](msg_input)
            weighted_msg = weight_e * msg
            agg_msg.index_add_(0, dst, weighted_msg)

        node_input = torch.cat([x_t, agg_msg], dim=-1)
        delta = self.node_update(node_input)
        x_next = x_t + delta if self.residual else delta
        return x_next


class AmortizedCausalDiscovery(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_edge_types=3, tau=0.5):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.tau = tau
        self.encoder = CausalEncoder(input_dim, hidden_dim, n_edge_types)
        self.decoder = CausalDecoder(input_dim, hidden_dim, n_edge_types)

    def build_fully_connected_edge_index(self, N):
        i = torch.arange(N).repeat_interleave(N)
        j = torch.arange(N).repeat(N)
        return torch.stack([i, j])  #[2, N*N]

    def forward(self, x_seq, hard=False):
        N, T, D = x_seq.shape
        edge_index = self.build_fully_connected_edge_index(N).to(x_seq.device)

        z, logits = self.encoder(x_seq, edge_index, tau=self.tau, hard=hard)

        losses = []
        for t in range(T - 1):
            x_t = x_seq[:, t, :]  #[N, D]
            x_target = x_seq[:, t + 1, :]  #[N, D]
            x_pred = self.decoder(x_t, z, edge_index)
            loss_t = F.mse_loss(x_pred, x_target, reduction='sum')
            losses.append(loss_t)
        recon_loss = torch.stack(losses).mean()

        p0 = 0.7
        prior = torch.tensor([p0] + [(1 - p0) / (self.n_edge_types - 1)] * (self.n_edge_types - 1))
        prior = prior.to(z.device).unsqueeze(0).expand_as(z)
        q = F.softmax(logits, dim=-1)
        kl_loss = F.kl_div(q.log(), prior, reduction='batchmean')

        return recon_loss, kl_loss, z

    def predict_causal_graph(self, x_seq):
        N = x_seq.shape[0]
        edge_index = self.build_fully_connected_edge_index(N).to(x_seq.device)
        with torch.no_grad():

            z, _ = self.encoder(x_seq, edge_index, tau=1e-3, hard=True)
            edge_types = torch.argmax(z, dim=-1)
            adj = edge_types.view(N, N)

            return adj