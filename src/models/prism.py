import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from common.abstract_recommender import GeneralRecommender


class LightGCN(nn.Module):
    """LightGCN for behavioral representation learning."""
    def __init__(self, n_users, n_items, embedding_dim, n_layers=2):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, graph):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        user_emb, item_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return user_emb, item_emb


class GatedMechanism(nn.Module):
    """Gated mechanism for disentanglement."""
    def __init__(self, embedding_dim):
        super(GatedMechanism, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, h_item):
        g = self.gate(h_item)
        h_pref_relevant = h_item * g
        h_pref_irrelevant = h_item * (1 - g)
        return h_pref_relevant, h_pref_irrelevant


class PRISM(GeneralRecommender):
    """
    PRISM: Progressive Refinement with calIbrated Simulation for Multimodal recommendation.

    Three key improvements over DANCE:
    1. Calibrated Counterfactual: User-wise normalized scores for discriminative R*
       (fixes DANCE's degenerate R* mean~0.535 problem)
    2. Iterative Counterfactual Refinement: Periodically regenerates R* from improved
       Phase-3 embeddings (EM-style), so the counterfactual becomes more accurate over time
    3. Progressive Item-Item Graph Refinement: Periodically blends feature-based
       item similarity with learned behavioral similarity during training
    """
    def __init__(self, config, dataset):
        super(PRISM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        gcn_layers = config['n_layers']

        # Dataset-specific hyperparameters
        dataset_defaults = {
            'baby':     {'beta': 0.7, 'kb': 5},
            'sports':   {'beta': 0.5, 'kb': 3},
            'clothing': {'beta': 0.9, 'kb': 5},
        }
        ds_name = config['dataset'].lower()
        defaults = dataset_defaults.get(ds_name, {'beta': 0.7, 'kb': 5})
        self.beta = config['beta'] if config['beta'] is not None else defaults['beta']
        self.kb = config['kb'] if config['kb'] is not None else defaults['kb']

        # Loss weights
        self.lambda_infonce = config['lambda_infonce']
        self.lambda_ortho = config['lambda_ortho']
        self.lambda_pref = config['lambda_pref']
        self.lambda_l2 = config['lambda_l2']
        self.temperature = config['temperature']

        # PRISM-specific: calibrated counterfactual temperature
        self.cf_temperature = config['cf_temperature'] if config['cf_temperature'] is not None else 0.5

        # PRISM-specific: progressive graph refinement
        self.refine_step = config['refine_step'] if config['refine_step'] is not None else 50
        self.gamma_init = config['gamma_init'] if config['gamma_init'] is not None else 0.9
        self.gamma_final = config['gamma_final'] if config['gamma_final'] is not None else 0.5
        self.knn_k = config['knn_k']
        self._refine_epoch = 0  # tracks how many refinements done

        self.use_counterfactual = False

        # Build interaction matrix and graphs
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self._build_norm_adj()
        self.norm_adj = self._sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        R_sparse = self.interaction_matrix.tocsr()
        self.register_buffer('R_factual', torch.FloatTensor(R_sparse.toarray()))

        self.current_graph = self.norm_adj

        # Behavioral View: LightGCN
        self.lightgcn = LightGCN(self.n_users, self.n_items, self.embedding_dim, gcn_layers)

        # Multimodal View: parameter-free item-item graph (same as DANCE)
        self._build_item_item_graph(self.knn_k)

        # Gated Mechanism for disentanglement
        self.gated_mechanism = GatedMechanism(self.embedding_dim)

    # ==================== Graph Construction ====================

    def _build_item_item_graph(self, k):
        """Build fused item-item graph from raw multimodal features. Called at init and during refinement."""
        text_features = self._get_text_features()
        image_features = self._get_image_features()

        with torch.no_grad():
            text_adj = self._build_knn_adj(text_features, k=k)
            image_adj = self._build_knn_adj(image_features, k=k)
            fused_adj = 0.9 * text_adj + 0.1 * image_adj

        self.register_buffer('fused_item_adj', fused_adj)
        # Keep original feature-based graph for progressive refinement
        self.register_buffer('feature_item_adj', fused_adj.clone())

    def _build_knn_adj(self, features, k=10):
        """Build normalized binary k-NN adjacency matrix."""
        features_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(features_norm, features_norm.t())
        _, knn_ind = torch.topk(sim_matrix, k, dim=-1)
        adj = torch.zeros_like(sim_matrix).scatter_(-1, knn_ind, 1.0)
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat = torch.diagflat(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat, adj), d_mat)

    def refine_item_graph(self, epoch):
        """
        PRISM Innovation 2: Progressive Item-Item Graph Refinement.
        Blends feature-based item similarity with learned behavioral similarity.
        gamma decays from gamma_init to gamma_final over refinements.
        """
        self._refine_epoch += 1
        # Linearly decay gamma: start feature-heavy, gradually trust behavior more
        total_refinements = 5  # expected refinements over training
        gamma = self.gamma_init - (self.gamma_init - self.gamma_final) * (self._refine_epoch / total_refinements)
        gamma = max(self.gamma_final, gamma)

        with torch.no_grad():
            # Build behavioral similarity from current learned embeddings
            h_beh = F.normalize(self.lightgcn.item_embedding.weight, p=2, dim=1)
            beh_sim = torch.mm(h_beh, h_beh.t())
            beh_adj = self._build_knn_adj_from_sim(beh_sim, k=self.knn_k)

            # Blend: feature graph weighted by gamma, behavioral graph by (1-gamma)
            refined_adj = gamma * self.feature_item_adj + (1 - gamma) * beh_adj
            self.fused_item_adj.copy_(refined_adj)

        return gamma

    def _build_knn_adj_from_sim(self, sim_matrix, k=10):
        """Build normalized binary k-NN adjacency from a precomputed similarity matrix."""
        _, knn_ind = torch.topk(sim_matrix, k, dim=-1)
        adj = torch.zeros_like(sim_matrix).scatter_(-1, knn_ind, 1.0)
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat = torch.diagflat(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat, adj), d_mat)

    def _build_norm_adj(self):
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return norm_adj.tocoo()

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _get_text_features(self):
        if self.t_feat is not None:
            return self.t_feat
        return torch.randn(self.n_items, self.embedding_dim, device=self.device)

    def _get_image_features(self):
        if self.v_feat is not None:
            return self.v_feat
        return torch.randn(self.n_items, self.embedding_dim, device=self.device)

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    # ==================== Counterfactual Graph ====================

    def build_calibrated_counterfactual(self, h_user, h_item, batch_size=512):
        """
        PRISM Innovation 1: Calibrated Counterfactual Generation.

        Instead of raw sigmoid(h_u · h_i) which produces near-uniform scores (~0.535),
        we apply user-wise mean-centering + temperature scaling to produce
        discriminative scores spread across (0, 1).

        This ensures the counterfactual graph genuinely differs from the factual graph,
        making Phase 3 actually beneficial.
        """
        n_users = h_user.shape[0]
        n_items = h_item.shape[0]
        R_counterfactual = torch.zeros(n_users, n_items, device=h_user.device)

        for u_start in range(0, n_users, batch_size):
            u_end = min(u_start + batch_size, n_users)
            batch_users = h_user[u_start:u_end]
            # Raw dot product scores
            scores = torch.mm(batch_users, h_item.t())
            # User-wise calibration: center by mean, scale by std, apply temperature
            mean = scores.mean(dim=1, keepdim=True)
            std = scores.std(dim=1, keepdim=True) + 1e-8
            scores_calibrated = (scores - mean) / (std * self.cf_temperature)
            R_counterfactual[u_start:u_end] = torch.sigmoid(scores_calibrated)

        return R_counterfactual

    def build_balanced_graph(self, R_counterfactual):
        R_integrated = self.beta * R_counterfactual + (1 - self.beta) * self.R_factual

        topk_val_user, topk_idx_user = torch.topk(R_integrated, self.kb, dim=1)
        R_balanced_user = torch.zeros_like(R_integrated)
        R_balanced_user.scatter_(1, topk_idx_user, 1.0)

        topk_val_item, topk_idx_item = torch.topk(R_integrated, self.kb, dim=0)
        R_balanced_item = torch.zeros_like(R_integrated)
        R_balanced_item.scatter_(0, topk_idx_item, 1.0)

        R_balanced = torch.clamp(R_balanced_user + R_balanced_item, 0, 1)
        return R_balanced

    def create_laplacian_matrix(self, R):
        n_users, n_items = R.shape
        R_cpu = R.cpu()
        A_cpu = torch.zeros(n_users + n_items, n_users + n_items)
        A_cpu[:n_users, n_users:] = R_cpu
        A_cpu[n_users:, :n_users] = R_cpu.t()
        degree = torch.sum(A_cpu, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        L_cpu = torch.mm(torch.mm(d_mat_inv_sqrt, A_cpu), d_mat_inv_sqrt)
        return L_cpu.to_sparse().to(R.device)

    def set_balanced_graph(self, R_counterfactual):
        R_balanced = self.build_balanced_graph(R_counterfactual)
        self.current_graph = self.create_laplacian_matrix(R_balanced)
        self.use_counterfactual = True

    def disable_counterfactual(self):
        self.use_counterfactual = False

    def enable_counterfactual(self):
        self.use_counterfactual = True

    # ==================== Loss Functions ====================

    def info_nce_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        batch_size = view1.size(0)
        logits = torch.mm(view1, view2.t()) / self.temperature
        labels = torch.arange(batch_size, device=view1.device)
        return F.cross_entropy(logits, labels)

    def orthogonal_loss(self, h_rel, h_irr):
        h_rel_norm = F.normalize(h_rel, p=2, dim=1)
        h_irr_norm = F.normalize(h_irr, p=2, dim=1)
        cosine_sim = torch.sum(h_rel_norm * h_irr_norm, dim=1)
        return torch.mean(torch.abs(cosine_sim))

    def preference_relevance_loss(self, user_emb, item_rel, item_irr):
        pos_score = torch.sum(user_emb * item_rel, dim=1)
        neg_score = torch.sum(user_emb * item_irr, dim=1)
        return -torch.mean(F.logsigmoid(pos_score - neg_score))

    def bpr_loss_fn(self, users, pos_items, neg_items, h_user, h_item):
        user_emb = h_user[users]
        pos_emb = h_item[pos_items]
        neg_emb = h_item[neg_items]
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    # ==================== Forward Pass ====================

    def forward(self, graph=None, return_embeddings=False):
        if graph is None:
            graph = self.current_graph

        h_user_beh, h_item_beh = self.lightgcn(graph)

        # Parameter-free item-item GCN (progressively refined)
        h_item_id = self.lightgcn.item_embedding.weight
        h_item_multi = torch.mm(self.fused_item_adj, h_item_id)

        h_pref_rel, h_pref_irr = self.gated_mechanism(h_item_multi)

        h_item_final = h_pref_rel + h_item_beh
        h_user_final = h_user_beh

        if return_embeddings:
            return h_user_final, h_item_final

        return h_user_beh, h_item_beh, h_pref_rel, h_pref_irr, h_user_final, h_item_final

    # ==================== Framework Interface ====================

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        h_user_beh, h_item_beh, h_pref_rel, h_pref_irr, h_user_final, h_item_final = self.forward()

        loss_bpr = self.bpr_loss_fn(users, pos_items, neg_items, h_user_beh, h_item_final)

        batch_user = h_user_beh[users]
        batch_rel = h_pref_rel[pos_items]
        batch_irr = h_pref_irr[pos_items]

        # InfoNCE: sample items for efficiency
        n_samples = min(2048, self.n_items)
        unique_items = torch.unique(pos_items)
        if len(unique_items) < n_samples:
            n_random = n_samples - len(unique_items)
            random_items = torch.randint(0, self.n_items, (n_random,), device=pos_items.device)
            sampled_items = torch.cat([unique_items, random_items])
        else:
            sampled_items = unique_items[:n_samples]
        loss_infonce = self.info_nce_loss(h_pref_rel[sampled_items], h_item_beh[sampled_items])

        loss_ortho = self.orthogonal_loss(h_pref_rel, h_pref_irr)
        loss_pref = self.preference_relevance_loss(batch_user, batch_rel, batch_irr)

        # L2 on batch embeddings only
        u_ego = self.lightgcn.user_embedding.weight[users]
        pos_ego = self.lightgcn.item_embedding.weight[pos_items]
        neg_ego = self.lightgcn.item_embedding.weight[neg_items]
        l2_reg = (1./2 * (u_ego**2).sum() + 1./2 * (pos_ego**2).sum() + 1./2 * (neg_ego**2).sum()) / users.shape[0]

        total_loss = (loss_bpr +
                      self.lambda_infonce * loss_infonce +
                      self.lambda_ortho * loss_ortho +
                      self.lambda_pref * loss_pref +
                      self.lambda_l2 * l2_reg)
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        h_user_final, h_item_final = self.forward(return_embeddings=True)
        u_embeddings = h_user_final[user]
        return torch.matmul(u_embeddings, h_item_final.transpose(0, 1))

    def get_embeddings(self):
        with torch.no_grad():
            h_user, h_item = self.forward(return_embeddings=True)
        return h_user, h_item
