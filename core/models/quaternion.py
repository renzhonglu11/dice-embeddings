from .base_model import *
from .static_funcs import quaternion_mul


def quaternion_mul_with_unit_norm(*, Q_1, Q_2):
    a_h, b_h, c_h, d_h = Q_1  # = {a_h + b_h i + c_h j + d_h k : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}

    # Normalize the relation to eliminate the scaling effect
    denominator = torch.sqrt(a_r ** 2 + b_r ** 2 + c_r ** 2 + d_r ** 2)
    p = a_r / denominator
    q = b_r / denominator
    u = c_r / denominator
    v = d_r / denominator
    #  Q'=E Hamilton product R
    r_val = a_h * p - b_h * q - c_h * u - d_h * v
    i_val = a_h * q + b_h * p + c_h * v - d_h * u
    j_val = a_h * u - b_h * v + c_h * p + d_h * q
    k_val = a_h * v + b_h * u - c_h * q + d_h * p
    return r_val, i_val, j_val, k_val


class QMult(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'QMult'

    def forward_triples(self, indexed_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(indexed_triple)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail_ent_emb, 4)

        # (2)
        # (2.1) Apply BN + Dropout on (1.2)-relations.
        # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.sum(r_val * emb_tail_real, dim=1)
        i_score = torch.sum(i_val * emb_tail_i, dim=1)
        j_score = torch.sum(j_val * emb_tail_j, dim=1)
        k_score = torch.sum(k_val * emb_tail_k, dim=1)
        return real_score + i_score + j_score + k_score

    def forward_k_vs_all(self, x):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(self.entity_embeddings.weight, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = emb_tail_real.transpose(1, 0), emb_tail_i.transpose(1,
                                                                                                                0), emb_tail_j.transpose(
            1, 0), emb_tail_k.transpose(1, 0)

        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.mm(r_val, emb_tail_real)
        i_score = torch.mm(i_val, emb_tail_i)
        j_score = torch.mm(j_val, emb_tail_j)
        k_score = torch.mm(k_val, emb_tail_k)

        return real_score + i_score + j_score + k_score

    def forward_k_vs_sample(self, x, target_entity_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        # (batch size, num. selected entity, dimension)
        tail_entity_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(target_entity_idx))
        # quaternion vectors
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.tensor_split(tail_entity_emb, 4, dim=2)

        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)
        emb_tail_j = emb_tail_j.transpose(1, 2)
        emb_tail_k = emb_tail_k.transpose(1, 2)

        # (batch size, 1, dimension)
        r_val = r_val.unsqueeze(1)
        i_val = i_val.unsqueeze(1)
        j_val = j_val.unsqueeze(1)
        k_val = k_val.unsqueeze(1)

        real_score = torch.bmm(r_val, emb_tail_real)
        i_score = torch.bmm(i_val, emb_tail_i)
        j_score = torch.bmm(j_val, emb_tail_j)
        k_score = torch.bmm(k_val, emb_tail_k)

        return (real_score + i_score + j_score + k_score).squeeze(1)


class oldQMult(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'QMult'
        # Temporary solutions: We need to find a day to
        del self.entity_embeddings
        del self.relation_embeddings
        del self.normalize_head_entity_embeddings
        del self.normalize_relation_embeddings
        del self.normalize_tail_entity_embeddings
        del self.input_dp_ent_real
        del self.input_dp_rel_real
        del self.hidden_dropout

        # Quaternion embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_ent_j.weight.data), xavier_normal_(self.emb_ent_k.weight.data)

        # Quaternion embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)
        xavier_normal_(self.emb_rel_j.weight.data), xavier_normal_(self.emb_rel_k.weight.data)

        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_j = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_k = torch.nn.Dropout(self.input_dropout_rate)
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_j = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_k = torch.nn.Dropout(self.input_dropout_rate)
        # Dropouts for quaternion embeddings obtained from quaternion multiplication.
        self.hidden_dp_real = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_i = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_j = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_k = torch.nn.Dropout(self.hidden_dropout_rate)

        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = self.normalizer_class(self.embedding_dim)
        self.bn_ent_i = self.normalizer_class(self.embedding_dim)
        self.bn_ent_j = self.normalizer_class(self.embedding_dim)
        self.bn_ent_k = self.normalizer_class(self.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = self.normalizer_class(self.embedding_dim)
        self.bn_rel_i = self.normalizer_class(self.embedding_dim)
        self.bn_rel_j = self.normalizer_class(self.embedding_dim)
        self.bn_rel_k = self.normalizer_class(self.embedding_dim)

        # Batch normalization for quaternion embeddings of relations.
        self.bn_hidden_real = self.normalizer_class(self.embedding_dim)
        self.bn_hidden_i = self.normalizer_class(self.embedding_dim)
        self.bn_hidden_j = self.normalizer_class(self.embedding_dim)
        self.bn_hidden_k = self.normalizer_class(self.embedding_dim)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data, self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data, self.emb_rel_k.weight.data), 1)
        return entity_emb.data.detach(), rel_emb.data.detach()

    def forward_k_vs_all(self, x):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)

        if self.apply_unit_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (3) Inner product of (2) with all entities.
            real_score = torch.mm(r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(j_val, self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(k_val, self.emb_ent_k.weight.transpose(1, 0))
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(emb_head_real)),
                     self.input_dp_ent_i(self.bn_ent_i(emb_head_i)),
                     self.input_dp_ent_j(self.bn_ent_j(emb_head_j)),
                     self.input_dp_ent_k(self.bn_ent_k(emb_head_k))),
                Q_2=(
                    self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                    self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                    self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                    self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))

            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Inner product
            real_score = torch.mm(self.hidden_dp_real(self.bn_hidden_real(r_val)),
                                  self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(self.bn_hidden_i(i_val)), self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(self.hidden_dp_j(self.bn_hidden_j(j_val)), self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(self.hidden_dp_k(self.bn_hidden_k(k_val)), self.emb_ent_k.weight.transpose(1, 0))

        return real_score + i_score + j_score + k_score

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)
        # (1.3) Quaternion embeddings of relations
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        emb_tail_j = self.emb_ent_j(e2_idx)
        emb_tail_k = self.emb_ent_k(e2_idx)

        if self.apply_unit_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (3) Inner product of (2) with tail entities.
            real_score = torch.sum(self.hidden_dp_real(r_val) * emb_tail_real, dim=1)
            i_score = torch.sum(self.hidden_dp_i(i_val) * emb_tail_i, dim=1)
            j_score = torch.sum(self.hidden_dp_j(j_val) * emb_tail_j, dim=1)
            k_score = torch.sum(self.hidden_dp_k(k_val) * emb_tail_k, dim=1)
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(emb_head_real)),
                     self.input_dp_ent_i(self.bn_ent_i(emb_head_i)),
                     self.input_dp_ent_j(self.bn_ent_j(emb_head_j)),
                     self.input_dp_ent_k(self.bn_ent_k(emb_head_k))),
                Q_2=(
                    self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                    self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                    self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                    self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))

            emb_tail_real = self.emb_ent_real(e2_idx)
            emb_tail_i = self.emb_ent_i(e2_idx)
            emb_tail_j = self.emb_ent_j(e2_idx)
            emb_tail_k = self.emb_ent_k(e2_idx)

            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Inner product
            real_score = torch.sum(self.hidden_dp_real(self.bn_hidden_real(r_val)) * emb_tail_real, dim=1)
            i_score = torch.sum(self.hidden_dp_i(self.bn_hidden_i(i_val)) * emb_tail_i, dim=1)
            j_score = torch.sum(self.hidden_dp_j(self.bn_hidden_j(j_val)) * emb_tail_j, dim=1)
            k_score = torch.sum(self.hidden_dp_k(self.bn_hidden_k(k_val)) * emb_tail_k, dim=1)

        return real_score + i_score + j_score + k_score

    def forward_without_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        h,r,t:=x
        QMult(h,r,t)-> score computed ithout dropout and batch norm.
        :param x:
        :return:
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)
        # (1.3) Quaternion embeddings of relations
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        emb_tail_j = self.emb_ent_j(e2_idx)
        emb_tail_k = self.emb_ent_k(e2_idx)
        # (2)
        # (2.1) Apply BN + Dropout on (1.2)-relations.
        # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.sum(r_val * emb_tail_real, dim=1)
        i_score = torch.sum(i_val * emb_tail_i, dim=1)
        j_score = torch.sum(j_val * emb_tail_j, dim=1)
        k_score = torch.sum(k_val * emb_tail_k, dim=1)

        return real_score + i_score + j_score + k_score


class ConvQ(BaseKGE):
    """ Convolutional Quaternion Knowledge Graph Embeddings

    @TODO: Implement ConvQ via integration residual connection with addition to distributed the gradients.

    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'ConvQ'

        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels  # 8 because of 8 real values in 2 quaternions
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = self.normalizer_class(self.embedding_dim)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, Q_1, Q_2):
        emb_ent_real, emb_ent_imag_i, emb_ent_imag_j, emb_ent_imag_k = Q_1
        emb_rel_real, emb_rel_imag_i, emb_rel_imag_j, emb_rel_imag_k = Q_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_j.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_k.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_j.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_k.view(-1, 1, 1, self.embedding_dim // 4)], 2)

        # n, c_in, h_in, w_in x.shape before conv. h_in=8, w_in embeddings
        x = self.conv2d(x)
        # n, c_out, h_out, w_out x.shape after conv.
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc1(x)))
        return torch.chunk(x, 4, dim=1)

    def forward_triples(self, indexed_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(indexed_triple)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail_ent_emb, 4)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3
        # (3)
        # (3.1) Apply BN + Dropout on (1.2).
        # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (4)
        # (4.1) Hadamard product of (2) with (3).
        # (4.2) Dropout on (4.1).
        # (4.3) Inner product
        real_score = torch.sum(conv_real * r_val * emb_tail_real, dim=1)
        i_score = torch.sum(conv_imag_i * i_val * emb_tail_i, dim=1)
        j_score = torch.sum(conv_imag_j * j_val * emb_tail_j, dim=1)
        k_score = torch.sum(conv_imag_k * k_val * emb_tail_k, dim=1)
        return real_score + i_score + j_score + k_score

    def forward_k_vs_all(self, x: torch.Tensor):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3

        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(self.entity_embeddings.weight, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = emb_tail_real.transpose(1, 0), emb_tail_i.transpose(1,
                                                                                                                0), emb_tail_j.transpose(
            1, 0), emb_tail_k.transpose(1, 0)

        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.mm(conv_real * r_val, emb_tail_real)
        i_score = torch.mm(conv_imag_i * i_val, emb_tail_i)
        j_score = torch.mm(conv_imag_j * j_val, emb_tail_j)
        k_score = torch.mm(conv_imag_k * k_val, emb_tail_k)

        return real_score + i_score + j_score + k_score


# TODO: Remove these classes
class oldQMult(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'QMult'
        # Quaternion embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_ent_j.weight.data), xavier_normal_(self.emb_ent_k.weight.data)

        # Quaternion embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)
        xavier_normal_(self.emb_rel_j.weight.data), xavier_normal_(self.emb_rel_k.weight.data)

        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_j = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_k = torch.nn.Dropout(self.input_dropout_rate)
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_j = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_k = torch.nn.Dropout(self.input_dropout_rate)
        # Dropouts for quaternion embeddings obtained from quaternion multiplication.
        self.hidden_dp_real = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_i = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_j = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_k = torch.nn.Dropout(self.hidden_dropout_rate)

        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = self.normalizer_class(self.embedding_dim)
        self.bn_ent_i = self.normalizer_class(self.embedding_dim)
        self.bn_ent_j = self.normalizer_class(self.embedding_dim)
        self.bn_ent_k = self.normalizer_class(self.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = self.normalizer_class(self.embedding_dim)
        self.bn_rel_i = self.normalizer_class(self.embedding_dim)
        self.bn_rel_j = self.normalizer_class(self.embedding_dim)
        self.bn_rel_k = self.normalizer_class(self.embedding_dim)

        # Batch normalization for quaternion embeddings of relations.
        self.bn_hidden_real = self.normalizer_class(self.embedding_dim)
        self.bn_hidden_i = self.normalizer_class(self.embedding_dim)
        self.bn_hidden_j = self.normalizer_class(self.embedding_dim)
        self.bn_hidden_k = self.normalizer_class(self.embedding_dim)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data, self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data, self.emb_rel_k.weight.data), 1)
        return entity_emb.data.detach(), rel_emb.data.detach()

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)

        if self.apply_unit_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (3) Inner product of (2) with all entities.
            real_score = torch.mm(r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(j_val, self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(k_val, self.emb_ent_k.weight.transpose(1, 0))
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(emb_head_real)),
                     self.input_dp_ent_i(self.bn_ent_i(emb_head_i)),
                     self.input_dp_ent_j(self.bn_ent_j(emb_head_j)),
                     self.input_dp_ent_k(self.bn_ent_k(emb_head_k))),
                Q_2=(
                    self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                    self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                    self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                    self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))

            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Inner product
            real_score = torch.mm(self.hidden_dp_real(self.bn_hidden_real(r_val)),
                                  self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(self.bn_hidden_i(i_val)), self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(self.hidden_dp_j(self.bn_hidden_j(j_val)), self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(self.hidden_dp_k(self.bn_hidden_k(k_val)), self.emb_ent_k.weight.transpose(1, 0))

        return real_score + i_score + j_score + k_score

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)
        # (1.3) Quaternion embeddings of relations
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        emb_tail_j = self.emb_ent_j(e2_idx)
        emb_tail_k = self.emb_ent_k(e2_idx)

        if self.apply_unit_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (3) Inner product of (2) with tail entities.
            real_score = torch.sum(self.hidden_dp_real(r_val) * emb_tail_real, dim=1)
            i_score = torch.sum(self.hidden_dp_i(i_val) * emb_tail_i, dim=1)
            j_score = torch.sum(self.hidden_dp_j(j_val) * emb_tail_j, dim=1)
            k_score = torch.sum(self.hidden_dp_k(k_val) * emb_tail_k, dim=1)
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(emb_head_real)),
                     self.input_dp_ent_i(self.bn_ent_i(emb_head_i)),
                     self.input_dp_ent_j(self.bn_ent_j(emb_head_j)),
                     self.input_dp_ent_k(self.bn_ent_k(emb_head_k))),
                Q_2=(
                    self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                    self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                    self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                    self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))

            emb_tail_real = self.emb_ent_real(e2_idx)
            emb_tail_i = self.emb_ent_i(e2_idx)
            emb_tail_j = self.emb_ent_j(e2_idx)
            emb_tail_k = self.emb_ent_k(e2_idx)

            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Inner product
            real_score = torch.sum(self.hidden_dp_real(self.bn_hidden_real(r_val)) * emb_tail_real, dim=1)
            i_score = torch.sum(self.hidden_dp_i(self.bn_hidden_i(i_val)) * emb_tail_i, dim=1)
            j_score = torch.sum(self.hidden_dp_j(self.bn_hidden_j(j_val)) * emb_tail_j, dim=1)
            k_score = torch.sum(self.hidden_dp_k(self.bn_hidden_k(k_val)) * emb_tail_k, dim=1)

        return real_score + i_score + j_score + k_score

    def forward_triples_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        h,r,t:=x
        QMult(h,r,t)-> score computed ithout dropout and batch norm.
        :param x:
        :return:
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)
        # (1.3) Quaternion embeddings of relations
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        emb_tail_j = self.emb_ent_j(e2_idx)
        emb_tail_k = self.emb_ent_k(e2_idx)
        # (2)
        # (2.1) Apply BN + Dropout on (1.2)-relations.
        # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.sum(r_val * emb_tail_real, dim=1)
        i_score = torch.sum(i_val * emb_tail_i, dim=1)
        j_score = torch.sum(j_val * emb_tail_j, dim=1)
        k_score = torch.sum(k_val * emb_tail_k, dim=1)

        return real_score + i_score + j_score + k_score

    def forward_k_vs_all_base(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)

        # (2)
        # (2.1) Apply BN + Dropout on (1.2)-relations.
        # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.mm(r_val,
                              self.emb_ent_real.weight.transpose(1, 0))
        i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
        j_score = torch.mm(j_val, self.emb_ent_j.weight.transpose(1, 0))
        k_score = torch.mm(k_val, self.emb_ent_k.weight.transpose(1, 0))

        return real_score + i_score + j_score + k_score


class QMultwoBNDP(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'QMult'
        # Quaternion embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_ent_j.weight.data), xavier_normal_(self.emb_ent_k.weight.data)

        # Quaternion embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)
        xavier_normal_(self.emb_rel_j.weight.data), xavier_normal_(self.emb_rel_k.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data, self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data, self.emb_rel_k.weight.data), 1)
        return entity_emb.data.detach(), rel_emb.data.detach()

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        """
        h,r,t:=x
        QMult(h,r,t)-> score computed ithout dropout and batch norm.
        :param x:
        :return:
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)
        # (1.3) Quaternion embeddings of relations
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        emb_tail_j = self.emb_ent_j(e2_idx)
        emb_tail_k = self.emb_ent_k(e2_idx)
        # (2)
        # (2.1) Apply BN + Dropout on (1.2)-relations.
        # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.sum(r_val * emb_tail_real, dim=1)
        i_score = torch.sum(i_val * emb_tail_i, dim=1)
        j_score = torch.sum(j_val * emb_tail_j, dim=1)
        k_score = torch.sum(k_val * emb_tail_k, dim=1)

        return real_score + i_score + j_score + k_score

    def forward_k_vs_all(self, x):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)

        # (2)
        # (2.1) Apply BN + Dropout on (1.2)-relations.
        # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        # (3)
        # (3.1) Dropout on (2)-result of quaternion multiplication.
        # (3.2) Inner product
        real_score = torch.mm(r_val,
                              self.emb_ent_real.weight.transpose(1, 0))
        i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
        j_score = torch.mm(j_val, self.emb_ent_j.weight.transpose(1, 0))
        k_score = torch.mm(k_val, self.emb_ent_k.weight.transpose(1, 0))

        return real_score + i_score + j_score + k_score
