import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

"""
    extract features using adjacent vector
"""


class DyGFormer2(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2,
                 num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(DyGFormer2, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)


        self.size = 64
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim,
                              bias=True),
            'edge': nn.Linear(in_features=self.size * self.edge_feat_dim, out_features=self.channel_embedding_dim,
                              bias=True),
            'location': nn.Linear(in_features=self.size, out_features=self.channel_embedding_dim,
                              bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim,
                              bias=True)
        })

        self.num_channels = 4

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim,
                                      out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """

        # 首先只进行空间信息提取，将原来的one-hop修改为adjacent matrix, 提取空间信息
        # 输入 是一个batch src_ids, tss;
        # 输出 一个

        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids,
                                                              node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids,
                                                              node_interact_times=node_interact_times)


        # src_fixed_nodes_neighbor_ids 得到 (batch_size, 64)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_fixed_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_fixed_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_fixed_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_fixed_nodes_neighbor_ids, src_fixed_nodes_edge_ids, src_fixed_nodes_neighbor_times = \
            self.fix_sequences(node_ids=src_node_ids,
                               node_interact_times=node_interact_times, # 没有用到
                               nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list,
                               nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               size=64)

        # dst_fixed_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_fixed_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_fixed_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_fixed_nodes_neighbor_ids, dst_fixed_nodes_edge_ids, dst_fixed_nodes_neighbor_times = \
            self.fix_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                               nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list,
                               nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               size=64)

        # 得到 adjacent vector:shape (batch_size * 64), edges:shape (batch_size * 64 * feature_dim)
        # 得到 nodes_neighbor_edge_raw_features

        src_fixed_nodes_edge_raw_features, src_fixed_adjacent_vector = \
            self.get_edges_features_and_vector_matrix(node_interact_times=node_interact_times, fixed_nodes_neighbor_ids=src_fixed_nodes_neighbor_ids,
                                                      fixed_nodes_edge_ids=src_fixed_nodes_edge_ids, fixed_nodes_neighbor_times=src_fixed_nodes_neighbor_times)

        dst_fixed_nodes_edge_raw_features, dst_fixed_adjacent_vector = \
            self.get_edges_features_and_vector_matrix(node_interact_times=node_interact_times,
                                                      fixed_nodes_neighbor_ids=dst_fixed_nodes_neighbor_ids,
                                                      fixed_nodes_edge_ids=dst_fixed_nodes_edge_ids,
                                                      fixed_nodes_neighbor_times=dst_fixed_nodes_neighbor_times)



        # 得到 nodes特征：shape (batch_size,node_dim), 得到time特征
        src_nodes_raw_features, dst_nodes_raw_features, node_interact_times_features = \
            self.get_node_and_time_features(src_nodes_ids= src_node_ids, dst_nodes_ids = dst_node_ids, node_interact_times=node_interact_times, time_encoder=self.time_encoder)

        # logger = logging.getLogger()
        # logger.info(src_nodes_raw_features.shape)
        # logger.info(src_fixed_nodes_edge_raw_features.shape)

        batch_size = len(src_node_ids)

        src_nodes_raw_features = self.projection_layer['node'](src_nodes_raw_features)
        dst_nodes_raw_features = self.projection_layer['node'](dst_nodes_raw_features)

        src_fixed_nodes_edge_raw_features = self.projection_layer['edge'](src_fixed_nodes_edge_raw_features.reshape(batch_size,self.size* self.edge_feat_dim))
        dst_fixed_nodes_edge_raw_features = self.projection_layer['edge'](dst_fixed_nodes_edge_raw_features.reshape(batch_size,self.size* self.edge_feat_dim))

        # logger = logging.getLogger()
        # logger.info(src_fixed_adjacent_vector.reshape(batch_size, self.size).shape)
        # logger.info(src_fixed_adjacent_vector.reshape(batch_size, self.size).dtype)
        # logger.info(src_fixed_nodes_edge_raw_features.dtype)
        # logger.info(src_nodes_raw_features.dtype)

        src_fixed_adjacent_vector = self.projection_layer['location'](src_fixed_adjacent_vector.reshape(batch_size, self.size))
        dst_fixed_adjacent_vector = self.projection_layer['location'](dst_fixed_adjacent_vector.reshape(batch_size, self.size))

        node_interact_times_features = self.projection_layer['time'](node_interact_times_features)



        src_data = [src_nodes_raw_features, src_fixed_nodes_edge_raw_features, src_fixed_adjacent_vector, node_interact_times_features]
        dst_data = [dst_nodes_raw_features, dst_fixed_nodes_edge_raw_features, dst_fixed_adjacent_vector, node_interact_times_features]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        src_data = torch.stack(src_data).reshape(batch_size, self.num_channels*self.channel_embedding_dim)
        dst_data = torch.stack(dst_data).reshape(batch_size, self.num_channels*self.channel_embedding_dim)
        #
        # logger = logging.getLogger()
        # logger.info(src_data.shape)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_data)

        return src_node_embeddings, dst_node_embeddings

    # 将提取的邻居序列调整到指定大小，不够填充零
    def fix_sequences(self,
                      node_ids: np.ndarray,
                      node_interact_times: np.ndarray,
                      nodes_neighbor_ids_list: list,
                      nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list,
                      size: int = 64):
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > size - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(size - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(size - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(size - 1):]
        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        fixed_nodes_neighbor_ids = np.zeros((len(node_ids), size)).astype(np.longlong)
        fixed_nodes_edge_ids = np.zeros((len(node_ids), size)).astype(np.longlong)
        fixed_nodes_neighbor_times = np.zeros((len(node_ids), size)).astype(np.float32)


        for idx in range(len(nodes_neighbor_ids_list)):
            fixed_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            fixed_nodes_edge_ids[idx, -1] = 0
            fixed_nodes_neighbor_times[idx, -1] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                fixed_nodes_neighbor_ids[idx, 0: len(nodes_neighbor_ids_list[idx])] = nodes_neighbor_ids_list[idx]
                fixed_nodes_edge_ids[idx, 0: len(nodes_neighbor_ids_list[idx])] = nodes_edge_ids_list[idx]
                fixed_nodes_neighbor_times[idx, 0: len(nodes_neighbor_ids_list[idx])] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return fixed_nodes_neighbor_ids, fixed_nodes_edge_ids, fixed_nodes_neighbor_times


    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(
                self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features


    def get_node_and_time_features(self, src_nodes_ids: np.ndarray, dst_nodes_ids: np.ndarray, node_interact_times: np.ndarray, time_encoder: TimeEncoder):

        # Tensor, shape (batch_size, node_feat_dim)
        src_nodes_raw_features = self.node_raw_features[torch.from_numpy(src_nodes_ids)]
        dst_nodes_raw_features = self.node_raw_features[torch.from_numpy(dst_nodes_ids)]

        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        nodes_time_features = time_encoder(timestamps=torch.from_numpy(np.diff(node_interact_times, append=node_interact_times[0])).float().to(self.device))

        return src_nodes_raw_features, dst_nodes_raw_features, nodes_time_features

    def get_edges_features_and_vector_matrix(self,
                                               node_interact_times: np.ndarray,
                                               fixed_nodes_neighbor_ids: np.ndarray,
                                               fixed_nodes_edge_ids: np.ndarray,
                                               fixed_nodes_neighbor_times: np.ndarray):

        # Tensor, shape (batch_size, 64, edge_feat_dim)
        fixed_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(fixed_nodes_edge_ids)]
        size = len(fixed_nodes_neighbor_ids[0])
        # Tensor, shape (batch_size, 64)
        fixed_adjacent_vector = np.zeros((len(node_interact_times), size)).astype(np.longlong)
        for idx in range(len(node_interact_times)):
            fixed_adjacent_vector[idx] = node_interact_times[idx] - fixed_nodes_neighbor_times[idx]
        fixed_adjacent_vector = torch.from_numpy(fixed_adjacent_vector).to(self.device).type(torch.float32)
        # three ndarrays with shape (batch_size, max_seq_length)
        return fixed_nodes_edge_raw_features, fixed_adjacent_vector

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor,
                    padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor,
                    # padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
                    patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features,  = [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            # patches_nodes_neighbor_co_occurrence_features.append(
            #     padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size,
                                                                                                      num_patches,
                                                                                                      patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(
            batch_size, num_patches, patch_size * self.time_feat_dim)
        #
        # patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features,
        #                                                             dim=1).reshape(batch_size, num_patches,
        #                                                                            patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim,
                      out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray,
                                dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids,
                                                                              dst_padded_nodes_neighbor_ids):
            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids,
                                                                         return_inverse=True, return_counts=True)
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(
                self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids,
                                                                         return_inverse=True, return_counts=True)
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(
                self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(
                torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(
                lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(
                torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(
            dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = \
        self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(
            0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=1)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output
