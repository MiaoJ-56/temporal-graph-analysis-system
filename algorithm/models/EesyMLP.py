import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder


class EasyMLP(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, time_feat_dim: int, channel_embedding_dim: int, device: str = 'cpu'):
        """
        EasyMLP model
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param device: str, device
        :return:
        """
        super(EasyMLP, self).__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        # self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        # self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim

        self.device = device

        self.time_encoder = InstanceTimeEncoder(time_dim=time_feat_dim)
        self.act = nn.ReLU()
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            # 'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'channel': nn.Linear(in_features=self.channel_embedding_dim, out_features=self.channel_embedding_dim, bias=True)
        })
        self.num_channels = 2

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """

        batch_size = len(src_node_ids)
        # Tensor, shape (batch_size, channel_embedding_dim)
        src_node_features = self.projection_layer['node'](self.node_raw_features[torch.from_numpy(src_node_ids)])
        dst_node_features = self.projection_layer['node'](self.node_raw_features[torch.from_numpy(dst_node_ids)])
        # edge_features = self.projection_layer['edge']()
        encoded_times = self.time_encoder(
            timestamps=torch.from_numpy(np.diff(node_interact_times, prepend=node_interact_times[0])).float().to(self.device))
        time_features = self.projection_layer['time'](encoded_times)

        src_node_features = self.projection_layer['channel'](self.act(src_node_features))
        dst_node_features = self.projection_layer['channel'](self.act(dst_node_features))
        time_features = self.projection_layer['channel'](self.act(time_features))

        src_data = [src_node_features, time_features]
        # Tensor, shape (batch_size, num_channels, channel_embedding_dim)
        src_data = torch.stack(src_data, dim=1)
        # Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_data = src_data.reshape(batch_size, self.num_channels * self.channel_embedding_dim)

        dst_data = [dst_node_features, time_features]
        # Tensor, shape (batch_size, num_channels, channel_embedding_dim)
        dst_data = torch.stack(dst_data, dim=1)
        # Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_data = dst_data.reshape(batch_size, self.num_channels * self.channel_embedding_dim)


        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(self.act(src_data))
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(self.act(dst_data))

        return src_node_embeddings, dst_node_embeddings


class InstanceTimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(InstanceTimeEncoder, self).__init__()

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