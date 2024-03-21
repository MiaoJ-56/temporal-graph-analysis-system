import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from algorithm.models.DyGFormer import DyGFormer
from algorithm.models.modules import MergeLayer
from algorithm.utils.utils import get_neighbor_sampler
from algorithm.utils.EarlyStopping import EarlyStopping

class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def get_link_prediction_data(dataset_name: str):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data
    """
    # Load data and train val test split
    graph_df = pd.read_csv("./uci/ml_{}.csv".format(dataset_name))
    edge_raw_features = np.load('./uci/ml_{}.npy'.format(dataset_name))
    node_raw_features = np.load('./uci/ml_{}_node.npy'.format(dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    return node_raw_features, edge_raw_features, full_data


# get data for training, validation and testing
node_raw_features, edge_raw_features, full_data = get_link_prediction_data(dataset_name='uci')

# initialize validation and test neighbor sampler to retrieve temporal graph
full_neighbor_sampler = get_neighbor_sampler(data=full_data, seed=1)
dynamic_backbone = DyGFormer(
    node_raw_features=node_raw_features, 
    edge_raw_features=edge_raw_features, 
    neighbor_sampler=full_neighbor_sampler,
    time_feat_dim=100, 
    channel_embedding_dim=50, 
    patch_size=1,
    num_layers=2, 
    num_heads=2, 
    dropout=0.1,
    max_input_sequence_length=32, 
    device='cpu')

link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                            hidden_dim=node_raw_features.shape[1], output_dim=1)
model = nn.Sequential(dynamic_backbone, link_predictor)

model_name = "DyGFormer"
seed = 0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# load the saved model
load_model_name = f'{model_name}_seed{seed}'
load_model_folder = f"./uci/{model_name}_seed{seed}/"
early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                save_model_name=load_model_name, logger=logger, model_name=model_name)
early_stopping.load_checkpoint(model, map_location='cpu')

model[0].set_neighbor_sampler(full_neighbor_sampler)
model.eval()
batch_src_node_ids = np.array([1], dtype=np.int64)
batch_dst_node_ids = np.array([2], dtype=np.int64)
batch_node_interact_times = np.array([1])
batch_src_node_embeddings, batch_dst_node_embeddings = \
    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids, node_interact_times=batch_node_interact_times)

probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()

print(probabilities)


def get_probabilities(src_node, dst_node, timestampe):
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data = get_link_prediction_data(dataset_name='uci')

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, seed=1)
    dynamic_backbone = DyGFormer(
        node_raw_features=node_raw_features, 
        edge_raw_features=edge_raw_features, 
        neighbor_sampler=full_neighbor_sampler,
        time_feat_dim=100, 
        channel_embedding_dim=50, 
        patch_size=1,
        num_layers=2, 
        num_heads=2, 
        dropout=0.1,
        max_input_sequence_length=32, 
        device='cpu')

    link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                hidden_dim=node_raw_features.shape[1], output_dim=1)
    model = nn.Sequential(dynamic_backbone, link_predictor)

    model_name = "DyGFormer"
    seed = 0
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # load the saved model
    load_model_name = f'{model_name}_seed{seed}'
    load_model_folder = f"./uci/{model_name}_seed{seed}/"
    early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                    save_model_name=load_model_name, logger=logger, model_name=model_name)
    early_stopping.load_checkpoint(model, map_location='cpu')

    model[0].set_neighbor_sampler(full_neighbor_sampler)
    model.eval()
    batch_src_node_ids = np.array([src_node], dtype=np.int64)
    batch_dst_node_ids = np.array([dst_node], dtype=np.int64)
    batch_node_interact_times = np.array([timestampe])
    batch_src_node_embeddings, batch_dst_node_embeddings = \
        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids, node_interact_times=batch_node_interact_times)

    probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()

    return batch_src_node_embeddings, batch_dst_node_embeddings, probabilities