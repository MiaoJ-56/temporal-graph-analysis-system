# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import torch
from model_test import get_probabilities

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="首页",
        page_icon="👋",
    )
    st.write("# 欢迎来的时序图分析系统! 👋")
    
    uploaded_file = st.file_uploader("上传数据集")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        dataframe[0:10]

    dataframe =pd.read_csv("./uci/ml_uci.csv", index_col=0)
    if dataframe is None:
        "defeat"
    st.write("# 数据集展示")
    st.write(dataframe.head(5))
    "# 统计数据"
    edge_raw_features = np.load("./uci/ml_uci.npy")
    node_raw_features = np.load('./uci/ml_uci_node.npy')
    info = {'dataset_name': 'uci',
            'num_nodes': node_raw_features.shape[0] - 1,
            'node_feat_dim': node_raw_features.shape[-1],
            'num_edges': edge_raw_features.shape[0] - 1,
            'edge_feat_dim': edge_raw_features.shape[-1]}
    info
    # 节点集合
    src_nodes = pd.unique(dataframe.u)
    dst_nodes = pd.unique(dataframe.i)
    # 时间
    ts = pd.unique(dataframe.ts)
    # ts = pd.to_datetime(dataframe.ts)
    # ts[0:5]
    "# 边预测任务"
    src_node = st.selectbox(
        '请选择预测的源节点?',
        (u for u in src_nodes)
    )
    dst_node = st.selectbox(
        '请选择预测的目的节点?',
        (v for v in dst_nodes)
    )
    predict_ts = st.selectbox(
        '请选择预测的时间基准点?',
        (t for t in ts))
    number = st.number_input('请输入基于时间基准点的偏移时间(毫秒)', step=1)

    
    st.write('您选择的源节点、目的节点分别是:', src_node,  dst_node)
    st.write('您选择的预测时间基准点是:', predict_ts)
    st.write('您设置的偏移时间是:', number,'最终预测时间为：',predict_ts+number)
    # 开始预测
    st.write("显示一个正样例:", dataframe[40000:40001])
    # dataframe[dataframe.ts==3636687]
    src_node_embeddings, dst_node_embeddings, probabilities = get_probabilities(src_node=src_node, dst_node=dst_node, timestampe=predict_ts+number)

    st.write("预测结果", format(probabilities.item(),'.3%'))
    # st.write("预测向量分别是", src_node_embeddings, dst_node_embeddings)
    
    # st.markdown(
    #     """
    #     数据、抽象、分析、建模、预测、下一步？

    #     **👈 快来试一试吧** 
        
    #     ### Who am I?
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into our [documentation](https://docs.streamlit.io)
    #     - Ask a question in our [community
    #       forums](https://discuss.streamlit.io)
    #     ### See more interesting things
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #       Dataset](https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    # """
    # )


if __name__ == "__main__":
    run()
