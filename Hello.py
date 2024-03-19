# # Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import streamlit as st
# from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)


# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#     )

#     st.write("# Welcome to Streamlit! 👋")

#     st.sidebar.success("Select a demo above.")

#     st.markdown(
#         """
#         Streamlit is an open-source app framework built specifically for
#         Machine Learning and Data Science projects.
#         **👈 Select a demo from the sidebar** to see some examples
#         of what Streamlit can do!
#         ### Want to learn more?
#         - Check out [streamlit.io](https://streamlit.io)
#         - Jump into our [documentation](https://docs.streamlit.io)
#         - Ask a question in our [community
#           forums](https://discuss.streamlit.io)
#         ### See more complex demos
#         - Use a neural net to [analyze the Udacity Self-driving Car Image
#           Dataset](https://github.com/streamlit/demo-self-driving)
#         - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
#     """
#     )


# if __name__ == "__main__":
#     run()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
df_cluster = None
data_file = None
n = 1

def input_file(data_file,n):
    data_file = st.file_uploader('Thêm dữ liệu',type=['csv'])
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.dataframe(df)
        selected_columns = st.multiselect('Lựa chọn dữ liệu phân cụm', df.columns.to_list())
        if selected_columns:
            df_cluster = pd.DataFrame(df[selected_columns])
            st.dataframe(df_cluster)
            Elbow(df_cluster)
            n = int(st.number_input('Nhập số cụm', min_value=2, key=int))
            runKmean(df_cluster,n)
        else:
            st.write('No columns selected')

def runKmean(df_cluster,n):
    if df_cluster is not None:
        kmeans = KMeans(n_clusters=n, init= 'k-means++',
                max_iter= 300, n_init=10)
        clusters = kmeans.fit_predict(df_cluster)
        # Tạo biểu đồ phân tán với các điểm dữ liệu được tô màu theo cụm
        plt.figure(figsize=(10, 6))
        plt.scatter(df_cluster.iloc[:, 0], df_cluster.iloc[:, 1], c=clusters, cmap='viridis', marker='o')
        # Đánh dấu tâm cụm
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot()

def runDbScan(df_cluster):
    return df_cluster

def Elbow(df_cluster):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df_cluster)
        wcss.append(kmeans.inertia_)
    # Plot the elbow graph
    st.set_option('deprecation.showPyplotGlobalUse', False) # To avoid deprecation warning
    plt.figure(figsize=(10,5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot() 

def run():
    st.set_page_config(
        page_title='Demo Sản Phẩm',
        page_icon='💻',
    )

    st.title('Nghiên cứu khai phá dữ liệu và ứng dụng trong phân loại kết quả học tập của sinh viên')
    input_file(data_file,n)
    
    with st.sidebar:
        st.title('Menu')
        radio_selection = st.radio('Lựa chọn thuật toán', ['K-MEANS','DBSCAN'])
        if radio_selection == 'K-MEANS':
            runKmean(df_cluster,n)
        else:
            runDbScan(df_cluster) 
        
# running main func
if __name__ == '__main__':
    run()   
