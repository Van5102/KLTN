import io
import streamlit as st
import pandas as pd
import numpy as np
import chardet
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
df_cluster = None
data_file = None
df = None
radio_selection = None
n = 1

def read_csv_with_file_uploader():
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # To detect file encoding
        encoding = chardet.detect(bytes_data)['encoding']
        # To convert to a string based IO:
        stringio = io.StringIO(bytes_data.decode(encoding))
        # To read file as string (assuming it's a CSV file):
        df = pd.read_csv(stringio)
        return df

def input_file(data_file, n, radio_selection, df_cluster):
    data_file = read_csv_with_file_uploader()
    if data_file is not None:
        df = data_file.dropna()
        st.dataframe(df)
        selected_columns = st.multiselect('Lựa chọn dữ liệu phân cụm', df.columns.to_list())
        if selected_columns:
            df_cluster = pd.DataFrame(df[selected_columns])
            st.dataframe(df_cluster)
            Elbow(df_cluster)
            if radio_selection == 'K-MEANS':
                n = int(st.number_input('Nhập số cụm', min_value=2, key=int))
                df_cluster = runKmean(df_cluster, n)
            else:
                df_cluster = runDbScan(df_cluster)
    return df_cluster

def export_clustered_data():
    global df_cluster

    if df_cluster is not None:
        data = df_cluster.sort_values('Cluster')
        output_filename = 'clustered_data.csv'
        data_csv = data.to_csv(index=False)
        if data_csv:
            st.download_button(label='TẢI VỀ KẾT QUẢ PHÂN CỤM',
                               data=data_csv,
                               file_name=output_filename)
        else:
            st.write('No data to export.')

def runKmean(df_cluster, n):
    if df_cluster is not None:
        kmeans = KMeans(
            n_clusters=n, init='k-means++', max_iter=300, n_init=10
        )
        clusters = kmeans.fit_predict(df_cluster)
        df_cluster['Cluster'] = kmeans.labels_

        # Tạo biểu đồ phân tán với các điểm dữ liệu được tô màu theo cụm
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_cluster.iloc[:, 0],
            df_cluster.iloc[:, 1],
            c=clusters,
            cmap='viridis',
            marker='o'
        )
        # Đánh dấu tâm cụm
        centers = kmeans.cluster_centers_
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c='red',
            s=200,
            alpha=0.75,
            marker='x'
        )
        plt.title('KMEANS Clustering')
        plt.xlabel('Spending Score')
        plt.ylabel('Annual Income')
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot()
    return df_cluster

def runDbScan(df_cluster):
    eps = st.slider('Chọn giá trị eps', min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    min_samples = st.slider('Chọn giá trị min_samples', min_value=1, max_value=200, value=5, step=1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_cluster)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df_cluster.iloc[:, 0],
        df_cluster.iloc[:, 1],
        c=clusters,
        cmap='viridis',
        marker='o'
    )
    n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise_ = list(clusters).count(-1)
    st.write('Số lượng cụm:', n_clusters_)
    st.write('Số lượng điểm nhiễu:', n_noise_)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot()
    return df_cluster

def Elbow(df_cluster):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df_cluster)
        wcss.append(kmeans.inertia_)
    # Plot the elbow graph
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To avoid deprecation warning
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot()

def run():
    global df_cluster
    st.set_page_config(
        page_title='Demo Sản Phẩm',
        page_icon='💻',
    )

    with st.sidebar:
        st.title('Menu')
        radio_selection = st.radio('Lựa chọn thuật toán', ['K-MEANS', 'DBSCAN'])
    st.title('Nghiên cứu khai phá dữ liệu và ứng dụng trong phân loại kết quả học tập của sinh viên')
    df_cluster = input_file(data_file, n, radio_selection, df_cluster)
    export_clustered_data()

# running main func
if __name__ == '__main__':
    run()