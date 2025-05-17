import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import scipy.cluster.hierarchy as sch
import streamlit as st


st.set_page_config(layout='wide')
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Mall Customer Segmentation")

#Load dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
except:
    print(f"Error loading data")
    df = None

if df is not None:
    st.markdown("#### Basic information on the dataset ")

    col1, col2 = st.columns(2)
    with col1:
        st.write('First 5 rows of the dataset:')
        st.write(df.head())

    with col2:
        st.write("Summary statistics:")
        st.write(df.describe())

    # Create subplots
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Distribution of Age',
            'Distribution of Annual Income',
            'Distribution of Spending Score',
            'Distribution of Gender',
        ),
    )

    # 1. Histogram of Age
    fig.add_trace(
        px.histogram(df, x='Age', nbins=20, marginal='rug').data[0],row=1, col=1,)
    # 2. Histogram of Annual Income
    fig.add_trace(
        px.histogram(df, x='Annual Income (k$)', nbins=20, marginal='rug').data[0], row=1,col=2,)
    # 3. Histogram of Spending Score
    fig.add_trace(
        px.histogram(df, x='Spending Score (1-100)', nbins=20, marginal='rug').data[0],row=2, col=1,)
    # 4. Count plot of Gender
    fig.add_trace(
        px.histogram(df, x='Gender').data[0],row=2,col=2,)

    # Update layout for better appearance
    fig.update_layout(
        title_text='Mall Customer Data Analysis',showlegend=False,
    )
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    st.markdown('### Clustering methods')
    option = st.selectbox("Select the clustering method", ('K-means Clustering', 'Hierarchical Clustering'))
    if option != 'None':
        # Selecting features for clustering
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        # Scaling the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if option == 'K-means Clustering':
            st.markdown('### K-means Clustering method')
            # Elbow Method
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
            fig_elbow_kmeans = px.line(x=range(1, 11),y=wcss,title='Elbow Method for K-Means',labels={'x': 'Number of clusters (K)', 'y': 'Within-Cluster Sum of Squares (WCSS)'})
            fig_elbow_kmeans.update_traces(marker=dict(size=8))
            st.plotly_chart(fig_elbow_kmeans, theme='streamlit', use_container_width=True)

            # Silhouette Analysis for K-Means
            silhouette_scores_kmeans = []
            range_n_clusters_kmeans = range(2, 11)
            for n_clusters in range_n_clusters_kmeans:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores_kmeans.append(silhouette_avg)
                st.write(f"For K-Means with n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

            fig_silhouette_kmeans = px.line(
                x=range_n_clusters_kmeans,
                y=silhouette_scores_kmeans,
                title='Silhouette Analysis for K-Means',
                labels={'x': 'Number of clusters (K)', 'y': 'Silhouette Score'}
            )
            fig_silhouette_kmeans.update_traces(marker=dict(size=8))
            fig_silhouette_kmeans.update_layout(xaxis=dict(tickvals=list(range_n_clusters_kmeans)))
            st.plotly_chart(fig_silhouette_kmeans, theme='streamlit', use_container_width=True)

            # Applying K-Means with an assumed optimal K (e.g., 5)
            optimal_k_kmeans = 5

            kmeans = KMeans(n_clusters=optimal_k_kmeans, init='k-means++', random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)

            centroids_scaled = kmeans.cluster_centers_
            centroids = scaler.inverse_transform(centroids_scaled)
            centroids_df = pd.DataFrame(centroids, columns=['Age', 'Spending Score (1-100)'])

            kmeans = KMeans(n_clusters=optimal_k_kmeans, init='k-means++', random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            df['KMeans_Cluster'] = kmeans_labels

            # Visualizing K-Means Clusters
            fig_scatter_kmeans = px.scatter(
                df,
                x='Annual Income (k$)',
                y='Spending Score (1-100)',
                color='KMeans_Cluster',
                title=f'K-Means Clustering (K={optimal_k_kmeans})',
                labels={'KMeans_Cluster': 'K-Means Cluster'},
                color_continuous_scale=px.colors.sequential.Plasma
            )
            fig_scatter_kmeans.add_trace(go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Centroids'
            ))
            st.plotly_chart(fig_scatter_kmeans, theme='streamlit', use_container_width=True)
            st.write("\nCluster Centroids:")
            st.write(centroids_df)

            # Hypothesis Testing for K-Means
            st.markdown("#### Hypothesis Testing for K-Means Clusters")
            st.markdown(
                "We can perform hypothesis tests to check if there are significant differences in 'Annual Income' and 'Spending Score' between the clusters.")

            for i in range(optimal_k_kmeans):
                cluster_data = df[df['KMeans_Cluster'] == i]
                st.write(f"Cluster {i} data:")
                st.write(cluster_data)  # Display cluster data

                # T-test for Annual Income
                t_income, p_income = stats.ttest_ind(
                    cluster_data['Annual Income (k$)'], df['Annual Income (k$)']
                )
                st.write(f"T-test for Annual Income (Cluster {i} vs. Overall):")
                st.write(f"T-statistic: {t_income}, P-value: {p_income}")

                # T-test for Spending Score
                t_spending, p_spending = stats.ttest_ind(
                    cluster_data['Spending Score (1-100)'], df['Spending Score (1-100)']
                )
                st.write(f"T-test for Spending Score (Cluster {i} vs. Overall):")
                st.write(f"T-statistic: {t_spending}, P-value: {p_spending}")

                alpha = 0.05  # Significance level
                if p_income < alpha:
                    st.write(
                        f"Result: For Cluster {i}, the average annual income is significantly different from the overall average. (p-value < {alpha})")
                else:
                    st.write(
                        f"Result: For Cluster {i}, the average annual income is not significantly different from the overall average. (p-value >= {alpha})")

                if p_spending < alpha:
                    st.write(
                        f"Result: For Cluster {i}, the average spending score is significantly different from the overall average. (p-value < {alpha})")
                else:
                    st.write(
                        f"Result: For Cluster {i}, the average spending score is not significantly different from the overall average. (p-value >= {alpha})")

    if option == 'Hierarchical Clustering':
            st.markdown('### Hierarchical Clustering method')
            # Dendrogram
            linked = sch.linkage(X_scaled, method='ward')
            fig_dendrogram_hierarchical = ff.create_dendrogram(
                X_scaled,
                labels=df.index.tolist(),
                linkagefun=lambda x: sch.linkage(x, method='ward')
            )
            fig_dendrogram_hierarchical.update_layout(
                title='Dendrogram for Hierarchical Clustering (Ward Linkage)',
                xaxis_title='Customers',
                yaxis_title='Euclidean distances'
            )
            st.plotly_chart(fig_dendrogram_hierarchical, theme='streamlit', use_container_width=True)

            # Silhouette Analysis for Hierarchical Clustering
            silhouette_scores_hierarchical = []
            range_n_clusters_hierarchical = range(2, 11)
            for n_clusters in range_n_clusters_hierarchical:
                agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                cluster_labels = agg_clustering.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores_hierarchical.append(silhouette_avg)
                st.write(f"For Hierarchical Clustering with n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

            fig_silhouette_hierarchical = px.line(
                x=range_n_clusters_hierarchical,
                y=silhouette_scores_hierarchical,
                title='Silhouette Analysis for Hierarchical Clustering',
                labels={'x': 'Number of clusters', 'y': 'Silhouette Score'}
            )
            fig_silhouette_hierarchical.update_traces(marker=dict(size=8))
            fig_silhouette_hierarchical.update_layout(xaxis=dict(tickvals=list(range_n_clusters_hierarchical)))
            st.plotly_chart(fig_silhouette_hierarchical, theme='streamlit', use_container_width=True)

            # Applying Hierarchical Clustering with an assumed optimal number of clusters (e.g., 5)
            optimal_n_clusters_hierarchical = 5
            agg_clustering = AgglomerativeClustering(n_clusters=optimal_n_clusters_hierarchical, linkage='ward')
            hierarchical_labels = agg_clustering.fit_predict(X_scaled)
            df['Hierarchical_Cluster'] = hierarchical_labels

            # Visualizing Hierarchical Clustering Clusters
            fig_scatter_hierarchical = px.scatter(
                df,
                x='Annual Income (k$)',
                y='Spending Score (1-100)',
                color='Hierarchical_Cluster',
                title=f'Hierarchical Clustering (Number of Clusters={optimal_n_clusters_hierarchical})',
                labels={'Hierarchical_Cluster': 'Hierarchical Cluster'},
                color_continuous_scale = px.colors.sequential.Plasma
            )
            st.plotly_chart(fig_scatter_hierarchical, theme='streamlit', use_container_width=True)

            # Hypothesis Testing for Hierarchical Clusters
            st.markdown("#### Hypothesis Testing for Hierarchical Clusters")
            st.markdown(
                "We can perform hypothesis tests to check if there are significant differences in 'Annual Income' and 'Spending Score' between the Hierarchical clusters."
            )

            for i in range(optimal_n_clusters_hierarchical):
                cluster_data = df[df['Hierarchical_Cluster'] == i]
                st.write(f"Cluster {i} data:")
                st.write(cluster_data)  # Display the data for the cluster

                # T-test for Annual Income
                t_income, p_income = stats.ttest_ind(
                    cluster_data['Annual Income (k$)'], df['Annual Income (k$)']
                )
                st.write(f"T-test for Annual   Income (Cluster {i} vs. Overall):")
                st.write(f"T-statistic: {t_income}, P-value: {p_income}")

                # T-test for Spending Score
                t_spending, p_spending = stats.ttest_ind(
                    cluster_data['Spending Score (1-100)'], df['Spending Score (1-100)']
                )
                st.write(f"T-test for Spending Score (Cluster {i} vs. Overall):")
                st.write(f"T-statistic: {t_spending}, P-value: {p_spending}")

                alpha = 0.05  # Significance level
                if p_income < alpha:
                    st.write(
                        f"Result: For Hierarchical Cluster {i}, the average annual income is significantly different from the overall average. (p-value < {alpha})"
                    )
                else:
                    st.write(
                        f"Result: For Hierarchical Cluster {i}, the average annual income is not significantly different from the overall average. (p-value >= {alpha})"
                    )

                if p_spending < alpha:
                    st.write(
                        f"Result: For Hierarchical Cluster {i}, the average spending score is significantly different from the overall average. (p-value < {alpha})"
                    )
                else:
                    st.write(
                        f"Result: For Hierarchical Cluster {i}, the average spending score is not significantly different from the overall average. (p-value >= {alpha})"
                    )