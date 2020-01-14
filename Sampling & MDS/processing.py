from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS

import numpy as np
import pandas as pd

class Data_Processer(object):
    def __init__(self, sample_rate = 0.3, no_of_columns = 12):
        self.df = None
        self.df_random_sample = None
        self.df_stratified = None
        self.pca = None
        self.pca_random = None
        self.pca_stratified = None
        self.top_attributes = None
        self.sample_rate = sample_rate
        self.no_of_columns = no_of_columns

    def get_task_1(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.fillna(0)

        self.df = self.df[list(self.df.columns[0:self.no_of_columns])]

        scaler = preprocessing.StandardScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns = self.df.columns)

        ### Random samples
        self.df_random_sample = self.df.sample(frac=self.sample_rate, replace=False, random_state=13)

        no_of_centroids = range(2, 10)

        sum_of_sq = np.zeros(shape=(len(no_of_centroids), 2))
        for i in no_of_centroids:
            kmeans = KMeans(n_clusters=i, random_state=123).fit(self.df)
            sum_of_square_i = sum(np.min(cdist(self.df, kmeans.cluster_centers_, 'euclidean'), axis=1)) / self.df.shape[0]
            sum_of_sq[i - 2] = [i, sum_of_square_i]

        ### Elbow
        elbow_idx = self.get_elbow_general(sum_of_sq)

        from collections import Counter

        list0 = []
        list1 = []
        list2 = []

        kmeans = KMeans(n_clusters=int(elbow_idx), random_state=123).fit(self.df)
        labels = kmeans.labels_
        for i in range(len(labels)):
            l = labels[i]
            if l == 0:
                list0.append(i)
            elif l == 1:
                list1.append(i)
            else:
                list2.append(i)

        stratified_sample_index = []
        np.random.shuffle(list0)
        np.random.shuffle(list1)
        np.random.shuffle(list2)

        stratified_sample_index.extend(list0[0: int(len(list0) * self.sample_rate)])
        stratified_sample_index.extend(list1[0: int(len(list1) * self.sample_rate)])
        stratified_sample_index.extend(list2[0: int(len(list2) * self.sample_rate)])

        stratified_samples = np.asarray([self.df.values[x] for x in stratified_sample_index])
        self.df_stratified = pd.DataFrame(stratified_samples, columns=self.df.columns)
        return self.df_random_sample.values.tolist(), sum_of_sq.tolist(), str(elbow_idx), self.df_stratified.values.tolist()

    def get_task_2(self, csv_file):

        if self.df_stratified is None :
            self.get_task_1(csv_file)

        self.pca = PCA()
        self.pca.fit(self.df.values)
        scree_all = list()

        cumulative_all = list()
        cumulative_all.append((1, self.pca.explained_variance_ratio_[0]))
        scree_all.append((1, self.pca.explained_variance_ratio_[0]))
        for idx, elem in enumerate(self.pca.explained_variance_ratio_[1:]):
            cumulative_all.append((idx + 2, cumulative_all[-1][1] + elem))
            scree_all.append((idx + 2, elem))

        self.pca_stratified = PCA()
        self.pca_stratified.fit(self.df_stratified.values)
        scree_strat = list()
        dimensionality = 0

        cumulative = list()
        cumulative.append((1, self.pca_stratified.explained_variance_ratio_[0]))
        scree_strat.append((1, self.pca_stratified.explained_variance_ratio_[0]))

        for idx, elem in enumerate(self.pca_stratified.explained_variance_ratio_[1:]):
            cumulative.append((idx + 2, cumulative[-1][1] + elem))
            scree_strat.append((idx + 2, elem))
            if dimensionality == 0 and cumulative[-1][1] > 0.75:
                dimensionality = idx + 2

        df_eigen = pd.DataFrame()
        df_eigen['Eigen value'] = self.pca_stratified.singular_values_
        df_eigen['Percentage of variance'] = self.pca_stratified.explained_variance_ratio_
        df_eigen['Cumulative Percentage of variance'] = cumulative

        df_pca_components = pd.DataFrame(self.pca_stratified.components_[0:3].T, columns=["PC1", "PC2", "PC3"])

        df_pca_loadings = pd.DataFrame(df_pca_components.values * np.sqrt(self.pca_stratified.explained_variance_ratio_[0:3]),
                                   columns=["PC1", "PC2", "PC3"], index=self.df.columns)
        df_pca_loadings['sum of squared loadings'] = df_pca_loadings['PC1'] ** 2 + df_pca_loadings['PC2'] ** 2 + \
                                                     df_pca_loadings['PC3'] ** 2

        df_pca_loadings = df_pca_loadings.sort_values(by=['sum of squared loadings'], ascending=False)
        self.top_attributes = list(df_pca_loadings.index[0:3])

        return str(dimensionality), scree_all, scree_strat, cumulative_all, cumulative,  self.top_attributes


    def get_task_3(self, csv_file):

        top_2_Eigen_vectors = self.pca_stratified.components_[0:2].T
        data_projections = np.matmul(self.df_stratified.values, top_2_Eigen_vectors)

        eucledian_matrix = euclidean_distances(self.df_stratified.values)
        correlation_matrix = np.corrcoef(self.df_stratified.values)

        model = MDS(n_components=2, random_state=1, dissimilarity='euclidean')
        eucledian_2d = model.fit_transform(eucledian_matrix)

        correlation_2d = model.fit_transform(correlation_matrix)

        data_projections_list = []
        for l in data_projections.tolist():
            data_projections_list.append(
                {"PCA_1": l[0], "PCA_2": l[1]})

        scatter_list = []
        scatter_2d_list = self.df_stratified[self.top_attributes].values.tolist()
        for l in scatter_2d_list:
            scatter_list.append({self.top_attributes[0]: l[0], self.top_attributes[1]: l[1], self.top_attributes[2]: l[2]})

        correlation_list = []
        for l in correlation_2d.tolist():
            correlation_list.append(
                {"x": l[0], "y": l[1]})

        eucledian_list = []
        for l in eucledian_2d.tolist():
            eucledian_list.append(
                {"x": l[0], "y": l[1]})


        return data_projections_list, eucledian_list, correlation_list, scatter_list



    def get_task_3_0(self, csv_file):

        top_2_Eigen_vectors = self.pca.components_[0:2].T
        data_projections = np.matmul(self.df.values, top_2_Eigen_vectors)

        eucledian_matrix = euclidean_distances(self.df.values)
        correlation_matrix = np.corrcoef(self.df.values)

        model = MDS(n_components=2, random_state=1, dissimilarity='euclidean')
        eucledian_2d = model.fit_transform(eucledian_matrix)

        correlation_2d = model.fit_transform(correlation_matrix)

        data_projections_list = []
        for l in data_projections.tolist():
            data_projections_list.append(
                {"PCA_1": l[0], "PCA_2": l[1]})

        scatter_list = []
        scatter_2d_list = self.df[self.top_attributes].values.tolist()
        for l in scatter_2d_list:
            scatter_list.append({self.top_attributes[0]: l[0], self.top_attributes[1]: l[1], self.top_attributes[2]: l[2]})

        correlation_list = []
        for l in correlation_2d.tolist():
            correlation_list.append(
                {"x": l[0], "y": l[1]})

        eucledian_list = []
        for l in eucledian_2d.tolist():
            eucledian_list.append(
                {"x": l[0], "y": l[1]})


        return data_projections_list, eucledian_list, correlation_list, scatter_list


    def get_elbow_general(self, data) :
        nPoints = len(data)
        allCoord = data.T

        firstPoint = data[0]
        # get vector between first and last point - this is the line
        lineVec = data[-1] - data[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))

        vecFromFirst = data - firstPoint

        scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel

        # distance to line is the norm of vecToLine
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

        # knee/elbow is the point with max distance value
        idxOfBestPoint = np.argmax(distToLine)

        print("Knee of the curve is at index =", idxOfBestPoint)
        print("Knee value =", data[idxOfBestPoint])
        return int(data[idxOfBestPoint][0])
