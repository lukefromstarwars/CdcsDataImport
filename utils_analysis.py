from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import kneighbors_graph

from utils import *


# region GRAPHICS
# ------------------------------------------------------------------------------------------


def get_object_methods(o):
    return [method for method in dir(o) if callable(getattr(o, method))]


def get_region_palette():
    col_bxl = "#FFEB17"
    col_vl = "#006A8D"
    col_wal = "#D91302"
    reg_ui = [col_bxl, col_vl, col_wal]
    sns.set_palette(reg_ui)
    pal = sns.color_palette()
    return pal


# ------------------------------------------------------------------------------------------
def get_birch_clusters(n_clusters):
    print_debug(n_clusters)
    return cluster.Birch(n_clusters=n_clusters)


def get_agglomerative_clusters(n_clusters, x_data):
    # -- connectivity matrix for structured Ward
    connectivity = kneighbors_graph(x_data, n_neighbors=20, include_self=False)
    # -- make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    return cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=n_clusters, connectivity=connectivity)


def get_spectral_clusters(n_clusters):
    return cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', gamma=1, degree=5)


def get_ward_clusters(n_clusters, x_data):
    # -- connectivity matrix for structured Ward
    connectivity = kneighbors_graph(x_data, n_neighbors=20, include_self=False)
    # -- make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    # -- set up clustering algorithms
    return cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity)


def get_minibatch_clusters(n_clusters):
    return cluster.MiniBatchKMeans(n_clusters=n_clusters)


def get_meanshift_clusters(x_data):
    bandwidth = cluster.estimate_bandwidth(x_data, quantile=.5)
    return cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)


# -- PROD : tol e-4, max_iter=500
# -- HACK : tol e-5, max_iter=800
def get_kmeans_clusters(n_clusters):
    return cluster.KMeans(n_clusters=n_clusters,
                          random_state=42,
                          n_init=10,
                          precompute_distances='auto',
                          max_iter=500,
                          tol=1e-2,
                          # n_jobs=-3,
                          init='k-means++')


def get_clusters(x_data, n_clusters: int, cluster_algorithm: str):
    # print_debug(x_data, n_clusters, cluster_algorithm)

    if cluster_algorithm == MyClusters.ALGO_KMeans:
        clf = get_kmeans_clusters(n_clusters)
    elif cluster_algorithm == MyClusters.ALGO_MiniBatchKMeans:
        clf = get_minibatch_clusters(n_clusters)
    elif cluster_algorithm == MyClusters.ALGO_Spectral:
        clf = get_spectral_clusters(n_clusters)
    elif cluster_algorithm == MyClusters.ALGO_MeanShift:
        clf = get_meanshift_clusters(x_data)
    elif cluster_algorithm == MyClusters.ALGO_Ward:
        clf = get_ward_clusters(n_clusters, x_data)
    elif cluster_algorithm == MyClusters.ALGO_Agglomerative:
        clf = get_agglomerative_clusters(n_clusters, x_data)
    elif cluster_algorithm == MyClusters.ALGO_Birch:
        clf = get_birch_clusters(n_clusters)

    cluster_labels = clf.fit_predict(x_data)

    if cluster_algorithm == MyClusters.ALGO_KMeans:
        return cluster_labels, clf.cluster_centers_
    else:
        return cluster_labels, []


def get_pca_parameters(df_x, n_comps):
    """

    :param df_x:
    :param n_comps:
    :return: dataframe of selected and transformed data for type in chosen column with pca and scaler for all kept years
    """
    pca = PCA(n_comps, random_state=42, whiten=True)
    X = np.array(df_x.astype(float))
    X_scaled, scaler = scale_array_between_0_1(X)
    x_data = pca.fit_transform(X_scaled)

    return x_data, pca, scaler


def scale_array_between_0_1(x_arr):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_arr.astype(float))
    return x_scaled, scaler


def do_pca_transformation(df, pca, scaler):
    X = np.array(df)
    X_scaled = scaler.fit_transform(X.astype(float))
    x_data = pca.fit_transform(X_scaled)
    return x_data


def get_pca_data(x_data, n_comps):
    """

    :param x_data:
    :param df_y:
    :param cluster_topic:
    :param cluster_by:
    :param n_comps:
    :return: dataframe of selected and transformed data for type in chosen column with pca and scaler for all kept years
    """
    scaler = MinMaxScaler()
    pca = PCA(n_comps, random_state=42, whiten=True)
    X_scaled = scaler.fit_transform(x_data.astype(float))
    x_data = pca.fit_transform(X_scaled)

    return x_data, pca, scaler


def get_cluster_values(df, pca, scaler, clf):
    X = np.array(df)
    X_scaled = scaler.transform(X.astype(float))
    x_data = pca.transform(X_scaled)
    clusters = clf.predict(x_data)
    return clusters + 1


def get_clusters_parameters(df, pca, scaler, n_clusters) -> (DataFrame, DataFrame):
    X = np.array(df)
    X_scaled = scaler.transform(X.astype(float))
    x_data = pca.transform(X_scaled)

    # -- Define clusters
    clf = get_kmeans_clusters(n_clusters)
    clf.fit(x_data)
    clusters = clf.labels_
    centers = clf.cluster_centers_

    # -- Get centers
    df_centers = DataFrame(scaler.inverse_transform(pca.inverse_transform(centers)), columns=df.columns)
    df_centers = scale_min_max_df(df_centers)
    df_centers[MyClusters.CLUSTER] = df_centers.index + 1
    df[MyClusters.CLUSTER] = clusters + 1

    return df, df_centers, clf


# region STATS

def get_knn_scores(df_x, df_y, category, col,
                   whiten=False, min_n_comps=2, max_n_comps=10,
                   max_clusters=20, kmeans_metric='euclidean', clustering_algorithms=[MyClusters.ALGO_KMeans]) -> DataFrame:
    sc = MinMaxScaler()
    df = DataFrame(columns=MyDfs.KNN_COLS)
    min_clusters = 3

    # --
    df_x_selected = df_x[df_y[col] == category]
    sample_size = len(df_x_selected)
    # print_debug(category, sample_size)
    X = np.array(df_x_selected)
    X_scaled = sc.fit_transform(X.astype(float))

    for n_comps in np.arange(min_n_comps, max_n_comps):
        # pca = PCA(n_comps, random_state=42, whiten=True)
        pca = PCA(n_comps, random_state=42, whiten=whiten)

        if sample_size > max_clusters:
            x_data = pca.fit_transform(X_scaled)

            for clustering_algorithm in clustering_algorithms:
                for n_clusters in np.arange(min_clusters, max_clusters):
                    # -- HACK - check if we get better results taking all clusters x all components
                    if len(df[(df[MyClusters.CLUSTER_NAME] == category)
                              & (df[MyClusters.N_CLUSTERS] == n_clusters)
                              & (df[MyClusters.METRIC] == kmeans_metric)
                              & (df[MyClusters.CLUSTERING_ALGORITHM] == kmeans_metric)]) != 0:
                        print_debug('Already covered')
                        break

                    else:
                        cluster_labels, centers = get_clusters(x_data, n_clusters, clustering_algorithm)
                        silhouette_avg = silhouette_score(x_data, cluster_labels)
                        sample_silhouette_values = silhouette_samples(x_data, cluster_labels, metric=kmeans_metric)

                        # print_debug('sample_silhouette_values', sample_silhouette_values, len(sample_silhouette_values))
                        # print_debug('clustering_algorithm', clustering_algorithm)
                        # print_debug('cluster_labels', cluster_labels, len(cluster_labels))
                        # print_debug('centers', centers)

                        below_0 = np.where(sample_silhouette_values < 0)
                        worst_assignment = sample_silhouette_values[below_0[0]].mean()
                        pctMissClassifiedSamples = len(below_0) / len(sample_silhouette_values)

                        if pctMissClassifiedSamples <= .001:
                            # print_debug(n_comps, n_clusters, pctMissClassifiedSamples, worst_assignment)
                            df.loc[len(df)] = [col, category, n_clusters, n_comps, silhouette_avg, kmeans_metric, clustering_algorithm, worst_assignment, pctMissClassifiedSamples]

                        else:
                            print_debug(worst_assignment, pctMissClassifiedSamples, below_0)
                            print_debug('{} values are negative'.format(pctMissClassifiedSamples))
                            break

    return df


# endregion


# region BASIC PLOT
def plot_histogram(x):
    plt.hist(x, color='gray', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # TMP
    # plt.show()


def plot_histogram_dv(x, y):
    plt.hist(list(x[y == 0]), alpha=0.5, label='Outcome=0')
    plt.hist(list(x[y == 1]), alpha=0.5, label='Outcome=1')
    plt.title("Histogram of '{var_name}' by Outcome Category".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    # TMP
    # plt.show()


def box_plot_summary(category, clrs, df_x, nb_clusters, var_name, var_french):
    grp = 'Groupements'

    fig, ax = plt.subplots()
    sns.set_palette(clrs)
    sns.boxplot(x='Cluster', y=var_name, data=df_x, ax=ax)
    sns.despine(offset=15, trim=True, bottom=True)
    plt.suptitle(category, fontsize=16, fontweight='bold')
    ax.set_xlabel(grp)
    ax.set_ylabel(var_french)
    fig.tight_layout(h_pad=.9)
    plt.subplots_adjust(top=.9)
    # TMP
    # plt.show()

    png_file_to_save = MyFolders.CLUSTER_GRAPHS + '{} - {} - Boxplot ({} {}).png'.format(category, var_name, nb_clusters, grp)
    print(png_file_to_save)
    plt.savefig(png_file_to_save, dpi=300)
    plt.close()


def plot_corr_matrix(df, fn, corr_threshold=.3, font_scale=1.2, labelsize=10):
    # TEST
    # df = dfCorr
    # fn = 'fn'
    # corr_threshold = .3
    # font_scale = 1.2
    # labelsize = 10
    # TEST end

    df_corr = df.corr()

    df_result = df_corr[(df_corr.abs() > corr_threshold) & (df_corr.abs() < 1)]
    df_result = df_result.dropna(thresh=1).dropna(thresh=1, axis=1)

    fig, ax = plt.subplots()
    pal = sns.diverging_palette(220, 20, n=10)
    sns.heatmap(df_result, ax=ax, cmap=pal, yticklabels=1, xticklabels=1)
    sns.set(font_scale=font_scale)
    plt.yticks(rotation='horizontal')
    plt.xticks(rotation='vertical')
    plt.tick_params(axis='both', labelsize=labelsize)
    plt.subplots_adjust(top=.95)
    # fig.tight_layout()
    fig.suptitle(fn, fontsize=12)

    plt.show()
    plt.close()

    save_as_xlsx(df_result, fn)
    save_plot_as_png(fig, fn)


# endregion

# region GRAPH

def graph_pca_scree(x_data, max_comps, ds_name, category, dump_to_pickle=False):
    fig, ax = plt.subplots()
    minor_locator = MultipleLocator(1)

    # --
    x = []
    y = []

    for n_comp in np.arange(1, max_comps):
        x_pca, pca, scaler = get_pca_parameters(x_data, n_comp)
        y = pca.explained_variance_

    # Find tipping point
    tipping_slope = .01
    df = DataFrame([x > tipping_slope for x in (y - np.roll(y, -1))], columns=['IsValid'])
    x_tip_1 = df[df['IsValid']].index.min() + 2
    x_tip_2 = df[df['IsValid']].index.max() + 1

    # print(y, np.roll(y, -1), (y - np.roll(y, -1))[1:])

    ax.plot(y, color='green', linewidth=.5)
    # ax.set_title(file_path, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Explained variance')
    ax.set_xticks(np.arange(0, max_comps, 5))
    ax.xaxis.set_minor_locator(minor_locator)
    ax.tick_params(axis='x', which='minor', bottom=True, length=4, width=.5)

    ax.axvline(x=x_tip_1, color="r", linestyle=":")
    ax.axvline(x=x_tip_2, color="b", linestyle=":")
    ax.axvline(x=2, color="g", linestyle=":")

    title = '{} - PCA Scree ({} features)'.format(sanitize_string(category), n_comp)
    plt.suptitle(title, fontsize=12, fontweight='bold')
    # plt.show()
    plt.close()

    # print_debug(file_path, x_tip_1, x_tip_2)

    # --
    file_path = '{}/{}/{}.fig'.format(sanitize_string(ds_name), 'SCREE', title)
    save_plot_as_png(fig, file_path, dump_to_pickle=dump_to_pickle)

    return x_tip_1, x_tip_2


def graph_pca_scree_by_category_values(df_x, df_y, ds_name, values_col, dump_to_pickle=False):
    categories = sorted(df_y[values_col].unique())
    cat_tipping_points = []
    max_comps = len(df_x.columns) + 1

    # -- Cluster
    for category in categories:
        x_data = df_x[df_y[values_col] == category]

        if len(x_data) > len(df_x.columns):
            tip_point_1, tip_point_2 = graph_pca_scree(x_data, max_comps, ds_name, category, dump_to_pickle=dump_to_pickle)
            cat_tipping_points.append([category, tip_point_1, tip_point_2])

    return DataFrame(cat_tipping_points, columns=[MyCols.CAT, MyClusters.MIN_NCOMPS, MyClusters.MAX_NCOMPS]).sort_values(MyCols.CAT)


# TO REMOVE : 18-09-2018
def graph_2factors_clusters(df_x, df_y, col, min_clusters=3, max_clusters=6, categories=[], clustering_algorithm=MyClusters.ALGO_KMeans):
    # CHECK
    # print('graph_2factors_clusters')

    n_comps = 2
    # col = DV.

    if not categories:
        categories = df_y[col].unique()

    # CHECK
    # print(cats)

    scaler = MinMaxScaler()

    # -- Cluster
    for category in categories:

        # # TODO: Debug HACK
        # n_comps = 2

        pca = PCA(n_comps, random_state=42)
        print(category, n_comps)

        if category is not None:
            df_data = df_x[df_y[col] == category]
            # df_data = drop_columns([col], df_data)
            get_cols_alphabetically(df_data)

            if len(df_data) > max_clusters:
                X = np.array(df_data)
                X_scaled = scaler.fit_transform(X.astype(float))
                X_pca = pca.fit_transform(X_scaled)

                graph_2d_clusters(X_pca, max_clusters, category, min_clusters, n_comps, clustering_algorithm)


# END REMOVE

def get_silhouette_avgs(x_data, folder_name, topic, cluster_by, n_best_features, n_comps, min_clusters=3, clustering_algorithm=MyClusters.ALGO_KMeans):
    silh_avgs = []

    # --
    for n_cluster in np.arange(min_clusters, MyClusters.N_CLUSTERS):
        if n_cluster < min(x_data.shape):
            cluster_labels, centers = get_clusters(x_data, n_cluster, clustering_algorithm)
            silhouette_avg = silhouette_score(x_data, cluster_labels)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(x_data, cluster_labels)

            # Check some value for each cluster are above average
            all_above_average = True
            for i in range(n_cluster):
                # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                all_above_average = (ith_cluster_silhouette_values > silhouette_avg).any()
                if not all_above_average:
                    break

            if (sample_silhouette_values >= 0).all() & all_above_average:
                silh_avgs.append([folder_name, topic, cluster_by, n_best_features, n_cluster, n_comps, silhouette_avg])
    # --
    return silh_avgs


def graph_2d_clusters(x_data, folder_name, topic, cluster_by, n_best_features, n_comps, min_clusters=3, clustering_algorithm=MyClusters.ALGO_KMeans):
    range_n_clusters = np.arange(min_clusters, MyClusters.N_CLUSTERS)
    silh_avgs = []

    for n_cluster in range_n_clusters:
        if len(x_data) > n_cluster:
            silh_avg = graph_2d_cluster(folder_name, cluster_by, x_data, n_best_features, n_cluster, n_comps, clustering_algorithm)
            silh_avgs.append([folder_name, topic, cluster_by, n_best_features, n_cluster, n_comps, silh_avg])

    return silh_avgs


def graph_2d_cluster(folder_name, cluster_by, x_data, n_best_features, n_clusters, n_comps, clustering_algorithm=MyClusters.ALGO_KMeans, miss_classified_pct=0, worst_assignment=0):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x_data) + (n_clusters + 1) * 10])

    cluster_labels, centers = get_clusters(x_data, n_clusters, clustering_algorithm)
    silhouette_avg = silhouette_score(x_data, cluster_labels)

    # print("For {} clusters, the average silhouette_score is : {}".format(n_clusters, silhouette_avg))
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x_data, cluster_labels)

    # Check some value in cluster are above the average
    all_above_average = True
    ith_cluster_sizes = []
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_sizes.append(ith_cluster_silhouette_values.size)
        all_above_average = (ith_cluster_silhouette_values > silhouette_avg).any()
        if not all_above_average:
            break

    if (sample_silhouette_values >= 0).all() & all_above_average:

        # below_0 = np.where(sample_silhouette_values < 0)
        # worst_assignment = sample_silhouette_values[below_0[0]].mean()
        # pctMissClassifiedSamples = len(below_0) / len(sample_silhouette_values)

        y_lower = 10

        cluster_colors = []
        colPal = sns.hls_palette(n_clusters, h=.5)

        indexes = get_list_index_by_rank(ith_cluster_sizes, True)
        cluster_labels = cluster_labels + len(indexes)
        cluster_labels = np.asarray([indexes[x - len(indexes)] for x in cluster_labels])

        # First plot
        # --
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # --
            color = colPal[i]
            cluster_colors.append(color)

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.5)

            ax1.plot(ith_cluster_silhouette_values, np.arange(y_lower, y_upper), color=color, alpha=0.9, linewidth=.2)
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouettes")
        ax1.set_xlabel("Val.")
        ax1.set_ylabel("Clusters")

        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        # --
        # itemColors = [cluster_colors[x] for x in cluster_labels]
        itemColors = [cluster_colors[x] for x in cluster_labels]
        centerColors = [cluster_colors[x] for x in indexes]
        ax2.scatter(x_data[:, 0], x_data[:, 1], marker='.', s=50, lw=0, alpha=0.7, c=itemColors)

        # Draw white circles at cluster centers
        # print_debug(centers)
        if len(centers) > 0:
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c=centerColors, alpha=.6, s=600, edgecolors=centerColors)
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c='None', alpha=1, s=600, edgecolors=centerColors)

            for i, c in enumerate(centers):
                # ax2.scatter(c[0], c[1], marker='$%d$' % (i + 1), alpha=1, s=100, color='w')
                ax2.scatter(c[0], c[1], marker='$%d$' % (indexes[i] + 1), alpha=1, s=150, color='w')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        title = 'Silhouette analysis (N={}) - {} with {} clusters and {} factors'.format(len(x_data), sanitize_string(cluster_by), int(n_clusters), int(n_comps))
        plt.margins(.1, .9)
        plt.suptitle(title, fontsize=14, fontweight='bold')

        # plt.tight_layout(pad=.2, h_pad=.2

        # --
        file_path = '{} - {} clusters - {} factors - {} features ({} - score={} - errors(%={}, {}))'.format(sanitize_string(cluster_by),
                                                                                                            int(n_clusters),
                                                                                                            int(n_comps),
                                                                                                            int(n_best_features),
                                                                                                            clustering_algorithm,
                                                                                                            as_percent(silhouette_avg),
                                                                                                            as_percent(miss_classified_pct),
                                                                                                            as_percent(worst_assignment))

        file_path = '{}/{}/F{}/{}'.format(sanitize_string(folder_name), 'CLUSTERS', n_best_features, file_path)

        save_plot_as_png(fig, file_path)

        # TMP
        # plt.show()
        plt.close()
        return silhouette_avg
    else:
        plt.close(fig)


def graph_3d_cluster(x_data, n_clusters, cluster_by, n_comps, clustering_algorithm=MyClusters.ALGO_KMeans):
    cluster_labels, centers = get_clusters(x_data, n_clusters, clustering_algorithm)

    # Labeling the clusters
    fig = plt.figure(figsize=plt.figaspect(0.8))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    graph_3d_view(fig, ax, 45, 45, centers, cluster_labels, n_clusters, x_data)

    # HACK 18.05.2018 - No real need to get rid of -> it's just offers other viewing options
    # ax = fig.add_subplot(221, projection='3d')
    # graph_3d_view(fig, ax, 45, 45, centers, cluster_labels, n_clusters, x_data)
    # ax = fig.add_subplot(222, projection='3d')
    # graph_3d_view(fig, ax, 90, 90, centers, cluster_labels, n_clusters, x_data)
    # ax = fig.add_subplot(223, projection='3d')
    # graph_3d_view(fig, ax, 0, 0, centers, cluster_labels, n_clusters, x_data)
    # ax = fig.add_subplot(224, projection='3d')
    # graph_3d_view(fig, ax, 90, 0, centers, cluster_labels, n_clusters, x_data)

    png_file_to_save = '{} - {} clusters with {} factors  (alg={})'.format(sanitize_string(cluster_by), n_clusters, n_comps, clustering_algorithm)
    plt.suptitle(png_file_to_save, fontsize=12)

    plt.show()
    fig.tight_layout()


def graph_3d_view(fig, ax, azim, elev, centers, cluster_labels, n_clusters, x_pca):
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)
    colPal = sns.hls_palette(n_clusters, h=.5)
    itemColors = [colPal[x] for x in cluster_labels]

    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c=cluster_colors, alpha=.8, s=500, edgecolors=cluster_colors)

    ax.view_init(elev, azim)
    ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=itemColors, s=5)

    if len(centers) > 0:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colPal, s=100, marker="*")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def graph_3d_clusters(df_x, df_y, cat, categories=[], min_clusters=3, max_clusters=9, close_plot=True):
    n_comps = 3
    # TESTING
    # min_clusters = 3
    # max_clusters = 12

    # cat = DV.prog_type_txt_cat
    if not categories:
        categories = df_y[cat].unique()

    scaler = MinMaxScaler()

    cols = ['CATEGORY', 'N_CLUSTERS', 'LABELS', 'CENTERS', 'SILHOUETTE_AVG']
    cat_clusters = DataFrame(columns=cols)

    # -- Cluster
    for category in categories:
        # category = 'Consultations spécialisées'

        pca = PCA(n_comps, random_state=42)
        # print(category, n_comps)

        if category is not None:
            df_x_selected = df_x[df_y[cat] == category]

            if len(df_x_selected) > max_clusters:
                X = np.array(df_x_selected)
                x_scaled = scaler.fit_transform(X.astype(float))
                x_pca = pca.fit_transform(x_scaled)
                graph_3d_cluster(x_pca, min_clusters, category, max_clusters, cat_clusters)

    return cat_clusters


# endregion

def reorder_by_cluster_size(df, nb_of_clusters):
    # reorder group columns based on cluster size
    df1 = df.iloc[:, : nb_of_clusters]
    df2 = df.iloc[:, nb_of_clusters:]
    df1 = df1.T.sort_values(by='COUNT', ascending=False).T
    df1.columns = np.arange(1, nb_of_clusters + 1)
    df = pd.concat([df1, df2], axis=1)
    return df


def rescale_from0to1_socio_cmps(df):
    # df = df_centers
    # TESTING
    # df = df_centers
    cols_lives_with = get_cols_with_prefix(df, 'LVN_WITH_WHOM_')
    cols_lives_where = get_cols_with_prefix(df, 'LVN_WHERE_')
    cols_work_inc = get_cols_with_prefix(df, 'INCOME_')
    cols_work_status = get_cols_with_prefix(df, 'LABOUR_')
    cols_use = ['COUNT_PRODS_NUM', 'FREQ_USE_ORD']
    cols_education = get_cols_with_prefix(df, 'EDUCATION_')

    cols_socio = cols_lives_with + cols_lives_where + cols_work_inc + cols_work_status + cols_use + cols_education
    # rescale to value between 0 and 1
    df1 = df[cols_socio]
    df2 = drop_columns(df, cols_socio)

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(np.array(df1.astype(float)))
    df1_scaled = DataFrame(x_scaled, columns=df1.columns)

    # CHECK
    # get_cols_alphabetically(df)
    # get_cols_alphabetically(df1)
    # get_cols_alphabetically(df1_scaled)
    # get_cols_alphabetically(df2)

    df = pd.concat([df1_scaled, df2], axis=1)

    get_cols_alphabetically(df)

    return df


def cleanup_cols(df):
    df_cleaned = df.T

    replace_in_col_names(df_cleaned, ' / ', '/')
    replace_in_col_names(df_cleaned, '/ ', '/')

    cols_cluster = [MyCols.COUNT, MyCols.PERCENT]
    cols_sex = ['SEX_BIN']
    cols_education = get_cols_with_prefix(df_cleaned, 'EDUCATION_')
    cols_lives_with = get_cols_with_prefix(df_cleaned, 'LVN_WITH_WHOM_')
    cols_lives_where = get_cols_with_prefix(df_cleaned, 'LVN_WHERE_')
    cols_work_inc = get_cols_with_prefix(df_cleaned, 'INCOME_')
    cols_work_status = get_cols_with_prefix(df_cleaned, 'LABOUR_')
    cols_use = ['COUNT_PRODS_NUM', 'FREQ_USE_ORD']

    cols_socio = cols_lives_with + cols_lives_where + cols_work_inc + cols_work_status

    cols_to_drop = cols_education + cols_socio + cols_use
    df_ord_age = drop_columns(df_cleaned, cols_sex + cols_cluster + cols_to_drop)

    df_education = df_cleaned[cols_education]
    df_ord_socio = df_cleaned[cols_socio]
    df_ord_clusters = df_cleaned[cols_cluster]
    df_ord_sex = df_cleaned[cols_sex]
    df_use = df_cleaned[cols_use]

    replace_in_col_names(df_ord_sex, 'SEX_BIN', 'Sexe_Homme/Femme')

    col_add_prefix(df_ord_age, '', 'Age_')
    df_ord_age = df_ord_age[['Age_MAX',
                             'Age_75%',
                             'Age_50%',
                             'Age_25%',
                             'Age_MIN',
                             'Age_MEAN',
                             'Age_STD']]
    replace_in_col_names(df_ord_age, 'AGE_NUM', '')
    replace_in_col_names(df_ord_age, 'MAX', 'Max')
    replace_in_col_names(df_ord_age, 'MIN', 'Min')
    replace_in_col_names(df_ord_age, 'MEAN', 'Moyenne')
    replace_in_col_names(df_ord_age, 'STD', 'Ecart type')

    replace_in_col_names(df_education, 'EDUCATION_LEVEL_ORD', 'Education_Niveau')
    replace_in_col_names(df_ord_socio, 'LVN_WITH_WHOM_', 'Vit (avec)_')
    replace_in_col_names(df_ord_socio, 'LVN_WHERE_', 'Vit (où)_')
    replace_in_col_names(df_ord_socio, 'INCOME_', 'Travail (revenu)_')
    replace_in_col_names(df_ord_socio, 'LABOUR_', 'Travail (occupation)_')

    replace_in_col_names(df_use, 'COUNT_PRODS_NUM', 'Consommation_Nb de produits')
    replace_in_col_names(df_use, 'FREQ_USE_ORD', 'Consommation_Fréquence')

    col_add_prefix(df_ord_clusters, '', 'Groupement_')
    # col_replace_in_name(df_ord_clusters, 'CLUSTER', 'GROUPEMENT')
    replace_in_col_names(df_ord_clusters, 'COUNT', '(N)')
    replace_in_col_names(df_ord_clusters, 'PERCENTAGE', '(%)')

    # CHECK
    # get_cols_alphabetically(df)
    # get_cols_alphabetically(df_cleaned)
    # get_cols_alphabetically(df_ord_age)
    # get_cols_alphabetically(df_ord_socio)
    # get_cols_alphabetically(df_ord_clusters)

    df_cleaned = pd.concat([df_ord_clusters, df_ord_sex, df_ord_age, df_education, df_use, df_ord_socio], axis=1).T
    split_ix = df_cleaned.index.str.split('_')
    df_cleaned.index = pd.MultiIndex.from_tuples([(x[0], x[1]) for x in split_ix])

    return df_cleaned


# endregion

# region DEPRECATED

@deprecated
def remove_highly_corr_dimensions(df, corr_threshold=.3):
    # df
    corr_threshold = .5
    plot_corr_matrix(df, corr_threshold)
    # df
    df_corr = df.corr()
    df_result = df_corr[(df_corr.abs() > corr_threshold) & (df_corr.abs() < 1)].dropna(thresh=1).dropna(thresh=1, axis=1)
    # get_cols_alphabetically(df_corr)
    # get_cols_alphabetically(df_result)
    fig, ax = plt.subplots()
    g = sns.heatmap(df_result, ax=ax)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    fig.tight_layout(h_pad=.9)
    plt.subplots_adjust(top=.9)
    # TMP
    # plt.show()

# endregion
