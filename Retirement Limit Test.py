import os
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as col
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib.patches import Ellipse
style.use('ggplot')

# **********Changing Directories**************
abc = os.getcwd
bbc = "/Users/amorriso/Downloads/Steelpan-Vibrations-master"
os.chdir(bbc)
ccd = os.getcwd
conf = os.path.exists("/Users/amorriso/Downloads/Steelpan-Vibrations-master/steelpan-vibrations-classifications.csv")


# ***********Splitting Objects in CSV File*****
coords6272559 = []
coords6272549 = []
coords6272553 = []
coords6272556 = []
coords6279621 = []
coords6279622 = []
coords6272563 = []
coords6279624 = []
with open("steelpan-vibrations-classifications.csv") as csvfile:
    csvrow = csv.reader(csvfile)
    for row in csvrow:
        z = row[5]
        if z == "Retirement Limit Test":
            # Isolating the Subject_Set and Frame Number of each classification
            classification_id = row[-2]
            cds = re.split('"|\{|\}|retired|:|null|Filename', classification_id)
            cds = list(filter(None, cds))
            subject_id = cds[0]
            classification_annotations = row[-3]
            cas = re.split('"|\{|\}|\[|\]|:|T\d'
                           '|task|T0|,|task|_|label|How many antinode regions do you see\?|value'
                           '|\s\(no concentric circles\)|T1'
                           '|Draw circles/ellipses around all the antinode regions that you see\.|x|y|rx|ry|tool|angle'
                           '|frame|details|Draw Circle/Ellipse Tool|Draw circles around all the fringes\s'
                           '|ou see\.|Ellipse Draw \d', classification_annotations)
            cas = list(filter(None, cas))
            l_cas = len(cas)
            
            # print (cas)
            # print (l_cas)
            # print (cds)

            # Building the lists that will be used for the arrays used in the Kmeans clustering
            # and as the dataframe in pandas
            if l_cas == 1:
                cas[0] = 0
                # Frames without antinode regions do not need to be plotted
                # So no data is appended to the subject sets

            elif l_cas == 9:
                cas[0] = 1
                coords1 = [float(cas[1]), float(cas[2]), subject_id, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6])]
                if subject_id == '6272559':
                    coords6272559.append(coords1)
                    print(coords6272559)
                elif subject_id == '6272549':
                    coords6272549.append(coords1)
                elif subject_id == '6272553':
                    coords6272553.append(coords1)
                elif subject_id == '6272556':
                    coords6272556.append(coords1)
                elif subject_id == '6279621':
                    coords6279621.append(coords1)
                elif subject_id == '6279622':
                    coords6279622.append(coords1)
                elif subject_id == '6272563':
                    coords6272563.append(coords1)
                elif subject_id == '6279624':
                    coords6279624.append(coords1)

            elif l_cas == 17:
                cas[0] = 2
                coords1 = [float(cas[1]), float(cas[2]), subject_id, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6])]
                coords2 = [float(cas[9]), float(cas[10]), subject_id, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14])]
                if subject_id == '6272559':
                    coords6272559.append(coords1)
                    coords6272559.append(coords2)
                elif subject_id == '6272549':
                    coords6272549.append(coords1)
                    coords6272549.append(coords2)
                elif subject_id == '6272553':
                    coords6272553.append(coords1)
                    coords6272553.append(coords2)
                elif subject_id == '6272556':
                    coords6272556.append(coords1)
                    coords6272556.append(coords2)
                elif subject_id == '6279621':
                    coords6279621.append(coords1)
                    coords6279621.append(coords2)
                elif subject_id == '6279622':
                    coords6279622.append(coords1)
                    coords6279622.append(coords2)
                elif subject_id == '6272563':
                    coords6272563.append(coords1)
                    coords6272563.append(coords2)
                elif subject_id == '6279624':
                    coords6279624.append(coords1)
                    coords6279624.append(coords2)

            elif l_cas == 25:
                cas[0] = 3
                coords1 = [float(cas[1]), float(cas[2]), subject_id, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6])]
                coords2 = [float(cas[9]), float(cas[10]), subject_id, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14])]
                coords3 = [float(cas[17]), float(cas[18]), subject_id, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22])]
                if subject_id == '6272559':
                    coords6272559.append(coords1)
                    coords6272559.append(coords2)
                    coords6272559.append(coords3)
                elif subject_id == '6272549':
                    coords6272549.append(coords1)
                    coords6272549.append(coords2)
                    coords6272549.append(coords3)
                elif subject_id == '6272553':
                    coords6272553.append(coords1)
                    coords6272553.append(coords2)
                    coords6272553.append(coords3)
                elif subject_id == '6272556':
                    coords6272556.append(coords1)
                    coords6272556.append(coords2)
                    coords6272556.append(coords3)
                elif subject_id == '6279621':
                    coords6279621.append(coords1)
                    coords6279621.append(coords2)
                    coords6279621.append(coords3)
                elif subject_id == '6279622':
                    coords6279622.append(coords1)
                    coords6279622.append(coords2)
                    coords6279622.append(coords3)
                elif subject_id == '6272563':
                    coords6272563.append(coords1)
                    coords6272563.append(coords2)
                    coords6272563.append(coords3)
                elif subject_id == '6279624':
                    coords6279624.append(coords1)
                    coords6279624.append(coords2)
                    coords6279624.append(coords3)

            elif l_cas == 33:
                cas[0] = 4
                coords1 = [float(cas[1]), float(cas[2]), subject_id, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6])]
                coords2 = [float(cas[9]), float(cas[10]), subject_id, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14])]
                coords3 = [float(cas[17]), float(cas[18]), subject_id, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22])]
                coords4 = [float(cas[25]), float(cas[26]), subject_id, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30])]
                if subject_id == '6272559':
                    coords6272559.append(coords1)
                    coords6272559.append(coords2)
                    coords6272559.append(coords3)
                    coords6272559.append(coords4)
                elif subject_id == '6272549':
                    coords6272549.append(coords1)
                    coords6272549.append(coords2)
                    coords6272549.append(coords3)
                    coords6272549.append(coords4)
                elif subject_id == '6272553':
                    coords6272553.append(coords1)
                    coords6272553.append(coords2)
                    coords6272553.append(coords3)
                    coords6272553.append(coords4)
                elif subject_id == '6272556':
                    coords6272556.append(coords1)
                    coords6272556.append(coords2)
                    coords6272556.append(coords3)
                    coords6272556.append(coords4)
                elif subject_id == '6279621':
                    coords6279621.append(coords1)
                    coords6279621.append(coords2)
                    coords6279621.append(coords3)
                    coords6279621.append(coords4)
                elif subject_id == '6279622':
                    coords6279622.append(coords1)
                    coords6279622.append(coords2)
                    coords6279622.append(coords3)
                    coords6279622.append(coords4)
                elif subject_id == '6272563':
                    coords6272563.append(coords1)
                    coords6272563.append(coords2)
                    coords6272563.append(coords3)
                    coords6272563.append(coords4)
                elif subject_id == '6279624':
                    coords6279624.append(coords1)
                    coords6279624.append(coords2)
                    coords6279624.append(coords3)
                    coords6279624.append(coords4)


def df_to_center_plt(coords_x):
    x_val = []
    y_val = []
    frng = []
    crds = []
    ell = []
    for centers in coords_x:
        x_val.append(centers[0])
        y_val.append(centers[1])
        frng.append(centers[3])
        crds.append([centers[0], centers[1]])
        ell.append(Ellipse(xy=[centers[0], centers[1]], width=centers[4], height=centers[5], angle=centers[6]))
    centers_raw = {'XVal': x_val,
                   'YVal': y_val,
                   'Fringe': frng}
    centers_df = pd.DataFrame(centers_raw, columns=['XVal', 'YVal', 'Fringe'])
    plt.scatter(centers_df.XVal, centers_df.YVal, s=20, c=cmap_1.to_rgba(centers_df.Fringe), alpha=.6)
    plt.xlim(0, 512)
    plt.ylim(0, 384)
    plt.show()
    dbscan(crds)
    draw_ellipse(ell)


def draw_ellipse(ell):
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    for e in ell:
        ax.add_artist(e)
        e.set_alpha(.3)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 384)
    plt.show()


def dbscan(crds):
    X = np.array(crds)
    db = DBSCAN(eps=18, min_samples=3).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlim(0, 512)
    plt.ylim(0, 384)

cmap_1 = cm.ScalarMappable(
    col.Normalize(1, 11, cm.gist_rainbow))

df_to_center_plt(coords6272559)
df_to_center_plt(coords6272549)
df_to_center_plt(coords6272553)
df_to_center_plt(coords6272556)
df_to_center_plt(coords6279621)
df_to_center_plt(coords6279622)
df_to_center_plt(coords6272563)
df_to_center_plt(coords6279624)

