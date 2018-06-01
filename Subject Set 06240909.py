import os
import csv
import re
import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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
bbc = "C:\\Users\\Joseph\\python"
os.chdir(bbc)
ccd = os.getcwd
conf = os.path.exists("C:\\Users\\Joseph\\python\\steelpan-vibrations-classifications.csv")

# ***********Splitting Objects in CSV File*****
coords06240909 = []
subject1 = '06240909'
with open("steelpan-vibrations-classifications.csv") as csvfile:
    csvrow = csv.reader(csvfile)
    for row in csvrow:
        z = row[5]
        if z == "Counting Fringes":
            # Isolating the Subject_Set and Frame_Number of each classification
            classification_id = row[-2]
            cds = re.split('"|\{|\}|retired|:|null|Filename|_|\.|png|,|proc', classification_id)
            cds = list(filter(None, cds))
            subject_set = cds[1]
            frame_num = int(cds[2])
            # cds_header = ['Subject_Ids','Subject_Set','Frame_Number']
            classification_annotations = row[-3]
            cas = re.split('"|\{|\}|\[|\]|:|T\d|\*|\\\\|\(|\)|\''
                           '|if you are unsure\.|see\.|opinions\.'
                           '|[a-z]|[A-Z]|\s|_|\?|\,|/|\D\.|_\.', classification_annotations)
            cas = list(filter(None, cas))
            l_cas = len(cas)

            if l_cas == 1:
                cas[0] = 0
                # Frames without antinode regions do not need to be plotted
                # So no data is appended to the subject sets
            elif l_cas == 9:
                cas[0] = 1
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                if subject_set == subject1:
                    coords06240909.append(coords1)
            elif l_cas == 17:
                cas[0] = 2
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                if subject_set == subject1:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
            elif l_cas == 25:
                cas[0] = 3
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                if subject_set == subject1:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
            elif l_cas == 33:
                cas[0] = 4
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30]), row[1]]
                if subject_set == subject1:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                    coords06240909.append(coords4)
            elif l_cas == 41:
                cas[0] = 5
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30]), row[1]]
                coords5 = [float(cas[33]), float(cas[34]), frame_num, int(cas[40]), float(cas[35]), float(cas[36]), float(cas[38]), row[1]]
                if subject_set == subject1:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                    coords06240909.append(coords4)
                    coords06240909.append(coords5)
            elif l_cas == 49:
                cas[0] = 6
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30]), row[1]]
                coords5 = [float(cas[33]), float(cas[34]), frame_num, int(cas[40]), float(cas[35]), float(cas[36]), float(cas[38]), row[1]]
                coords6 = [float(cas[41]), float(cas[42]), frame_num, int(cas[48]), float(cas[43]), float(cas[44]), float(cas[46]), row[1]]
                if subject_set == subject1:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                    coords06240909.append(coords4)
                    coords06240909.append(coords5)
                    coords06240909.append(coords6)
for datum in coords06240909:
    if datum[3] == 0:
        datum[3] = np.NaN
    rx = datum[4]
    ry = datum[5]
    area_ellipse = 3.14159265359*rx*ry
    datum.append(area_ellipse)

aggregated_data_init = {'x_crd': [],
                        'y_crd': [],
                        'fringe': [],
                        'frame': [],
                        'area': [],
                        'label': []}
aggregated_data = pd.DataFrame(aggregated_data_init, columns=['x_crd','y_crd','fringe','frame','area','label'])
antinode1 = pd.DataFrame(aggregated_data_init, columns=['x_crd','y_crd','fringe','frame','area','label'])
antinode2 = pd.DataFrame(aggregated_data_init, columns=['x_crd','y_crd','fringe','frame','area','label'])
antinode3 = pd.DataFrame(aggregated_data_init, columns=['x_crd','y_crd','fringe','frame','area','label'])
antinode4 = pd.DataFrame(aggregated_data_init, columns=['x_crd','y_crd','fringe','frame','area','label'])
antinode5 = pd.DataFrame(aggregated_data_init, columns=['x_crd','y_crd','fringe','frame','area','label'])
df = pd.DataFrame(data=coords06240909, columns=['X', 'Y', 'Frame', 'Fringes', 'rX', 'rY', 'Angle', 'Volunteer', 'Area'])
for name, group in df.groupby('Frame'):
    min_sample_size = math.floor(len(list(group.groupby('Volunteer').size()))/2)
    if min_sample_size == 0:
        min_sample_size = 1
    subset = group[['X', 'Y']]
    tuples = [list(x) for x in subset.values]
    X = np.array(tuples)
    db = DBSCAN(eps=18, min_samples=(min_sample_size)).fit(X)
    labels = db.labels_
    cluster_raw = {'x_crd': list(group['X']),
                   'y_crd': list(group['Y']),
                   'fringe': list(group['Fringes']),
                   'frame': [int(name)]*(len(list(group['X']))),
                   'area': list(group['Area']),
                   'label': labels}
    clusters = pd.DataFrame(cluster_raw, columns=['x_crd','y_crd','fringe','frame','area','label'])
    avg_df = clusters.groupby(['label'], as_index=False).mean()
    avg_df_f = avg_df[avg_df.label != -1]
    aggregated_data = aggregated_data.append(avg_df_f,ignore_index=True)
for index, row in aggregated_data.iterrows():
    if 60 < row['x_crd'] < 150 and 80 < row['y_crd'] < 180:
        antinode1 = antinode1.append(row, ignore_index=True)
    elif 270 < row['x_crd'] < 330 and 100 < row['y_crd'] < 165:
        antinode2 = antinode2.append(row, ignore_index=True)
    elif 100 < row['x_crd'] < 200 and 300 < row['y_crd'] < 350:
        antinode3 = antinode3.append(row, ignore_index=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aggregated_data.x_crd, aggregated_data.y_crd, aggregated_data.frame, c='r', marker='o')
ax.set_xlabel('X Coord')
ax.set_ylabel('Y Coord')
ax.set_zlabel('Frame')
ax.set_xlim(0, 512)
ax.set_ylim(0, 384)
ax.set_zlim(0, 2000)
plt.show()

def plot_amp_time(antinode):
    plt.xlim(0, 2000)
    plt.ylim(0, 11)
    plt.xlabel('Time(Frame)')
    plt.ylabel('Amplitude(Fringe)')
    plt.show(plt.scatter(antinode.frame, antinode.fringe, s=20))
    plt.xlim(0, 2000)
    plt.xlabel('Time(Frame)')
    plt.ylabel('Amplitude(Area)')
    plt.show(plt.scatter(antinode.frame, antinode.area, s=20))

plot_amp_time(antinode1)
plot_amp_time(antinode2)
plot_amp_time(antinode3)
