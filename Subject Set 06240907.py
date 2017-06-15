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
coords06240907 = []
subject1 = '06240907'
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
            # Isolating the relevant pieces of data from classification annotations
            classification_annotations = row[-3]
            cas = re.split('"|\{|\}|\[|\]|:|T\d|\*|\\\\|\(|\)|\''
                           '|if you are unsure\.|see\.|opinions\.'
                           '|[a-z]|[A-Z]|\s|_|\?|\,|/|\D\.|_\.', classification_annotations)
            cas = list(filter(None, cas))
            l_cas = len(cas)

            # Appending the data for a single subject set together
            # Datum are in the form of [x,y,frame,fringe,rx,ry,angle,volunteer]
            # Had to take in account of the classifications annotations being different lengths
            if l_cas == 1:
                cas[0] = 0
                # Frames without antinode regions do not need to be plotted
                # So no data is appended to the subject sets
            elif l_cas == 9:
                cas[0] = 1
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                if subject_set == subject1:
                    coords06240907.append(coords1)
            elif l_cas == 17:
                cas[0] = 2
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                if subject_set == subject1:
                    coords06240907.append(coords1)
                    coords06240907.append(coords2)
            elif l_cas == 25:
                cas[0] = 3
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                if subject_set == subject1:
                    coords06240907.append(coords1)
                    coords06240907.append(coords2)
                    coords06240907.append(coords3)
            elif l_cas == 33:
                cas[0] = 4
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30]), row[1]]
                if subject_set == subject1:
                    coords06240907.append(coords1)
                    coords06240907.append(coords2)
                    coords06240907.append(coords3)
                    coords06240907.append(coords4)
            elif l_cas == 41:
                cas[0] = 5
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30]), row[1]]
                coords5 = [float(cas[33]), float(cas[34]), frame_num, int(cas[40]), float(cas[35]), float(cas[36]), float(cas[38]), row[1]]
                if subject_set == subject1:
                    coords06240907.append(coords1)
                    coords06240907.append(coords2)
                    coords06240907.append(coords3)
                    coords06240907.append(coords4)
                    coords06240907.append(coords5)
            elif l_cas == 49:
                cas[0] = 6
                coords1 = [float(cas[1]), float(cas[2]), frame_num, int(cas[8]), float(cas[3]), float(cas[4]), float(cas[6]), row[1]]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, int(cas[16]), float(cas[11]), float(cas[12]), float(cas[14]), row[1]]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, int(cas[24]), float(cas[19]), float(cas[20]), float(cas[22]), row[1]]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, int(cas[32]), float(cas[27]), float(cas[28]), float(cas[30]), row[1]]
                coords5 = [float(cas[33]), float(cas[34]), frame_num, int(cas[40]), float(cas[35]), float(cas[36]), float(cas[38]), row[1]]
                coords6 = [float(cas[41]), float(cas[42]), frame_num, int(cas[48]), float(cas[43]), float(cas[44]), float(cas[46]), row[1]]
                if subject_set == subject1:
                    coords06240907.append(coords1)
                    coords06240907.append(coords2)
                    coords06240907.append(coords3)
                    coords06240907.append(coords4)
                    coords06240907.append(coords5)
                    coords06240907.append(coords6)
                    
# Getting rid of classifications with '0' fringes
# Appending the area of an antinode region to the data
for datum in coords06240907:
    if datum[3] == 0:
        datum[3] = np.NaN
    rx = datum[4]
    ry = datum[5]
    area_ellipse = 3.14159265359*rx*ry
    datum.append(area_ellipse)

# Initializing the dataframe that will hold all the data
# and the dataframes that will hold each antinode region
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

# Creating a dataframe for all the data points
df = pd.DataFrame(data=coords06240907, columns=['X', 'Y', 'Frame', 'Fringes', 'rX', 'rY', 'Angle', 'Volunteer', 'Area'])
for name, group in df.groupby('Frame'):
    min_sample_size = math.floor(len(list(group.groupby('Volunteer').size()))/2)
    if min_sample_size == 0:
        min_sample_size = 1
    subset = group[['X', 'Y']]
    tuples = [list(x) for x in subset.values]
    # Clustering the centers for each frame, then averaging out the values for each cluster
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
    # Appending the average of each frame together
    aggregated_data = aggregated_data.append(avg_df_f,ignore_index=True)

# Identifying where each region is locating and isolating each one
for index, row in aggregated_data.iterrows():
    if 0 < row['x_crd'] < 115 and 175 < row['y_crd'] < 300:
        antinode1 = antinode1.append(row, ignore_index=True)
    if 200 < row['x_crd'] < 270 and 20 < row['y_crd'] < 65:
        antinode2 = antinode2.append(row, ignore_index=True)
    if 271 < row['x_crd'] < 350 and 20 < row['y_crd'] < 100:
        antinode3 = antinode3.append(row, ignore_index=True)
    if 230 < row['x_crd'] < 330 and 150 < row['y_crd'] < 250:
        antinode4 = antinode4 .append(row, ignore_index=True)
    if 260 < row['x_crd'] < 330 and 300 < row['y_crd'] < 370:
        antinode5 = antinode5.append(row, ignore_index=True)

# Plotting each antinode's amplitude vs. time graphs
# Amplitude measured in fringes and area of antinode region
plt.show(plt.scatter(antinode1.frame, antinode1.fringe, s=20))
plt.show(plt.scatter(antinode1.frame, antinode1.area, s=20))
plt.show(plt.scatter(antinode2.frame, antinode2.fringe, s=20))
plt.show(plt.scatter(antinode2.frame, antinode2.area, s=20))
plt.show(plt.scatter(antinode3.frame, antinode3.fringe, s=20))
plt.show(plt.scatter(antinode3.frame, antinode3.area, s=20))
plt.show(plt.scatter(antinode4.frame, antinode4.fringe, s=20))
plt.show(plt.scatter(antinode4.frame, antinode4.area, s=20))
plt.show(plt.scatter(antinode5.frame, antinode5.fringe, s=20))
plt.show(plt.scatter(antinode5.frame, antinode5.area, s=20))

# 3D plot that tracks location of antinode centers through time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aggregated_data.x_crd, aggregated_data.y_crd, aggregated_data.frame, c='r', marker='o')
ax.set_xlabel('X Coord')
ax.set_ylabel('Y Coord')
ax.set_zlabel('Frame')
ax.set_xlim(0,512)
ax.set_ylim(0,384)
ax.set_zlim(0,2000)
plt.show()
