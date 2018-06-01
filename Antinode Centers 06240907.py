import os
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')
import matplotlib.cm as cm
import matplotlib.colors as col

# **********Changing Directories**************
# ab = os.path
# print (ab)
abc = os.getcwd
# abc = os.path
# print(abc)
bbc = "/Users/amorriso/Downloads/Steelpan-Vibrations-master"
os.chdir(bbc)
ccd = os.getcwd
# print(ccd)
conf = os.path.exists("/Users/amorriso/Downloads/Steelpan-Vibrations-master/steelpan-vibrations-classifications.csv")
# print(conf)

# ***********Splitting Objects in CSV File*****
coords06240907 = []
coords06240909 = []
coords06240910 = []
coords06241902 = []
subject1 = '06240907'
subject2 = '06240909'
subject3 = '06240910'
subject4 = '06241902'
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
            cas = re.split('"|\{|\}|\[|\]|:|T\d'
                           '|task|T0|,|task|_|label|How many antinode regions do you see\?|value'
                           '|\s\(no concentric circles\)'
                           '|Draw circles around all the antinode regions you see\.|x|y|rx|ry|tool|angle'
                           '|frame|details|Ellipse Draw Tool|Draw circles around all the fringes\s'
                           '|ou see\.|Ellipse Draw \d', classification_annotations)
            cas = list(filter(None, cas))
            l_cas = len(cas)

            # Building the lists that will be used for the arrays used in the Kmeans clustering
            # and as the dataframe in pandas
            if l_cas == 1:
                cas[0] = 0
                # Frames without antinode regions do not need to be plotted
                # So no data is appended to the subject sets
            elif l_cas == 9:
                cas[0] = 1
                region1 = [frame_num, cas[8], float(cas[1]), float(cas[2])]
                coords1 = [float(cas[1]), float(cas[2]), frame_num, cas[8]]
                if subject_set == subject1:
                    if cas[8] != 0:
                        coords06240907.append(coords1)
                elif subject_set == subject2:
                    coords06240909.append(coords1)
                elif subject_set == subject3:
                    coords06240910.append(coords1)
                elif subject_set == subject4:
                    coords06241902.append(coords1)
            elif l_cas == 17:
                cas[0] = 2
                region1 = [frame_num, cas[8], float(cas[1]), float(cas[2])]
                coords1 = [float(cas[1]), float(cas[2]), frame_num, cas[8]]
                region2 = [frame_num, cas[16], float(cas[9]), float(cas[10])]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, cas[16]]
                if subject_set == subject1:
                    if cas[8] != 0:
                        coords06240907.append(coords1)
                    if cas[16] != 0:
                        coords06240907.append(coords2)
                elif subject_set == subject2:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                elif subject_set == subject3:
                    coords06240910.append(coords1)
                    coords06240910.append(coords2)
                elif subject_set == subject4:
                    coords06241902.append(coords1)
                    coords06241902.append(coords2)
            elif l_cas == 25:
                cas[0] = 3
                region1 = [frame_num, cas[8], float(cas[1]), float(cas[2])]
                coords1 = [float(cas[1]), float(cas[2]), frame_num, cas[8]]
                region2 = [frame_num, cas[16], float(cas[9]), float(cas[10])]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, cas[16]]
                region3 = [frame_num, cas[24], float(cas[17]), float(cas[18])]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, cas[24]]
                if subject_set == subject1:
                    if cas[8] != 0:
                        coords06240907.append(coords1)
                    if cas[16] != 0:
                        coords06240907.append(coords2)
                    if cas[24] != 0:
                        coords06240907.append(coords3)
                elif subject_set == subject2:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                elif subject_set == subject3:
                    coords06240910.append(coords1)
                    coords06240910.append(coords2)
                    coords06240910.append(coords3)
                elif subject_set == subject4:
                    coords06241902.append(coords1)
                    coords06241902.append(coords2)
                    coords06241902.append(coords3)
            elif l_cas == 33:
                cas[0] = 4
                region1 = [frame_num, cas[8], float(cas[1]), float(cas[2])]
                coords1 = [float(cas[1]), float(cas[2]), frame_num, cas[8]]
                region2 = [frame_num, cas[16], float(cas[9]), float(cas[10])]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, cas[16]]
                region3 = [frame_num, cas[24], float(cas[17]), float(cas[18])]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, cas[24]]
                region4 = [frame_num, cas[32], float(cas[25]), float(cas[26])]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, cas[32]]
                if subject_set == subject1:
                    if cas[8] != 0:
                        coords06240907.append(coords1)
                    if cas[16] != 0:
                        coords06240907.append(coords2)
                    if cas[24] != 0:
                        coords06240907.append(coords3)
                    if cas[32] != 0:
                        coords06240907.append(coords4)
                elif subject_set == subject2:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                    coords06240909.append(coords4)
                elif subject_set == subject3:
                    coords06240910.append(coords1)
                    coords06240910.append(coords2)
                    coords06240910.append(coords3)
                    coords06240910.append(coords4)
                elif subject_set == subject4:
                    coords06241902.append(coords1)
                    coords06241902.append(coords2)
                    coords06241902.append(coords3)
                    coords06241902.append(coords4)
            elif l_cas == 41:
                cas[0] = 5
                region1 = [frame_num, cas[8], float(cas[1]), float(cas[2])]
                coords1 = [float(cas[1]), float(cas[2]), frame_num, cas[8]]
                region2 = [frame_num, cas[16], float(cas[9]), float(cas[10])]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, cas[16]]
                region3 = [frame_num, cas[24], float(cas[17]), float(cas[18])]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, cas[24]]
                region4 = [frame_num, cas[32], float(cas[25]), float(cas[26])]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, cas[32]]
                region5 = [frame_num, cas[40], float(cas[33]), float(cas[34])]
                coords5 = [float(cas[33]), float(cas[34]), frame_num, cas[40]]
                if subject_set == subject1:
                    if cas[8] != 0:
                        coords06240907.append(coords1)
                    if cas[16] != 0:
                        coords06240907.append(coords2)
                    if cas[24] != 0:
                        coords06240907.append(coords3)
                    if cas[32] != 0:
                        coords06240907.append(coords4)
                    if cas[40] != 0:
                        coords06240907.append(coords5)
                elif subject_set == subject2:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                    coords06240909.append(coords4)
                    coords06240909.append(coords5)
                elif subject_set == subject3:
                    coords06240910.append(coords1)
                    coords06240910.append(coords2)
                    coords06240910.append(coords3)
                    coords06240910.append(coords4)
                    coords06240910.append(coords5)
                elif subject_set == subject4:
                    coords06241902.append(coords1)
                    coords06241902.append(coords2)
                    coords06241902.append(coords3)
                    coords06241902.append(coords4)
                    coords06241902.append(coords5)
            elif l_cas == 49:
                cas[0] = 6
                region1 = [frame_num, cas[8], float(cas[1]), float(cas[2])]
                coords1 = [float(cas[1]), float(cas[2]), frame_num, cas[8]]
                region2 = [frame_num, cas[16], float(cas[9]), float(cas[10])]
                coords2 = [float(cas[9]), float(cas[10]), frame_num, cas[16]]
                region3 = [frame_num, cas[24], float(cas[17]), float(cas[18])]
                coords3 = [float(cas[17]), float(cas[18]), frame_num, cas[24]]
                region4 = [frame_num, cas[32], float(cas[25]), float(cas[26])]
                coords4 = [float(cas[25]), float(cas[26]), frame_num, cas[32]]
                region5 = [frame_num, cas[40], float(cas[33]), float(cas[34])]
                coords5 = [float(cas[33]), float(cas[34]), frame_num, cas[40]]
                region6 = [frame_num, cas[48], float(cas[41]), float(cas[42])]
                coords6 = [float(cas[41]), float(cas[42]), frame_num, cas[48]]
                if subject_set == subject1:
                    if cas[8] != 0:
                        coords06240907.append(coords1)
                    if cas[16] != 0:
                        coords06240907.append(coords2)
                    if cas[24] != 0:
                        coords06240907.append(coords3)
                    if cas[32] != 0:
                        coords06240907.append(coords4)
                    if cas[40] != 0:
                        coords06240907.append(coords5)
                    if cas[48] != 0:
                        coords06240907.append(coords6)
                elif subject_set == subject2:
                    coords06240909.append(coords1)
                    coords06240909.append(coords2)
                    coords06240909.append(coords3)
                    coords06240909.append(coords4)
                    coords06240909.append(coords5)
                    coords06240909.append(coords6)
                elif subject_set == subject3:
                    coords06240910.append(coords1)
                    coords06240910.append(coords2)
                    coords06240910.append(coords3)
                    coords06240910.append(coords4)
                    coords06240910.append(coords5)
                    coords06240910.append(coords6)
                elif subject_set == subject4:
                    coords06241902.append(coords1)
                    coords06241902.append(coords2)
                    coords06241902.append(coords3)
                    coords06241902.append(coords4)
                    coords06241902.append(coords5)
                    coords06241902.append(coords6)
x_val_1 = []
y_val_1 = []
frm_1 = []
time_1 = []
frng_1 = []
for centers in coords06240907:
    x_val_1.append(centers[0])
    y_val_1.append(centers[1])
    frm_1.append(centers[2])
color_scatter_raw_1 = {'XVal': x_val_1,
                       'YVal': y_val_1,
                       'Frame': frm_1}
colored_centers_1 = pd.DataFrame(color_scatter_raw_1, columns=['XVal', 'YVal', 'Frame'])
# print (colored_centers_1)
cmap_1 = cm.ScalarMappable(
    col.Normalize(colored_centers_1.Frame.min(), colored_centers_1.Frame.max()),
    cm.BuPu)
plt.show(plt.scatter(colored_centers_1.XVal, colored_centers_1.YVal, s=20, alpha=.4)) #c=cmap_1.to_rgba(colored_centers_1.Frame)))

plt.xlabel('time (Frames)')
plt.ylabel('Amplitude (Fringes)')

for region_1 in coords06240907:
    if region_1[0]>260 and region_1[0]<330 and region_1[1]>300 and region_1[1]<370 and region_1[3]!='0':
        time_1.append(int(region_1[2]))
        frng_1.append(int(region_1[3]))
antinode_raw_1 = {'Time': time_1,
                  'Amplitude': frng_1}
antinode_1 = pd.DataFrame(antinode_raw_1, columns=['Time', 'Amplitude'])
Avg_antinode_1 = antinode_1.groupby(['Time'], as_index=False).mean()
# plt.show(plt.scatter(Avg_antinode_1.Time, Avg_antinode_1.Amplitude, s=20))

time_2 = []
frng_2 = []
for region_2 in coords06240907:
    if region_2[0]>10 and region_2[0]<150 and region_2[1]>140 and region_2[1]<305 and region_2[3]!='0':
        time_2.append(int(region_2[2]))
        frng_2.append(int(region_2[3]))
antinode_raw_2 = {'Time': time_2,
                  'Amplitude': frng_2}
antinode_2 = pd.DataFrame(antinode_raw_2, columns=['Time', 'Amplitude'])
Avg_antinode_2 = antinode_2.groupby(['Time'], as_index=False).mean()
# plt.show(plt.scatter(Avg_antinode_2.Time, Avg_antinode_2.Amplitude, s=20))

time_3 = []
frng_3 = []
for region_3 in coords06240907:
    if region_3[0]>215 and region_3[0]<276 and region_3[1]>30 and region_3[1]<72 and region_3[3]!='0':
        time_3.append(int(region_3[2]))
        frng_3.append(int(region_3[3]))
antinode_raw_3 = {'Time': time_3,
                  'Amplitude': frng_3}
antinode_3 = pd.DataFrame(antinode_raw_3, columns=['Time', 'Amplitude'])
Avg_antinode_3 = antinode_3.groupby(['Time'], as_index=False).mean()
# plt.show(plt.scatter(Avg_antinode_3.Time, Avg_antinode_3.Amplitude, s=20))

time_4 = []
frng_4 = []
for region_4 in coords06240907:
    if region_4[0]>276 and region_4[0]<335 and region_4[1]>30 and region_4[1]<85 and region_4[3]!='0':
        time_4.append(int(region_4[2]))
        frng_4.append(int(region_4[3]))
antinode_raw_4 = {'Time': time_4,
                  'Amplitude': frng_4}
antinode_4 = pd.DataFrame(antinode_raw_4, columns=['Time', 'Amplitude'])
Avg_antinode_4 = antinode_4.groupby(['Time'], as_index=False).mean()
# plt.show(plt.scatter(Avg_antinode_4.Time, Avg_antinode_4.Amplitude, s=20))

time_5 = []
frng_5 = []
for region_5 in coords06240907:
    if region_5[0]>230 and region_5[0]<337 and region_5[1]>160 and region_5[1]<250 and region_5[3]!='0':
        time_5.append(int(region_5[2]))
        frng_5.append(int(region_5[3]))
antinode_raw_5 = {'Time': time_5,
                  'Amplitude': frng_5}
antinode_5 = pd.DataFrame(antinode_raw_5, columns=['Time', 'Amplitude'])
Avg_antinode_5 = antinode_5.groupby(['Time'], as_index=False).mean()
# plt.show(plt.scatter(Avg_antinode_5.Time, Avg_antinode_5.Amplitude, s=20))

time_6 = []
frng_6 = []
for region_6 in coords06240907:
    if region_6[0] > 255 and region_6[0] < 315 and region_6[1] > 160 and region_6[1] < 230 and region_6[3] != '0':
        time_6.append(int(region_6[2]))
        frng_6.append(int(region_6[3]))
antinode_raw_6 = {'Time': time_6,
                  'Amplitude': frng_6}
antinode_6 = pd.DataFrame(antinode_raw_6, columns=['Time', 'Amplitude'])
Avg_antinode_6 = antinode_6.groupby(['Time'], as_index=False).mean()
# plt.show(plt.scatter(Avg_antinode_6.Time, Avg_antinode_6.Amplitude, s=20))

