
# coding: utf-8

# In[1]:


import tkinter as tk
#from tkinter import Frame as Fr
import tkinter.filedialog 
import pandas as pd
import os
import shutil 
import matplotlib
from matplotlib.figure import Figure



matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg


from astropy.stats import circmean
from astropy import units as u

import json
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as col
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib.patches import Ellipse
style.use('ggplot')

from itertools import filterfalse

import csv

cmap_1 = cm.ScalarMappable(col.Normalize(1, 11, cm.gist_rainbow))


# In[2]:


class App(tk.Tk):
    
    
    #this changes the sorting range
    def setSort(self, num):
    
        self.sizeMoreThan = num
        self.moreThan = str(self.sizeMoreThan)
        self.infoBox.config(text="Sorting set to classifications above " + self.moreThan )
        app.after(2000, self.csvEdit)
    
    
        
    #Initialize everything in the App class
    
    def __init__(self):       
        
        tk.Tk.__init__(self)
        frameM = tk.Frame(width=800, height=600)
        frameM.grid()
        print("frame should be drawn")
        
        
        
        
        
        
        
        
       #establish a 3*3 grid on the interface 
        frameM.grid_rowconfigure(0, weight=1)
        frameM.grid_columnconfigure(0, weight=1)
        frameM.grid_rowconfigure(1, weight=1)
        frameM.grid_columnconfigure(1, weight=1)
        frameM.grid_rowconfigure(2, weight=1)
        frameM.grid_columnconfigure(2, weight=1)
        
        frameM.grid_propagate(False)
        
      
        
        
        
        #Visual console in the interface

        self.infoBox = tk.Label(frameM, text = "Configure sorting file to begin",bd=2)
        
        self.infoBox.grid(row=0, column=1, sticky=tk.N)
        
#Buttons
        
    
        self.openButton = tk.Button(frameM, text = "Open CSV File", command=self.csvEdit,borderwidth=1)
        self.openButton.grid(row=0, column=0, sticky=tk.NW)
        
        self.confirmButton = tk.Button(frameM, text = "Graph Classifications", state = tk.DISABLED, relief = tk.SUNKEN, command=self.graph,borderwidth=1)
        self.confirmButton.grid(sticky=tk.NE, row = 0, column =2)
        
        
        self.saveButton = tk.Button(frameM, text = "Save Image", state = tk.DISABLED, relief = tk.SUNKEN, command=self.graph,borderwidth=1)
        self.saveButton.grid(sticky=tk.S, row=2,column=1)
        
        self.saveCSVButton = tk.Button(frameM, text = "save sorted CSV", state = tk.DISABLED, relief = tk.SUNKEN, command=self.saveCSV, borderwidth=1)
        self.saveCSVButton.grid(row=2, sticky=tk.SW)
        
        self.exitButton = tk.Button(frameM, text = "Quit", command=on_closing, borderwidth=1, fg = "red")
        self.exitButton.grid(column=2, sticky=tk.SE, row=2)
        
       
    
        #The dropdown which changes sorting rules
        
        self.menubar = tk.Menu(self)
        
        
        menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Configure Sort", menu=menu)
        menu.add_command(label="More than 1", command= lambda: self.setSort(1))
        menu.add_command(label="More than 2", command= lambda: self.setSort(2))
        menu.add_command(label="More than 3", command= lambda: self.setSort(3))
        menu.add_command(label="More than 4", command= lambda: self.setSort(4))
        menu.add_command(label="More than 5", command= lambda: self.setSort(5))
        menu.add_command(label="More than 6", command= lambda: self.setSort(6))
        menu.add_command(label="More than 7", command= lambda: self.setSort(7))
        menu.add_command(label="More than 8", command= lambda: self.setSort(8))
        menu.add_command(label="More than 9", command= lambda: self.setSort(9))
        menu.add_command(label="More than 10", command= lambda: self.setSort(10))
        menu.add_command(label="More than 11", command= lambda: self.setSort(11))
        menu.add_command(label="More than 12", command= lambda: self.setSort(12))
        menu.add_command(label="More than 13", command= lambda: self.setSort(13))
        menu.add_command(label="Max of 14+ ", command= lambda: self.setSort(14))
        
        self.config(menu=self.menubar)
        
        
        
        
        self.sizeMoreThan = None
        if self.sizeMoreThan is None:
            self.sizeMoreThan = 4
        
        
        
        
        self.moreThan = str(self.sizeMoreThan)
        
        
        
        
        
        
#graph canvas
        
        #The canvas that MatplotLib is integrated
        self.graphCanvas = Plot(frameM, width=700, height=475)
        
        self.graphCanvas.grid(sticky=S, column=0, row=1, columnspan=3)
        
       
        
    def graph(self):
        
        #Calls the canvas to integrate matplotlib graph
        
        print("Call MatplotLIB Class!")
        self.infoBox.config(text="Graphing data", fg = "black")
        plot = Plot()
        self.confirmButton.config(state='disabled', relief = SUNKEN)
        self.saveCSVButton.config(state='disabled', relief = SUNKEN)
        self.openButton.config(state='disabled', relief = SUNKEN)
        
    def saveCSV(self):
        
        #Saves current sorted CSV
        
        shutil.copyfile('sg_temp.csv', 'sorted-classifications-moreThan_'+ self.moreThan + '.csv')
        print("[SteelGraph] sorted CSV saved as sorted-classifications-moreThan_" + self.moreThan + ".csv")
        self.infoBox.config(text="sorted CSV saved as sorted-classifications-moreThan_" + self.moreThan + ".csv", fg="black")
        app.after(2000, self.accessGranted)
        
        
    def accessGranted(self):
        
        #Enables buttons when they are safe to trigger
        
        self.confirmButton.config(state='normal', relief = RAISED) 
        print("[SteelGraph] Ready to plot data")     
        self.saveCSVButton.config(state='normal', relief = RAISED)
        self.infoBox.config(text="Ready to Graph ", fg = "green")
        
        
        
        
    def csvEdit(self):
    
    
        #This function takes out unnecessary parts of the csv, and prepares it for graphing
        
        filename =tk.filedialog.askopenfilename(initialdir = "C:\\Users\Matt\Desktop\physics", title = "Select file",filetypes = (("csv FILES ONLY","*.csv"),))
        self.infoBox.config(text="Sorting Classifications ", fg="blacK")
        
        print("[SteelGraph] Opened: ", filename)
        #Open CSV and sort classifications
        
        App()
        df = pd.read_csv(filename)
    
        

        #Sorts the data with respect to Subject_ids
        class_counts = df['subject_ids'].value_counts().rename('class_counts')
        df = df.merge(class_counts.to_frame(), left_on='subject_ids', right_index=True)
        df2 = df[df.class_counts > self.sizeMoreThan] 
        df3 = df2.drop(df2.columns[14], axis=1) #Takes out sorting column
        df4 = df3[~df['workflow_name'].str.contains('Retirement Limit Test')]


        #Print to CSV, deleting unnecessary index
        df4.to_csv('sg_temp.csv', index = False)
        print("[SteelGraph] sorted csv created...")
        app.after(1000, self.accessGranted) 
        
        
    


# In[ ]:



    


# In[3]:


class Plot(tk.Canvas):  
    
    #Tkinter canvas class
    
    
    def df_to_center_plt(self, coords_x):
        x_val = []
        y_val = []
        frng = []
        crds = []
        ell = []

        for centers in coords_x:
            # The IndexErrors *might* have been fixed, but this has not been checked yet.
            try:
                x_val.append(centers[0])
            except IndexError:
                pass
            try:
                y_val.append(centers[1])
            except IndexError:
                pass
            try:
                frng.append(centers[3])
            except IndexError:
                pass
            try:
                crds.append([centers[0], centers[1]])
            except IndexError:
                pass
            try:
                ell.append(Ellipse(xy=[centers[0], centers[1]], width=centers[4], height=centers[5], angle=centers[6]))
            except IndexError:
                pass
        centers_raw = {'XVal': x_val,
                        'YVal': y_val,
                        'Fringe': frng}

        centers_df = pd.DataFrame(centers_raw, columns=['XVal', 'YVal', 'Fringe'])
        plt.scatter(centers_df.XVal, centers_df.YVal, s=20, c=cmap_1.to_rgba(centers_df.Fringe), alpha=.6)
        plt.xlim(0, 512)
        plt.ylim(0, 384)
        plt.title('Subject id = %s'%(coords_x[0][2]))
        self.canvas.show()
        bad_xy = self.dbscan(crds)
            #print("bad_xy = ",bad_xy)
            #for each in ell:
            #    print("ell.center = ",each.center)
            #    print("ell.angle = ",each.angle)
            #    print("ell.width = ",each.width)
            #    print("ell.height = ",each.height)
        self.draw_ellipse(ell)

            ###Filter out the bad_xy entries - we will only look at bad x coordinates and remove all entries matching the bad_x.
            ###There is probably a better way to do this, or at least do more cross checking.
        bad_x = 0
        new_coords_x = []
        keepgood = coords_x
        for i in range(len(bad_xy)):
            try:
                for j in range(len(bad_xy[i])):
                    try:

                        bad_x = bad_xy[i][j][0]
                        # bad_y is not (currently) being used.
                        bad_y = bad_xy[i][j][1]
                        #print("bad_x = ",bad_x)
                        # This list comprehension step is what removes the bad ellipse
                        keepgood = [item for item in keepgood if item[:][0] != bad_x]

                    except IndexError:
                        pass
                        #if bad_x != coords_x[i][0]:
                        #        print(coords_x[i])
                    #if bad_x == coords_x[i][0]:
                    #    print("Bad Entry!")
                    #else:
                    #    print(coords_x[i])
            except IndexError:
                pass
        averages = self.dbscan_average(keepgood)
            #print("averages = ",averages)
        average_list = []
            #print(len(averages[0]))
        if len(averages[0])>0:
            for i in range(len(averages[0])):
                average_list.append([averages[0][i],averages[1][i],keepgood[0][2],averages[2][i],averages[3][i],averages[4][i],averages[5][i]])

    
        
        
        
        
        
        
        
        
        
        
        
    def dbscan(self, crds):
        bad_xy = []  #might need to change this
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

            # These are the definitely "good" xy values.
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=14)
            #print("\n Good? xy = ",xy)
            #print("X = ",X)
            # These are the "bad" xy values. Note that some maybe-bad and maybe-good are included here.
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=6)
            #print("\n Bad? xy = ",xy)
            bad_xy.append(xy)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.xlim(0, 512)
        plt.ylim(0, 384)

        return bad_xy    





    def draw_ellipse(self, ell):
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        for e in ell:
            ax.add_artist(e)
            e.set_alpha(.3)
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 384)
        self.canvas.show()



    
            



    def dbscan_average(self, keepgood):
        db=[]
        gooddata = []

        for i in range(len(keepgood)):
            if keepgood[i][3] != None:
                try:
                    gooddata.append([float(keepgood[i][0]),float(keepgood[i][1]),int(keepgood[i][2]),int(keepgood[i][3]),float(keepgood[i][4]),float(keepgood[i][5]),float(keepgood[i][6])])
                except ValueError:
                    pass      # or whatever

            #print("gooddata = ",gooddata)
        bad_xy = []  #might need to change this
        X = np.array(gooddata)
            #print("X = ",X)
            #print("\n len(X) = ",len(X))
            #X = X[:,[0,1]]
            #try:
            #    db = DBSCAN(eps=18, min_samples=3).fit(X[:,[0,1]])
            #except IndexError:
            #    try:
            #        db = DBSCAN(eps=18, min_samples=2).fit(X[:,[0,1]])
            #    except IndexError:
            #        pass
            #    except AttributeError:
            #        pass
        if len(X)>0:
            db = DBSCAN(eps=18, min_samples=3).fit(X[:,[0,1]])

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)

            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

                #fig = plt.figure(1)
                #ax = fig.add_subplot(212, aspect='equal')
            x_mean = []
            y_mean = []
            fringe_count_mean = []
            rx_mean = []
            ry_mean = []
            angle_mean = []

            for k, col in zip(unique_labels, colors):
                if k == -1:
                        # Black used for noise.
                    col = 'k'

                x = []
                y = []
                fringe_count = []
                rx = []
                ry = []
                angle = []

                class_member_mask = (labels == k)
                    #print("class_member_mask =",class_member_mask)
                    # These are the definitely "good" xy values.
                xy = X[class_member_mask & core_samples_mask]
                    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                    #         markeredgecolor='k', markersize=14)
                    #print("k = ",k)
                for each in range(len(xy)):
                        #print("x = ",xy[each][0])
                    x.append(xy[each][0])
                        #print("y = ",xy[each][1])
                    y.append(xy[each][1])
                        #print("fringe_count = ",xy[each][3])
                    fringe_count.append(xy[each][3])
                        #print("rx = ",xy[each][4])
                    rx.append(xy[each][4])
                        #print("ry = ",xy[each][5])
                    ry.append(xy[each][5])
                        #print("angle = ",xy[each][6])
                    angle.append(xy[each][6])
                x_mean.append(np.mean(x))
                y_mean.append(np.mean(y))
                fringe_count_mean.append(np.mean(fringe_count))
                rx_mean.append(np.mean(rx))
                ry_mean.append(np.mean(ry))
                angles = np.array(angle)*u.deg
                angle_mean.append(circmean(angles).value)
                    #angle_mean = [x / 2 for x in angle_mean]
                    #print("\n Good? xy = ",xy)
                    #print("X = ",X)
                    # These are the "bad" xy values. Note that some maybe-bad and maybe-good are included here.
                xy = X[class_member_mask & ~core_samples_mask]
                    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                    #         markeredgecolor='k', markersize=6)
                    #print("\n Bad? xy = ",xy)
                bad_xy.append(xy)

            #    plt.title('From average - Estimated number of clusters: %d' % n_clusters_)
            #    plt.xlim(0, 512)
            #    plt.ylim(0, 384)
            #    print("x_mean = ",x_mean)
            #    print("y_mean = ",y_mean)
            #    print("fringe_count_mean = ",fringe_count_mean)
            #    print("rx_mean = ",rx_mean)
            #    print("ry_mean = ",ry_mean)

            #    print("angle_mean = ",angle_mean)

            averages = [x_mean,y_mean,fringe_count_mean,rx_mean,ry_mean,angle_mean]
            ell=[]
            for i in range(len(averages[0])):
                ell.append(Ellipse(xy=[x_mean[i],y_mean[i]],width=2*rx_mean[i],height=2*ry_mean[i],angle=angle_mean[i]))

            with open("all-subject-ids.csv") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                subjects = [r for r in reader]

                #print(subjects)    
            for i in range(len(subjects)):
                if subjects[i][0] == keepgood[0][2]:
                    print(subjects[i][1])
                    filetoopen = subjects[i][1]
                    img = plt.imread("./images/"+filetoopen)
                    fig, ax_new = plt.subplots(figsize=(9, 8), dpi= 72, facecolor='w', edgecolor='k')
                    ax_new.imshow(img,origin='lower',extent=[0, 512, 0, 384],cmap='gray')
                    for e in ell:
                        ax_new.add_artist(e)
                        e.set_alpha(.3)
                    ax_new.set_xlim(0, 512)
                    ax_new.set_ylim(0, 384)
                    plt.savefig("./output/"+subjects[i][0]+".png",bbox_inches='tight')
                        #plt.show()   

                #print(filetoopen)        
                #subjects.index(keepgood[0][2])
                #print(keepgood[0][2])    


        else:
            averages = [[],[],[],[],[]]

        return averages    
        
        
 ###############################################################################################333
###################################################################################################3
        

        
    def draw(self):
    # new-steelpan-vibrations-classifications.csv should be generated from csvcount routine
        with open("new-steelpan-vibrations-classifications-five_or_more-061818.csv") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            data = [r for r in reader]


        allsubjectids = []
        for i in range(len(data)):
            allsubjectids.append(int(data[i][13]))

        #Remove duplicate subject_ids
        uniquesubjectids = (set(allsubjectids))

        # Initialize empty dictionary of our subjects to be checked
        d = {}
        # Fill dictionary with keys from uniquesubjectids
        for k in uniquesubjectids:
            d['coords'+str(k)] = []


        # Parse the raw classification data for the subjects to be checked    
        for i in range(len(data)):
            parsed_json = json.loads(data[i][11])
            if data[i][5] != "Retirement Limit Test" and int(data[i][13]) in uniquesubjectids:
                if len(parsed_json)==2:
                    for j in range(len(parsed_json[1]['value'])):
                        if len(parsed_json[1]['value'][j]['details']) == 1:
                            if isinstance(parsed_json[1]['value'][j]['details'][0]['value'], str):
                                try:
                                    fringe_count = int(parsed_json[1]['value'][j]['details'][0]['value'])
                                except ValueError:
                                    pass

                            else:
                                fringe_count = parsed_json[1]['value'][j]['details'][0]['value']

                            datalist = [parsed_json[1]['value'][j]['x'],parsed_json[1]['value'][j]['y'],data[i][13],fringe_count,parsed_json[1]['value'][j]['rx'],parsed_json[1]['value'][j]['ry'],parsed_json[1]['value'][j]['angle']]

                            d['coords'+data[i][13]].append(datalist)

                elif len(parsed_json)==1:
                    # If there are no fringes recorded, fill every field except subject_id with 0 -> parse later
                    datalist = [0,0,data[i][13],0,0,0,0]
                    d['coords'+data[i][13]].append(datalist)
                    #print(data[i][13]," No antinodes found!")

        print(len(data))     

        for key, value in d.items():
                #print(len(value))
                if len(value)  >0 and value[0][0] != 'None':
                #   if len(value)  >0 and None not in value[key]:
                    print("key =",key,"value = ", value,"\n")
                    #if len(d.items()) > 0:
                #        df_to_center_plt("{}".format(key))
                    self.df_to_center_plt(value)

                    #df_to_center_plt(key)


    
    def __init__(self, *args, **kwargs):
        tk.Canvas.__init__(self, *args, **kwargs)
        
        self.create_line(0, 0, 200, 100)
        self.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))
        self.create_rectangle(0, 0, 700, 475, fill="white")
        
        
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
       # a.plot(AggregatingClassifications)
        
        self.canvas = FigureCanvasTkAgg(f, self)
        #canvas.draw()
        
        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.grid()
        self.draw()


# In[4]:


def on_closing():
    
    #Safely exits the program
    if os.path.isfile('./sorted-classifications.csv'):
            os.remove('sorted-classifications.csv')
            print("[SteelGraph] Sorted File Removed")
    else:
        print("[SteelGraph] No File Found")
    
    
    print("[SteelGraph] bye")
    app.destroy()
    
    


# In[5]:


if __name__ == "__main__":

    #The main function that runs the program
    
    app = App()
    app.resizable(0,0)
    #app.title("SteelGraph")
    app.protocol("WM_DELETE_WINDOW", on_closing)    
    app.mainloop()


# In[ ]:


#Emergency stop

#app.destroy()

