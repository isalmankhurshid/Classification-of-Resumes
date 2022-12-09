# Loading Libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import os
from PIL import ImageTk,Image

window = tk.Tk()
window.title('Classification of Resume')
# Set window size
window.geometry("720x480")
window.config(background="#54787d")
t = Label(window,text="Classification of Resume",
                            fg="white",bg="black",font=('Times', 24))

t.pack(fill=X);
frame = Frame(window, bg='#c6cca5',
              bd = 15)
frame2=Frame(window, bg='#c6cca5',bd=15)
frame3=Frame(window, bg='#c6cca5')
frame4=Frame(window, bg='#c6cca5',bd=15)

def open_report():
    path="E:\\report.txt"
    os.startfile(path)


def open_bar_graph():
    path = "E:\\a.png"
    im = Image.open(path)
    im.show()

def open_pie_chart():
    path = "E:\\b.png"
    im = Image.open(path)
    im.show()

def browseFiles():
    global file_path
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    filename = filedialog.askopenfilename(initialdir=desktop,
                                          title="Select a File",
                                          filetypes=(("CSV files",
                                                      "*.Csv*"),
                                                     ("All files",
                                                      "*.*")))
    file_path=filename
    resume_dataset.configure(text="File Selected: " + filename)


def analyze():
    global report,classes
    if file_path!="":
        resumeDataSet = pd.read_csv(r"C:\Users\SalMan\OneDrive\Desktop\MCA\Sem 3\CS-437 NLP\Project\UpdatedResumeDataSet.csv", encoding='utf-8')
        # EDA
        plt.figure(figsize=(15, 15))
        plt.xticks(rotation=90)
        sns.countplot(y="Category", data=resumeDataSet)
        plt.savefig('E:/a.png')
        # Pie-chart
        targetCounts = resumeDataSet['Category'].value_counts().reset_index()['Category']
        targetLabels = resumeDataSet['Category'].value_counts().reset_index()['index']
        # Make square figures and axes
        plt.figure(1, figsize=(25, 25))
        the_grid = GridSpec(2, 2)
        plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
        source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, )
        plt.savefig('E:b.png')

        # imageWindow

        # Data Preprocessing
        def cleanResume(resumeText):
            resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
            resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
            resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
            resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
            resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ',
                                resumeText)  # remove punctuations
            resumeText = re.sub(r'[^x00-x7f]', r' ', resumeText)
            resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
            return resumeText

        resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
        var_mod = ['Category']
        le = LabelEncoder()
        for i in var_mod:
            resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
        requiredText = resumeDataSet['cleaned_resume'].values
        requiredTarget = resumeDataSet['Category'].values
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english',
            max_features=1500)
        word_vectorizer.fit(requiredText)
        WordFeatures = word_vectorizer.transform(requiredText)
        # Model Building
        X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
        print(X_train.shape)
        print(X_test.shape)
        clf = OneVsRestClassifier(KNeighborsClassifier())
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        # Results

        # print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
        # print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
        # print("\n Classification report for classifier %s:\nn%sn" % (clf, metrics.classification_report(y_test, prediction)))
        # report=metrics.classification_report(y_test, prediction)
        classes=le.classes_
        f = open("E:\\report.txt", "w+")
        f.write("Training data: "+str(X_train.shape)+"\n")
        f.write("Validation date: "+str(X_test.shape)+"\n\n")
        f.write('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)) + "\n")
        f.write('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)) + "\n\n")
        f.write("Classification report for classifier %s:\nClasses%s" % (clf, metrics.classification_report(y_test, prediction)) + "\n\n")
        i=0
        for element in le.classes_:
            f.write(str(i)+" --> "+element+"\n")
            i+=1
        f.close()
        messagebox.showinfo("Result",'Done')
    else:
        messagebox.showerror("showerror", "Select a File")

button_explore = Button(frame,
                        text="Choose File",
                        command=browseFiles,width=15)


button_bar = Button(frame2,
                        text="Bar Graph",
                        command=open_bar_graph,width=15)

button_pie = Button(frame2,
                        text="Pie Chart",
                        command=open_pie_chart,width=15)
button_report = Button(frame2,
                        text="Report",
                        command=open_report,width=15)
button_analyze = Button(frame3,
                        text="Analyze",
                        command=analyze,width=15,padx=10,pady=10)

resume_dataset = Label(frame, text="Select Resume Dataset",fg="black",bg="#c6cca5",font=('Helvetica', 10, 'bold'))
file_path=""
report=""
classes=""
button_explore.grid(column=1,row=1,sticky='w', padx=20, pady=20);
resume_dataset.grid(column=2,row=1,sticky='w')
button_analyze.grid(column=1,row=2,sticky='e')


result = Label(frame2, text="Results",fg="black",bg="#c6cca5",font=('Helvetica', 14, 'bold'))
result.grid(column=2,row=3 )
button_bar.grid(column=1,row=4 ,padx=20, pady=20)
button_pie.grid(column=2,row=4)
button_report.grid(column=3,row=4,padx=20, pady=20)

frame.pack(expand=True)
frame3.pack(side=TOP)
frame2.pack(expand=True)
# Loading Data
window.mainloop()