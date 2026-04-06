# B. Titanic Survival Case Study - Object Oriented

# 1. Necessary DS libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# 2. all imports from sklearn for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # ML algorithm
# All remaining classification methods
from sklearn.tree import DecisionTreeClassifier # Decision Tree or DT
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbour 
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.svm import SVC # Support Vector Machine

# 3. Evalution Metrics of all models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# 4. Save the model using pickle for future use .pkl
import joblib
import pickle

class TitanicSurvival:

    def loadDataset(self):
        # 1. Load dataset
        self.df = pd.read_csv('titanic.csv')

    def understandingDataset(self):
        # 2. basic understanding of dataset
        print('--- Shape (rows,cols) of Dataset ---')
        print(self.df.shape) # no of rows and columns - 600 rows and 5 columns
        print('--- Columns of dataset ---')
        print(self.df.columns) # print dataset columns
        print('--- Data Types of columns ---')
        print(self.df.dtypes) # data types of the columns
        print('--- Dataset Information')
        print(self.df.info()) # complete information of dataset

        # 3. first and last 5 records
        print('--- First 5 Records ---')
        print(self.df.head())
        print('--- Last 5 Records ---')
        print(self.df.tail())

    def findMissingValues(self):
        # 4. Find missing values in dataset
        print('--- Missing Values ---')
        print(self.df.isnull().sum()) # there are no missing values in Iris dataset

    def statsAnalysis(self):
        # 5. Stats Analysis
        print('--- Stats. Analysis ---')
        print(self.df.describe()) # gives complete stat analysis

    def exploratoryDataAnalysis(self):
        # 6. Exploratory data analysis
        print('--- Distribution of Categories ---')
        print(self.df['Sex'].value_counts())
        print(self.df['Survived'].value_counts())
        print(self.df['Pclass'].value_counts())
        print(self.df['Embarked'].value_counts())

    def dataVisualization(self):
        # 7. Data Visualization
        gender_label=['Male','Female']
        survived_label=['Survived','Died']
        pclass_label=['C1','C2','C3']
        embarked_label=['S','C','Q']

        gender_data=self.df['Sex'].value_counts()
        survived_data=self.df['Survived'].value_counts()
        pclass_data=self.df['Pclass'].value_counts()
        embarked_data=self.df['Embarked'].value_counts()

        plt.figure(figsize=(10,5))

        plt.suptitle('Distibution analysis on Titanic Data')
        plt.subplot(2,2,1)
        plt.grid(True)
        plt.bar(gender_label,gender_data)

        plt.subplot(2,2,2)
        plt.grid(True)
        plt.bar(survived_label,survived_data,color='red')

        plt.subplot(2,2,3)
        plt.grid(True)
        plt.bar(pclass_label,pclass_data,color='green')

        plt.subplot(2,2,4)
        plt.grid(True)
        plt.bar(embarked_label,embarked_data,color='grey')
        plt.savefig('distanalysis1.png')
        plt.show()

        plt.figure(figsize=(10,5))  
        plt.title('Boxplot on Gender')  
        sns.boxplot(x='Sex',y='Age',data=self.df)
        plt.savefig('boxplot1.png')
        plt.show()

        plt.figure(figsize=(10,5))
        plt.grid(True)
        plt.title("Histogram on the age of Pasengers")
        sns.distplot(self.df['Age'].dropna(),color='darkgreen',bins=30)
        plt.savefig('agedist1.png')
        plt.show()

        plt.figure(figsize=(10,5))
        plt.grid(True)
        p = sns.countplot(x = "Embarked", hue = "Survived", data = self.df, palette=["C1", "C0"])
        p.set_xticklabels(["Southampton","Cherbourg","Queenstown"])
        p.legend(labels = ["Deceased", "Survived"])
        p.set_title("Survival based on embarking point.")
        plt.savefig('survival.png')
        plt.show()

    def dataPreprocessing(self):
        # 8. Data Pre-processing
        # Delete the column not required to train the model [Cabin, Fare]
        self.df = self.df.drop(columns=['PassengerId','Cabin','Fare','Name','Ticket'], axis=1)
        print(self.df.info())

        # 9. Find and replace missing values
        # Numeric column missing data by mean() of the column
        # Object column missing data by mode() of the column
        print(self.df.isnull().sum())
        self.df['Age'].fillna(self.df['Age'].mean(),inplace=True)
        print()
        print(self.df.isnull().sum())

        # 10. Data transformation Categorical to Numeric - Sex and Embarked
        print(self.df.dtypes)
        self.df.replace({'Sex':{'male':0,'female':1}, \
                            'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
        print(self.df.dtypes) 

    def splitDataset(self):
        # 11. Split dataset into input and output variables
        X = self.df.drop(columns=['Survived']) # All input variables
        y = self.df['Survived'] # Output variable or labbled variable

        print(X)
        print(y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X,y, test_size=0.2, random_state=1)
    
    def allModelTraining(self):

        # 13. Comparative Analysis of all Algorithms
        models = [] # create an empty list called models
        models.append(('LR', LogisticRegression()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DT', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))

        print(models)
        # models list would be created with 5 algorithm functions
        accs = []  # create an empty list to store results of the algo.
        names = []    # create an empty list to stores names of the algo.

        for name, model in models: # name = DT, model = DT.....,.....()
            model.fit(self.X_train, self.Y_train)
            pred = model.predict(self.X_test)
            acc = round(accuracy_score(self.Y_test,pred) * 100,2)

            names.append(name) # ['LR','KNN','DT']
            accs.append(acc) # [98.33,95.90,100.00]
            print('Model Name : ',name,'Accuracy',acc,'%')

        plt.bar(names,accs)
        plt.grid(True)
        plt.title("Comparative Analysis of classification methods")
        plt.savefig('companalysis.png')
        plt.show()

    def saveModel(self):
        # 14. Save the model
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.Y_train)
        joblib.dump(model,"DT_model.pkl")

    
    def runPipeLine(self):
        self.loadDataset()
        self.understandingDataset()
        self.findMissingValues()
        self.statsAnalysis()
        self.exploratoryDataAnalysis()
        self.dataVisualization()
        self.dataPreprocessing()
        self.splitDataset()
        self.allModelTraining()
        self.saveModel()


# main program
obj = TitanicSurvival()
obj.runPipeLine()         