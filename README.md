<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
# Importing Libraries
import pandas as pd                                                 
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read the dataset from drive
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
df.head()

# Finding Missing Values
df.isnull().sum()

# Check For Duplicates
df.duplicated().sum()

# Remove Unnecessary Columns
df=df.drop(['Surname', 'Geography','Gender'], axis=1)

# Normalize the dataset                                         
scaler=StandardScaler()                                             
df=pd.DataFrame(scaler.fit_transform(df))
df.head()

# Split the dataset into input and output
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values
print('Input:\n',X,'\nOutput:\n',Y)

# Splitting the data for training & Testing
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)

# X Train and Test
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)

# Y Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)
```
## OUTPUT:
### Dataset:
![Screenshot 2024-02-25 172812](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/1bd8e7fb-4c0c-4a29-a5c8-16eb6b7cc330)

### Null Values:
![Screenshot 2024-02-25 172838](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/76fdcfa0-0556-48c3-988f-935c50e6911a)

### Normalized:
![Screenshot 2024-02-25 172859](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/242e5515-c64e-49bb-b1f2-41731b981666)

### Data Splitting:
![Screenshot 2024-02-25 172924](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/3c9cc008-8e76-4457-a1a9-054bd7ab99d2)
![Screenshot 2024-02-25 172940](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/e5e363f3-11d9-4f34-894a-f83abbd10457)

### Train Data:
![Screenshot 2024-02-25 173013](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/c19d8d91-f7d1-4ccc-a764-2f3e17fc9d09)
![Screenshot 2024-02-25 173043](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/f22565ff-bded-4f30-84f6-88252c7a6887)

### Test Data:
![Screenshot 2024-02-25 173027](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/768d194b-c554-477b-80d8-3e35c7e90ba6)
![Screenshot 2024-02-25 173101](https://github.com/vasundrasriravi/Ex-1-NN/assets/119393983/3f513c2c-233b-4142-84c3-f0776543260e)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
