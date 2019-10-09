

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
%matplotlib inline
```

## Reading in the dataset


```python
# read in the dataset and preview it
full_dataset=pd.read_csv("train.csv")
full_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeNo</th>
      <th>Division</th>
      <th>Qualification</th>
      <th>Gender</th>
      <th>Channel_of_Recruitment</th>
      <th>Trainings_Attended</th>
      <th>Year_of_birth</th>
      <th>Last_performance_score</th>
      <th>Year_of_recruitment</th>
      <th>Targets_met</th>
      <th>Previous_Award</th>
      <th>Training_score_average</th>
      <th>State_Of_Origin</th>
      <th>Foreign_schooled</th>
      <th>Marital_Status</th>
      <th>Past_Disciplinary_Action</th>
      <th>Previous_IntraDepartmental_Movement</th>
      <th>No_of_previous_employers</th>
      <th>Promoted_or_Not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YAK/S/00001</td>
      <td>Commercial Sales and Marketing</td>
      <td>MSc, MBA and PhD</td>
      <td>Female</td>
      <td>Direct Internal process</td>
      <td>2</td>
      <td>1986</td>
      <td>12.5</td>
      <td>2011</td>
      <td>1</td>
      <td>0</td>
      <td>41</td>
      <td>ANAMBRA</td>
      <td>No</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YAK/S/00002</td>
      <td>Customer Support and Field Operations</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Agency and others</td>
      <td>2</td>
      <td>1991</td>
      <td>12.5</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>ANAMBRA</td>
      <td>Yes</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YAK/S/00003</td>
      <td>Commercial Sales and Marketing</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Direct Internal process</td>
      <td>2</td>
      <td>1987</td>
      <td>7.5</td>
      <td>2012</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>KATSINA</td>
      <td>Yes</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>YAK/S/00004</td>
      <td>Commercial Sales and Marketing</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Agency and others</td>
      <td>3</td>
      <td>1982</td>
      <td>2.5</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>NIGER</td>
      <td>Yes</td>
      <td>Single</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YAK/S/00006</td>
      <td>Information and Strategy</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Direct Internal process</td>
      <td>3</td>
      <td>1990</td>
      <td>7.5</td>
      <td>2012</td>
      <td>0</td>
      <td>0</td>
      <td>77</td>
      <td>AKWA IBOM</td>
      <td>Yes</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Slight Exploratory Data Analysis of the data to know what we are working with Plus comments


```python
# get a general feel of the data
full_dataset.info()
# note that the 'qualification' has some missing data
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 38312 entries, 0 to 38311
    Data columns (total 19 columns):
    EmployeeNo                             38312 non-null object
    Division                               38312 non-null object
    Qualification                          36633 non-null object
    Gender                                 38312 non-null object
    Channel_of_Recruitment                 38312 non-null object
    Trainings_Attended                     38312 non-null int64
    Year_of_birth                          38312 non-null int64
    Last_performance_score                 38312 non-null float64
    Year_of_recruitment                    38312 non-null int64
    Targets_met                            38312 non-null int64
    Previous_Award                         38312 non-null int64
    Training_score_average                 38312 non-null int64
    State_Of_Origin                        38312 non-null object
    Foreign_schooled                       38312 non-null object
    Marital_Status                         38312 non-null object
    Past_Disciplinary_Action               38312 non-null object
    Previous_IntraDepartmental_Movement    38312 non-null object
    No_of_previous_employers               38312 non-null object
    Promoted_or_Not                        38312 non-null int64
    dtypes: float64(1), int64(7), object(11)
    memory usage: 5.6+ MB
    


```python
# visualise some featues that are highly correlated with each other, on the numerical variables
sns.heatmap(full_dataset.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27761a31a90>




![png](output_6_1.png)



```python
# visualise the buckets to see the distribution of the numerical data
full_dataset.hist(bins=50,figsize=(30,25))
# side-note convert the ages to respective buckets, there is an outlier that is older than 1960
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000027761DBF8D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027761E08B00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027761E450B8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000027761E75668>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027761EA8C18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027761EE7208>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000027761F177B8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027761F48DA0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027761F48DD8>]],
          dtype=object)




![png](output_7_1.png)



```python
# get a top-bottom statistical insight into the entire dataset
full_dataset.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Trainings_Attended</th>
      <td>38312.0</td>
      <td>2.253680</td>
      <td>0.609443</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Year_of_birth</th>
      <td>38312.0</td>
      <td>1986.209334</td>
      <td>7.646047</td>
      <td>1950.0</td>
      <td>1982.0</td>
      <td>1988.0</td>
      <td>1992.0</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>Last_performance_score</th>
      <td>38312.0</td>
      <td>7.698959</td>
      <td>3.744135</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>7.5</td>
      <td>10.0</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>Year_of_recruitment</th>
      <td>38312.0</td>
      <td>2013.139695</td>
      <td>4.261451</td>
      <td>1982.0</td>
      <td>2012.0</td>
      <td>2014.0</td>
      <td>2016.0</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>Targets_met</th>
      <td>38312.0</td>
      <td>0.352996</td>
      <td>0.477908</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Previous_Award</th>
      <td>38312.0</td>
      <td>0.023152</td>
      <td>0.150388</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Training_score_average</th>
      <td>38312.0</td>
      <td>55.366465</td>
      <td>13.362741</td>
      <td>31.0</td>
      <td>43.0</td>
      <td>52.0</td>
      <td>68.0</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>Promoted_or_Not</th>
      <td>38312.0</td>
      <td>0.084595</td>
      <td>0.278282</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check out the people who were born before 1960, since they are so few(12), we then turn them into category "people < 1960"
# the age is also normally distributed/slightly tail heavy,so there is no need for much transformation.
full_dataset[full_dataset['Year_of_birth']<1960].shape[0]
# min 1950   max 2001
```




    12




```python
full_dataset['No_of_previous_employers'].value_counts()
```




    1              18867
    0              13272
    2               1918
    3               1587
    4               1324
    5                943
    More than 5      401
    Name: No_of_previous_employers, dtype: int64



## Indepth Exploratory data analysis and Sidenotes


```python
full_dataset.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeNo</th>
      <th>Division</th>
      <th>Qualification</th>
      <th>Gender</th>
      <th>Channel_of_Recruitment</th>
      <th>Trainings_Attended</th>
      <th>Year_of_birth</th>
      <th>Last_performance_score</th>
      <th>Year_of_recruitment</th>
      <th>Targets_met</th>
      <th>Previous_Award</th>
      <th>Training_score_average</th>
      <th>State_Of_Origin</th>
      <th>Foreign_schooled</th>
      <th>Marital_Status</th>
      <th>Past_Disciplinary_Action</th>
      <th>Previous_IntraDepartmental_Movement</th>
      <th>No_of_previous_employers</th>
      <th>Promoted_or_Not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YAK/S/00001</td>
      <td>Commercial Sales and Marketing</td>
      <td>MSc, MBA and PhD</td>
      <td>Female</td>
      <td>Direct Internal process</td>
      <td>2</td>
      <td>1986</td>
      <td>12.5</td>
      <td>2011</td>
      <td>1</td>
      <td>0</td>
      <td>41</td>
      <td>ANAMBRA</td>
      <td>No</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YAK/S/00002</td>
      <td>Customer Support and Field Operations</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Agency and others</td>
      <td>2</td>
      <td>1991</td>
      <td>12.5</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>ANAMBRA</td>
      <td>Yes</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_dataset['Marital_Status'].value_counts()
# turn any not-sure into Single to prevent our algorithm from learning garbage
```




    Married     31022
    Single       6927
    Not_Sure      363
    Name: Marital_Status, dtype: int64




```python
full_dataset['State_Of_Origin'].value_counts()
# i should probably map state of each state to the three major tribes to know if they are yoruba, igbo or hausa
```




    LAGOS          6204
    FCT            2389
    OGUN           2302
    RIVERS         2222
    ANAMBRA        1710
    KANO           1649
    DELTA          1594
    OYO            1508
    KADUNA         1399
    IMO            1307
    EDO            1259
    ENUGU          1025
    ABIA            950
    OSUN            929
    ONDO            875
    NIGER           857
    KWARA           765
    PLATEAU         739
    AKWA IBOM       673
    NASSARAWA       632
    KATSINA         615
    ADAMAWA         605
    BENUE           579
    BAUCHI          557
    KOGI            542
    SOKOTO          499
    CROSS RIVER     495
    EKITI           455
    BORNO           447
    TARABA          400
    KEBBI           393
    BAYELSA         324
    EBONYI          313
    GOMBE           291
    ZAMFARA         290
    JIGAWA          262
    YOBE            257
    Name: State_Of_Origin, dtype: int64




```python
full_dataset['Trainings_Attended'].value_counts()
# since greater than 5 training is pretty few, i should consider making it anything>5 = 5
```




    2     30981
    3      5631
    4      1244
    5       316
    6        93
    7        28
    8         6
    10        5
    11        4
    9         4
    Name: Trainings_Attended, dtype: int64




```python
full_dataset['Channel_of_Recruitment'].value_counts()
# nothing here too. we just categorically encode them
```




    Agency and others                  21310
    Direct Internal process            16194
    Referral and Special candidates      808
    Name: Channel_of_Recruitment, dtype: int64




```python
full_dataset['Gender'].value_counts()
# genders are fair, i guess. nothing much to see here, we just simply categorically encode it
```




    Male      26880
    Female    11432
    Name: Gender, dtype: int64




```python
full_dataset['Qualification'].value_counts()
# this seems pretty distributed too, but we categorically encode them wrt their level 3,2,1
# fill in missing values with Non-University Education
```




    First Degree or HND         25578
    MSc, MBA and PhD            10469
    Non-University Education      586
    Name: Qualification, dtype: int64




```python
full_dataset['Division'].value_counts()
# pretty well distributed, no further engineering needed, simply categorically encoding them
```




    Commercial Sales and Marketing                 11695
    Customer Support and Field Operations           7973
    Sourcing and Purchasing                         5052
    Information Technology and Solution Support     4952
    Information and Strategy                        3721
    Business Finance Operations                     1786
    People/HR Management                            1704
    Regulatory and Legal services                    733
    Research and Innovation                          696
    Name: Division, dtype: int64




```python
# note that the dataset it gr
full_dataset['Promoted_or_Not'].value_counts()
```




    0    35071
    1     3241
    Name: Promoted_or_Not, dtype: int64




```python

```

## Data Spliting

#### Split into Train,Testing and validation set (60%,25%,15%)


```python
X=full_dataset.drop(['Promoted_or_Not'],1)
y=full_dataset['Promoted_or_Not']
```


```python
# split into train and test/validation
X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.4, random_state=42)

# split into test and validation
X_test,X_validation,y_test,y_validation=train_test_split(X_test_valid,y_test_valid,test_size=0.375)
```


```python
# size of my batches of data
print("train size of rows:{} and columns:{}".format(*X_train.shape))
print("train size of rows:{} and columns:{}".format(*X_test.shape))
print("train size of rows:{} and columns:{}".format(*X_validation.shape))
```

    train size of rows:22987 and columns:18
    train size of rows:9578 and columns:18
    train size of rows:5747 and columns:18
    


```python
pd.value_counts(full_dataset["Qualification"])
```




    First Degree or HND         25578
    MSc, MBA and PhD            10469
    Non-University Education      586
    Name: Qualification, dtype: int64




```python
# visualise the average number of people from different qualification whp were promoted
full_dataset.groupby("Qualification")['Promoted_or_Not'].mean().plot(kind="bar")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x277634625c0>




![png](output_28_1.png)



```python
full_dataset.groupby("Marital_Status")['Promoted_or_Not'].mean().plot(kind="bar")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27762dd04a8>




![png](output_29_1.png)



```python
full_dataset.groupby("Past_Disciplinary_Action")['Promoted_or_Not'].mean().plot(kind="bar")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27763436ba8>




![png](output_30_1.png)


## Custom mappings we can use to transform our data


```python

```


```python
# create a mappinng function which returns a dictionary mapping of the average values of that category which are promoted
# this was developed out of sheer desperation to try and boost my score :/

def mapCategoricalToAverage(name):
    mapping=dict(full_dataset.groupby(name)['Promoted_or_Not'].mean())
    return lambda x: mapping[x]

qualification_num=input_data['Qualification'].map(mapCategoricalToAverage("Qualification")).values
foreign_schooled_num=input_data['Foreign_schooled'].map(mapCategoricalToAverage("Foreign_schooled")).values
```

## Specify my data transformations


```python
from sklearn.base import BaseEstimator,TransformerMixin
full_dataset.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeNo</th>
      <th>Division</th>
      <th>Qualification</th>
      <th>Gender</th>
      <th>Channel_of_Recruitment</th>
      <th>Trainings_Attended</th>
      <th>Year_of_birth</th>
      <th>Last_performance_score</th>
      <th>Year_of_recruitment</th>
      <th>Targets_met</th>
      <th>Previous_Award</th>
      <th>Training_score_average</th>
      <th>State_Of_Origin</th>
      <th>Foreign_schooled</th>
      <th>Marital_Status</th>
      <th>Past_Disciplinary_Action</th>
      <th>Previous_IntraDepartmental_Movement</th>
      <th>No_of_previous_employers</th>
      <th>Promoted_or_Not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YAK/S/00001</td>
      <td>Commercial Sales and Marketing</td>
      <td>MSc, MBA and PhD</td>
      <td>Female</td>
      <td>Direct Internal process</td>
      <td>2</td>
      <td>1986</td>
      <td>12.5</td>
      <td>2011</td>
      <td>1</td>
      <td>0</td>
      <td>41</td>
      <td>ANAMBRA</td>
      <td>No</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>YAK/S/00002</td>
      <td>Customer Support and Field Operations</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Agency and others</td>
      <td>2</td>
      <td>1991</td>
      <td>12.5</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>ANAMBRA</td>
      <td>Yes</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YAK/S/00003</td>
      <td>Commercial Sales and Marketing</td>
      <td>First Degree or HND</td>
      <td>Male</td>
      <td>Direct Internal process</td>
      <td>2</td>
      <td>1987</td>
      <td>7.5</td>
      <td>2012</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>KATSINA</td>
      <td>Yes</td>
      <td>Married</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# specify the numeric columns we dont want to transform
numeric_columns=['Trainings_Attended', 'Year_of_birth', 'Last_performance_score', 'Year_of_recruitment', 'Targets_met', 'Previous_Award', 'Training_score_average']
class myTransformer():
    def __init__(self):
        self.division_binarizer=LabelBinarizer()
        self.channel_binariser=LabelBinarizer()
        self.tribe_binariser=LabelBinarizer()
        self.dataScaler=StandardScaler()
    
    # Fit all the binarisers on the training data
    def fit(self,input_data):
        self.division_binarizer=self.division_binarizer.fit(input_data['Division'])
        self.channel_binariser=self.channel_binariser.fit(input_data['Channel_of_Recruitment'])
        self.tribe_binariser=self.tribe_binariser.fit(input_data['State_Of_Origin'])
        
    # Transform the input data using the fitted binarisers
    def transform(self,full_dataset,train=False):
        #     making a copy of the input because we dont want to change the input in the main function
        input_data=full_dataset.copy()
        # label binarise the dvision
        division_binarised=self.division_binarizer.transform(input_data['Division'])

        # categorise the qualifications
        input_data['Qualification']=input_data['Qualification'].fillna("Non-University Education")
        qualification_num=input_data['Qualification'].map(mapCategoricalToAverage("Qualification")).values

        # categorise the gender
        gender_num=input_data["Gender"].map(mapCategoricalToAverage("Gender")).values

        # binarise the channel
        channel_binarised=self.channel_binariser.transform(input_data['Channel_of_Recruitment'])

        # map state of origin to tribe and binarise it
        state_binarised=self.tribe_binariser.transform(input_data['State_Of_Origin'])

        # map foreign schooled
        foreign_schooled_num=input_data['Foreign_schooled'].map(mapCategoricalToAverage("Foreign_schooled")).values

        # map marital status
#         marital_status_num=input_data['Marital_Status'].map(lambda x: "Single" if x=="Not_Sure" else x).map(maritalMap).values

        # map past disciplinary actions
        past_discipline_num=input_data['Past_Disciplinary_Action'].map(mapCategoricalToAverage("Past_Disciplinary_Action")).values

        # map interdep movement
        interdep_movement_num=input_data['Previous_IntraDepartmental_Movement'].map(mapCategoricalToAverage("Previous_IntraDepartmental_Movement")).values

        # map employer
        previous_employer_count=input_data['No_of_previous_employers'].map(mapCategoricalToAverage("No_of_previous_employers")).values

        numeric_data=input_data[numeric_columns].values
        
        # Create new variables
        qualification_times_scoreavg=(qualification_num*input_data['Training_score_average']).values
        department_times_scoreavg=(input_data['Division'].map(dept_to_number) * input_data['Training_score_average']).values
        department_in_number=input_data['Division'].map(dept_to_number)

        
        # this concatenates all the data
        fully_transformed=np.c_[qualification_times_scoreavg,department_times_scoreavg,department_in_number,division_binarised,qualification_num,gender_num,channel_binarised,state_binarised,foreign_schooled_num,past_discipline_num,interdep_movement_num,previous_employer_count,numeric_data]
#         fully_transformed=np.c_[division_binarised,qualification_num,gender_num,channel_binarised,state_binarised,foreign_schooled_num,marital_status_num,past_discipline_num,interdep_movement_num,previous_employer_count,numeric_data]
        return fully_transformed
```


```python

```

## Spliting the training and testing and validation data


```python
input_dataset=full_dataset.drop("Promoted_or_Not",1)
output_dataset=full_dataset["Promoted_or_Not"]
```


```python
x_train,x_test_valid,y_train,y_test_valid=train_test_split(input_dataset,output_dataset,test_size=0.35,random_state=42)
x_test,x_valid,y_test,y_valid=train_test_split(x_test_valid,y_test_valid,test_size=0.3,random_state=42)
```


```python

```

## Transforming the train_set and the test_set and validation_set


```python
transformer=myTransformer()
transformer.fit(x_train)
```


```python
transformed_x_train=transformer.transform(x_train,train=True)
transformed_x_test=transformer.transform(x_test)
transformed_x_valid=transformer.transform(x_valid)
```


```python
pd.value_counts(y_train)
```




    0    22785
    1     2117
    Name: Promoted_or_Not, dtype: int64



#### the ratio of 10 to 0 is so imbalanced, we need to balance them by augumenting the data


```python
transformed_x_train.shape
```




    (24902, 65)



## Augumenting the imbalanced dataset


```python
# use adasyn to balance the dataset
def makeOverSamplesADASYN(X,y):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
     from imblearn.over_sampling import ADASYN 
     sm = ADASYN()
     X,y = sm.fit_sample(X, y)
     return(X,y)
    
balanced_x_train,balanced_y_train=makeOverSamplesADASYN(transformed_x_train,y_train)
```

    Using TensorFlow backend.
    


```python
pd.value_counts(balanced_y_train)
```




    1    23159
    0    22785
    dtype: int64



#### Finally, our training set is now balanced as the values of 1 and 0 are almost equal


```python

```

## Finally training the model

#### using xg-boost as my base model


```python
from xgboost import XGBClassifier
from sklearn import metrics
```


```python
clf = XGBClassifier(base_score=0.7,booster="dart",n_estimators=3000,
                    max_depth=8,learning_rate=0.01,objective='binary:logistic',subsample=0.9,reg_lambda=0.03)
eval_set  = [(transformed_x_train,y_train), (transformed_x_test,y_test)]
#.900057
# clf.fit(transformed_x_train, y_train, eval_set=eval_set,eval_metric="auc", early_stopping_rounds=200)
```

#### testing the xgboost model


```python
y_train_pred = clf.predict(transformed_x_train)
# how did our model perform on the train set?
count_misclassified = (y_train != y_train_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_train_pred, y_train)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 1324
    Accuracy: 0.95
    


```python
y_valid_pred = clf.predict(transformed_x_valid)
# how did our model perform on the test set?
count_misclassified = (y_valid != y_valid_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_valid_pred, y_valid)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 251
    Accuracy: 0.94
    


```python

```


```python

```

## checking out lgmboost


```python
import lightgbm 
lightgbm.LGBMClassifier()
```




    LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                   importance_type='split', learning_rate=0.1, max_depth=-1,
                   min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                   n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                   random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                   subsample=1.0, subsample_for_bin=200000, subsample_freq=0)




```python
gbm = lightgbm.LGBMClassifier(boosting_type="dart",n_estimators=10000,learning_rate=0.01,num_leaves = 15,max_depth=6,subsample=0.9
                              ,colsample_bytree=0.3,reg_lambda=0.9,early_stopping_rounds=50)
```


```python
gbm.fit(balanced_x_train,balanced_y_train,eval_metric='auc',eval_set=[(transformed_x_test, y_test)],early_stopping_rounds=10000)
```

    C:\Users\Admin\Anaconda3\lib\site-packages\lightgbm\engine.py:123: UserWarning: Found `early_stopping_rounds` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    C:\Users\Admin\Anaconda3\lib\site-packages\lightgbm\callback.py:189: UserWarning: Early stopping is not available in dart mode
      warnings.warn('Early stopping is not available in dart mode')
    

    [1]	valid_0's auc: 0.648033	valid_0's binary_logloss: 0.696119
    [2]	valid_0's auc: 0.628972	valid_0's binary_logloss: 0.691835
    [3]	valid_0's auc: 0.67337	valid_0's binary_logloss: 0.688189
    [4]	valid_0's auc: 0.700827	valid_0's binary_logloss: 0.684839
    [5]	valid_0's auc: 0.697513	valid_0's binary_logloss: 0.681222
    [6]	valid_0's auc: 0.696789	valid_0's binary_logloss: 0.67745
    [7]	valid_0's auc: 0.767384	valid_0's binary_logloss: 0.672668
    [8]	valid_0's auc: 0.786281	valid_0's binary_logloss: 0.673566
    [9]	valid_0's auc: 0.778618	valid_0's binary_logloss: 0.669788
    [10]	valid_0's auc: 0.783175	valid_0's binary_logloss: 0.667282
    [11]	valid_0's auc: 0.775282	valid_0's binary_logloss: 0.663691
    [12]	valid_0's auc: 0.782378	valid_0's binary_logloss: 0.665207
    [13]	valid_0's auc: 0.776653	valid_0's binary_logloss: 0.662109
    [14]	valid_0's auc: 0.772848	valid_0's binary_logloss: 0.658787
    [15]	valid_0's auc: 0.777655	valid_0's binary_logloss: 0.65484
    [16]	valid_0's auc: 0.782634	valid_0's binary_logloss: 0.651615
    [17]	valid_0's auc: 0.776717	valid_0's binary_logloss: 0.648081
    [18]	valid_0's auc: 0.77416	valid_0's binary_logloss: 0.645519
    [19]	valid_0's auc: 0.795601	valid_0's binary_logloss: 0.641013
    [20]	valid_0's auc: 0.805823	valid_0's binary_logloss: 0.637087
    [21]	valid_0's auc: 0.805497	valid_0's binary_logloss: 0.63767
    [22]	valid_0's auc: 0.80415	valid_0's binary_logloss: 0.633549
    [23]	valid_0's auc: 0.806034	valid_0's binary_logloss: 0.630276
    [24]	valid_0's auc: 0.807206	valid_0's binary_logloss: 0.627562
    [25]	valid_0's auc: 0.807087	valid_0's binary_logloss: 0.624357
    [26]	valid_0's auc: 0.8354	valid_0's binary_logloss: 0.61966
    [27]	valid_0's auc: 0.832961	valid_0's binary_logloss: 0.616409
    [28]	valid_0's auc: 0.832804	valid_0's binary_logloss: 0.617565
    [29]	valid_0's auc: 0.831856	valid_0's binary_logloss: 0.614612
    [30]	valid_0's auc: 0.838063	valid_0's binary_logloss: 0.610294
    [31]	valid_0's auc: 0.835783	valid_0's binary_logloss: 0.610286
    [32]	valid_0's auc: 0.835325	valid_0's binary_logloss: 0.606281
    [33]	valid_0's auc: 0.835427	valid_0's binary_logloss: 0.603674
    [34]	valid_0's auc: 0.833958	valid_0's binary_logloss: 0.599797
    [35]	valid_0's auc: 0.834301	valid_0's binary_logloss: 0.601419
    [36]	valid_0's auc: 0.834062	valid_0's binary_logloss: 0.603258
    [37]	valid_0's auc: 0.83433	valid_0's binary_logloss: 0.600677
    [38]	valid_0's auc: 0.833163	valid_0's binary_logloss: 0.597675
    [39]	valid_0's auc: 0.842367	valid_0's binary_logloss: 0.593304
    [40]	valid_0's auc: 0.842074	valid_0's binary_logloss: 0.593087
    [41]	valid_0's auc: 0.842812	valid_0's binary_logloss: 0.594903
    [42]	valid_0's auc: 0.841298	valid_0's binary_logloss: 0.591963
    [43]	valid_0's auc: 0.840062	valid_0's binary_logloss: 0.593688
    [44]	valid_0's auc: 0.839001	valid_0's binary_logloss: 0.590267
    [45]	valid_0's auc: 0.840309	valid_0's binary_logloss: 0.58731
    [46]	valid_0's auc: 0.840397	valid_0's binary_logloss: 0.588606
    [47]	valid_0's auc: 0.842323	valid_0's binary_logloss: 0.584899
    [48]	valid_0's auc: 0.844359	valid_0's binary_logloss: 0.584119
    [49]	valid_0's auc: 0.844533	valid_0's binary_logloss: 0.585478
    [50]	valid_0's auc: 0.844105	valid_0's binary_logloss: 0.588023
    [51]	valid_0's auc: 0.847378	valid_0's binary_logloss: 0.584043
    [52]	valid_0's auc: 0.849856	valid_0's binary_logloss: 0.581797
    [53]	valid_0's auc: 0.849574	valid_0's binary_logloss: 0.58314
    [54]	valid_0's auc: 0.848121	valid_0's binary_logloss: 0.579468
    [55]	valid_0's auc: 0.847573	valid_0's binary_logloss: 0.576535
    [56]	valid_0's auc: 0.847325	valid_0's binary_logloss: 0.578438
    [57]	valid_0's auc: 0.845594	valid_0's binary_logloss: 0.574816
    [58]	valid_0's auc: 0.846129	valid_0's binary_logloss: 0.576065
    [59]	valid_0's auc: 0.846033	valid_0's binary_logloss: 0.577762
    [60]	valid_0's auc: 0.842922	valid_0's binary_logloss: 0.574356
    [61]	valid_0's auc: 0.84357	valid_0's binary_logloss: 0.575203
    [62]	valid_0's auc: 0.843351	valid_0's binary_logloss: 0.573266
    [63]	valid_0's auc: 0.843253	valid_0's binary_logloss: 0.570407
    [64]	valid_0's auc: 0.844705	valid_0's binary_logloss: 0.571959
    [65]	valid_0's auc: 0.844051	valid_0's binary_logloss: 0.573785
    [66]	valid_0's auc: 0.84397	valid_0's binary_logloss: 0.571164
    [67]	valid_0's auc: 0.845661	valid_0's binary_logloss: 0.569763
    [68]	valid_0's auc: 0.843749	valid_0's binary_logloss: 0.566341
    [69]	valid_0's auc: 0.844062	valid_0's binary_logloss: 0.567915
    [70]	valid_0's auc: 0.844955	valid_0's binary_logloss: 0.569184
    [71]	valid_0's auc: 0.844523	valid_0's binary_logloss: 0.570025
    [72]	valid_0's auc: 0.845107	valid_0's binary_logloss: 0.568026
    [73]	valid_0's auc: 0.848304	valid_0's binary_logloss: 0.565169
    [74]	valid_0's auc: 0.848114	valid_0's binary_logloss: 0.566623
    [75]	valid_0's auc: 0.845478	valid_0's binary_logloss: 0.563256
    [76]	valid_0's auc: 0.845093	valid_0's binary_logloss: 0.563206
    [77]	valid_0's auc: 0.845204	valid_0's binary_logloss: 0.564243
    [78]	valid_0's auc: 0.845349	valid_0's binary_logloss: 0.565325
    [79]	valid_0's auc: 0.847818	valid_0's binary_logloss: 0.56338
    [80]	valid_0's auc: 0.848969	valid_0's binary_logloss: 0.560391
    [81]	valid_0's auc: 0.849462	valid_0's binary_logloss: 0.561599
    [82]	valid_0's auc: 0.849291	valid_0's binary_logloss: 0.559848
    [83]	valid_0's auc: 0.850011	valid_0's binary_logloss: 0.561282
    [84]	valid_0's auc: 0.850079	valid_0's binary_logloss: 0.563359
    [85]	valid_0's auc: 0.849809	valid_0's binary_logloss: 0.564638
    [86]	valid_0's auc: 0.850426	valid_0's binary_logloss: 0.561949
    [87]	valid_0's auc: 0.847573	valid_0's binary_logloss: 0.558802
    [88]	valid_0's auc: 0.847044	valid_0's binary_logloss: 0.560602
    [89]	valid_0's auc: 0.846623	valid_0's binary_logloss: 0.561718
    [90]	valid_0's auc: 0.846448	valid_0's binary_logloss: 0.563212
    [91]	valid_0's auc: 0.846637	valid_0's binary_logloss: 0.563925
    [92]	valid_0's auc: 0.846502	valid_0's binary_logloss: 0.562197
    [93]	valid_0's auc: 0.847002	valid_0's binary_logloss: 0.559285
    [94]	valid_0's auc: 0.846154	valid_0's binary_logloss: 0.56054
    [95]	valid_0's auc: 0.846575	valid_0's binary_logloss: 0.561569
    [96]	valid_0's auc: 0.846874	valid_0's binary_logloss: 0.562883
    [97]	valid_0's auc: 0.847082	valid_0's binary_logloss: 0.55984
    [98]	valid_0's auc: 0.847267	valid_0's binary_logloss: 0.560962
    [99]	valid_0's auc: 0.847156	valid_0's binary_logloss: 0.558235
    [100]	valid_0's auc: 0.848342	valid_0's binary_logloss: 0.556936
    [101]	valid_0's auc: 0.848305	valid_0's binary_logloss: 0.55819
    [102]	valid_0's auc: 0.848296	valid_0's binary_logloss: 0.559618
    [103]	valid_0's auc: 0.848079	valid_0's binary_logloss: 0.560925
    [104]	valid_0's auc: 0.848867	valid_0's binary_logloss: 0.55861
    [105]	valid_0's auc: 0.848718	valid_0's binary_logloss: 0.559731
    [106]	valid_0's auc: 0.848657	valid_0's binary_logloss: 0.561004
    [107]	valid_0's auc: 0.848534	valid_0's binary_logloss: 0.562265
    [108]	valid_0's auc: 0.847471	valid_0's binary_logloss: 0.558719
    [109]	valid_0's auc: 0.847505	valid_0's binary_logloss: 0.55997
    [110]	valid_0's auc: 0.847649	valid_0's binary_logloss: 0.561341
    [111]	valid_0's auc: 0.84836	valid_0's binary_logloss: 0.558932
    [112]	valid_0's auc: 0.848244	valid_0's binary_logloss: 0.555944
    [113]	valid_0's auc: 0.848678	valid_0's binary_logloss: 0.553942
    [114]	valid_0's auc: 0.848188	valid_0's binary_logloss: 0.551634
    [115]	valid_0's auc: 0.848258	valid_0's binary_logloss: 0.549224
    [116]	valid_0's auc: 0.848544	valid_0's binary_logloss: 0.548228
    [117]	valid_0's auc: 0.848378	valid_0's binary_logloss: 0.549087
    [118]	valid_0's auc: 0.847845	valid_0's binary_logloss: 0.546824
    [119]	valid_0's auc: 0.847732	valid_0's binary_logloss: 0.547799
    [120]	valid_0's auc: 0.847446	valid_0's binary_logloss: 0.549102
    [121]	valid_0's auc: 0.847446	valid_0's binary_logloss: 0.550352
    [122]	valid_0's auc: 0.847023	valid_0's binary_logloss: 0.55182
    [123]	valid_0's auc: 0.846764	valid_0's binary_logloss: 0.548995
    [124]	valid_0's auc: 0.846231	valid_0's binary_logloss: 0.546926
    [125]	valid_0's auc: 0.846165	valid_0's binary_logloss: 0.547715
    [126]	valid_0's auc: 0.846085	valid_0's binary_logloss: 0.548756
    [127]	valid_0's auc: 0.84589	valid_0's binary_logloss: 0.546098
    [128]	valid_0's auc: 0.846098	valid_0's binary_logloss: 0.547004
    [129]	valid_0's auc: 0.845728	valid_0's binary_logloss: 0.544432
    [130]	valid_0's auc: 0.843167	valid_0's binary_logloss: 0.541572
    [131]	valid_0's auc: 0.843582	valid_0's binary_logloss: 0.542547
    [132]	valid_0's auc: 0.84538	valid_0's binary_logloss: 0.540394
    [133]	valid_0's auc: 0.846012	valid_0's binary_logloss: 0.539345
    [134]	valid_0's auc: 0.846066	valid_0's binary_logloss: 0.540459
    [135]	valid_0's auc: 0.846462	valid_0's binary_logloss: 0.537938
    [136]	valid_0's auc: 0.846771	valid_0's binary_logloss: 0.539066
    [137]	valid_0's auc: 0.846832	valid_0's binary_logloss: 0.540236
    [138]	valid_0's auc: 0.846775	valid_0's binary_logloss: 0.537667
    [139]	valid_0's auc: 0.846837	valid_0's binary_logloss: 0.535593
    [140]	valid_0's auc: 0.84706	valid_0's binary_logloss: 0.537044
    [141]	valid_0's auc: 0.847161	valid_0's binary_logloss: 0.538136
    [142]	valid_0's auc: 0.847429	valid_0's binary_logloss: 0.539186
    [143]	valid_0's auc: 0.847383	valid_0's binary_logloss: 0.536694
    [144]	valid_0's auc: 0.847455	valid_0's binary_logloss: 0.537614
    [145]	valid_0's auc: 0.847353	valid_0's binary_logloss: 0.535276
    [146]	valid_0's auc: 0.847276	valid_0's binary_logloss: 0.536416
    [147]	valid_0's auc: 0.847469	valid_0's binary_logloss: 0.537665
    [148]	valid_0's auc: 0.847541	valid_0's binary_logloss: 0.538841
    [149]	valid_0's auc: 0.847059	valid_0's binary_logloss: 0.535747
    [150]	valid_0's auc: 0.846901	valid_0's binary_logloss: 0.533638
    [151]	valid_0's auc: 0.846773	valid_0's binary_logloss: 0.534819
    [152]	valid_0's auc: 0.846784	valid_0's binary_logloss: 0.53594
    [153]	valid_0's auc: 0.846511	valid_0's binary_logloss: 0.534128
    [154]	valid_0's auc: 0.84651	valid_0's binary_logloss: 0.535081
    [155]	valid_0's auc: 0.846768	valid_0's binary_logloss: 0.531852
    [156]	valid_0's auc: 0.846576	valid_0's binary_logloss: 0.533173
    [157]	valid_0's auc: 0.846682	valid_0's binary_logloss: 0.533808
    [158]	valid_0's auc: 0.846591	valid_0's binary_logloss: 0.535109
    [159]	valid_0's auc: 0.846944	valid_0's binary_logloss: 0.536469
    [160]	valid_0's auc: 0.846644	valid_0's binary_logloss: 0.537837
    [161]	valid_0's auc: 0.844553	valid_0's binary_logloss: 0.534866
    [162]	valid_0's auc: 0.84437	valid_0's binary_logloss: 0.535625
    [163]	valid_0's auc: 0.842004	valid_0's binary_logloss: 0.532844
    [164]	valid_0's auc: 0.842757	valid_0's binary_logloss: 0.530484
    [165]	valid_0's auc: 0.842738	valid_0's binary_logloss: 0.52912
    [166]	valid_0's auc: 0.843493	valid_0's binary_logloss: 0.527197
    [167]	valid_0's auc: 0.842978	valid_0's binary_logloss: 0.524894
    [168]	valid_0's auc: 0.842139	valid_0's binary_logloss: 0.521853
    [169]	valid_0's auc: 0.84297	valid_0's binary_logloss: 0.519969
    [170]	valid_0's auc: 0.843022	valid_0's binary_logloss: 0.521085
    [171]	valid_0's auc: 0.842704	valid_0's binary_logloss: 0.518899
    [172]	valid_0's auc: 0.842541	valid_0's binary_logloss: 0.51996
    [173]	valid_0's auc: 0.847067	valid_0's binary_logloss: 0.516715
    [174]	valid_0's auc: 0.846344	valid_0's binary_logloss: 0.514041
    [175]	valid_0's auc: 0.84607	valid_0's binary_logloss: 0.511894
    [176]	valid_0's auc: 0.845973	valid_0's binary_logloss: 0.512832
    [177]	valid_0's auc: 0.845918	valid_0's binary_logloss: 0.513977
    [178]	valid_0's auc: 0.846176	valid_0's binary_logloss: 0.515086
    [179]	valid_0's auc: 0.8465	valid_0's binary_logloss: 0.512737
    [180]	valid_0's auc: 0.846543	valid_0's binary_logloss: 0.5139
    [181]	valid_0's auc: 0.846864	valid_0's binary_logloss: 0.511913
    [182]	valid_0's auc: 0.846778	valid_0's binary_logloss: 0.509778
    [183]	valid_0's auc: 0.8456	valid_0's binary_logloss: 0.506969
    [184]	valid_0's auc: 0.846013	valid_0's binary_logloss: 0.504763
    [185]	valid_0's auc: 0.845855	valid_0's binary_logloss: 0.506312
    [186]	valid_0's auc: 0.844327	valid_0's binary_logloss: 0.503733
    [187]	valid_0's auc: 0.845936	valid_0's binary_logloss: 0.501721
    [188]	valid_0's auc: 0.845385	valid_0's binary_logloss: 0.499814
    [189]	valid_0's auc: 0.845439	valid_0's binary_logloss: 0.501076
    [190]	valid_0's auc: 0.845613	valid_0's binary_logloss: 0.502254
    [191]	valid_0's auc: 0.845169	valid_0's binary_logloss: 0.500356
    [192]	valid_0's auc: 0.844565	valid_0's binary_logloss: 0.498336
    [193]	valid_0's auc: 0.844444	valid_0's binary_logloss: 0.496508
    [194]	valid_0's auc: 0.844248	valid_0's binary_logloss: 0.497782
    [195]	valid_0's auc: 0.844288	valid_0's binary_logloss: 0.498712
    [196]	valid_0's auc: 0.844538	valid_0's binary_logloss: 0.497322
    [197]	valid_0's auc: 0.844496	valid_0's binary_logloss: 0.498317
    [198]	valid_0's auc: 0.844543	valid_0's binary_logloss: 0.49935
    [199]	valid_0's auc: 0.84427	valid_0's binary_logloss: 0.497366
    [200]	valid_0's auc: 0.844254	valid_0's binary_logloss: 0.498589
    [201]	valid_0's auc: 0.844036	valid_0's binary_logloss: 0.499823
    [202]	valid_0's auc: 0.843973	valid_0's binary_logloss: 0.500818
    [203]	valid_0's auc: 0.843191	valid_0's binary_logloss: 0.498989
    [204]	valid_0's auc: 0.843182	valid_0's binary_logloss: 0.499949
    [205]	valid_0's auc: 0.843316	valid_0's binary_logloss: 0.500933
    [206]	valid_0's auc: 0.844263	valid_0's binary_logloss: 0.499902
    [207]	valid_0's auc: 0.844465	valid_0's binary_logloss: 0.500977
    [208]	valid_0's auc: 0.844937	valid_0's binary_logloss: 0.498959
    [209]	valid_0's auc: 0.844843	valid_0's binary_logloss: 0.500069
    [210]	valid_0's auc: 0.844819	valid_0's binary_logloss: 0.501086
    [211]	valid_0's auc: 0.844766	valid_0's binary_logloss: 0.502127
    [212]	valid_0's auc: 0.845336	valid_0's binary_logloss: 0.499689
    [213]	valid_0's auc: 0.845853	valid_0's binary_logloss: 0.49753
    [214]	valid_0's auc: 0.847202	valid_0's binary_logloss: 0.495786
    [215]	valid_0's auc: 0.847462	valid_0's binary_logloss: 0.494067
    [216]	valid_0's auc: 0.847548	valid_0's binary_logloss: 0.495183
    [217]	valid_0's auc: 0.847248	valid_0's binary_logloss: 0.492664
    [218]	valid_0's auc: 0.846089	valid_0's binary_logloss: 0.490241
    [219]	valid_0's auc: 0.846008	valid_0's binary_logloss: 0.49114
    [220]	valid_0's auc: 0.845946	valid_0's binary_logloss: 0.49029
    [221]	valid_0's auc: 0.845805	valid_0's binary_logloss: 0.491378
    [222]	valid_0's auc: 0.846389	valid_0's binary_logloss: 0.490601
    [223]	valid_0's auc: 0.845093	valid_0's binary_logloss: 0.488255
    [224]	valid_0's auc: 0.845175	valid_0's binary_logloss: 0.489354
    [225]	valid_0's auc: 0.844983	valid_0's binary_logloss: 0.490671
    [226]	valid_0's auc: 0.845122	valid_0's binary_logloss: 0.491748
    [227]	valid_0's auc: 0.845134	valid_0's binary_logloss: 0.492681
    [228]	valid_0's auc: 0.845036	valid_0's binary_logloss: 0.490544
    [229]	valid_0's auc: 0.845078	valid_0's binary_logloss: 0.491498
    [230]	valid_0's auc: 0.844328	valid_0's binary_logloss: 0.489192
    [231]	valid_0's auc: 0.842466	valid_0's binary_logloss: 0.48691
    [232]	valid_0's auc: 0.843627	valid_0's binary_logloss: 0.484937
    [233]	valid_0's auc: 0.843561	valid_0's binary_logloss: 0.48589
    [234]	valid_0's auc: 0.843981	valid_0's binary_logloss: 0.48378
    [235]	valid_0's auc: 0.844063	valid_0's binary_logloss: 0.484746
    [236]	valid_0's auc: 0.844884	valid_0's binary_logloss: 0.482536
    [237]	valid_0's auc: 0.844872	valid_0's binary_logloss: 0.481091
    [238]	valid_0's auc: 0.845581	valid_0's binary_logloss: 0.479606
    [239]	valid_0's auc: 0.84881	valid_0's binary_logloss: 0.476973
    [240]	valid_0's auc: 0.848567	valid_0's binary_logloss: 0.475292
    [241]	valid_0's auc: 0.848556	valid_0's binary_logloss: 0.476446
    [242]	valid_0's auc: 0.848229	valid_0's binary_logloss: 0.474615
    [243]	valid_0's auc: 0.848285	valid_0's binary_logloss: 0.475829
    [244]	valid_0's auc: 0.848429	valid_0's binary_logloss: 0.476772
    [245]	valid_0's auc: 0.848446	valid_0's binary_logloss: 0.475073
    [246]	valid_0's auc: 0.848003	valid_0's binary_logloss: 0.473328
    [247]	valid_0's auc: 0.847379	valid_0's binary_logloss: 0.471
    [248]	valid_0's auc: 0.847393	valid_0's binary_logloss: 0.472221
    [249]	valid_0's auc: 0.847335	valid_0's binary_logloss: 0.473117
    [250]	valid_0's auc: 0.848141	valid_0's binary_logloss: 0.471399
    [251]	valid_0's auc: 0.848246	valid_0's binary_logloss: 0.472502
    [252]	valid_0's auc: 0.848173	valid_0's binary_logloss: 0.473452
    [253]	valid_0's auc: 0.849191	valid_0's binary_logloss: 0.471716
    [254]	valid_0's auc: 0.848613	valid_0's binary_logloss: 0.469996
    [255]	valid_0's auc: 0.848433	valid_0's binary_logloss: 0.46843
    [256]	valid_0's auc: 0.848456	valid_0's binary_logloss: 0.467295
    [257]	valid_0's auc: 0.848326	valid_0's binary_logloss: 0.465661
    [258]	valid_0's auc: 0.84906	valid_0's binary_logloss: 0.463917
    [259]	valid_0's auc: 0.849278	valid_0's binary_logloss: 0.465008
    [260]	valid_0's auc: 0.849253	valid_0's binary_logloss: 0.466138
    [261]	valid_0's auc: 0.849268	valid_0's binary_logloss: 0.467237
    [262]	valid_0's auc: 0.849252	valid_0's binary_logloss: 0.464975
    [263]	valid_0's auc: 0.849137	valid_0's binary_logloss: 0.462897
    [264]	valid_0's auc: 0.848346	valid_0's binary_logloss: 0.460767
    [265]	valid_0's auc: 0.847392	valid_0's binary_logloss: 0.458708
    [266]	valid_0's auc: 0.847206	valid_0's binary_logloss: 0.457572
    [267]	valid_0's auc: 0.847369	valid_0's binary_logloss: 0.458597
    [268]	valid_0's auc: 0.847513	valid_0's binary_logloss: 0.456956
    [269]	valid_0's auc: 0.847575	valid_0's binary_logloss: 0.457951
    [270]	valid_0's auc: 0.847815	valid_0's binary_logloss: 0.456113
    [271]	valid_0's auc: 0.847858	valid_0's binary_logloss: 0.457153
    [272]	valid_0's auc: 0.848744	valid_0's binary_logloss: 0.455605
    [273]	valid_0's auc: 0.850323	valid_0's binary_logloss: 0.453391
    [274]	valid_0's auc: 0.849999	valid_0's binary_logloss: 0.454422
    [275]	valid_0's auc: 0.84972	valid_0's binary_logloss: 0.452204
    [276]	valid_0's auc: 0.849891	valid_0's binary_logloss: 0.453329
    [277]	valid_0's auc: 0.850213	valid_0's binary_logloss: 0.452086
    [278]	valid_0's auc: 0.850181	valid_0's binary_logloss: 0.453048
    [279]	valid_0's auc: 0.850173	valid_0's binary_logloss: 0.451614
    [280]	valid_0's auc: 0.850242	valid_0's binary_logloss: 0.452712
    [281]	valid_0's auc: 0.850292	valid_0's binary_logloss: 0.453614
    [282]	valid_0's auc: 0.850989	valid_0's binary_logloss: 0.451385
    [283]	valid_0's auc: 0.850936	valid_0's binary_logloss: 0.45256
    [284]	valid_0's auc: 0.852012	valid_0's binary_logloss: 0.450404
    [285]	valid_0's auc: 0.851767	valid_0's binary_logloss: 0.448402
    [286]	valid_0's auc: 0.853284	valid_0's binary_logloss: 0.447252
    [287]	valid_0's auc: 0.852175	valid_0's binary_logloss: 0.445481
    [288]	valid_0's auc: 0.852264	valid_0's binary_logloss: 0.44646
    [289]	valid_0's auc: 0.85191	valid_0's binary_logloss: 0.444926
    [290]	valid_0's auc: 0.850635	valid_0's binary_logloss: 0.443194
    [291]	valid_0's auc: 0.85084	valid_0's binary_logloss: 0.441993
    [292]	valid_0's auc: 0.850769	valid_0's binary_logloss: 0.443114
    [293]	valid_0's auc: 0.850652	valid_0's binary_logloss: 0.444122
    [294]	valid_0's auc: 0.849867	valid_0's binary_logloss: 0.442172
    [295]	valid_0's auc: 0.849866	valid_0's binary_logloss: 0.443247
    [296]	valid_0's auc: 0.849912	valid_0's binary_logloss: 0.444229
    [297]	valid_0's auc: 0.84922	valid_0's binary_logloss: 0.442278
    [298]	valid_0's auc: 0.849299	valid_0's binary_logloss: 0.443246
    [299]	valid_0's auc: 0.849499	valid_0's binary_logloss: 0.4417
    [300]	valid_0's auc: 0.849572	valid_0's binary_logloss: 0.442656
    [301]	valid_0's auc: 0.849418	valid_0's binary_logloss: 0.443644
    [302]	valid_0's auc: 0.849679	valid_0's binary_logloss: 0.44206
    [303]	valid_0's auc: 0.849684	valid_0's binary_logloss: 0.442985
    [304]	valid_0's auc: 0.849698	valid_0's binary_logloss: 0.443947
    [305]	valid_0's auc: 0.849627	valid_0's binary_logloss: 0.445024
    [306]	valid_0's auc: 0.849517	valid_0's binary_logloss: 0.445981
    [307]	valid_0's auc: 0.849466	valid_0's binary_logloss: 0.446785
    [308]	valid_0's auc: 0.849346	valid_0's binary_logloss: 0.445473
    [309]	valid_0's auc: 0.849486	valid_0's binary_logloss: 0.446421
    [310]	valid_0's auc: 0.850204	valid_0's binary_logloss: 0.444325
    [311]	valid_0's auc: 0.85037	valid_0's binary_logloss: 0.44329
    [312]	valid_0's auc: 0.850145	valid_0's binary_logloss: 0.441748
    [313]	valid_0's auc: 0.849994	valid_0's binary_logloss: 0.440188
    [314]	valid_0's auc: 0.850026	valid_0's binary_logloss: 0.441005
    [315]	valid_0's auc: 0.849874	valid_0's binary_logloss: 0.442001
    [316]	valid_0's auc: 0.849399	valid_0's binary_logloss: 0.440034
    [317]	valid_0's auc: 0.849366	valid_0's binary_logloss: 0.438383
    [318]	valid_0's auc: 0.849122	valid_0's binary_logloss: 0.437002
    [319]	valid_0's auc: 0.849114	valid_0's binary_logloss: 0.437924
    [320]	valid_0's auc: 0.848205	valid_0's binary_logloss: 0.43606
    [321]	valid_0's auc: 0.848141	valid_0's binary_logloss: 0.436936
    [322]	valid_0's auc: 0.848324	valid_0's binary_logloss: 0.436173
    [323]	valid_0's auc: 0.848361	valid_0's binary_logloss: 0.434453
    [324]	valid_0's auc: 0.84841	valid_0's binary_logloss: 0.435546
    [325]	valid_0's auc: 0.848338	valid_0's binary_logloss: 0.434142
    [326]	valid_0's auc: 0.84833	valid_0's binary_logloss: 0.435162
    [327]	valid_0's auc: 0.847932	valid_0's binary_logloss: 0.433751
    [328]	valid_0's auc: 0.847929	valid_0's binary_logloss: 0.434567
    [329]	valid_0's auc: 0.848346	valid_0's binary_logloss: 0.433369
    [330]	valid_0's auc: 0.848402	valid_0's binary_logloss: 0.434244
    [331]	valid_0's auc: 0.848449	valid_0's binary_logloss: 0.435175
    [332]	valid_0's auc: 0.848458	valid_0's binary_logloss: 0.435995
    [333]	valid_0's auc: 0.848513	valid_0's binary_logloss: 0.436993
    [334]	valid_0's auc: 0.848517	valid_0's binary_logloss: 0.438005
    [335]	valid_0's auc: 0.848821	valid_0's binary_logloss: 0.437016
    [336]	valid_0's auc: 0.84889	valid_0's binary_logloss: 0.437954
    [337]	valid_0's auc: 0.847956	valid_0's binary_logloss: 0.436284
    [338]	valid_0's auc: 0.847681	valid_0's binary_logloss: 0.434997
    [339]	valid_0's auc: 0.847615	valid_0's binary_logloss: 0.435804
    [340]	valid_0's auc: 0.847611	valid_0's binary_logloss: 0.43675
    [341]	valid_0's auc: 0.847505	valid_0's binary_logloss: 0.435308
    [342]	valid_0's auc: 0.847142	valid_0's binary_logloss: 0.433397
    [343]	valid_0's auc: 0.847896	valid_0's binary_logloss: 0.432085
    [344]	valid_0's auc: 0.847937	valid_0's binary_logloss: 0.432916
    [345]	valid_0's auc: 0.848211	valid_0's binary_logloss: 0.431531
    [346]	valid_0's auc: 0.849308	valid_0's binary_logloss: 0.430345
    [347]	valid_0's auc: 0.849352	valid_0's binary_logloss: 0.431123
    [348]	valid_0's auc: 0.849745	valid_0's binary_logloss: 0.429482
    [349]	valid_0's auc: 0.851081	valid_0's binary_logloss: 0.427524
    [350]	valid_0's auc: 0.85113	valid_0's binary_logloss: 0.42847
    [351]	valid_0's auc: 0.852281	valid_0's binary_logloss: 0.426391
    [352]	valid_0's auc: 0.852398	valid_0's binary_logloss: 0.424805
    [353]	valid_0's auc: 0.852473	valid_0's binary_logloss: 0.423006
    [354]	valid_0's auc: 0.85264	valid_0's binary_logloss: 0.423771
    [355]	valid_0's auc: 0.852581	valid_0's binary_logloss: 0.424667
    [356]	valid_0's auc: 0.852659	valid_0's binary_logloss: 0.423434
    [357]	valid_0's auc: 0.852649	valid_0's binary_logloss: 0.424256
    [358]	valid_0's auc: 0.852691	valid_0's binary_logloss: 0.423089
    [359]	valid_0's auc: 0.852714	valid_0's binary_logloss: 0.423999
    [360]	valid_0's auc: 0.852381	valid_0's binary_logloss: 0.422558
    [361]	valid_0's auc: 0.852379	valid_0's binary_logloss: 0.421141
    [362]	valid_0's auc: 0.85247	valid_0's binary_logloss: 0.419787
    [363]	valid_0's auc: 0.852477	valid_0's binary_logloss: 0.420635
    [364]	valid_0's auc: 0.852421	valid_0's binary_logloss: 0.421474
    [365]	valid_0's auc: 0.85237	valid_0's binary_logloss: 0.420172
    [366]	valid_0's auc: 0.852319	valid_0's binary_logloss: 0.421078
    [367]	valid_0's auc: 0.85233	valid_0's binary_logloss: 0.422037
    [368]	valid_0's auc: 0.852757	valid_0's binary_logloss: 0.42053
    [369]	valid_0's auc: 0.852765	valid_0's binary_logloss: 0.42129
    [370]	valid_0's auc: 0.852546	valid_0's binary_logloss: 0.42016
    [371]	valid_0's auc: 0.852537	valid_0's binary_logloss: 0.421046
    [372]	valid_0's auc: 0.853041	valid_0's binary_logloss: 0.419622
    [373]	valid_0's auc: 0.853039	valid_0's binary_logloss: 0.418069
    [374]	valid_0's auc: 0.853021	valid_0's binary_logloss: 0.416721
    [375]	valid_0's auc: 0.853433	valid_0's binary_logloss: 0.415227
    [376]	valid_0's auc: 0.853518	valid_0's binary_logloss: 0.416024
    [377]	valid_0's auc: 0.85337	valid_0's binary_logloss: 0.414632
    [378]	valid_0's auc: 0.85343	valid_0's binary_logloss: 0.415567
    [379]	valid_0's auc: 0.853486	valid_0's binary_logloss: 0.416399
    [380]	valid_0's auc: 0.854682	valid_0's binary_logloss: 0.415091
    [381]	valid_0's auc: 0.854717	valid_0's binary_logloss: 0.415967
    [382]	valid_0's auc: 0.854647	valid_0's binary_logloss: 0.416823
    [383]	valid_0's auc: 0.854935	valid_0's binary_logloss: 0.415309
    [384]	valid_0's auc: 0.855753	valid_0's binary_logloss: 0.4134
    [385]	valid_0's auc: 0.855708	valid_0's binary_logloss: 0.414317
    [386]	valid_0's auc: 0.857074	valid_0's binary_logloss: 0.412231
    [387]	valid_0's auc: 0.857172	valid_0's binary_logloss: 0.4109
    [388]	valid_0's auc: 0.858623	valid_0's binary_logloss: 0.408989
    [389]	valid_0's auc: 0.858726	valid_0's binary_logloss: 0.407655
    [390]	valid_0's auc: 0.858727	valid_0's binary_logloss: 0.406594
    [391]	valid_0's auc: 0.858709	valid_0's binary_logloss: 0.407518
    [392]	valid_0's auc: 0.858981	valid_0's binary_logloss: 0.405782
    [393]	valid_0's auc: 0.858799	valid_0's binary_logloss: 0.404546
    [394]	valid_0's auc: 0.858793	valid_0's binary_logloss: 0.403308
    [395]	valid_0's auc: 0.858579	valid_0's binary_logloss: 0.401598
    [396]	valid_0's auc: 0.858577	valid_0's binary_logloss: 0.402457
    [397]	valid_0's auc: 0.858433	valid_0's binary_logloss: 0.401175
    [398]	valid_0's auc: 0.858415	valid_0's binary_logloss: 0.400244
    [399]	valid_0's auc: 0.858249	valid_0's binary_logloss: 0.399114
    [400]	valid_0's auc: 0.858019	valid_0's binary_logloss: 0.39788
    [401]	valid_0's auc: 0.857951	valid_0's binary_logloss: 0.398641
    [402]	valid_0's auc: 0.858156	valid_0's binary_logloss: 0.397466
    [403]	valid_0's auc: 0.858164	valid_0's binary_logloss: 0.398246
    [404]	valid_0's auc: 0.858314	valid_0's binary_logloss: 0.396941
    [405]	valid_0's auc: 0.858537	valid_0's binary_logloss: 0.395714
    [406]	valid_0's auc: 0.858527	valid_0's binary_logloss: 0.396527
    [407]	valid_0's auc: 0.858722	valid_0's binary_logloss: 0.394956
    [408]	valid_0's auc: 0.858661	valid_0's binary_logloss: 0.395847
    [409]	valid_0's auc: 0.858592	valid_0's binary_logloss: 0.394885
    [410]	valid_0's auc: 0.858453	valid_0's binary_logloss: 0.39383
    [411]	valid_0's auc: 0.858278	valid_0's binary_logloss: 0.392725
    [412]	valid_0's auc: 0.858272	valid_0's binary_logloss: 0.393496
    [413]	valid_0's auc: 0.858153	valid_0's binary_logloss: 0.394321
    [414]	valid_0's auc: 0.858884	valid_0's binary_logloss: 0.392575
    [415]	valid_0's auc: 0.858374	valid_0's binary_logloss: 0.391059
    [416]	valid_0's auc: 0.858151	valid_0's binary_logloss: 0.389486
    [417]	valid_0's auc: 0.858085	valid_0's binary_logloss: 0.390358
    [418]	valid_0's auc: 0.858051	valid_0's binary_logloss: 0.391195
    [419]	valid_0's auc: 0.858006	valid_0's binary_logloss: 0.389876
    [420]	valid_0's auc: 0.85801	valid_0's binary_logloss: 0.390706
    [421]	valid_0's auc: 0.858598	valid_0's binary_logloss: 0.38957
    [422]	valid_0's auc: 0.858329	valid_0's binary_logloss: 0.388469
    [423]	valid_0's auc: 0.859144	valid_0's binary_logloss: 0.387917
    [424]	valid_0's auc: 0.85913	valid_0's binary_logloss: 0.38686
    [425]	valid_0's auc: 0.859077	valid_0's binary_logloss: 0.387715
    [426]	valid_0's auc: 0.85894	valid_0's binary_logloss: 0.38669
    [427]	valid_0's auc: 0.859151	valid_0's binary_logloss: 0.385741
    [428]	valid_0's auc: 0.859086	valid_0's binary_logloss: 0.386581
    [429]	valid_0's auc: 0.858969	valid_0's binary_logloss: 0.385553
    [430]	valid_0's auc: 0.858943	valid_0's binary_logloss: 0.386261
    [431]	valid_0's auc: 0.859086	valid_0's binary_logloss: 0.385055
    [432]	valid_0's auc: 0.85847	valid_0's binary_logloss: 0.383737
    [433]	valid_0's auc: 0.859275	valid_0's binary_logloss: 0.383154
    [434]	valid_0's auc: 0.859249	valid_0's binary_logloss: 0.383917
    [435]	valid_0's auc: 0.859232	valid_0's binary_logloss: 0.384699
    [436]	valid_0's auc: 0.859191	valid_0's binary_logloss: 0.383574
    [437]	valid_0's auc: 0.859374	valid_0's binary_logloss: 0.382887
    [438]	valid_0's auc: 0.859374	valid_0's binary_logloss: 0.383611
    [439]	valid_0's auc: 0.8594	valid_0's binary_logloss: 0.384345
    [440]	valid_0's auc: 0.85983	valid_0's binary_logloss: 0.383198
    [441]	valid_0's auc: 0.859895	valid_0's binary_logloss: 0.381982
    [442]	valid_0's auc: 0.859725	valid_0's binary_logloss: 0.380898
    [443]	valid_0's auc: 0.859713	valid_0's binary_logloss: 0.381609
    [444]	valid_0's auc: 0.859784	valid_0's binary_logloss: 0.380418
    [445]	valid_0's auc: 0.859381	valid_0's binary_logloss: 0.379188
    [446]	valid_0's auc: 0.859373	valid_0's binary_logloss: 0.379985
    [447]	valid_0's auc: 0.859235	valid_0's binary_logloss: 0.378995
    [448]	valid_0's auc: 0.860171	valid_0's binary_logloss: 0.377427
    [449]	valid_0's auc: 0.860126	valid_0's binary_logloss: 0.376954
    [450]	valid_0's auc: 0.860096	valid_0's binary_logloss: 0.37581
    [451]	valid_0's auc: 0.860233	valid_0's binary_logloss: 0.374826
    [452]	valid_0's auc: 0.860188	valid_0's binary_logloss: 0.375576
    [453]	valid_0's auc: 0.860179	valid_0's binary_logloss: 0.376305
    [454]	valid_0's auc: 0.860154	valid_0's binary_logloss: 0.377022
    [455]	valid_0's auc: 0.860254	valid_0's binary_logloss: 0.375925
    [456]	valid_0's auc: 0.860194	valid_0's binary_logloss: 0.376672
    [457]	valid_0's auc: 0.860174	valid_0's binary_logloss: 0.377418
    [458]	valid_0's auc: 0.859566	valid_0's binary_logloss: 0.376306
    [459]	valid_0's auc: 0.860307	valid_0's binary_logloss: 0.37469
    [460]	valid_0's auc: 0.860289	valid_0's binary_logloss: 0.375352
    [461]	valid_0's auc: 0.86015	valid_0's binary_logloss: 0.373882
    [462]	valid_0's auc: 0.860171	valid_0's binary_logloss: 0.374628
    [463]	valid_0's auc: 0.860147	valid_0's binary_logloss: 0.37537
    [464]	valid_0's auc: 0.860259	valid_0's binary_logloss: 0.374347
    [465]	valid_0's auc: 0.859922	valid_0's binary_logloss: 0.373045
    [466]	valid_0's auc: 0.859944	valid_0's binary_logloss: 0.373751
    [467]	valid_0's auc: 0.861059	valid_0's binary_logloss: 0.372069
    [468]	valid_0's auc: 0.861038	valid_0's binary_logloss: 0.372739
    [469]	valid_0's auc: 0.86045	valid_0's binary_logloss: 0.371535
    [470]	valid_0's auc: 0.860423	valid_0's binary_logloss: 0.372225
    [471]	valid_0's auc: 0.860636	valid_0's binary_logloss: 0.371265
    [472]	valid_0's auc: 0.860617	valid_0's binary_logloss: 0.371938
    [473]	valid_0's auc: 0.860657	valid_0's binary_logloss: 0.370874
    [474]	valid_0's auc: 0.86064	valid_0's binary_logloss: 0.369786
    [475]	valid_0's auc: 0.86053	valid_0's binary_logloss: 0.368816
    [476]	valid_0's auc: 0.86053	valid_0's binary_logloss: 0.369467
    [477]	valid_0's auc: 0.860504	valid_0's binary_logloss: 0.368669
    [478]	valid_0's auc: 0.860525	valid_0's binary_logloss: 0.369417
    [479]	valid_0's auc: 0.860527	valid_0's binary_logloss: 0.370134
    [480]	valid_0's auc: 0.860779	valid_0's binary_logloss: 0.368723
    [481]	valid_0's auc: 0.860649	valid_0's binary_logloss: 0.367788
    [482]	valid_0's auc: 0.860105	valid_0's binary_logloss: 0.366642
    [483]	valid_0's auc: 0.861318	valid_0's binary_logloss: 0.365072
    [484]	valid_0's auc: 0.860888	valid_0's binary_logloss: 0.363806
    [485]	valid_0's auc: 0.860902	valid_0's binary_logloss: 0.364443
    [486]	valid_0's auc: 0.860445	valid_0's binary_logloss: 0.363228
    [487]	valid_0's auc: 0.861229	valid_0's binary_logloss: 0.361792
    [488]	valid_0's auc: 0.861042	valid_0's binary_logloss: 0.360895
    [489]	valid_0's auc: 0.860836	valid_0's binary_logloss: 0.359615
    [490]	valid_0's auc: 0.860804	valid_0's binary_logloss: 0.360303
    [491]	valid_0's auc: 0.860835	valid_0's binary_logloss: 0.360966
    [492]	valid_0's auc: 0.860809	valid_0's binary_logloss: 0.361611
    [493]	valid_0's auc: 0.860833	valid_0's binary_logloss: 0.36231
    [494]	valid_0's auc: 0.860739	valid_0's binary_logloss: 0.361444
    [495]	valid_0's auc: 0.860773	valid_0's binary_logloss: 0.360498
    [496]	valid_0's auc: 0.860738	valid_0's binary_logloss: 0.359588
    [497]	valid_0's auc: 0.860492	valid_0's binary_logloss: 0.358462
    [498]	valid_0's auc: 0.860486	valid_0's binary_logloss: 0.357615
    [499]	valid_0's auc: 0.86047	valid_0's binary_logloss: 0.35825
    [500]	valid_0's auc: 0.860197	valid_0's binary_logloss: 0.357013
    [501]	valid_0's auc: 0.860162	valid_0's binary_logloss: 0.35771
    [502]	valid_0's auc: 0.860158	valid_0's binary_logloss: 0.35651
    [503]	valid_0's auc: 0.860111	valid_0's binary_logloss: 0.355637
    [504]	valid_0's auc: 0.86004	valid_0's binary_logloss: 0.356319
    [505]	valid_0's auc: 0.860071	valid_0's binary_logloss: 0.356916
    [506]	valid_0's auc: 0.859822	valid_0's binary_logloss: 0.355837
    [507]	valid_0's auc: 0.860005	valid_0's binary_logloss: 0.354864
    [508]	valid_0's auc: 0.860503	valid_0's binary_logloss: 0.353618
    [509]	valid_0's auc: 0.860398	valid_0's binary_logloss: 0.352758
    [510]	valid_0's auc: 0.860529	valid_0's binary_logloss: 0.351823
    [511]	valid_0's auc: 0.860575	valid_0's binary_logloss: 0.352503
    [512]	valid_0's auc: 0.861269	valid_0's binary_logloss: 0.35164
    [513]	valid_0's auc: 0.861482	valid_0's binary_logloss: 0.350938
    [514]	valid_0's auc: 0.861234	valid_0's binary_logloss: 0.349904
    [515]	valid_0's auc: 0.861395	valid_0's binary_logloss: 0.348958
    [516]	valid_0's auc: 0.861369	valid_0's binary_logloss: 0.349595
    [517]	valid_0's auc: 0.861373	valid_0's binary_logloss: 0.350209
    [518]	valid_0's auc: 0.861321	valid_0's binary_logloss: 0.350721
    [519]	valid_0's auc: 0.861287	valid_0's binary_logloss: 0.351352
    [520]	valid_0's auc: 0.861299	valid_0's binary_logloss: 0.350558
    [521]	valid_0's auc: 0.86114	valid_0's binary_logloss: 0.349422
    [522]	valid_0's auc: 0.861139	valid_0's binary_logloss: 0.350052
    [523]	valid_0's auc: 0.861129	valid_0's binary_logloss: 0.350713
    [524]	valid_0's auc: 0.861086	valid_0's binary_logloss: 0.351376
    [525]	valid_0's auc: 0.861084	valid_0's binary_logloss: 0.351957
    [526]	valid_0's auc: 0.860687	valid_0's binary_logloss: 0.35084
    [527]	valid_0's auc: 0.860657	valid_0's binary_logloss: 0.351431
    [528]	valid_0's auc: 0.860223	valid_0's binary_logloss: 0.350398
    [529]	valid_0's auc: 0.860407	valid_0's binary_logloss: 0.349749
    [530]	valid_0's auc: 0.860368	valid_0's binary_logloss: 0.348904
    [531]	valid_0's auc: 0.860328	valid_0's binary_logloss: 0.349495
    [532]	valid_0's auc: 0.86052	valid_0's binary_logloss: 0.348707
    [533]	valid_0's auc: 0.860482	valid_0's binary_logloss: 0.349317
    [534]	valid_0's auc: 0.860582	valid_0's binary_logloss: 0.348673
    [535]	valid_0's auc: 0.861264	valid_0's binary_logloss: 0.347307
    [536]	valid_0's auc: 0.861262	valid_0's binary_logloss: 0.347827
    [537]	valid_0's auc: 0.861326	valid_0's binary_logloss: 0.346747
    [538]	valid_0's auc: 0.861377	valid_0's binary_logloss: 0.346078
    [539]	valid_0's auc: 0.861336	valid_0's binary_logloss: 0.34673
    [540]	valid_0's auc: 0.86136	valid_0's binary_logloss: 0.347351
    [541]	valid_0's auc: 0.861368	valid_0's binary_logloss: 0.347943
    [542]	valid_0's auc: 0.861413	valid_0's binary_logloss: 0.347072
    [543]	valid_0's auc: 0.861409	valid_0's binary_logloss: 0.347743
    [544]	valid_0's auc: 0.861369	valid_0's binary_logloss: 0.348339
    [545]	valid_0's auc: 0.860904	valid_0's binary_logloss: 0.347373
    [546]	valid_0's auc: 0.861962	valid_0's binary_logloss: 0.346005
    [547]	valid_0's auc: 0.862303	valid_0's binary_logloss: 0.34519
    [548]	valid_0's auc: 0.862048	valid_0's binary_logloss: 0.344183
    [549]	valid_0's auc: 0.861991	valid_0's binary_logloss: 0.344769
    [550]	valid_0's auc: 0.861993	valid_0's binary_logloss: 0.344097
    [551]	valid_0's auc: 0.86212	valid_0's binary_logloss: 0.34316
    [552]	valid_0's auc: 0.862345	valid_0's binary_logloss: 0.342424
    [553]	valid_0's auc: 0.862327	valid_0's binary_logloss: 0.343095
    [554]	valid_0's auc: 0.862297	valid_0's binary_logloss: 0.343629
    [555]	valid_0's auc: 0.861864	valid_0's binary_logloss: 0.342684
    [556]	valid_0's auc: 0.861847	valid_0's binary_logloss: 0.343263
    [557]	valid_0's auc: 0.861836	valid_0's binary_logloss: 0.343866
    [558]	valid_0's auc: 0.862486	valid_0's binary_logloss: 0.342595
    [559]	valid_0's auc: 0.863048	valid_0's binary_logloss: 0.341334
    [560]	valid_0's auc: 0.863102	valid_0's binary_logloss: 0.340511
    [561]	valid_0's auc: 0.863063	valid_0's binary_logloss: 0.341097
    [562]	valid_0's auc: 0.86304	valid_0's binary_logloss: 0.341652
    [563]	valid_0's auc: 0.863027	valid_0's binary_logloss: 0.342191
    [564]	valid_0's auc: 0.862994	valid_0's binary_logloss: 0.342862
    [565]	valid_0's auc: 0.863913	valid_0's binary_logloss: 0.341505
    [566]	valid_0's auc: 0.864003	valid_0's binary_logloss: 0.34074
    [567]	valid_0's auc: 0.863445	valid_0's binary_logloss: 0.33989
    [568]	valid_0's auc: 0.863454	valid_0's binary_logloss: 0.340414
    [569]	valid_0's auc: 0.863426	valid_0's binary_logloss: 0.341002
    [570]	valid_0's auc: 0.863456	valid_0's binary_logloss: 0.341505
    [571]	valid_0's auc: 0.863129	valid_0's binary_logloss: 0.340572
    [572]	valid_0's auc: 0.863053	valid_0's binary_logloss: 0.33983
    [573]	valid_0's auc: 0.863032	valid_0's binary_logloss: 0.340348
    [574]	valid_0's auc: 0.863037	valid_0's binary_logloss: 0.340908
    [575]	valid_0's auc: 0.862807	valid_0's binary_logloss: 0.340024
    [576]	valid_0's auc: 0.862826	valid_0's binary_logloss: 0.340633
    [577]	valid_0's auc: 0.862784	valid_0's binary_logloss: 0.341183
    [578]	valid_0's auc: 0.862768	valid_0's binary_logloss: 0.341756
    [579]	valid_0's auc: 0.862783	valid_0's binary_logloss: 0.342289
    [580]	valid_0's auc: 0.862729	valid_0's binary_logloss: 0.342837
    [581]	valid_0's auc: 0.863303	valid_0's binary_logloss: 0.341727
    [582]	valid_0's auc: 0.863337	valid_0's binary_logloss: 0.342257
    [583]	valid_0's auc: 0.863387	valid_0's binary_logloss: 0.341442
    [584]	valid_0's auc: 0.863335	valid_0's binary_logloss: 0.342019
    [585]	valid_0's auc: 0.863543	valid_0's binary_logloss: 0.341257
    [586]	valid_0's auc: 0.86355	valid_0's binary_logloss: 0.341759
    [587]	valid_0's auc: 0.863711	valid_0's binary_logloss: 0.34104
    [588]	valid_0's auc: 0.863723	valid_0's binary_logloss: 0.3416
    [589]	valid_0's auc: 0.863711	valid_0's binary_logloss: 0.342199
    [590]	valid_0's auc: 0.863725	valid_0's binary_logloss: 0.342663
    [591]	valid_0's auc: 0.863724	valid_0's binary_logloss: 0.343265
    [592]	valid_0's auc: 0.863717	valid_0's binary_logloss: 0.343826
    [593]	valid_0's auc: 0.863762	valid_0's binary_logloss: 0.343137
    [594]	valid_0's auc: 0.863737	valid_0's binary_logloss: 0.343667
    [595]	valid_0's auc: 0.863737	valid_0's binary_logloss: 0.34415
    [596]	valid_0's auc: 0.863665	valid_0's binary_logloss: 0.343304
    [597]	valid_0's auc: 0.863659	valid_0's binary_logloss: 0.343858
    [598]	valid_0's auc: 0.863655	valid_0's binary_logloss: 0.344437
    [599]	valid_0's auc: 0.864472	valid_0's binary_logloss: 0.343155
    [600]	valid_0's auc: 0.864564	valid_0's binary_logloss: 0.342365
    [601]	valid_0's auc: 0.864387	valid_0's binary_logloss: 0.34163
    [602]	valid_0's auc: 0.864364	valid_0's binary_logloss: 0.342156
    [603]	valid_0's auc: 0.864288	valid_0's binary_logloss: 0.341378
    [604]	valid_0's auc: 0.864282	valid_0's binary_logloss: 0.341904
    [605]	valid_0's auc: 0.864291	valid_0's binary_logloss: 0.342428
    [606]	valid_0's auc: 0.864302	valid_0's binary_logloss: 0.342942
    [607]	valid_0's auc: 0.8643	valid_0's binary_logloss: 0.34341
    [608]	valid_0's auc: 0.86423	valid_0's binary_logloss: 0.342599
    [609]	valid_0's auc: 0.863867	valid_0's binary_logloss: 0.341573
    [610]	valid_0's auc: 0.863286	valid_0's binary_logloss: 0.34072
    [611]	valid_0's auc: 0.863313	valid_0's binary_logloss: 0.339963
    [612]	valid_0's auc: 0.863431	valid_0's binary_logloss: 0.338898
    [613]	valid_0's auc: 0.863429	valid_0's binary_logloss: 0.339431
    [614]	valid_0's auc: 0.863255	valid_0's binary_logloss: 0.33866
    [615]	valid_0's auc: 0.863044	valid_0's binary_logloss: 0.337646
    [616]	valid_0's auc: 0.86308	valid_0's binary_logloss: 0.33691
    [617]	valid_0's auc: 0.863107	valid_0's binary_logloss: 0.337392
    [618]	valid_0's auc: 0.863138	valid_0's binary_logloss: 0.33664
    [619]	valid_0's auc: 0.862723	valid_0's binary_logloss: 0.33571
    [620]	valid_0's auc: 0.862597	valid_0's binary_logloss: 0.33493
    [621]	valid_0's auc: 0.862602	valid_0's binary_logloss: 0.335379
    [622]	valid_0's auc: 0.862583	valid_0's binary_logloss: 0.335882
    [623]	valid_0's auc: 0.862745	valid_0's binary_logloss: 0.335058
    [624]	valid_0's auc: 0.862705	valid_0's binary_logloss: 0.33556
    [625]	valid_0's auc: 0.86263	valid_0's binary_logloss: 0.334877
    [626]	valid_0's auc: 0.862775	valid_0's binary_logloss: 0.333844
    [627]	valid_0's auc: 0.862792	valid_0's binary_logloss: 0.333068
    [628]	valid_0's auc: 0.862953	valid_0's binary_logloss: 0.332042
    [629]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.332537
    [630]	valid_0's auc: 0.862956	valid_0's binary_logloss: 0.333048
    [631]	valid_0's auc: 0.863072	valid_0's binary_logloss: 0.332242
    [632]	valid_0's auc: 0.863172	valid_0's binary_logloss: 0.331488
    [633]	valid_0's auc: 0.863169	valid_0's binary_logloss: 0.332001
    [634]	valid_0's auc: 0.863133	valid_0's binary_logloss: 0.332515
    [635]	valid_0's auc: 0.863109	valid_0's binary_logloss: 0.331858
    [636]	valid_0's auc: 0.863146	valid_0's binary_logloss: 0.331227
    [637]	valid_0's auc: 0.863177	valid_0's binary_logloss: 0.330371
    [638]	valid_0's auc: 0.863146	valid_0's binary_logloss: 0.330895
    [639]	valid_0's auc: 0.863129	valid_0's binary_logloss: 0.331432
    [640]	valid_0's auc: 0.863258	valid_0's binary_logloss: 0.330643
    [641]	valid_0's auc: 0.863257	valid_0's binary_logloss: 0.331129
    [642]	valid_0's auc: 0.86329	valid_0's binary_logloss: 0.331514
    [643]	valid_0's auc: 0.862725	valid_0's binary_logloss: 0.330756
    [644]	valid_0's auc: 0.862756	valid_0's binary_logloss: 0.331268
    [645]	valid_0's auc: 0.862744	valid_0's binary_logloss: 0.330761
    [646]	valid_0's auc: 0.86282	valid_0's binary_logloss: 0.330065
    [647]	valid_0's auc: 0.862803	valid_0's binary_logloss: 0.329353
    [648]	valid_0's auc: 0.86275	valid_0's binary_logloss: 0.328623
    [649]	valid_0's auc: 0.862717	valid_0's binary_logloss: 0.329119
    [650]	valid_0's auc: 0.86271	valid_0's binary_logloss: 0.329568
    [651]	valid_0's auc: 0.862334	valid_0's binary_logloss: 0.328689
    [652]	valid_0's auc: 0.862339	valid_0's binary_logloss: 0.329122
    [653]	valid_0's auc: 0.86234	valid_0's binary_logloss: 0.329568
    [654]	valid_0's auc: 0.86234	valid_0's binary_logloss: 0.330001
    [655]	valid_0's auc: 0.86245	valid_0's binary_logloss: 0.329231
    [656]	valid_0's auc: 0.862522	valid_0's binary_logloss: 0.328605
    [657]	valid_0's auc: 0.862548	valid_0's binary_logloss: 0.329091
    [658]	valid_0's auc: 0.862219	valid_0's binary_logloss: 0.328248
    [659]	valid_0's auc: 0.862217	valid_0's binary_logloss: 0.328712
    [660]	valid_0's auc: 0.862095	valid_0's binary_logloss: 0.327779
    [661]	valid_0's auc: 0.862085	valid_0's binary_logloss: 0.328239
    [662]	valid_0's auc: 0.861586	valid_0's binary_logloss: 0.327562
    [663]	valid_0's auc: 0.861592	valid_0's binary_logloss: 0.328002
    [664]	valid_0's auc: 0.861584	valid_0's binary_logloss: 0.328465
    [665]	valid_0's auc: 0.861598	valid_0's binary_logloss: 0.32889
    [666]	valid_0's auc: 0.861625	valid_0's binary_logloss: 0.329333
    [667]	valid_0's auc: 0.861565	valid_0's binary_logloss: 0.329852
    [668]	valid_0's auc: 0.861612	valid_0's binary_logloss: 0.330292
    [669]	valid_0's auc: 0.861909	valid_0's binary_logloss: 0.329765
    [670]	valid_0's auc: 0.86189	valid_0's binary_logloss: 0.329089
    [671]	valid_0's auc: 0.86189	valid_0's binary_logloss: 0.329462
    [672]	valid_0's auc: 0.862088	valid_0's binary_logloss: 0.32892
    [673]	valid_0's auc: 0.862094	valid_0's binary_logloss: 0.329327
    [674]	valid_0's auc: 0.862042	valid_0's binary_logloss: 0.328719
    [675]	valid_0's auc: 0.862392	valid_0's binary_logloss: 0.327945
    [676]	valid_0's auc: 0.862563	valid_0's binary_logloss: 0.327195
    [677]	valid_0's auc: 0.862797	valid_0's binary_logloss: 0.326451
    [678]	valid_0's auc: 0.862762	valid_0's binary_logloss: 0.326845
    [679]	valid_0's auc: 0.862754	valid_0's binary_logloss: 0.326327
    [680]	valid_0's auc: 0.863258	valid_0's binary_logloss: 0.325604
    [681]	valid_0's auc: 0.863244	valid_0's binary_logloss: 0.32601
    [682]	valid_0's auc: 0.863265	valid_0's binary_logloss: 0.326435
    [683]	valid_0's auc: 0.863843	valid_0's binary_logloss: 0.325319
    [684]	valid_0's auc: 0.864733	valid_0's binary_logloss: 0.324111
    [685]	valid_0's auc: 0.86471	valid_0's binary_logloss: 0.324613
    [686]	valid_0's auc: 0.864629	valid_0's binary_logloss: 0.323856
    [687]	valid_0's auc: 0.864583	valid_0's binary_logloss: 0.324329
    [688]	valid_0's auc: 0.864586	valid_0's binary_logloss: 0.324811
    [689]	valid_0's auc: 0.864595	valid_0's binary_logloss: 0.324152
    [690]	valid_0's auc: 0.864473	valid_0's binary_logloss: 0.323523
    [691]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.323962
    [692]	valid_0's auc: 0.86448	valid_0's binary_logloss: 0.324415
    [693]	valid_0's auc: 0.864402	valid_0's binary_logloss: 0.323808
    [694]	valid_0's auc: 0.864646	valid_0's binary_logloss: 0.32316
    [695]	valid_0's auc: 0.864674	valid_0's binary_logloss: 0.323588
    [696]	valid_0's auc: 0.864868	valid_0's binary_logloss: 0.322935
    [697]	valid_0's auc: 0.864782	valid_0's binary_logloss: 0.32241
    [698]	valid_0's auc: 0.864807	valid_0's binary_logloss: 0.321774
    [699]	valid_0's auc: 0.864818	valid_0's binary_logloss: 0.322247
    [700]	valid_0's auc: 0.864567	valid_0's binary_logloss: 0.321437
    [701]	valid_0's auc: 0.86473	valid_0's binary_logloss: 0.320774
    [702]	valid_0's auc: 0.865298	valid_0's binary_logloss: 0.319711
    [703]	valid_0's auc: 0.865295	valid_0's binary_logloss: 0.320096
    [704]	valid_0's auc: 0.86532	valid_0's binary_logloss: 0.320552
    [705]	valid_0's auc: 0.865357	valid_0's binary_logloss: 0.320974
    [706]	valid_0's auc: 0.865385	valid_0's binary_logloss: 0.321351
    [707]	valid_0's auc: 0.865401	valid_0's binary_logloss: 0.321735
    [708]	valid_0's auc: 0.865189	valid_0's binary_logloss: 0.320927
    [709]	valid_0's auc: 0.865082	valid_0's binary_logloss: 0.320243
    [710]	valid_0's auc: 0.865205	valid_0's binary_logloss: 0.319667
    [711]	valid_0's auc: 0.865224	valid_0's binary_logloss: 0.320025
    [712]	valid_0's auc: 0.865247	valid_0's binary_logloss: 0.320437
    [713]	valid_0's auc: 0.865231	valid_0's binary_logloss: 0.319798
    [714]	valid_0's auc: 0.865257	valid_0's binary_logloss: 0.31908
    [715]	valid_0's auc: 0.865256	valid_0's binary_logloss: 0.319555
    [716]	valid_0's auc: 0.865465	valid_0's binary_logloss: 0.318639
    [717]	valid_0's auc: 0.865497	valid_0's binary_logloss: 0.319051
    [718]	valid_0's auc: 0.865464	valid_0's binary_logloss: 0.319477
    [719]	valid_0's auc: 0.865492	valid_0's binary_logloss: 0.319939
    [720]	valid_0's auc: 0.865481	valid_0's binary_logloss: 0.320388
    [721]	valid_0's auc: 0.865579	valid_0's binary_logloss: 0.319715
    [722]	valid_0's auc: 0.865589	valid_0's binary_logloss: 0.320128
    [723]	valid_0's auc: 0.865441	valid_0's binary_logloss: 0.319157
    [724]	valid_0's auc: 0.86555	valid_0's binary_logloss: 0.318911
    [725]	valid_0's auc: 0.865533	valid_0's binary_logloss: 0.319339
    [726]	valid_0's auc: 0.865888	valid_0's binary_logloss: 0.318605
    [727]	valid_0's auc: 0.865882	valid_0's binary_logloss: 0.31776
    [728]	valid_0's auc: 0.865771	valid_0's binary_logloss: 0.31715
    [729]	valid_0's auc: 0.865783	valid_0's binary_logloss: 0.317543
    [730]	valid_0's auc: 0.86578	valid_0's binary_logloss: 0.31701
    [731]	valid_0's auc: 0.865784	valid_0's binary_logloss: 0.317408
    [732]	valid_0's auc: 0.865795	valid_0's binary_logloss: 0.317827
    [733]	valid_0's auc: 0.865815	valid_0's binary_logloss: 0.317238
    [734]	valid_0's auc: 0.865821	valid_0's binary_logloss: 0.317658
    [735]	valid_0's auc: 0.866188	valid_0's binary_logloss: 0.31667
    [736]	valid_0's auc: 0.866196	valid_0's binary_logloss: 0.317081
    [737]	valid_0's auc: 0.866227	valid_0's binary_logloss: 0.317452
    [738]	valid_0's auc: 0.866171	valid_0's binary_logloss: 0.316831
    [739]	valid_0's auc: 0.866191	valid_0's binary_logloss: 0.316198
    [740]	valid_0's auc: 0.866323	valid_0's binary_logloss: 0.315625
    [741]	valid_0's auc: 0.866306	valid_0's binary_logloss: 0.315964
    [742]	valid_0's auc: 0.866265	valid_0's binary_logloss: 0.315515
    [743]	valid_0's auc: 0.866201	valid_0's binary_logloss: 0.314861
    [744]	valid_0's auc: 0.866696	valid_0's binary_logloss: 0.314219
    [745]	valid_0's auc: 0.866697	valid_0's binary_logloss: 0.314574
    [746]	valid_0's auc: 0.866685	valid_0's binary_logloss: 0.31496
    [747]	valid_0's auc: 0.866458	valid_0's binary_logloss: 0.314024
    [748]	valid_0's auc: 0.86688	valid_0's binary_logloss: 0.313456
    [749]	valid_0's auc: 0.866897	valid_0's binary_logloss: 0.313867
    [750]	valid_0's auc: 0.866664	valid_0's binary_logloss: 0.313067
    [751]	valid_0's auc: 0.866641	valid_0's binary_logloss: 0.312541
    [752]	valid_0's auc: 0.866668	valid_0's binary_logloss: 0.312939
    [753]	valid_0's auc: 0.866556	valid_0's binary_logloss: 0.312305
    [754]	valid_0's auc: 0.866558	valid_0's binary_logloss: 0.312737
    [755]	valid_0's auc: 0.866327	valid_0's binary_logloss: 0.311954
    [756]	valid_0's auc: 0.866339	valid_0's binary_logloss: 0.312343
    [757]	valid_0's auc: 0.866329	valid_0's binary_logloss: 0.312689
    [758]	valid_0's auc: 0.866311	valid_0's binary_logloss: 0.312059
    [759]	valid_0's auc: 0.866561	valid_0's binary_logloss: 0.311598
    [760]	valid_0's auc: 0.866551	valid_0's binary_logloss: 0.312069
    [761]	valid_0's auc: 0.866737	valid_0's binary_logloss: 0.311562
    [762]	valid_0's auc: 0.866308	valid_0's binary_logloss: 0.31097
    [763]	valid_0's auc: 0.866287	valid_0's binary_logloss: 0.311323
    [764]	valid_0's auc: 0.866309	valid_0's binary_logloss: 0.310695
    [765]	valid_0's auc: 0.866306	valid_0's binary_logloss: 0.311043
    [766]	valid_0's auc: 0.866201	valid_0's binary_logloss: 0.310237
    [767]	valid_0's auc: 0.866171	valid_0's binary_logloss: 0.310624
    [768]	valid_0's auc: 0.866172	valid_0's binary_logloss: 0.310973
    [769]	valid_0's auc: 0.866506	valid_0's binary_logloss: 0.310044
    [770]	valid_0's auc: 0.866558	valid_0's binary_logloss: 0.310394
    [771]	valid_0's auc: 0.866565	valid_0's binary_logloss: 0.310718
    [772]	valid_0's auc: 0.866518	valid_0's binary_logloss: 0.311091
    [773]	valid_0's auc: 0.866533	valid_0's binary_logloss: 0.311489
    [774]	valid_0's auc: 0.866522	valid_0's binary_logloss: 0.311792
    [775]	valid_0's auc: 0.866397	valid_0's binary_logloss: 0.311155
    [776]	valid_0's auc: 0.866767	valid_0's binary_logloss: 0.31017
    [777]	valid_0's auc: 0.866751	valid_0's binary_logloss: 0.309518
    [778]	valid_0's auc: 0.866697	valid_0's binary_logloss: 0.308917
    [779]	valid_0's auc: 0.866708	valid_0's binary_logloss: 0.309317
    [780]	valid_0's auc: 0.866724	valid_0's binary_logloss: 0.309703
    [781]	valid_0's auc: 0.866798	valid_0's binary_logloss: 0.308998
    [782]	valid_0's auc: 0.866947	valid_0's binary_logloss: 0.308313
    [783]	valid_0's auc: 0.866934	valid_0's binary_logloss: 0.308753
    [784]	valid_0's auc: 0.866892	valid_0's binary_logloss: 0.309177
    [785]	valid_0's auc: 0.866893	valid_0's binary_logloss: 0.308556
    [786]	valid_0's auc: 0.866907	valid_0's binary_logloss: 0.308926
    [787]	valid_0's auc: 0.866907	valid_0's binary_logloss: 0.309352
    [788]	valid_0's auc: 0.866888	valid_0's binary_logloss: 0.309734
    [789]	valid_0's auc: 0.866727	valid_0's binary_logloss: 0.30905
    [790]	valid_0's auc: 0.866749	valid_0's binary_logloss: 0.308478
    [791]	valid_0's auc: 0.867002	valid_0's binary_logloss: 0.307624
    [792]	valid_0's auc: 0.867034	valid_0's binary_logloss: 0.307982
    [793]	valid_0's auc: 0.867018	valid_0's binary_logloss: 0.307402
    [794]	valid_0's auc: 0.867022	valid_0's binary_logloss: 0.307764
    [795]	valid_0's auc: 0.867125	valid_0's binary_logloss: 0.30726
    [796]	valid_0's auc: 0.867141	valid_0's binary_logloss: 0.307587
    [797]	valid_0's auc: 0.867151	valid_0's binary_logloss: 0.307986
    [798]	valid_0's auc: 0.867142	valid_0's binary_logloss: 0.308315
    [799]	valid_0's auc: 0.867078	valid_0's binary_logloss: 0.307762
    [800]	valid_0's auc: 0.867054	valid_0's binary_logloss: 0.308146
    [801]	valid_0's auc: 0.866828	valid_0's binary_logloss: 0.307458
    [802]	valid_0's auc: 0.866812	valid_0's binary_logloss: 0.307754
    [803]	valid_0's auc: 0.866837	valid_0's binary_logloss: 0.30805
    [804]	valid_0's auc: 0.866875	valid_0's binary_logloss: 0.308413
    [805]	valid_0's auc: 0.866869	valid_0's binary_logloss: 0.308741
    [806]	valid_0's auc: 0.866886	valid_0's binary_logloss: 0.309058
    [807]	valid_0's auc: 0.867628	valid_0's binary_logloss: 0.307966
    [808]	valid_0's auc: 0.867841	valid_0's binary_logloss: 0.307208
    [809]	valid_0's auc: 0.867845	valid_0's binary_logloss: 0.307529
    [810]	valid_0's auc: 0.867676	valid_0's binary_logloss: 0.306727
    [811]	valid_0's auc: 0.867451	valid_0's binary_logloss: 0.305977
    [812]	valid_0's auc: 0.867462	valid_0's binary_logloss: 0.306338
    [813]	valid_0's auc: 0.867432	valid_0's binary_logloss: 0.305677
    [814]	valid_0's auc: 0.86742	valid_0's binary_logloss: 0.305988
    [815]	valid_0's auc: 0.867392	valid_0's binary_logloss: 0.305395
    [816]	valid_0's auc: 0.867439	valid_0's binary_logloss: 0.304769
    [817]	valid_0's auc: 0.867432	valid_0's binary_logloss: 0.305083
    [818]	valid_0's auc: 0.867862	valid_0's binary_logloss: 0.304118
    [819]	valid_0's auc: 0.86784	valid_0's binary_logloss: 0.304442
    [820]	valid_0's auc: 0.867816	valid_0's binary_logloss: 0.304837
    [821]	valid_0's auc: 0.867984	valid_0's binary_logloss: 0.30453
    [822]	valid_0's auc: 0.867974	valid_0's binary_logloss: 0.304839
    [823]	valid_0's auc: 0.867701	valid_0's binary_logloss: 0.304117
    [824]	valid_0's auc: 0.867826	valid_0's binary_logloss: 0.303539
    [825]	valid_0's auc: 0.867818	valid_0's binary_logloss: 0.30302
    [826]	valid_0's auc: 0.867856	valid_0's binary_logloss: 0.303352
    [827]	valid_0's auc: 0.867868	valid_0's binary_logloss: 0.303684
    [828]	valid_0's auc: 0.867896	valid_0's binary_logloss: 0.303071
    [829]	valid_0's auc: 0.867887	valid_0's binary_logloss: 0.302583
    [830]	valid_0's auc: 0.867892	valid_0's binary_logloss: 0.301994
    [831]	valid_0's auc: 0.867976	valid_0's binary_logloss: 0.301454
    [832]	valid_0's auc: 0.867882	valid_0's binary_logloss: 0.300953
    [833]	valid_0's auc: 0.867793	valid_0's binary_logloss: 0.300236
    [834]	valid_0's auc: 0.867785	valid_0's binary_logloss: 0.300536
    [835]	valid_0's auc: 0.867782	valid_0's binary_logloss: 0.300827
    [836]	valid_0's auc: 0.867733	valid_0's binary_logloss: 0.300136
    [837]	valid_0's auc: 0.867698	valid_0's binary_logloss: 0.299613
    [838]	valid_0's auc: 0.867721	valid_0's binary_logloss: 0.300017
    [839]	valid_0's auc: 0.867438	valid_0's binary_logloss: 0.299391
    [840]	valid_0's auc: 0.867465	valid_0's binary_logloss: 0.298858
    [841]	valid_0's auc: 0.86717	valid_0's binary_logloss: 0.298267
    [842]	valid_0's auc: 0.867202	valid_0's binary_logloss: 0.297797
    [843]	valid_0's auc: 0.867227	valid_0's binary_logloss: 0.297388
    [844]	valid_0's auc: 0.867232	valid_0's binary_logloss: 0.297711
    [845]	valid_0's auc: 0.867248	valid_0's binary_logloss: 0.29804
    [846]	valid_0's auc: 0.867276	valid_0's binary_logloss: 0.298335
    [847]	valid_0's auc: 0.867233	valid_0's binary_logloss: 0.297816
    [848]	valid_0's auc: 0.867248	valid_0's binary_logloss: 0.298147
    [849]	valid_0's auc: 0.867268	valid_0's binary_logloss: 0.298413
    [850]	valid_0's auc: 0.867406	valid_0's binary_logloss: 0.29791
    [851]	valid_0's auc: 0.867389	valid_0's binary_logloss: 0.298247
    [852]	valid_0's auc: 0.867274	valid_0's binary_logloss: 0.297789
    [853]	valid_0's auc: 0.867289	valid_0's binary_logloss: 0.297038
    [854]	valid_0's auc: 0.867304	valid_0's binary_logloss: 0.297314
    [855]	valid_0's auc: 0.86724	valid_0's binary_logloss: 0.296822
    [856]	valid_0's auc: 0.867261	valid_0's binary_logloss: 0.296298
    [857]	valid_0's auc: 0.867255	valid_0's binary_logloss: 0.29658
    [858]	valid_0's auc: 0.867243	valid_0's binary_logloss: 0.296889
    [859]	valid_0's auc: 0.867247	valid_0's binary_logloss: 0.297203
    [860]	valid_0's auc: 0.867044	valid_0's binary_logloss: 0.296522
    [861]	valid_0's auc: 0.866959	valid_0's binary_logloss: 0.295832
    [862]	valid_0's auc: 0.86716	valid_0's binary_logloss: 0.295373
    [863]	valid_0's auc: 0.867147	valid_0's binary_logloss: 0.295664
    [864]	valid_0's auc: 0.867144	valid_0's binary_logloss: 0.295996
    [865]	valid_0's auc: 0.867117	valid_0's binary_logloss: 0.295667
    [866]	valid_0's auc: 0.867132	valid_0's binary_logloss: 0.295221
    [867]	valid_0's auc: 0.867112	valid_0's binary_logloss: 0.29471
    [868]	valid_0's auc: 0.867112	valid_0's binary_logloss: 0.295019
    [869]	valid_0's auc: 0.867007	valid_0's binary_logloss: 0.294347
    [870]	valid_0's auc: 0.867013	valid_0's binary_logloss: 0.294694
    [871]	valid_0's auc: 0.866996	valid_0's binary_logloss: 0.295018
    [872]	valid_0's auc: 0.867027	valid_0's binary_logloss: 0.294494
    [873]	valid_0's auc: 0.867036	valid_0's binary_logloss: 0.294805
    [874]	valid_0's auc: 0.867034	valid_0's binary_logloss: 0.295083
    [875]	valid_0's auc: 0.867022	valid_0's binary_logloss: 0.295417
    [876]	valid_0's auc: 0.867071	valid_0's binary_logloss: 0.294873
    [877]	valid_0's auc: 0.867085	valid_0's binary_logloss: 0.295146
    [878]	valid_0's auc: 0.867069	valid_0's binary_logloss: 0.294618
    [879]	valid_0's auc: 0.867078	valid_0's binary_logloss: 0.294918
    [880]	valid_0's auc: 0.867062	valid_0's binary_logloss: 0.29526
    [881]	valid_0's auc: 0.867086	valid_0's binary_logloss: 0.295565
    [882]	valid_0's auc: 0.867088	valid_0's binary_logloss: 0.29504
    [883]	valid_0's auc: 0.867073	valid_0's binary_logloss: 0.295332
    [884]	valid_0's auc: 0.867417	valid_0's binary_logloss: 0.294582
    [885]	valid_0's auc: 0.867441	valid_0's binary_logloss: 0.294052
    [886]	valid_0's auc: 0.867588	valid_0's binary_logloss: 0.293505
    [887]	valid_0's auc: 0.86752	valid_0's binary_logloss: 0.29285
    [888]	valid_0's auc: 0.867453	valid_0's binary_logloss: 0.292562
    [889]	valid_0's auc: 0.867444	valid_0's binary_logloss: 0.292917
    [890]	valid_0's auc: 0.867423	valid_0's binary_logloss: 0.292521
    [891]	valid_0's auc: 0.867465	valid_0's binary_logloss: 0.29205
    [892]	valid_0's auc: 0.867337	valid_0's binary_logloss: 0.291331
    [893]	valid_0's auc: 0.867356	valid_0's binary_logloss: 0.291537
    [894]	valid_0's auc: 0.867339	valid_0's binary_logloss: 0.291155
    [895]	valid_0's auc: 0.867359	valid_0's binary_logloss: 0.291425
    [896]	valid_0's auc: 0.867381	valid_0's binary_logloss: 0.291717
    [897]	valid_0's auc: 0.867385	valid_0's binary_logloss: 0.292051
    [898]	valid_0's auc: 0.86737	valid_0's binary_logloss: 0.292355
    [899]	valid_0's auc: 0.86745	valid_0's binary_logloss: 0.292149
    [900]	valid_0's auc: 0.867531	valid_0's binary_logloss: 0.291715
    [901]	valid_0's auc: 0.867537	valid_0's binary_logloss: 0.291185
    [902]	valid_0's auc: 0.867559	valid_0's binary_logloss: 0.291437
    [903]	valid_0's auc: 0.867905	valid_0's binary_logloss: 0.290863
    [904]	valid_0's auc: 0.867744	valid_0's binary_logloss: 0.290275
    [905]	valid_0's auc: 0.86775	valid_0's binary_logloss: 0.290564
    [906]	valid_0's auc: 0.867725	valid_0's binary_logloss: 0.290929
    [907]	valid_0's auc: 0.867981	valid_0's binary_logloss: 0.290404
    [908]	valid_0's auc: 0.867965	valid_0's binary_logloss: 0.290701
    [909]	valid_0's auc: 0.867976	valid_0's binary_logloss: 0.290962
    [910]	valid_0's auc: 0.868009	valid_0's binary_logloss: 0.291296
    [911]	valid_0's auc: 0.867999	valid_0's binary_logloss: 0.290866
    [912]	valid_0's auc: 0.868008	valid_0's binary_logloss: 0.291138
    [913]	valid_0's auc: 0.868013	valid_0's binary_logloss: 0.291392
    [914]	valid_0's auc: 0.868013	valid_0's binary_logloss: 0.290972
    [915]	valid_0's auc: 0.868006	valid_0's binary_logloss: 0.290466
    [916]	valid_0's auc: 0.868013	valid_0's binary_logloss: 0.290777
    [917]	valid_0's auc: 0.868022	valid_0's binary_logloss: 0.291055
    [918]	valid_0's auc: 0.868026	valid_0's binary_logloss: 0.291334
    [919]	valid_0's auc: 0.868016	valid_0's binary_logloss: 0.291664
    [920]	valid_0's auc: 0.868054	valid_0's binary_logloss: 0.291945
    [921]	valid_0's auc: 0.868037	valid_0's binary_logloss: 0.292231
    [922]	valid_0's auc: 0.868026	valid_0's binary_logloss: 0.29253
    [923]	valid_0's auc: 0.868038	valid_0's binary_logloss: 0.292775
    [924]	valid_0's auc: 0.868124	valid_0's binary_logloss: 0.292195
    [925]	valid_0's auc: 0.868018	valid_0's binary_logloss: 0.291739
    [926]	valid_0's auc: 0.868683	valid_0's binary_logloss: 0.29088
    [927]	valid_0's auc: 0.868692	valid_0's binary_logloss: 0.29117
    [928]	valid_0's auc: 0.868669	valid_0's binary_logloss: 0.291438
    [929]	valid_0's auc: 0.868405	valid_0's binary_logloss: 0.290867
    [930]	valid_0's auc: 0.868493	valid_0's binary_logloss: 0.290398
    [931]	valid_0's auc: 0.868503	valid_0's binary_logloss: 0.290645
    [932]	valid_0's auc: 0.868497	valid_0's binary_logloss: 0.290931
    [933]	valid_0's auc: 0.868496	valid_0's binary_logloss: 0.290448
    [934]	valid_0's auc: 0.868297	valid_0's binary_logloss: 0.289869
    [935]	valid_0's auc: 0.868301	valid_0's binary_logloss: 0.289426
    [936]	valid_0's auc: 0.868281	valid_0's binary_logloss: 0.289746
    [937]	valid_0's auc: 0.86831	valid_0's binary_logloss: 0.290094
    [938]	valid_0's auc: 0.868322	valid_0's binary_logloss: 0.289652
    [939]	valid_0's auc: 0.868326	valid_0's binary_logloss: 0.289912
    [940]	valid_0's auc: 0.86829	valid_0's binary_logloss: 0.289424
    [941]	valid_0's auc: 0.868302	valid_0's binary_logloss: 0.28966
    [942]	valid_0's auc: 0.867946	valid_0's binary_logloss: 0.289179
    [943]	valid_0's auc: 0.867962	valid_0's binary_logloss: 0.289452
    [944]	valid_0's auc: 0.867975	valid_0's binary_logloss: 0.288939
    [945]	valid_0's auc: 0.867979	valid_0's binary_logloss: 0.289272
    [946]	valid_0's auc: 0.868007	valid_0's binary_logloss: 0.289504
    [947]	valid_0's auc: 0.867769	valid_0's binary_logloss: 0.288937
    [948]	valid_0's auc: 0.867779	valid_0's binary_logloss: 0.288451
    [949]	valid_0's auc: 0.868219	valid_0's binary_logloss: 0.28778
    [950]	valid_0's auc: 0.868071	valid_0's binary_logloss: 0.287244
    [951]	valid_0's auc: 0.868384	valid_0's binary_logloss: 0.286571
    [952]	valid_0's auc: 0.86838	valid_0's binary_logloss: 0.286845
    [953]	valid_0's auc: 0.868376	valid_0's binary_logloss: 0.287109
    [954]	valid_0's auc: 0.868387	valid_0's binary_logloss: 0.287369
    [955]	valid_0's auc: 0.868377	valid_0's binary_logloss: 0.286931
    [956]	valid_0's auc: 0.868981	valid_0's binary_logloss: 0.286076
    [957]	valid_0's auc: 0.869071	valid_0's binary_logloss: 0.285519
    [958]	valid_0's auc: 0.869049	valid_0's binary_logloss: 0.285783
    [959]	valid_0's auc: 0.86911	valid_0's binary_logloss: 0.285321
    [960]	valid_0's auc: 0.868988	valid_0's binary_logloss: 0.28477
    [961]	valid_0's auc: 0.869209	valid_0's binary_logloss: 0.284289
    [962]	valid_0's auc: 0.86911	valid_0's binary_logloss: 0.283649
    [963]	valid_0's auc: 0.869231	valid_0's binary_logloss: 0.283125
    [964]	valid_0's auc: 0.869226	valid_0's binary_logloss: 0.283354
    [965]	valid_0's auc: 0.869289	valid_0's binary_logloss: 0.282974
    [966]	valid_0's auc: 0.869276	valid_0's binary_logloss: 0.283256
    [967]	valid_0's auc: 0.869275	valid_0's binary_logloss: 0.283519
    [968]	valid_0's auc: 0.869267	valid_0's binary_logloss: 0.283784
    [969]	valid_0's auc: 0.869274	valid_0's binary_logloss: 0.28404
    [970]	valid_0's auc: 0.869592	valid_0's binary_logloss: 0.283335
    [971]	valid_0's auc: 0.869604	valid_0's binary_logloss: 0.282892
    [972]	valid_0's auc: 0.870355	valid_0's binary_logloss: 0.282027
    [973]	valid_0's auc: 0.870367	valid_0's binary_logloss: 0.282361
    [974]	valid_0's auc: 0.870359	valid_0's binary_logloss: 0.282636
    [975]	valid_0's auc: 0.870352	valid_0's binary_logloss: 0.282931
    [976]	valid_0's auc: 0.870213	valid_0's binary_logloss: 0.282392
    [977]	valid_0's auc: 0.870245	valid_0's binary_logloss: 0.282636
    [978]	valid_0's auc: 0.870041	valid_0's binary_logloss: 0.28215
    [979]	valid_0's auc: 0.87006	valid_0's binary_logloss: 0.282401
    [980]	valid_0's auc: 0.870163	valid_0's binary_logloss: 0.281861
    [981]	valid_0's auc: 0.870163	valid_0's binary_logloss: 0.282101
    [982]	valid_0's auc: 0.870144	valid_0's binary_logloss: 0.282376
    [983]	valid_0's auc: 0.869946	valid_0's binary_logloss: 0.28185
    [984]	valid_0's auc: 0.869795	valid_0's binary_logloss: 0.28134
    [985]	valid_0's auc: 0.869812	valid_0's binary_logloss: 0.28159
    [986]	valid_0's auc: 0.869824	valid_0's binary_logloss: 0.281836
    [987]	valid_0's auc: 0.869804	valid_0's binary_logloss: 0.282102
    [988]	valid_0's auc: 0.869614	valid_0's binary_logloss: 0.28162
    [989]	valid_0's auc: 0.869634	valid_0's binary_logloss: 0.281889
    [990]	valid_0's auc: 0.869651	valid_0's binary_logloss: 0.282083
    [991]	valid_0's auc: 0.869586	valid_0's binary_logloss: 0.281663
    [992]	valid_0's auc: 0.869916	valid_0's binary_logloss: 0.281145
    [993]	valid_0's auc: 0.869921	valid_0's binary_logloss: 0.281383
    [994]	valid_0's auc: 0.870726	valid_0's binary_logloss: 0.280522
    [995]	valid_0's auc: 0.870745	valid_0's binary_logloss: 0.280785
    [996]	valid_0's auc: 0.870694	valid_0's binary_logloss: 0.280418
    [997]	valid_0's auc: 0.870508	valid_0's binary_logloss: 0.279894
    [998]	valid_0's auc: 0.870523	valid_0's binary_logloss: 0.280115
    [999]	valid_0's auc: 0.871192	valid_0's binary_logloss: 0.279361
    [1000]	valid_0's auc: 0.871188	valid_0's binary_logloss: 0.279578
    [1001]	valid_0's auc: 0.871194	valid_0's binary_logloss: 0.27982
    [1002]	valid_0's auc: 0.871193	valid_0's binary_logloss: 0.280094
    [1003]	valid_0's auc: 0.871314	valid_0's binary_logloss: 0.279686
    [1004]	valid_0's auc: 0.871302	valid_0's binary_logloss: 0.279305
    [1005]	valid_0's auc: 0.871283	valid_0's binary_logloss: 0.278944
    [1006]	valid_0's auc: 0.871309	valid_0's binary_logloss: 0.279149
    [1007]	valid_0's auc: 0.871595	valid_0's binary_logloss: 0.278748
    [1008]	valid_0's auc: 0.871637	valid_0's binary_logloss: 0.278277
    [1009]	valid_0's auc: 0.871577	valid_0's binary_logloss: 0.277829
    [1010]	valid_0's auc: 0.871601	valid_0's binary_logloss: 0.277479
    [1011]	valid_0's auc: 0.871599	valid_0's binary_logloss: 0.277722
    [1012]	valid_0's auc: 0.871582	valid_0's binary_logloss: 0.277469
    [1013]	valid_0's auc: 0.87159	valid_0's binary_logloss: 0.277731
    [1014]	valid_0's auc: 0.87225	valid_0's binary_logloss: 0.276884
    [1015]	valid_0's auc: 0.872285	valid_0's binary_logloss: 0.276408
    [1016]	valid_0's auc: 0.872123	valid_0's binary_logloss: 0.27597
    [1017]	valid_0's auc: 0.872129	valid_0's binary_logloss: 0.276163
    [1018]	valid_0's auc: 0.872119	valid_0's binary_logloss: 0.276351
    [1019]	valid_0's auc: 0.872131	valid_0's binary_logloss: 0.276611
    [1020]	valid_0's auc: 0.872108	valid_0's binary_logloss: 0.276054
    [1021]	valid_0's auc: 0.87211	valid_0's binary_logloss: 0.275799
    [1022]	valid_0's auc: 0.872113	valid_0's binary_logloss: 0.276026
    [1023]	valid_0's auc: 0.87205	valid_0's binary_logloss: 0.275556
    [1024]	valid_0's auc: 0.872116	valid_0's binary_logloss: 0.275097
    [1025]	valid_0's auc: 0.872105	valid_0's binary_logloss: 0.275347
    [1026]	valid_0's auc: 0.872089	valid_0's binary_logloss: 0.275586
    [1027]	valid_0's auc: 0.872797	valid_0's binary_logloss: 0.274746
    [1028]	valid_0's auc: 0.87282	valid_0's binary_logloss: 0.27496
    [1029]	valid_0's auc: 0.872832	valid_0's binary_logloss: 0.274438
    [1030]	valid_0's auc: 0.872821	valid_0's binary_logloss: 0.274716
    [1031]	valid_0's auc: 0.872799	valid_0's binary_logloss: 0.274961
    [1032]	valid_0's auc: 0.872744	valid_0's binary_logloss: 0.2746
    [1033]	valid_0's auc: 0.872733	valid_0's binary_logloss: 0.274837
    [1034]	valid_0's auc: 0.873002	valid_0's binary_logloss: 0.274283
    [1035]	valid_0's auc: 0.873082	valid_0's binary_logloss: 0.274091
    [1036]	valid_0's auc: 0.873043	valid_0's binary_logloss: 0.273691
    [1037]	valid_0's auc: 0.873114	valid_0's binary_logloss: 0.27325
    [1038]	valid_0's auc: 0.873142	valid_0's binary_logloss: 0.273496
    [1039]	valid_0's auc: 0.873169	valid_0's binary_logloss: 0.273691
    [1040]	valid_0's auc: 0.873167	valid_0's binary_logloss: 0.273925
    [1041]	valid_0's auc: 0.873452	valid_0's binary_logloss: 0.273218
    [1042]	valid_0's auc: 0.873464	valid_0's binary_logloss: 0.273401
    [1043]	valid_0's auc: 0.873467	valid_0's binary_logloss: 0.272974
    [1044]	valid_0's auc: 0.873378	valid_0's binary_logloss: 0.272584
    [1045]	valid_0's auc: 0.873886	valid_0's binary_logloss: 0.271928
    [1046]	valid_0's auc: 0.873914	valid_0's binary_logloss: 0.27213
    [1047]	valid_0's auc: 0.873917	valid_0's binary_logloss: 0.272379
    [1048]	valid_0's auc: 0.873854	valid_0's binary_logloss: 0.272028
    [1049]	valid_0's auc: 0.87384	valid_0's binary_logloss: 0.272258
    [1050]	valid_0's auc: 0.873662	valid_0's binary_logloss: 0.271812
    [1051]	valid_0's auc: 0.873664	valid_0's binary_logloss: 0.271984
    [1052]	valid_0's auc: 0.873686	valid_0's binary_logloss: 0.272153
    [1053]	valid_0's auc: 0.873634	valid_0's binary_logloss: 0.271783
    [1054]	valid_0's auc: 0.873751	valid_0's binary_logloss: 0.271403
    [1055]	valid_0's auc: 0.873741	valid_0's binary_logloss: 0.271644
    [1056]	valid_0's auc: 0.87369	valid_0's binary_logloss: 0.271062
    [1057]	valid_0's auc: 0.873692	valid_0's binary_logloss: 0.27132
    [1058]	valid_0's auc: 0.873701	valid_0's binary_logloss: 0.271557
    [1059]	valid_0's auc: 0.873704	valid_0's binary_logloss: 0.271767
    [1060]	valid_0's auc: 0.873704	valid_0's binary_logloss: 0.271981
    [1061]	valid_0's auc: 0.873763	valid_0's binary_logloss: 0.272195
    [1062]	valid_0's auc: 0.87378	valid_0's binary_logloss: 0.272411
    [1063]	valid_0's auc: 0.873903	valid_0's binary_logloss: 0.271754
    [1064]	valid_0's auc: 0.873928	valid_0's binary_logloss: 0.271949
    [1065]	valid_0's auc: 0.873948	valid_0's binary_logloss: 0.272123
    [1066]	valid_0's auc: 0.873989	valid_0's binary_logloss: 0.271697
    [1067]	valid_0's auc: 0.87399	valid_0's binary_logloss: 0.271881
    [1068]	valid_0's auc: 0.873859	valid_0's binary_logloss: 0.271536
    [1069]	valid_0's auc: 0.873846	valid_0's binary_logloss: 0.271762
    [1070]	valid_0's auc: 0.874335	valid_0's binary_logloss: 0.271024
    [1071]	valid_0's auc: 0.874326	valid_0's binary_logloss: 0.270651
    [1072]	valid_0's auc: 0.874317	valid_0's binary_logloss: 0.270921
    [1073]	valid_0's auc: 0.874265	valid_0's binary_logloss: 0.270435
    [1074]	valid_0's auc: 0.874282	valid_0's binary_logloss: 0.270652
    [1075]	valid_0's auc: 0.874507	valid_0's binary_logloss: 0.270054
    [1076]	valid_0's auc: 0.874519	valid_0's binary_logloss: 0.270242
    [1077]	valid_0's auc: 0.874552	valid_0's binary_logloss: 0.270461
    [1078]	valid_0's auc: 0.87481	valid_0's binary_logloss: 0.269811
    [1079]	valid_0's auc: 0.87482	valid_0's binary_logloss: 0.270048
    [1080]	valid_0's auc: 0.875374	valid_0's binary_logloss: 0.269261
    [1081]	valid_0's auc: 0.875388	valid_0's binary_logloss: 0.268907
    [1082]	valid_0's auc: 0.875522	valid_0's binary_logloss: 0.26852
    [1083]	valid_0's auc: 0.875424	valid_0's binary_logloss: 0.26819
    [1084]	valid_0's auc: 0.875433	valid_0's binary_logloss: 0.268414
    [1085]	valid_0's auc: 0.87543	valid_0's binary_logloss: 0.268664
    [1086]	valid_0's auc: 0.87541	valid_0's binary_logloss: 0.26888
    [1087]	valid_0's auc: 0.875428	valid_0's binary_logloss: 0.269081
    [1088]	valid_0's auc: 0.875429	valid_0's binary_logloss: 0.26871
    [1089]	valid_0's auc: 0.875419	valid_0's binary_logloss: 0.268925
    [1090]	valid_0's auc: 0.875412	valid_0's binary_logloss: 0.269153
    [1091]	valid_0's auc: 0.875247	valid_0's binary_logloss: 0.268709
    [1092]	valid_0's auc: 0.875184	valid_0's binary_logloss: 0.268351
    [1093]	valid_0's auc: 0.875201	valid_0's binary_logloss: 0.268516
    [1094]	valid_0's auc: 0.875208	valid_0's binary_logloss: 0.268064
    [1095]	valid_0's auc: 0.87517	valid_0's binary_logloss: 0.267717
    [1096]	valid_0's auc: 0.875189	valid_0's binary_logloss: 0.267968
    [1097]	valid_0's auc: 0.875178	valid_0's binary_logloss: 0.267627
    [1098]	valid_0's auc: 0.875062	valid_0's binary_logloss: 0.267282
    [1099]	valid_0's auc: 0.87507	valid_0's binary_logloss: 0.267487
    [1100]	valid_0's auc: 0.875045	valid_0's binary_logloss: 0.267079
    [1101]	valid_0's auc: 0.875072	valid_0's binary_logloss: 0.267296
    [1102]	valid_0's auc: 0.875063	valid_0's binary_logloss: 0.267028
    [1103]	valid_0's auc: 0.875065	valid_0's binary_logloss: 0.267291
    [1104]	valid_0's auc: 0.875064	valid_0's binary_logloss: 0.267519
    [1105]	valid_0's auc: 0.875069	valid_0's binary_logloss: 0.267702
    [1106]	valid_0's auc: 0.875063	valid_0's binary_logloss: 0.267875
    [1107]	valid_0's auc: 0.875076	valid_0's binary_logloss: 0.268048
    [1108]	valid_0's auc: 0.875075	valid_0's binary_logloss: 0.268255
    [1109]	valid_0's auc: 0.87501	valid_0's binary_logloss: 0.26789
    [1110]	valid_0's auc: 0.87502	valid_0's binary_logloss: 0.268069
    [1111]	valid_0's auc: 0.875029	valid_0's binary_logloss: 0.268258
    [1112]	valid_0's auc: 0.875035	valid_0's binary_logloss: 0.268429
    [1113]	valid_0's auc: 0.875035	valid_0's binary_logloss: 0.268681
    [1114]	valid_0's auc: 0.875045	valid_0's binary_logloss: 0.268875
    [1115]	valid_0's auc: 0.875051	valid_0's binary_logloss: 0.269055
    [1116]	valid_0's auc: 0.87505	valid_0's binary_logloss: 0.26924
    [1117]	valid_0's auc: 0.875057	valid_0's binary_logloss: 0.269417
    [1118]	valid_0's auc: 0.875609	valid_0's binary_logloss: 0.268691
    [1119]	valid_0's auc: 0.875615	valid_0's binary_logloss: 0.268934
    [1120]	valid_0's auc: 0.875618	valid_0's binary_logloss: 0.269159
    [1121]	valid_0's auc: 0.875572	valid_0's binary_logloss: 0.268764
    [1122]	valid_0's auc: 0.875556	valid_0's binary_logloss: 0.268417
    [1123]	valid_0's auc: 0.876097	valid_0's binary_logloss: 0.267617
    [1124]	valid_0's auc: 0.876101	valid_0's binary_logloss: 0.267802
    [1125]	valid_0's auc: 0.876045	valid_0's binary_logloss: 0.26753
    [1126]	valid_0's auc: 0.876046	valid_0's binary_logloss: 0.267134
    [1127]	valid_0's auc: 0.876058	valid_0's binary_logloss: 0.267355
    [1128]	valid_0's auc: 0.875872	valid_0's binary_logloss: 0.266967
    [1129]	valid_0's auc: 0.875861	valid_0's binary_logloss: 0.266528
    [1130]	valid_0's auc: 0.875874	valid_0's binary_logloss: 0.266688
    [1131]	valid_0's auc: 0.875871	valid_0's binary_logloss: 0.266876
    [1132]	valid_0's auc: 0.875785	valid_0's binary_logloss: 0.266494
    [1133]	valid_0's auc: 0.875775	valid_0's binary_logloss: 0.266119
    [1134]	valid_0's auc: 0.876003	valid_0's binary_logloss: 0.265632
    [1135]	valid_0's auc: 0.875952	valid_0's binary_logloss: 0.26536
    [1136]	valid_0's auc: 0.875948	valid_0's binary_logloss: 0.265553
    [1137]	valid_0's auc: 0.875954	valid_0's binary_logloss: 0.26578
    [1138]	valid_0's auc: 0.875955	valid_0's binary_logloss: 0.265965
    [1139]	valid_0's auc: 0.875952	valid_0's binary_logloss: 0.266147
    [1140]	valid_0's auc: 0.875821	valid_0's binary_logloss: 0.265677
    [1141]	valid_0's auc: 0.875836	valid_0's binary_logloss: 0.265883
    [1142]	valid_0's auc: 0.875831	valid_0's binary_logloss: 0.266079
    [1143]	valid_0's auc: 0.87585	valid_0's binary_logloss: 0.265723
    [1144]	valid_0's auc: 0.875847	valid_0's binary_logloss: 0.265315
    [1145]	valid_0's auc: 0.875836	valid_0's binary_logloss: 0.264983
    [1146]	valid_0's auc: 0.875839	valid_0's binary_logloss: 0.265166
    [1147]	valid_0's auc: 0.875826	valid_0's binary_logloss: 0.264836
    [1148]	valid_0's auc: 0.875807	valid_0's binary_logloss: 0.265042
    [1149]	valid_0's auc: 0.875946	valid_0's binary_logloss: 0.264629
    [1150]	valid_0's auc: 0.875951	valid_0's binary_logloss: 0.264827
    [1151]	valid_0's auc: 0.875972	valid_0's binary_logloss: 0.265014
    [1152]	valid_0's auc: 0.875987	valid_0's binary_logloss: 0.265223
    [1153]	valid_0's auc: 0.875984	valid_0's binary_logloss: 0.265449
    [1154]	valid_0's auc: 0.87599	valid_0's binary_logloss: 0.265102
    [1155]	valid_0's auc: 0.875988	valid_0's binary_logloss: 0.265326
    [1156]	valid_0's auc: 0.875976	valid_0's binary_logloss: 0.264959
    [1157]	valid_0's auc: 0.875976	valid_0's binary_logloss: 0.264625
    [1158]	valid_0's auc: 0.875981	valid_0's binary_logloss: 0.264803
    [1159]	valid_0's auc: 0.875999	valid_0's binary_logloss: 0.264957
    [1160]	valid_0's auc: 0.876568	valid_0's binary_logloss: 0.264199
    [1161]	valid_0's auc: 0.876559	valid_0's binary_logloss: 0.264403
    [1162]	valid_0's auc: 0.876644	valid_0's binary_logloss: 0.263955
    [1163]	valid_0's auc: 0.876709	valid_0's binary_logloss: 0.263592
    [1164]	valid_0's auc: 0.876713	valid_0's binary_logloss: 0.263777
    [1165]	valid_0's auc: 0.876726	valid_0's binary_logloss: 0.264041
    [1166]	valid_0's auc: 0.876757	valid_0's binary_logloss: 0.263701
    [1167]	valid_0's auc: 0.876583	valid_0's binary_logloss: 0.263302
    [1168]	valid_0's auc: 0.876594	valid_0's binary_logloss: 0.263458
    [1169]	valid_0's auc: 0.876603	valid_0's binary_logloss: 0.263675
    [1170]	valid_0's auc: 0.876612	valid_0's binary_logloss: 0.263901
    [1171]	valid_0's auc: 0.876752	valid_0's binary_logloss: 0.263662
    [1172]	valid_0's auc: 0.876762	valid_0's binary_logloss: 0.263889
    [1173]	valid_0's auc: 0.876758	valid_0's binary_logloss: 0.263567
    [1174]	valid_0's auc: 0.876769	valid_0's binary_logloss: 0.263771
    [1175]	valid_0's auc: 0.876777	valid_0's binary_logloss: 0.263937
    [1176]	valid_0's auc: 0.876787	valid_0's binary_logloss: 0.263647
    [1177]	valid_0's auc: 0.876812	valid_0's binary_logloss: 0.263807
    [1178]	valid_0's auc: 0.876819	valid_0's binary_logloss: 0.264013
    [1179]	valid_0's auc: 0.876787	valid_0's binary_logloss: 0.263688
    [1180]	valid_0's auc: 0.87681	valid_0's binary_logloss: 0.263843
    [1181]	valid_0's auc: 0.876828	valid_0's binary_logloss: 0.264
    [1182]	valid_0's auc: 0.876766	valid_0's binary_logloss: 0.263699
    [1183]	valid_0's auc: 0.876756	valid_0's binary_logloss: 0.263312
    [1184]	valid_0's auc: 0.87664	valid_0's binary_logloss: 0.263026
    [1185]	valid_0's auc: 0.876563	valid_0's binary_logloss: 0.262515
    [1186]	valid_0's auc: 0.876562	valid_0's binary_logloss: 0.262746
    [1187]	valid_0's auc: 0.876762	valid_0's binary_logloss: 0.262201
    [1188]	valid_0's auc: 0.876763	valid_0's binary_logloss: 0.261912
    [1189]	valid_0's auc: 0.876773	valid_0's binary_logloss: 0.262073
    [1190]	valid_0's auc: 0.876787	valid_0's binary_logloss: 0.262281
    [1191]	valid_0's auc: 0.87678	valid_0's binary_logloss: 0.262488
    [1192]	valid_0's auc: 0.876791	valid_0's binary_logloss: 0.262699
    [1193]	valid_0's auc: 0.876803	valid_0's binary_logloss: 0.262883
    [1194]	valid_0's auc: 0.876719	valid_0's binary_logloss: 0.262565
    [1195]	valid_0's auc: 0.876745	valid_0's binary_logloss: 0.262382
    [1196]	valid_0's auc: 0.876849	valid_0's binary_logloss: 0.262052
    [1197]	valid_0's auc: 0.876702	valid_0's binary_logloss: 0.26169
    [1198]	valid_0's auc: 0.876675	valid_0's binary_logloss: 0.261357
    [1199]	valid_0's auc: 0.876672	valid_0's binary_logloss: 0.261613
    [1200]	valid_0's auc: 0.876674	valid_0's binary_logloss: 0.261829
    [1201]	valid_0's auc: 0.876651	valid_0's binary_logloss: 0.261526
    [1202]	valid_0's auc: 0.876774	valid_0's binary_logloss: 0.261226
    [1203]	valid_0's auc: 0.87694	valid_0's binary_logloss: 0.260805
    [1204]	valid_0's auc: 0.876948	valid_0's binary_logloss: 0.260957
    [1205]	valid_0's auc: 0.876953	valid_0's binary_logloss: 0.261129
    [1206]	valid_0's auc: 0.876941	valid_0's binary_logloss: 0.260789
    [1207]	valid_0's auc: 0.876973	valid_0's binary_logloss: 0.260501
    [1208]	valid_0's auc: 0.877012	valid_0's binary_logloss: 0.26013
    [1209]	valid_0's auc: 0.877042	valid_0's binary_logloss: 0.259813
    [1210]	valid_0's auc: 0.876968	valid_0's binary_logloss: 0.259557
    [1211]	valid_0's auc: 0.876823	valid_0's binary_logloss: 0.259174
    [1212]	valid_0's auc: 0.876831	valid_0's binary_logloss: 0.259318
    [1213]	valid_0's auc: 0.876876	valid_0's binary_logloss: 0.258964
    [1214]	valid_0's auc: 0.876813	valid_0's binary_logloss: 0.258468
    [1215]	valid_0's auc: 0.876823	valid_0's binary_logloss: 0.258641
    [1216]	valid_0's auc: 0.876835	valid_0's binary_logloss: 0.258808
    [1217]	valid_0's auc: 0.876912	valid_0's binary_logloss: 0.258431
    [1218]	valid_0's auc: 0.876826	valid_0's binary_logloss: 0.257972
    [1219]	valid_0's auc: 0.876827	valid_0's binary_logloss: 0.258138
    [1220]	valid_0's auc: 0.876918	valid_0's binary_logloss: 0.257801
    [1221]	valid_0's auc: 0.876921	valid_0's binary_logloss: 0.257966
    [1222]	valid_0's auc: 0.876961	valid_0's binary_logloss: 0.257735
    [1223]	valid_0's auc: 0.876978	valid_0's binary_logloss: 0.257901
    [1224]	valid_0's auc: 0.877559	valid_0's binary_logloss: 0.257177
    [1225]	valid_0's auc: 0.877569	valid_0's binary_logloss: 0.257325
    [1226]	valid_0's auc: 0.877512	valid_0's binary_logloss: 0.25702
    [1227]	valid_0's auc: 0.877517	valid_0's binary_logloss: 0.257161
    [1228]	valid_0's auc: 0.877486	valid_0's binary_logloss: 0.256834
    [1229]	valid_0's auc: 0.877958	valid_0's binary_logloss: 0.256209
    [1230]	valid_0's auc: 0.877997	valid_0's binary_logloss: 0.256427
    [1231]	valid_0's auc: 0.877999	valid_0's binary_logloss: 0.256564
    [1232]	valid_0's auc: 0.878004	valid_0's binary_logloss: 0.256702
    [1233]	valid_0's auc: 0.878013	valid_0's binary_logloss: 0.256901
    [1234]	valid_0's auc: 0.878021	valid_0's binary_logloss: 0.257078
    [1235]	valid_0's auc: 0.878034	valid_0's binary_logloss: 0.257295
    [1236]	valid_0's auc: 0.877998	valid_0's binary_logloss: 0.257004
    [1237]	valid_0's auc: 0.877954	valid_0's binary_logloss: 0.256695
    [1238]	valid_0's auc: 0.877945	valid_0's binary_logloss: 0.25687
    [1239]	valid_0's auc: 0.877952	valid_0's binary_logloss: 0.257021
    [1240]	valid_0's auc: 0.877963	valid_0's binary_logloss: 0.257217
    [1241]	valid_0's auc: 0.877995	valid_0's binary_logloss: 0.257414
    [1242]	valid_0's auc: 0.878059	valid_0's binary_logloss: 0.257204
    [1243]	valid_0's auc: 0.878069	valid_0's binary_logloss: 0.257343
    [1244]	valid_0's auc: 0.8781	valid_0's binary_logloss: 0.256998
    [1245]	valid_0's auc: 0.878215	valid_0's binary_logloss: 0.256625
    [1246]	valid_0's auc: 0.878125	valid_0's binary_logloss: 0.256198
    [1247]	valid_0's auc: 0.878138	valid_0's binary_logloss: 0.256355
    [1248]	valid_0's auc: 0.878154	valid_0's binary_logloss: 0.256554
    [1249]	valid_0's auc: 0.878168	valid_0's binary_logloss: 0.256732
    [1250]	valid_0's auc: 0.878211	valid_0's binary_logloss: 0.256433
    [1251]	valid_0's auc: 0.878183	valid_0's binary_logloss: 0.2561
    [1252]	valid_0's auc: 0.878205	valid_0's binary_logloss: 0.256263
    [1253]	valid_0's auc: 0.878213	valid_0's binary_logloss: 0.256395
    [1254]	valid_0's auc: 0.878062	valid_0's binary_logloss: 0.255952
    [1255]	valid_0's auc: 0.878114	valid_0's binary_logloss: 0.255449
    [1256]	valid_0's auc: 0.878128	valid_0's binary_logloss: 0.255649
    [1257]	valid_0's auc: 0.877961	valid_0's binary_logloss: 0.255289
    [1258]	valid_0's auc: 0.877973	valid_0's binary_logloss: 0.255463
    [1259]	valid_0's auc: 0.877955	valid_0's binary_logloss: 0.255062
    [1260]	valid_0's auc: 0.877971	valid_0's binary_logloss: 0.255259
    [1261]	valid_0's auc: 0.877963	valid_0's binary_logloss: 0.254942
    [1262]	valid_0's auc: 0.877973	valid_0's binary_logloss: 0.255122
    [1263]	valid_0's auc: 0.877829	valid_0's binary_logloss: 0.25474
    [1264]	valid_0's auc: 0.877814	valid_0's binary_logloss: 0.254912
    [1265]	valid_0's auc: 0.877841	valid_0's binary_logloss: 0.255088
    [1266]	valid_0's auc: 0.87785	valid_0's binary_logloss: 0.255247
    [1267]	valid_0's auc: 0.878302	valid_0's binary_logloss: 0.254916
    [1268]	valid_0's auc: 0.878313	valid_0's binary_logloss: 0.255076
    [1269]	valid_0's auc: 0.878313	valid_0's binary_logloss: 0.255286
    [1270]	valid_0's auc: 0.878312	valid_0's binary_logloss: 0.255015
    [1271]	valid_0's auc: 0.878334	valid_0's binary_logloss: 0.254689
    [1272]	valid_0's auc: 0.878348	valid_0's binary_logloss: 0.254841
    [1273]	valid_0's auc: 0.878354	valid_0's binary_logloss: 0.254968
    [1274]	valid_0's auc: 0.878358	valid_0's binary_logloss: 0.255103
    [1275]	valid_0's auc: 0.878331	valid_0's binary_logloss: 0.254817
    [1276]	valid_0's auc: 0.878346	valid_0's binary_logloss: 0.254989
    [1277]	valid_0's auc: 0.878348	valid_0's binary_logloss: 0.255167
    [1278]	valid_0's auc: 0.878324	valid_0's binary_logloss: 0.254878
    [1279]	valid_0's auc: 0.878336	valid_0's binary_logloss: 0.255032
    [1280]	valid_0's auc: 0.878328	valid_0's binary_logloss: 0.255224
    [1281]	valid_0's auc: 0.878339	valid_0's binary_logloss: 0.255404
    [1282]	valid_0's auc: 0.878351	valid_0's binary_logloss: 0.255592
    [1283]	valid_0's auc: 0.878591	valid_0's binary_logloss: 0.255084
    [1284]	valid_0's auc: 0.878603	valid_0's binary_logloss: 0.255231
    [1285]	valid_0's auc: 0.87874	valid_0's binary_logloss: 0.254934
    [1286]	valid_0's auc: 0.878738	valid_0's binary_logloss: 0.255126
    [1287]	valid_0's auc: 0.878755	valid_0's binary_logloss: 0.255323
    [1288]	valid_0's auc: 0.878771	valid_0's binary_logloss: 0.255502
    [1289]	valid_0's auc: 0.878796	valid_0's binary_logloss: 0.255659
    [1290]	valid_0's auc: 0.878807	valid_0's binary_logloss: 0.255855
    [1291]	valid_0's auc: 0.878758	valid_0's binary_logloss: 0.255568
    [1292]	valid_0's auc: 0.878805	valid_0's binary_logloss: 0.255252
    [1293]	valid_0's auc: 0.878821	valid_0's binary_logloss: 0.255374
    [1294]	valid_0's auc: 0.878835	valid_0's binary_logloss: 0.255557
    [1295]	valid_0's auc: 0.878803	valid_0's binary_logloss: 0.255208
    [1296]	valid_0's auc: 0.878833	valid_0's binary_logloss: 0.255357
    [1297]	valid_0's auc: 0.878858	valid_0's binary_logloss: 0.25513
    [1298]	valid_0's auc: 0.878869	valid_0's binary_logloss: 0.254831
    [1299]	valid_0's auc: 0.878837	valid_0's binary_logloss: 0.254551
    [1300]	valid_0's auc: 0.878758	valid_0's binary_logloss: 0.254175
    [1301]	valid_0's auc: 0.878738	valid_0's binary_logloss: 0.253877
    [1302]	valid_0's auc: 0.878783	valid_0's binary_logloss: 0.253503
    [1303]	valid_0's auc: 0.878798	valid_0's binary_logloss: 0.25363
    [1304]	valid_0's auc: 0.878791	valid_0's binary_logloss: 0.253794
    [1305]	valid_0's auc: 0.879146	valid_0's binary_logloss: 0.253196
    [1306]	valid_0's auc: 0.879234	valid_0's binary_logloss: 0.252886
    [1307]	valid_0's auc: 0.87923	valid_0's binary_logloss: 0.253045
    [1308]	valid_0's auc: 0.879303	valid_0's binary_logloss: 0.25276
    [1309]	valid_0's auc: 0.879274	valid_0's binary_logloss: 0.252463
    [1310]	valid_0's auc: 0.879357	valid_0's binary_logloss: 0.252145
    [1311]	valid_0's auc: 0.879366	valid_0's binary_logloss: 0.252314
    [1312]	valid_0's auc: 0.879382	valid_0's binary_logloss: 0.252441
    [1313]	valid_0's auc: 0.879388	valid_0's binary_logloss: 0.252606
    [1314]	valid_0's auc: 0.879533	valid_0's binary_logloss: 0.252313
    [1315]	valid_0's auc: 0.879484	valid_0's binary_logloss: 0.252139
    [1316]	valid_0's auc: 0.879484	valid_0's binary_logloss: 0.252297
    [1317]	valid_0's auc: 0.879461	valid_0's binary_logloss: 0.252062
    [1318]	valid_0's auc: 0.879468	valid_0's binary_logloss: 0.252187
    [1319]	valid_0's auc: 0.879476	valid_0's binary_logloss: 0.252356
    [1320]	valid_0's auc: 0.879476	valid_0's binary_logloss: 0.252527
    [1321]	valid_0's auc: 0.879602	valid_0's binary_logloss: 0.252134
    [1322]	valid_0's auc: 0.879558	valid_0's binary_logloss: 0.251846
    [1323]	valid_0's auc: 0.879564	valid_0's binary_logloss: 0.25203
    [1324]	valid_0's auc: 0.879562	valid_0's binary_logloss: 0.251721
    [1325]	valid_0's auc: 0.879394	valid_0's binary_logloss: 0.251385
    [1326]	valid_0's auc: 0.87941	valid_0's binary_logloss: 0.25111
    [1327]	valid_0's auc: 0.879523	valid_0's binary_logloss: 0.250732
    [1328]	valid_0's auc: 0.879593	valid_0's binary_logloss: 0.250386
    [1329]	valid_0's auc: 0.879882	valid_0's binary_logloss: 0.249985
    [1330]	valid_0's auc: 0.879884	valid_0's binary_logloss: 0.250148
    [1331]	valid_0's auc: 0.880053	valid_0's binary_logloss: 0.249765
    [1332]	valid_0's auc: 0.880064	valid_0's binary_logloss: 0.249944
    [1333]	valid_0's auc: 0.880088	valid_0's binary_logloss: 0.249651
    [1334]	valid_0's auc: 0.88009	valid_0's binary_logloss: 0.249451
    [1335]	valid_0's auc: 0.880106	valid_0's binary_logloss: 0.249635
    [1336]	valid_0's auc: 0.880118	valid_0's binary_logloss: 0.249759
    [1337]	valid_0's auc: 0.880126	valid_0's binary_logloss: 0.249962
    [1338]	valid_0's auc: 0.88014	valid_0's binary_logloss: 0.250119
    [1339]	valid_0's auc: 0.880156	valid_0's binary_logloss: 0.250318
    [1340]	valid_0's auc: 0.880173	valid_0's binary_logloss: 0.250507
    [1341]	valid_0's auc: 0.880188	valid_0's binary_logloss: 0.250688
    [1342]	valid_0's auc: 0.880013	valid_0's binary_logloss: 0.250414
    [1343]	valid_0's auc: 0.880014	valid_0's binary_logloss: 0.250538
    [1344]	valid_0's auc: 0.880039	valid_0's binary_logloss: 0.250721
    [1345]	valid_0's auc: 0.880053	valid_0's binary_logloss: 0.250866
    [1346]	valid_0's auc: 0.880056	valid_0's binary_logloss: 0.251053
    [1347]	valid_0's auc: 0.880126	valid_0's binary_logloss: 0.250785
    [1348]	valid_0's auc: 0.880125	valid_0's binary_logloss: 0.250951
    [1349]	valid_0's auc: 0.880172	valid_0's binary_logloss: 0.250672
    [1350]	valid_0's auc: 0.88019	valid_0's binary_logloss: 0.250856
    [1351]	valid_0's auc: 0.880175	valid_0's binary_logloss: 0.250575
    [1352]	valid_0's auc: 0.880185	valid_0's binary_logloss: 0.250748
    [1353]	valid_0's auc: 0.88014	valid_0's binary_logloss: 0.250432
    [1354]	valid_0's auc: 0.880083	valid_0's binary_logloss: 0.250158
    [1355]	valid_0's auc: 0.880066	valid_0's binary_logloss: 0.249891
    [1356]	valid_0's auc: 0.880056	valid_0's binary_logloss: 0.249472
    [1357]	valid_0's auc: 0.879906	valid_0's binary_logloss: 0.249162
    [1358]	valid_0's auc: 0.879907	valid_0's binary_logloss: 0.249341
    [1359]	valid_0's auc: 0.879982	valid_0's binary_logloss: 0.248917
    [1360]	valid_0's auc: 0.879994	valid_0's binary_logloss: 0.249082
    [1361]	valid_0's auc: 0.880013	valid_0's binary_logloss: 0.24883
    [1362]	valid_0's auc: 0.88016	valid_0's binary_logloss: 0.248552
    [1363]	valid_0's auc: 0.880264	valid_0's binary_logloss: 0.248306
    [1364]	valid_0's auc: 0.880244	valid_0's binary_logloss: 0.24802
    [1365]	valid_0's auc: 0.880273	valid_0's binary_logloss: 0.247803
    [1366]	valid_0's auc: 0.880271	valid_0's binary_logloss: 0.247531
    [1367]	valid_0's auc: 0.880194	valid_0's binary_logloss: 0.247254
    [1368]	valid_0's auc: 0.880258	valid_0's binary_logloss: 0.246956
    [1369]	valid_0's auc: 0.880274	valid_0's binary_logloss: 0.246677
    [1370]	valid_0's auc: 0.880274	valid_0's binary_logloss: 0.246823
    [1371]	valid_0's auc: 0.880276	valid_0's binary_logloss: 0.246994
    [1372]	valid_0's auc: 0.880284	valid_0's binary_logloss: 0.2467
    [1373]	valid_0's auc: 0.880272	valid_0's binary_logloss: 0.246443
    [1374]	valid_0's auc: 0.880288	valid_0's binary_logloss: 0.246587
    [1375]	valid_0's auc: 0.880219	valid_0's binary_logloss: 0.246327
    [1376]	valid_0's auc: 0.880217	valid_0's binary_logloss: 0.246486
    [1377]	valid_0's auc: 0.880224	valid_0's binary_logloss: 0.246643
    [1378]	valid_0's auc: 0.880241	valid_0's binary_logloss: 0.246813
    [1379]	valid_0's auc: 0.880281	valid_0's binary_logloss: 0.24656
    [1380]	valid_0's auc: 0.880298	valid_0's binary_logloss: 0.246316
    [1381]	valid_0's auc: 0.880168	valid_0's binary_logloss: 0.245931
    [1382]	valid_0's auc: 0.880183	valid_0's binary_logloss: 0.246092
    [1383]	valid_0's auc: 0.880195	valid_0's binary_logloss: 0.246229
    [1384]	valid_0's auc: 0.880198	valid_0's binary_logloss: 0.246364
    [1385]	valid_0's auc: 0.880217	valid_0's binary_logloss: 0.246474
    [1386]	valid_0's auc: 0.880212	valid_0's binary_logloss: 0.246656
    [1387]	valid_0's auc: 0.880212	valid_0's binary_logloss: 0.246385
    [1388]	valid_0's auc: 0.880196	valid_0's binary_logloss: 0.246091
    [1389]	valid_0's auc: 0.880193	valid_0's binary_logloss: 0.246231
    [1390]	valid_0's auc: 0.880173	valid_0's binary_logloss: 0.245983
    [1391]	valid_0's auc: 0.88019	valid_0's binary_logloss: 0.24612
    [1392]	valid_0's auc: 0.880655	valid_0's binary_logloss: 0.245574
    [1393]	valid_0's auc: 0.880799	valid_0's binary_logloss: 0.245182
    [1394]	valid_0's auc: 0.880811	valid_0's binary_logloss: 0.245353
    [1395]	valid_0's auc: 0.880779	valid_0's binary_logloss: 0.245117
    [1396]	valid_0's auc: 0.88089	valid_0's binary_logloss: 0.244696
    [1397]	valid_0's auc: 0.880874	valid_0's binary_logloss: 0.244453
    [1398]	valid_0's auc: 0.880886	valid_0's binary_logloss: 0.244597
    [1399]	valid_0's auc: 0.880896	valid_0's binary_logloss: 0.244717
    [1400]	valid_0's auc: 0.880983	valid_0's binary_logloss: 0.244424
    [1401]	valid_0's auc: 0.880991	valid_0's binary_logloss: 0.244545
    [1402]	valid_0's auc: 0.880992	valid_0's binary_logloss: 0.244694
    [1403]	valid_0's auc: 0.881017	valid_0's binary_logloss: 0.244512
    [1404]	valid_0's auc: 0.881036	valid_0's binary_logloss: 0.244667
    [1405]	valid_0's auc: 0.881036	valid_0's binary_logloss: 0.2444
    [1406]	valid_0's auc: 0.881045	valid_0's binary_logloss: 0.244555
    [1407]	valid_0's auc: 0.88093	valid_0's binary_logloss: 0.244273
    [1408]	valid_0's auc: 0.880934	valid_0's binary_logloss: 0.244429
    [1409]	valid_0's auc: 0.880978	valid_0's binary_logloss: 0.244201
    [1410]	valid_0's auc: 0.880992	valid_0's binary_logloss: 0.244395
    [1411]	valid_0's auc: 0.880992	valid_0's binary_logloss: 0.244548
    [1412]	valid_0's auc: 0.880982	valid_0's binary_logloss: 0.244311
    [1413]	valid_0's auc: 0.881441	valid_0's binary_logloss: 0.243728
    [1414]	valid_0's auc: 0.881454	valid_0's binary_logloss: 0.243869
    [1415]	valid_0's auc: 0.88146	valid_0's binary_logloss: 0.244012
    [1416]	valid_0's auc: 0.88155	valid_0's binary_logloss: 0.24375
    [1417]	valid_0's auc: 0.881558	valid_0's binary_logloss: 0.243889
    [1418]	valid_0's auc: 0.88157	valid_0's binary_logloss: 0.244018
    [1419]	valid_0's auc: 0.881484	valid_0's binary_logloss: 0.243739
    [1420]	valid_0's auc: 0.881496	valid_0's binary_logloss: 0.243888
    [1421]	valid_0's auc: 0.881555	valid_0's binary_logloss: 0.243664
    [1422]	valid_0's auc: 0.881557	valid_0's binary_logloss: 0.243775
    [1423]	valid_0's auc: 0.881951	valid_0's binary_logloss: 0.24316
    [1424]	valid_0's auc: 0.881954	valid_0's binary_logloss: 0.243313
    [1425]	valid_0's auc: 0.881958	valid_0's binary_logloss: 0.243436
    [1426]	valid_0's auc: 0.881946	valid_0's binary_logloss: 0.243178
    [1427]	valid_0's auc: 0.881913	valid_0's binary_logloss: 0.242946
    [1428]	valid_0's auc: 0.881923	valid_0's binary_logloss: 0.24308
    [1429]	valid_0's auc: 0.881879	valid_0's binary_logloss: 0.242882
    [1430]	valid_0's auc: 0.881895	valid_0's binary_logloss: 0.24301
    [1431]	valid_0's auc: 0.881899	valid_0's binary_logloss: 0.243167
    [1432]	valid_0's auc: 0.881909	valid_0's binary_logloss: 0.2433
    [1433]	valid_0's auc: 0.882228	valid_0's binary_logloss: 0.242795
    [1434]	valid_0's auc: 0.882239	valid_0's binary_logloss: 0.242946
    [1435]	valid_0's auc: 0.882245	valid_0's binary_logloss: 0.243086
    [1436]	valid_0's auc: 0.882261	valid_0's binary_logloss: 0.243193
    [1437]	valid_0's auc: 0.882679	valid_0's binary_logloss: 0.242712
    [1438]	valid_0's auc: 0.882686	valid_0's binary_logloss: 0.242872
    [1439]	valid_0's auc: 0.883072	valid_0's binary_logloss: 0.242388
    [1440]	valid_0's auc: 0.883069	valid_0's binary_logloss: 0.24251
    [1441]	valid_0's auc: 0.883076	valid_0's binary_logloss: 0.242614
    [1442]	valid_0's auc: 0.883087	valid_0's binary_logloss: 0.242745
    [1443]	valid_0's auc: 0.883043	valid_0's binary_logloss: 0.242466
    [1444]	valid_0's auc: 0.883055	valid_0's binary_logloss: 0.242611
    [1445]	valid_0's auc: 0.883098	valid_0's binary_logloss: 0.242272
    [1446]	valid_0's auc: 0.883116	valid_0's binary_logloss: 0.242025
    [1447]	valid_0's auc: 0.883124	valid_0's binary_logloss: 0.242168
    [1448]	valid_0's auc: 0.883131	valid_0's binary_logloss: 0.242278
    [1449]	valid_0's auc: 0.883126	valid_0's binary_logloss: 0.242436
    [1450]	valid_0's auc: 0.883201	valid_0's binary_logloss: 0.24213
    [1451]	valid_0's auc: 0.883258	valid_0's binary_logloss: 0.241849
    [1452]	valid_0's auc: 0.883281	valid_0's binary_logloss: 0.24201
    [1453]	valid_0's auc: 0.883288	valid_0's binary_logloss: 0.242124
    [1454]	valid_0's auc: 0.883191	valid_0's binary_logloss: 0.241848
    [1455]	valid_0's auc: 0.883157	valid_0's binary_logloss: 0.24161
    [1456]	valid_0's auc: 0.883336	valid_0's binary_logloss: 0.241143
    [1457]	valid_0's auc: 0.883422	valid_0's binary_logloss: 0.240918
    [1458]	valid_0's auc: 0.883528	valid_0's binary_logloss: 0.240651
    [1459]	valid_0's auc: 0.883589	valid_0's binary_logloss: 0.240467
    [1460]	valid_0's auc: 0.883598	valid_0's binary_logloss: 0.240592
    [1461]	valid_0's auc: 0.883639	valid_0's binary_logloss: 0.240294
    [1462]	valid_0's auc: 0.883576	valid_0's binary_logloss: 0.240067
    [1463]	valid_0's auc: 0.883619	valid_0's binary_logloss: 0.239823
    [1464]	valid_0's auc: 0.883626	valid_0's binary_logloss: 0.239963
    [1465]	valid_0's auc: 0.883638	valid_0's binary_logloss: 0.240099
    [1466]	valid_0's auc: 0.883649	valid_0's binary_logloss: 0.240256
    [1467]	valid_0's auc: 0.883662	valid_0's binary_logloss: 0.240362
    [1468]	valid_0's auc: 0.88366	valid_0's binary_logloss: 0.240495
    [1469]	valid_0's auc: 0.883679	valid_0's binary_logloss: 0.240612
    [1470]	valid_0's auc: 0.883651	valid_0's binary_logloss: 0.240395
    [1471]	valid_0's auc: 0.883662	valid_0's binary_logloss: 0.240531
    [1472]	valid_0's auc: 0.883677	valid_0's binary_logloss: 0.240663
    [1473]	valid_0's auc: 0.883683	valid_0's binary_logloss: 0.240788
    [1474]	valid_0's auc: 0.883691	valid_0's binary_logloss: 0.240908
    [1475]	valid_0's auc: 0.883689	valid_0's binary_logloss: 0.241028
    [1476]	valid_0's auc: 0.883696	valid_0's binary_logloss: 0.241166
    [1477]	valid_0's auc: 0.883821	valid_0's binary_logloss: 0.240842
    [1478]	valid_0's auc: 0.883811	valid_0's binary_logloss: 0.240607
    [1479]	valid_0's auc: 0.883823	valid_0's binary_logloss: 0.24075
    [1480]	valid_0's auc: 0.883835	valid_0's binary_logloss: 0.240497
    [1481]	valid_0's auc: 0.883925	valid_0's binary_logloss: 0.240116
    [1482]	valid_0's auc: 0.883923	valid_0's binary_logloss: 0.239876
    [1483]	valid_0's auc: 0.884053	valid_0's binary_logloss: 0.239512
    [1484]	valid_0's auc: 0.884136	valid_0's binary_logloss: 0.23926
    [1485]	valid_0's auc: 0.884144	valid_0's binary_logloss: 0.239387
    [1486]	valid_0's auc: 0.884153	valid_0's binary_logloss: 0.239534
    [1487]	valid_0's auc: 0.884163	valid_0's binary_logloss: 0.239666
    [1488]	valid_0's auc: 0.884173	valid_0's binary_logloss: 0.239814
    [1489]	valid_0's auc: 0.884187	valid_0's binary_logloss: 0.239955
    [1490]	valid_0's auc: 0.884202	valid_0's binary_logloss: 0.240118
    [1491]	valid_0's auc: 0.884174	valid_0's binary_logloss: 0.239887
    [1492]	valid_0's auc: 0.884191	valid_0's binary_logloss: 0.23998
    [1493]	valid_0's auc: 0.884196	valid_0's binary_logloss: 0.240096
    [1494]	valid_0's auc: 0.884199	valid_0's binary_logloss: 0.240222
    [1495]	valid_0's auc: 0.884204	valid_0's binary_logloss: 0.240374
    [1496]	valid_0's auc: 0.884217	valid_0's binary_logloss: 0.240471
    [1497]	valid_0's auc: 0.884224	valid_0's binary_logloss: 0.240157
    [1498]	valid_0's auc: 0.884211	valid_0's binary_logloss: 0.239928
    [1499]	valid_0's auc: 0.884216	valid_0's binary_logloss: 0.240043
    [1500]	valid_0's auc: 0.884224	valid_0's binary_logloss: 0.240144
    [1501]	valid_0's auc: 0.884223	valid_0's binary_logloss: 0.240293
    [1502]	valid_0's auc: 0.88423	valid_0's binary_logloss: 0.240426
    [1503]	valid_0's auc: 0.884251	valid_0's binary_logloss: 0.240551
    [1504]	valid_0's auc: 0.884216	valid_0's binary_logloss: 0.240313
    [1505]	valid_0's auc: 0.884219	valid_0's binary_logloss: 0.240408
    [1506]	valid_0's auc: 0.884226	valid_0's binary_logloss: 0.240528
    [1507]	valid_0's auc: 0.8842	valid_0's binary_logloss: 0.240171
    [1508]	valid_0's auc: 0.884207	valid_0's binary_logloss: 0.24033
    [1509]	valid_0's auc: 0.884212	valid_0's binary_logloss: 0.240439
    [1510]	valid_0's auc: 0.884216	valid_0's binary_logloss: 0.24058
    [1511]	valid_0's auc: 0.884145	valid_0's binary_logloss: 0.240248
    [1512]	valid_0's auc: 0.884171	valid_0's binary_logloss: 0.239891
    [1513]	valid_0's auc: 0.884175	valid_0's binary_logloss: 0.240047
    [1514]	valid_0's auc: 0.884195	valid_0's binary_logloss: 0.240198
    [1515]	valid_0's auc: 0.884211	valid_0's binary_logloss: 0.240309
    [1516]	valid_0's auc: 0.884216	valid_0's binary_logloss: 0.240418
    [1517]	valid_0's auc: 0.884234	valid_0's binary_logloss: 0.240547
    [1518]	valid_0's auc: 0.884201	valid_0's binary_logloss: 0.240311
    [1519]	valid_0's auc: 0.88422	valid_0's binary_logloss: 0.240479
    [1520]	valid_0's auc: 0.884113	valid_0's binary_logloss: 0.240228
    [1521]	valid_0's auc: 0.884146	valid_0's binary_logloss: 0.239998
    [1522]	valid_0's auc: 0.884156	valid_0's binary_logloss: 0.240131
    [1523]	valid_0's auc: 0.884369	valid_0's binary_logloss: 0.239635
    [1524]	valid_0's auc: 0.88438	valid_0's binary_logloss: 0.239752
    [1525]	valid_0's auc: 0.884383	valid_0's binary_logloss: 0.239841
    [1526]	valid_0's auc: 0.884393	valid_0's binary_logloss: 0.239641
    [1527]	valid_0's auc: 0.884476	valid_0's binary_logloss: 0.239335
    [1528]	valid_0's auc: 0.884493	valid_0's binary_logloss: 0.23944
    [1529]	valid_0's auc: 0.884501	valid_0's binary_logloss: 0.239554
    [1530]	valid_0's auc: 0.884512	valid_0's binary_logloss: 0.239652
    [1531]	valid_0's auc: 0.884527	valid_0's binary_logloss: 0.239782
    [1532]	valid_0's auc: 0.884554	valid_0's binary_logloss: 0.239506
    [1533]	valid_0's auc: 0.884567	valid_0's binary_logloss: 0.239618
    [1534]	valid_0's auc: 0.884577	valid_0's binary_logloss: 0.239377
    [1535]	valid_0's auc: 0.884588	valid_0's binary_logloss: 0.239094
    [1536]	valid_0's auc: 0.8846	valid_0's binary_logloss: 0.239207
    [1537]	valid_0's auc: 0.884656	valid_0's binary_logloss: 0.239006
    [1538]	valid_0's auc: 0.884673	valid_0's binary_logloss: 0.23912
    [1539]	valid_0's auc: 0.88468	valid_0's binary_logloss: 0.238861
    [1540]	valid_0's auc: 0.884686	valid_0's binary_logloss: 0.23902
    [1541]	valid_0's auc: 0.884715	valid_0's binary_logloss: 0.238751
    [1542]	valid_0's auc: 0.884716	valid_0's binary_logloss: 0.238863
    [1543]	valid_0's auc: 0.884916	valid_0's binary_logloss: 0.238413
    [1544]	valid_0's auc: 0.885039	valid_0's binary_logloss: 0.238078
    [1545]	valid_0's auc: 0.88504	valid_0's binary_logloss: 0.238229
    [1546]	valid_0's auc: 0.885047	valid_0's binary_logloss: 0.238357
    [1547]	valid_0's auc: 0.885069	valid_0's binary_logloss: 0.238483
    [1548]	valid_0's auc: 0.885066	valid_0's binary_logloss: 0.238283
    [1549]	valid_0's auc: 0.884959	valid_0's binary_logloss: 0.238098
    [1550]	valid_0's auc: 0.884941	valid_0's binary_logloss: 0.23775
    [1551]	valid_0's auc: 0.884893	valid_0's binary_logloss: 0.237487
    [1552]	valid_0's auc: 0.884902	valid_0's binary_logloss: 0.237606
    [1553]	valid_0's auc: 0.884892	valid_0's binary_logloss: 0.237406
    [1554]	valid_0's auc: 0.88488	valid_0's binary_logloss: 0.237074
    [1555]	valid_0's auc: 0.884806	valid_0's binary_logloss: 0.236871
    [1556]	valid_0's auc: 0.884823	valid_0's binary_logloss: 0.237018
    [1557]	valid_0's auc: 0.884834	valid_0's binary_logloss: 0.237129
    [1558]	valid_0's auc: 0.884844	valid_0's binary_logloss: 0.23725
    [1559]	valid_0's auc: 0.884798	valid_0's binary_logloss: 0.23701
    [1560]	valid_0's auc: 0.884805	valid_0's binary_logloss: 0.237133
    [1561]	valid_0's auc: 0.884706	valid_0's binary_logloss: 0.236875
    [1562]	valid_0's auc: 0.884713	valid_0's binary_logloss: 0.237013
    [1563]	valid_0's auc: 0.884723	valid_0's binary_logloss: 0.237143
    [1564]	valid_0's auc: 0.884685	valid_0's binary_logloss: 0.236929
    [1565]	valid_0's auc: 0.884693	valid_0's binary_logloss: 0.237077
    [1566]	valid_0's auc: 0.884662	valid_0's binary_logloss: 0.236846
    [1567]	valid_0's auc: 0.884669	valid_0's binary_logloss: 0.236964
    [1568]	valid_0's auc: 0.884636	valid_0's binary_logloss: 0.236754
    [1569]	valid_0's auc: 0.884626	valid_0's binary_logloss: 0.236497
    [1570]	valid_0's auc: 0.884592	valid_0's binary_logloss: 0.236203
    [1571]	valid_0's auc: 0.884548	valid_0's binary_logloss: 0.235999
    [1572]	valid_0's auc: 0.884614	valid_0's binary_logloss: 0.235767
    [1573]	valid_0's auc: 0.884786	valid_0's binary_logloss: 0.235481
    [1574]	valid_0's auc: 0.884722	valid_0's binary_logloss: 0.235278
    [1575]	valid_0's auc: 0.88479	valid_0's binary_logloss: 0.235038
    [1576]	valid_0's auc: 0.884863	valid_0's binary_logloss: 0.234666
    [1577]	valid_0's auc: 0.884868	valid_0's binary_logloss: 0.234334
    [1578]	valid_0's auc: 0.884812	valid_0's binary_logloss: 0.234137
    [1579]	valid_0's auc: 0.884818	valid_0's binary_logloss: 0.23422
    [1580]	valid_0's auc: 0.88483	valid_0's binary_logloss: 0.234341
    [1581]	valid_0's auc: 0.884873	valid_0's binary_logloss: 0.234128
    [1582]	valid_0's auc: 0.884825	valid_0's binary_logloss: 0.233905
    [1583]	valid_0's auc: 0.884838	valid_0's binary_logloss: 0.234003
    [1584]	valid_0's auc: 0.884837	valid_0's binary_logloss: 0.233833
    [1585]	valid_0's auc: 0.884923	valid_0's binary_logloss: 0.23344
    [1586]	valid_0's auc: 0.884904	valid_0's binary_logloss: 0.233222
    [1587]	valid_0's auc: 0.884915	valid_0's binary_logloss: 0.233333
    [1588]	valid_0's auc: 0.884926	valid_0's binary_logloss: 0.233456
    [1589]	valid_0's auc: 0.88492	valid_0's binary_logloss: 0.233225
    [1590]	valid_0's auc: 0.884918	valid_0's binary_logloss: 0.233327
    [1591]	valid_0's auc: 0.884923	valid_0's binary_logloss: 0.233432
    [1592]	valid_0's auc: 0.88493	valid_0's binary_logloss: 0.233548
    [1593]	valid_0's auc: 0.884937	valid_0's binary_logloss: 0.233655
    [1594]	valid_0's auc: 0.884945	valid_0's binary_logloss: 0.233755
    [1595]	valid_0's auc: 0.884877	valid_0's binary_logloss: 0.233523
    [1596]	valid_0's auc: 0.8848	valid_0's binary_logloss: 0.233281
    [1597]	valid_0's auc: 0.884813	valid_0's binary_logloss: 0.233403
    [1598]	valid_0's auc: 0.884798	valid_0's binary_logloss: 0.233241
    [1599]	valid_0's auc: 0.884795	valid_0's binary_logloss: 0.232942
    [1600]	valid_0's auc: 0.884689	valid_0's binary_logloss: 0.232723
    [1601]	valid_0's auc: 0.884604	valid_0's binary_logloss: 0.232492
    [1602]	valid_0's auc: 0.884585	valid_0's binary_logloss: 0.232333
    [1603]	valid_0's auc: 0.884585	valid_0's binary_logloss: 0.232466
    [1604]	valid_0's auc: 0.884857	valid_0's binary_logloss: 0.231975
    [1605]	valid_0's auc: 0.884878	valid_0's binary_logloss: 0.232081
    [1606]	valid_0's auc: 0.884813	valid_0's binary_logloss: 0.231907
    [1607]	valid_0's auc: 0.884811	valid_0's binary_logloss: 0.23171
    [1608]	valid_0's auc: 0.884813	valid_0's binary_logloss: 0.231846
    [1609]	valid_0's auc: 0.884817	valid_0's binary_logloss: 0.231953
    [1610]	valid_0's auc: 0.884821	valid_0's binary_logloss: 0.232049
    [1611]	valid_0's auc: 0.884828	valid_0's binary_logloss: 0.232141
    [1612]	valid_0's auc: 0.884843	valid_0's binary_logloss: 0.232235
    [1613]	valid_0's auc: 0.884848	valid_0's binary_logloss: 0.232339
    [1614]	valid_0's auc: 0.88487	valid_0's binary_logloss: 0.232137
    [1615]	valid_0's auc: 0.884853	valid_0's binary_logloss: 0.231936
    [1616]	valid_0's auc: 0.884856	valid_0's binary_logloss: 0.232064
    [1617]	valid_0's auc: 0.884858	valid_0's binary_logloss: 0.232148
    [1618]	valid_0's auc: 0.884867	valid_0's binary_logloss: 0.232274
    [1619]	valid_0's auc: 0.884887	valid_0's binary_logloss: 0.232353
    [1620]	valid_0's auc: 0.885071	valid_0's binary_logloss: 0.232039
    [1621]	valid_0's auc: 0.885007	valid_0's binary_logloss: 0.231744
    [1622]	valid_0's auc: 0.88494	valid_0's binary_logloss: 0.231496
    [1623]	valid_0's auc: 0.884933	valid_0's binary_logloss: 0.231344
    [1624]	valid_0's auc: 0.885103	valid_0's binary_logloss: 0.230976
    [1625]	valid_0's auc: 0.885071	valid_0's binary_logloss: 0.230783
    [1626]	valid_0's auc: 0.885345	valid_0's binary_logloss: 0.230344
    [1627]	valid_0's auc: 0.885359	valid_0's binary_logloss: 0.230449
    [1628]	valid_0's auc: 0.885378	valid_0's binary_logloss: 0.230522
    [1629]	valid_0's auc: 0.885385	valid_0's binary_logloss: 0.230617
    [1630]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.230457
    [1631]	valid_0's auc: 0.88534	valid_0's binary_logloss: 0.230229
    [1632]	valid_0's auc: 0.885351	valid_0's binary_logloss: 0.230321
    [1633]	valid_0's auc: 0.885359	valid_0's binary_logloss: 0.230408
    [1634]	valid_0's auc: 0.885372	valid_0's binary_logloss: 0.230512
    [1635]	valid_0's auc: 0.885384	valid_0's binary_logloss: 0.230626
    [1636]	valid_0's auc: 0.885366	valid_0's binary_logloss: 0.23037
    [1637]	valid_0's auc: 0.885369	valid_0's binary_logloss: 0.230152
    [1638]	valid_0's auc: 0.885384	valid_0's binary_logloss: 0.229927
    [1639]	valid_0's auc: 0.885331	valid_0's binary_logloss: 0.229745
    [1640]	valid_0's auc: 0.885345	valid_0's binary_logloss: 0.229853
    [1641]	valid_0's auc: 0.885339	valid_0's binary_logloss: 0.23
    [1642]	valid_0's auc: 0.885352	valid_0's binary_logloss: 0.230084
    [1643]	valid_0's auc: 0.885347	valid_0's binary_logloss: 0.230226
    [1644]	valid_0's auc: 0.885358	valid_0's binary_logloss: 0.230304
    [1645]	valid_0's auc: 0.885366	valid_0's binary_logloss: 0.230419
    [1646]	valid_0's auc: 0.885377	valid_0's binary_logloss: 0.230527
    [1647]	valid_0's auc: 0.88539	valid_0's binary_logloss: 0.230631
    [1648]	valid_0's auc: 0.885339	valid_0's binary_logloss: 0.230397
    [1649]	valid_0's auc: 0.885348	valid_0's binary_logloss: 0.230489
    [1650]	valid_0's auc: 0.885356	valid_0's binary_logloss: 0.230594
    [1651]	valid_0's auc: 0.885353	valid_0's binary_logloss: 0.230434
    [1652]	valid_0's auc: 0.885363	valid_0's binary_logloss: 0.230547
    [1653]	valid_0's auc: 0.885372	valid_0's binary_logloss: 0.230341
    [1654]	valid_0's auc: 0.885315	valid_0's binary_logloss: 0.230139
    [1655]	valid_0's auc: 0.885327	valid_0's binary_logloss: 0.230233
    [1656]	valid_0's auc: 0.885329	valid_0's binary_logloss: 0.230037
    [1657]	valid_0's auc: 0.885292	valid_0's binary_logloss: 0.229899
    [1658]	valid_0's auc: 0.885274	valid_0's binary_logloss: 0.229708
    [1659]	valid_0's auc: 0.885292	valid_0's binary_logloss: 0.229479
    [1660]	valid_0's auc: 0.885305	valid_0's binary_logloss: 0.229576
    [1661]	valid_0's auc: 0.88531	valid_0's binary_logloss: 0.229704
    [1662]	valid_0's auc: 0.885261	valid_0's binary_logloss: 0.229515
    [1663]	valid_0's auc: 0.885207	valid_0's binary_logloss: 0.229289
    [1664]	valid_0's auc: 0.88522	valid_0's binary_logloss: 0.229113
    [1665]	valid_0's auc: 0.885215	valid_0's binary_logloss: 0.229256
    [1666]	valid_0's auc: 0.885216	valid_0's binary_logloss: 0.229368
    [1667]	valid_0's auc: 0.885235	valid_0's binary_logloss: 0.229163
    [1668]	valid_0's auc: 0.885242	valid_0's binary_logloss: 0.229274
    [1669]	valid_0's auc: 0.885255	valid_0's binary_logloss: 0.229373
    [1670]	valid_0's auc: 0.885195	valid_0's binary_logloss: 0.229188
    [1671]	valid_0's auc: 0.88555	valid_0's binary_logloss: 0.228853
    [1672]	valid_0's auc: 0.88556	valid_0's binary_logloss: 0.228937
    [1673]	valid_0's auc: 0.885566	valid_0's binary_logloss: 0.229027
    [1674]	valid_0's auc: 0.885572	valid_0's binary_logloss: 0.229095
    [1675]	valid_0's auc: 0.88551	valid_0's binary_logloss: 0.228888
    [1676]	valid_0's auc: 0.88551	valid_0's binary_logloss: 0.228701
    [1677]	valid_0's auc: 0.885504	valid_0's binary_logloss: 0.228569
    [1678]	valid_0's auc: 0.88556	valid_0's binary_logloss: 0.228383
    [1679]	valid_0's auc: 0.885521	valid_0's binary_logloss: 0.228221
    [1680]	valid_0's auc: 0.885505	valid_0's binary_logloss: 0.228035
    [1681]	valid_0's auc: 0.885868	valid_0's binary_logloss: 0.227696
    [1682]	valid_0's auc: 0.885877	valid_0's binary_logloss: 0.227776
    [1683]	valid_0's auc: 0.885883	valid_0's binary_logloss: 0.227913
    [1684]	valid_0's auc: 0.885887	valid_0's binary_logloss: 0.227992
    [1685]	valid_0's auc: 0.885777	valid_0's binary_logloss: 0.227739
    [1686]	valid_0's auc: 0.885931	valid_0's binary_logloss: 0.227367
    [1687]	valid_0's auc: 0.885854	valid_0's binary_logloss: 0.227208
    [1688]	valid_0's auc: 0.885879	valid_0's binary_logloss: 0.227318
    [1689]	valid_0's auc: 0.885863	valid_0's binary_logloss: 0.227144
    [1690]	valid_0's auc: 0.885842	valid_0's binary_logloss: 0.226876
    [1691]	valid_0's auc: 0.885845	valid_0's binary_logloss: 0.226687
    [1692]	valid_0's auc: 0.885858	valid_0's binary_logloss: 0.226769
    [1693]	valid_0's auc: 0.885852	valid_0's binary_logloss: 0.226645
    [1694]	valid_0's auc: 0.885861	valid_0's binary_logloss: 0.226475
    [1695]	valid_0's auc: 0.885779	valid_0's binary_logloss: 0.226283
    [1696]	valid_0's auc: 0.885775	valid_0's binary_logloss: 0.226385
    [1697]	valid_0's auc: 0.885764	valid_0's binary_logloss: 0.226541
    [1698]	valid_0's auc: 0.885755	valid_0's binary_logloss: 0.2264
    [1699]	valid_0's auc: 0.885752	valid_0's binary_logloss: 0.226231
    [1700]	valid_0's auc: 0.885753	valid_0's binary_logloss: 0.22635
    [1701]	valid_0's auc: 0.88575	valid_0's binary_logloss: 0.226222
    [1702]	valid_0's auc: 0.885741	valid_0's binary_logloss: 0.226054
    [1703]	valid_0's auc: 0.885828	valid_0's binary_logloss: 0.225853
    [1704]	valid_0's auc: 0.885828	valid_0's binary_logloss: 0.225661
    [1705]	valid_0's auc: 0.885838	valid_0's binary_logloss: 0.225754
    [1706]	valid_0's auc: 0.885847	valid_0's binary_logloss: 0.225856
    [1707]	valid_0's auc: 0.885856	valid_0's binary_logloss: 0.225937
    [1708]	valid_0's auc: 0.885856	valid_0's binary_logloss: 0.226029
    [1709]	valid_0's auc: 0.885875	valid_0's binary_logloss: 0.225834
    [1710]	valid_0's auc: 0.885886	valid_0's binary_logloss: 0.225926
    [1711]	valid_0's auc: 0.885888	valid_0's binary_logloss: 0.226043
    [1712]	valid_0's auc: 0.88588	valid_0's binary_logloss: 0.2259
    [1713]	valid_0's auc: 0.885892	valid_0's binary_logloss: 0.225996
    [1714]	valid_0's auc: 0.885919	valid_0's binary_logloss: 0.225825
    [1715]	valid_0's auc: 0.885923	valid_0's binary_logloss: 0.225951
    [1716]	valid_0's auc: 0.88586	valid_0's binary_logloss: 0.225817
    [1717]	valid_0's auc: 0.885853	valid_0's binary_logloss: 0.225919
    [1718]	valid_0's auc: 0.885858	valid_0's binary_logloss: 0.226041
    [1719]	valid_0's auc: 0.885854	valid_0's binary_logloss: 0.226127
    [1720]	valid_0's auc: 0.885861	valid_0's binary_logloss: 0.225963
    [1721]	valid_0's auc: 0.885884	valid_0's binary_logloss: 0.22575
    [1722]	valid_0's auc: 0.885886	valid_0's binary_logloss: 0.225895
    [1723]	valid_0's auc: 0.885781	valid_0's binary_logloss: 0.225735
    [1724]	valid_0's auc: 0.885958	valid_0's binary_logloss: 0.225346
    [1725]	valid_0's auc: 0.885968	valid_0's binary_logloss: 0.225466
    [1726]	valid_0's auc: 0.885969	valid_0's binary_logloss: 0.225551
    [1727]	valid_0's auc: 0.886017	valid_0's binary_logloss: 0.225374
    [1728]	valid_0's auc: 0.885959	valid_0's binary_logloss: 0.225202
    [1729]	valid_0's auc: 0.88596	valid_0's binary_logloss: 0.225285
    [1730]	valid_0's auc: 0.886007	valid_0's binary_logloss: 0.225102
    [1731]	valid_0's auc: 0.886015	valid_0's binary_logloss: 0.225197
    [1732]	valid_0's auc: 0.886012	valid_0's binary_logloss: 0.225348
    [1733]	valid_0's auc: 0.885994	valid_0's binary_logloss: 0.225204
    [1734]	valid_0's auc: 0.886001	valid_0's binary_logloss: 0.225275
    [1735]	valid_0's auc: 0.886008	valid_0's binary_logloss: 0.225366
    [1736]	valid_0's auc: 0.886007	valid_0's binary_logloss: 0.225502
    [1737]	valid_0's auc: 0.886026	valid_0's binary_logloss: 0.225585
    [1738]	valid_0's auc: 0.886011	valid_0's binary_logloss: 0.22543
    [1739]	valid_0's auc: 0.886038	valid_0's binary_logloss: 0.22526
    [1740]	valid_0's auc: 0.886057	valid_0's binary_logloss: 0.225098
    [1741]	valid_0's auc: 0.886057	valid_0's binary_logloss: 0.224919
    [1742]	valid_0's auc: 0.886052	valid_0's binary_logloss: 0.224729
    [1743]	valid_0's auc: 0.88606	valid_0's binary_logloss: 0.224843
    [1744]	valid_0's auc: 0.886363	valid_0's binary_logloss: 0.224571
    [1745]	valid_0's auc: 0.886365	valid_0's binary_logloss: 0.224393
    [1746]	valid_0's auc: 0.886361	valid_0's binary_logloss: 0.224495
    [1747]	valid_0's auc: 0.886304	valid_0's binary_logloss: 0.224344
    [1748]	valid_0's auc: 0.886305	valid_0's binary_logloss: 0.224442
    [1749]	valid_0's auc: 0.88631	valid_0's binary_logloss: 0.224543
    [1750]	valid_0's auc: 0.886484	valid_0's binary_logloss: 0.224181
    [1751]	valid_0's auc: 0.886466	valid_0's binary_logloss: 0.224078
    [1752]	valid_0's auc: 0.886476	valid_0's binary_logloss: 0.224167
    [1753]	valid_0's auc: 0.886713	valid_0's binary_logloss: 0.223761
    [1754]	valid_0's auc: 0.887056	valid_0's binary_logloss: 0.223319
    [1755]	valid_0's auc: 0.887184	valid_0's binary_logloss: 0.22297
    [1756]	valid_0's auc: 0.887189	valid_0's binary_logloss: 0.223042
    [1757]	valid_0's auc: 0.887201	valid_0's binary_logloss: 0.223127
    [1758]	valid_0's auc: 0.887193	valid_0's binary_logloss: 0.22324
    [1759]	valid_0's auc: 0.887197	valid_0's binary_logloss: 0.223031
    [1760]	valid_0's auc: 0.887201	valid_0's binary_logloss: 0.223129
    [1761]	valid_0's auc: 0.887214	valid_0's binary_logloss: 0.22324
    [1762]	valid_0's auc: 0.887227	valid_0's binary_logloss: 0.223312
    [1763]	valid_0's auc: 0.887225	valid_0's binary_logloss: 0.22344
    [1764]	valid_0's auc: 0.887162	valid_0's binary_logloss: 0.223298
    [1765]	valid_0's auc: 0.887161	valid_0's binary_logloss: 0.223387
    [1766]	valid_0's auc: 0.887193	valid_0's binary_logloss: 0.223209
    [1767]	valid_0's auc: 0.887177	valid_0's binary_logloss: 0.22306
    [1768]	valid_0's auc: 0.887427	valid_0's binary_logloss: 0.222654
    [1769]	valid_0's auc: 0.887313	valid_0's binary_logloss: 0.222517
    [1770]	valid_0's auc: 0.887311	valid_0's binary_logloss: 0.222605
    [1771]	valid_0's auc: 0.887336	valid_0's binary_logloss: 0.222435
    [1772]	valid_0's auc: 0.88735	valid_0's binary_logloss: 0.222525
    [1773]	valid_0's auc: 0.887364	valid_0's binary_logloss: 0.222605
    [1774]	valid_0's auc: 0.887368	valid_0's binary_logloss: 0.222448
    [1775]	valid_0's auc: 0.88737	valid_0's binary_logloss: 0.222526
    [1776]	valid_0's auc: 0.887345	valid_0's binary_logloss: 0.222346
    [1777]	valid_0's auc: 0.88735	valid_0's binary_logloss: 0.222435
    [1778]	valid_0's auc: 0.887564	valid_0's binary_logloss: 0.222185
    [1779]	valid_0's auc: 0.887553	valid_0's binary_logloss: 0.222302
    [1780]	valid_0's auc: 0.887534	valid_0's binary_logloss: 0.222155
    [1781]	valid_0's auc: 0.887586	valid_0's binary_logloss: 0.221972
    [1782]	valid_0's auc: 0.887592	valid_0's binary_logloss: 0.222049
    [1783]	valid_0's auc: 0.887566	valid_0's binary_logloss: 0.221928
    [1784]	valid_0's auc: 0.887599	valid_0's binary_logloss: 0.221679
    [1785]	valid_0's auc: 0.887609	valid_0's binary_logloss: 0.221783
    [1786]	valid_0's auc: 0.887621	valid_0's binary_logloss: 0.221878
    [1787]	valid_0's auc: 0.887585	valid_0's binary_logloss: 0.221598
    [1788]	valid_0's auc: 0.887529	valid_0's binary_logloss: 0.221446
    [1789]	valid_0's auc: 0.887525	valid_0's binary_logloss: 0.22157
    [1790]	valid_0's auc: 0.887546	valid_0's binary_logloss: 0.221415
    [1791]	valid_0's auc: 0.887546	valid_0's binary_logloss: 0.221544
    [1792]	valid_0's auc: 0.88782	valid_0's binary_logloss: 0.221123
    [1793]	valid_0's auc: 0.887829	valid_0's binary_logloss: 0.221221
    [1794]	valid_0's auc: 0.887833	valid_0's binary_logloss: 0.221334
    [1795]	valid_0's auc: 0.887812	valid_0's binary_logloss: 0.221215
    [1796]	valid_0's auc: 0.887816	valid_0's binary_logloss: 0.22128
    [1797]	valid_0's auc: 0.887813	valid_0's binary_logloss: 0.221383
    [1798]	valid_0's auc: 0.88782	valid_0's binary_logloss: 0.221468
    [1799]	valid_0's auc: 0.887833	valid_0's binary_logloss: 0.221532
    [1800]	valid_0's auc: 0.887952	valid_0's binary_logloss: 0.221202
    [1801]	valid_0's auc: 0.88788	valid_0's binary_logloss: 0.221072
    [1802]	valid_0's auc: 0.88775	valid_0's binary_logloss: 0.220957
    [1803]	valid_0's auc: 0.887757	valid_0's binary_logloss: 0.220807
    [1804]	valid_0's auc: 0.887766	valid_0's binary_logloss: 0.220658
    [1805]	valid_0's auc: 0.887654	valid_0's binary_logloss: 0.220529
    [1806]	valid_0's auc: 0.887577	valid_0's binary_logloss: 0.22038
    [1807]	valid_0's auc: 0.887585	valid_0's binary_logloss: 0.220463
    [1808]	valid_0's auc: 0.887653	valid_0's binary_logloss: 0.2203
    [1809]	valid_0's auc: 0.887664	valid_0's binary_logloss: 0.220376
    [1810]	valid_0's auc: 0.887796	valid_0's binary_logloss: 0.220149
    [1811]	valid_0's auc: 0.887797	valid_0's binary_logloss: 0.220221
    [1812]	valid_0's auc: 0.887824	valid_0's binary_logloss: 0.219991
    [1813]	valid_0's auc: 0.887848	valid_0's binary_logloss: 0.219847
    [1814]	valid_0's auc: 0.88785	valid_0's binary_logloss: 0.219928
    [1815]	valid_0's auc: 0.887813	valid_0's binary_logloss: 0.219774
    [1816]	valid_0's auc: 0.887822	valid_0's binary_logloss: 0.219865
    [1817]	valid_0's auc: 0.88783	valid_0's binary_logloss: 0.219731
    [1818]	valid_0's auc: 0.887875	valid_0's binary_logloss: 0.219569
    [1819]	valid_0's auc: 0.887884	valid_0's binary_logloss: 0.219655
    [1820]	valid_0's auc: 0.88788	valid_0's binary_logloss: 0.219525
    [1821]	valid_0's auc: 0.887885	valid_0's binary_logloss: 0.219624
    [1822]	valid_0's auc: 0.887882	valid_0's binary_logloss: 0.219745
    [1823]	valid_0's auc: 0.88807	valid_0's binary_logloss: 0.219391
    [1824]	valid_0's auc: 0.888079	valid_0's binary_logloss: 0.21946
    [1825]	valid_0's auc: 0.888085	valid_0's binary_logloss: 0.21953
    [1826]	valid_0's auc: 0.888077	valid_0's binary_logloss: 0.219391
    [1827]	valid_0's auc: 0.888078	valid_0's binary_logloss: 0.219503
    [1828]	valid_0's auc: 0.888087	valid_0's binary_logloss: 0.21932
    [1829]	valid_0's auc: 0.888096	valid_0's binary_logloss: 0.21939
    [1830]	valid_0's auc: 0.888103	valid_0's binary_logloss: 0.219471
    [1831]	valid_0's auc: 0.888077	valid_0's binary_logloss: 0.219358
    [1832]	valid_0's auc: 0.88801	valid_0's binary_logloss: 0.219236
    [1833]	valid_0's auc: 0.888006	valid_0's binary_logloss: 0.219328
    [1834]	valid_0's auc: 0.887994	valid_0's binary_logloss: 0.219227
    [1835]	valid_0's auc: 0.88798	valid_0's binary_logloss: 0.21935
    [1836]	valid_0's auc: 0.887959	valid_0's binary_logloss: 0.219157
    [1837]	valid_0's auc: 0.887953	valid_0's binary_logloss: 0.219256
    [1838]	valid_0's auc: 0.887963	valid_0's binary_logloss: 0.219322
    [1839]	valid_0's auc: 0.887942	valid_0's binary_logloss: 0.21915
    [1840]	valid_0's auc: 0.887941	valid_0's binary_logloss: 0.219256
    [1841]	valid_0's auc: 0.887968	valid_0's binary_logloss: 0.219148
    [1842]	valid_0's auc: 0.887974	valid_0's binary_logloss: 0.219236
    [1843]	valid_0's auc: 0.887978	valid_0's binary_logloss: 0.219328
    [1844]	valid_0's auc: 0.887985	valid_0's binary_logloss: 0.219429
    [1845]	valid_0's auc: 0.887979	valid_0's binary_logloss: 0.219527
    [1846]	valid_0's auc: 0.887989	valid_0's binary_logloss: 0.219387
    [1847]	valid_0's auc: 0.887988	valid_0's binary_logloss: 0.219479
    [1848]	valid_0's auc: 0.887944	valid_0's binary_logloss: 0.219334
    [1849]	valid_0's auc: 0.88793	valid_0's binary_logloss: 0.219215
    [1850]	valid_0's auc: 0.887952	valid_0's binary_logloss: 0.219086
    [1851]	valid_0's auc: 0.887954	valid_0's binary_logloss: 0.219153
    [1852]	valid_0's auc: 0.887961	valid_0's binary_logloss: 0.21922
    [1853]	valid_0's auc: 0.887964	valid_0's binary_logloss: 0.219306
    [1854]	valid_0's auc: 0.88809	valid_0's binary_logloss: 0.218942
    [1855]	valid_0's auc: 0.888091	valid_0's binary_logloss: 0.219038
    [1856]	valid_0's auc: 0.888103	valid_0's binary_logloss: 0.219152
    [1857]	valid_0's auc: 0.888072	valid_0's binary_logloss: 0.218946
    [1858]	valid_0's auc: 0.888068	valid_0's binary_logloss: 0.21906
    [1859]	valid_0's auc: 0.887987	valid_0's binary_logloss: 0.218909
    [1860]	valid_0's auc: 0.887998	valid_0's binary_logloss: 0.218967
    [1861]	valid_0's auc: 0.888015	valid_0's binary_logloss: 0.218808
    [1862]	valid_0's auc: 0.888018	valid_0's binary_logloss: 0.218877
    [1863]	valid_0's auc: 0.888276	valid_0's binary_logloss: 0.218593
    [1864]	valid_0's auc: 0.888293	valid_0's binary_logloss: 0.218441
    [1865]	valid_0's auc: 0.88836	valid_0's binary_logloss: 0.21832
    [1866]	valid_0's auc: 0.888366	valid_0's binary_logloss: 0.218395
    [1867]	valid_0's auc: 0.888487	valid_0's binary_logloss: 0.218158
    [1868]	valid_0's auc: 0.888503	valid_0's binary_logloss: 0.218243
    [1869]	valid_0's auc: 0.888522	valid_0's binary_logloss: 0.218076
    [1870]	valid_0's auc: 0.888531	valid_0's binary_logloss: 0.218166
    [1871]	valid_0's auc: 0.888536	valid_0's binary_logloss: 0.218006
    [1872]	valid_0's auc: 0.888748	valid_0's binary_logloss: 0.217616
    [1873]	valid_0's auc: 0.88876	valid_0's binary_logloss: 0.217727
    [1874]	valid_0's auc: 0.888755	valid_0's binary_logloss: 0.217575
    [1875]	valid_0's auc: 0.888766	valid_0's binary_logloss: 0.217645
    [1876]	valid_0's auc: 0.888785	valid_0's binary_logloss: 0.217461
    [1877]	valid_0's auc: 0.888793	valid_0's binary_logloss: 0.21756
    [1878]	valid_0's auc: 0.888785	valid_0's binary_logloss: 0.217686
    [1879]	valid_0's auc: 0.888781	valid_0's binary_logloss: 0.217768
    [1880]	valid_0's auc: 0.888789	valid_0's binary_logloss: 0.217862
    [1881]	valid_0's auc: 0.888798	valid_0's binary_logloss: 0.217949
    [1882]	valid_0's auc: 0.888788	valid_0's binary_logloss: 0.217809
    [1883]	valid_0's auc: 0.888796	valid_0's binary_logloss: 0.217882
    [1884]	valid_0's auc: 0.888793	valid_0's binary_logloss: 0.21803
    [1885]	valid_0's auc: 0.88877	valid_0's binary_logloss: 0.217866
    [1886]	valid_0's auc: 0.888772	valid_0's binary_logloss: 0.217948
    [1887]	valid_0's auc: 0.888778	valid_0's binary_logloss: 0.218065
    [1888]	valid_0's auc: 0.888804	valid_0's binary_logloss: 0.217804
    [1889]	valid_0's auc: 0.888804	valid_0's binary_logloss: 0.217899
    [1890]	valid_0's auc: 0.888802	valid_0's binary_logloss: 0.217987
    [1891]	valid_0's auc: 0.888789	valid_0's binary_logloss: 0.217842
    [1892]	valid_0's auc: 0.888715	valid_0's binary_logloss: 0.217691
    [1893]	valid_0's auc: 0.888753	valid_0's binary_logloss: 0.217511
    [1894]	valid_0's auc: 0.888743	valid_0's binary_logloss: 0.217322
    [1895]	valid_0's auc: 0.888754	valid_0's binary_logloss: 0.217174
    [1896]	valid_0's auc: 0.888714	valid_0's binary_logloss: 0.217002
    [1897]	valid_0's auc: 0.88884	valid_0's binary_logloss: 0.21667
    [1898]	valid_0's auc: 0.888844	valid_0's binary_logloss: 0.216544
    [1899]	valid_0's auc: 0.888856	valid_0's binary_logloss: 0.216622
    [1900]	valid_0's auc: 0.888859	valid_0's binary_logloss: 0.216706
    [1901]	valid_0's auc: 0.888865	valid_0's binary_logloss: 0.216791
    [1902]	valid_0's auc: 0.888873	valid_0's binary_logloss: 0.216861
    [1903]	valid_0's auc: 0.888895	valid_0's binary_logloss: 0.216729
    [1904]	valid_0's auc: 0.888897	valid_0's binary_logloss: 0.216807
    [1905]	valid_0's auc: 0.888889	valid_0's binary_logloss: 0.216932
    [1906]	valid_0's auc: 0.888888	valid_0's binary_logloss: 0.217008
    [1907]	valid_0's auc: 0.888889	valid_0's binary_logloss: 0.217114
    [1908]	valid_0's auc: 0.888865	valid_0's binary_logloss: 0.216914
    [1909]	valid_0's auc: 0.88886	valid_0's binary_logloss: 0.216762
    [1910]	valid_0's auc: 0.888868	valid_0's binary_logloss: 0.216845
    [1911]	valid_0's auc: 0.888874	valid_0's binary_logloss: 0.216939
    [1912]	valid_0's auc: 0.888895	valid_0's binary_logloss: 0.216815
    [1913]	valid_0's auc: 0.888895	valid_0's binary_logloss: 0.216891
    [1914]	valid_0's auc: 0.888899	valid_0's binary_logloss: 0.216969
    [1915]	valid_0's auc: 0.888898	valid_0's binary_logloss: 0.217041
    [1916]	valid_0's auc: 0.888907	valid_0's binary_logloss: 0.217112
    [1917]	valid_0's auc: 0.888936	valid_0's binary_logloss: 0.216976
    [1918]	valid_0's auc: 0.888942	valid_0's binary_logloss: 0.217044
    [1919]	valid_0's auc: 0.888941	valid_0's binary_logloss: 0.217125
    [1920]	valid_0's auc: 0.888958	valid_0's binary_logloss: 0.217216
    [1921]	valid_0's auc: 0.888962	valid_0's binary_logloss: 0.217323
    [1922]	valid_0's auc: 0.888872	valid_0's binary_logloss: 0.217191
    [1923]	valid_0's auc: 0.888969	valid_0's binary_logloss: 0.216977
    [1924]	valid_0's auc: 0.888979	valid_0's binary_logloss: 0.217053
    [1925]	valid_0's auc: 0.88899	valid_0's binary_logloss: 0.217141
    [1926]	valid_0's auc: 0.888932	valid_0's binary_logloss: 0.21696
    [1927]	valid_0's auc: 0.888936	valid_0's binary_logloss: 0.217029
    [1928]	valid_0's auc: 0.888947	valid_0's binary_logloss: 0.21709
    [1929]	valid_0's auc: 0.888946	valid_0's binary_logloss: 0.217171
    [1930]	valid_0's auc: 0.888874	valid_0's binary_logloss: 0.217044
    [1931]	valid_0's auc: 0.888854	valid_0's binary_logloss: 0.216859
    [1932]	valid_0's auc: 0.888837	valid_0's binary_logloss: 0.216956
    [1933]	valid_0's auc: 0.888813	valid_0's binary_logloss: 0.216809
    [1934]	valid_0's auc: 0.888848	valid_0's binary_logloss: 0.216539
    [1935]	valid_0's auc: 0.888866	valid_0's binary_logloss: 0.216613
    [1936]	valid_0's auc: 0.888858	valid_0's binary_logloss: 0.2165
    [1937]	valid_0's auc: 0.888865	valid_0's binary_logloss: 0.216559
    [1938]	valid_0's auc: 0.888863	valid_0's binary_logloss: 0.216386
    [1939]	valid_0's auc: 0.889056	valid_0's binary_logloss: 0.21603
    [1940]	valid_0's auc: 0.889227	valid_0's binary_logloss: 0.215646
    [1941]	valid_0's auc: 0.889232	valid_0's binary_logloss: 0.21572
    [1942]	valid_0's auc: 0.889184	valid_0's binary_logloss: 0.215599
    [1943]	valid_0's auc: 0.889178	valid_0's binary_logloss: 0.215444
    [1944]	valid_0's auc: 0.889172	valid_0's binary_logloss: 0.215324
    [1945]	valid_0's auc: 0.889413	valid_0's binary_logloss: 0.215024
    [1946]	valid_0's auc: 0.889413	valid_0's binary_logloss: 0.21513
    [1947]	valid_0's auc: 0.889421	valid_0's binary_logloss: 0.215206
    [1948]	valid_0's auc: 0.889348	valid_0's binary_logloss: 0.215076
    [1949]	valid_0's auc: 0.889582	valid_0's binary_logloss: 0.214805
    [1950]	valid_0's auc: 0.889587	valid_0's binary_logloss: 0.214678
    [1951]	valid_0's auc: 0.889589	valid_0's binary_logloss: 0.214557
    [1952]	valid_0's auc: 0.889602	valid_0's binary_logloss: 0.214649
    [1953]	valid_0's auc: 0.889701	valid_0's binary_logloss: 0.214351
    [1954]	valid_0's auc: 0.88969	valid_0's binary_logloss: 0.214433
    [1955]	valid_0's auc: 0.889699	valid_0's binary_logloss: 0.214476
    [1956]	valid_0's auc: 0.889702	valid_0's binary_logloss: 0.214586
    [1957]	valid_0's auc: 0.889706	valid_0's binary_logloss: 0.214642
    [1958]	valid_0's auc: 0.889709	valid_0's binary_logloss: 0.214715
    [1959]	valid_0's auc: 0.889678	valid_0's binary_logloss: 0.214541
    [1960]	valid_0's auc: 0.889683	valid_0's binary_logloss: 0.214624
    [1961]	valid_0's auc: 0.88969	valid_0's binary_logloss: 0.214711
    [1962]	valid_0's auc: 0.88964	valid_0's binary_logloss: 0.2146
    [1963]	valid_0's auc: 0.889646	valid_0's binary_logloss: 0.21466
    [1964]	valid_0's auc: 0.889618	valid_0's binary_logloss: 0.214557
    [1965]	valid_0's auc: 0.889846	valid_0's binary_logloss: 0.214284
    [1966]	valid_0's auc: 0.88984	valid_0's binary_logloss: 0.214378
    [1967]	valid_0's auc: 0.890043	valid_0's binary_logloss: 0.214126
    [1968]	valid_0's auc: 0.890045	valid_0's binary_logloss: 0.214032
    [1969]	valid_0's auc: 0.890049	valid_0's binary_logloss: 0.214093
    [1970]	valid_0's auc: 0.890054	valid_0's binary_logloss: 0.214178
    [1971]	valid_0's auc: 0.889999	valid_0's binary_logloss: 0.214032
    [1972]	valid_0's auc: 0.889994	valid_0's binary_logloss: 0.214103
    [1973]	valid_0's auc: 0.890005	valid_0's binary_logloss: 0.214183
    [1974]	valid_0's auc: 0.889962	valid_0's binary_logloss: 0.214093
    [1975]	valid_0's auc: 0.889948	valid_0's binary_logloss: 0.213953
    [1976]	valid_0's auc: 0.889957	valid_0's binary_logloss: 0.214013
    [1977]	valid_0's auc: 0.889966	valid_0's binary_logloss: 0.214079
    [1978]	valid_0's auc: 0.889925	valid_0's binary_logloss: 0.213954
    [1979]	valid_0's auc: 0.889923	valid_0's binary_logloss: 0.21383
    [1980]	valid_0's auc: 0.889928	valid_0's binary_logloss: 0.213892
    [1981]	valid_0's auc: 0.889932	valid_0's binary_logloss: 0.213967
    [1982]	valid_0's auc: 0.889942	valid_0's binary_logloss: 0.214042
    [1983]	valid_0's auc: 0.889919	valid_0's binary_logloss: 0.213913
    [1984]	valid_0's auc: 0.889913	valid_0's binary_logloss: 0.21378
    [1985]	valid_0's auc: 0.889916	valid_0's binary_logloss: 0.213884
    [1986]	valid_0's auc: 0.889917	valid_0's binary_logloss: 0.213958
    [1987]	valid_0's auc: 0.889921	valid_0's binary_logloss: 0.214038
    [1988]	valid_0's auc: 0.889923	valid_0's binary_logloss: 0.214098
    [1989]	valid_0's auc: 0.88989	valid_0's binary_logloss: 0.214
    [1990]	valid_0's auc: 0.889892	valid_0's binary_logloss: 0.214087
    [1991]	valid_0's auc: 0.889865	valid_0's binary_logloss: 0.21401
    [1992]	valid_0's auc: 0.889825	valid_0's binary_logloss: 0.213835
    [1993]	valid_0's auc: 0.889827	valid_0's binary_logloss: 0.213906
    [1994]	valid_0's auc: 0.889838	valid_0's binary_logloss: 0.213645
    [1995]	valid_0's auc: 0.889811	valid_0's binary_logloss: 0.213454
    [1996]	valid_0's auc: 0.889813	valid_0's binary_logloss: 0.213537
    [1997]	valid_0's auc: 0.889818	valid_0's binary_logloss: 0.213606
    [1998]	valid_0's auc: 0.889858	valid_0's binary_logloss: 0.213445
    [1999]	valid_0's auc: 0.889852	valid_0's binary_logloss: 0.213334
    [2000]	valid_0's auc: 0.889832	valid_0's binary_logloss: 0.213219
    [2001]	valid_0's auc: 0.889842	valid_0's binary_logloss: 0.213295
    [2002]	valid_0's auc: 0.889844	valid_0's binary_logloss: 0.213382
    [2003]	valid_0's auc: 0.889841	valid_0's binary_logloss: 0.213469
    [2004]	valid_0's auc: 0.889801	valid_0's binary_logloss: 0.213319
    [2005]	valid_0's auc: 0.889808	valid_0's binary_logloss: 0.213372
    [2006]	valid_0's auc: 0.889818	valid_0's binary_logloss: 0.213428
    [2007]	valid_0's auc: 0.889826	valid_0's binary_logloss: 0.213304
    [2008]	valid_0's auc: 0.889772	valid_0's binary_logloss: 0.213174
    [2009]	valid_0's auc: 0.889777	valid_0's binary_logloss: 0.21325
    [2010]	valid_0's auc: 0.889763	valid_0's binary_logloss: 0.213134
    [2011]	valid_0's auc: 0.889715	valid_0's binary_logloss: 0.213014
    [2012]	valid_0's auc: 0.889715	valid_0's binary_logloss: 0.212907
    [2013]	valid_0's auc: 0.889719	valid_0's binary_logloss: 0.212983
    [2014]	valid_0's auc: 0.889669	valid_0's binary_logloss: 0.212856
    [2015]	valid_0's auc: 0.889588	valid_0's binary_logloss: 0.212719
    [2016]	valid_0's auc: 0.88961	valid_0's binary_logloss: 0.212575
    [2017]	valid_0's auc: 0.889587	valid_0's binary_logloss: 0.212463
    [2018]	valid_0's auc: 0.889599	valid_0's binary_logloss: 0.212318
    [2019]	valid_0's auc: 0.889651	valid_0's binary_logloss: 0.212152
    [2020]	valid_0's auc: 0.889615	valid_0's binary_logloss: 0.212034
    [2021]	valid_0's auc: 0.889609	valid_0's binary_logloss: 0.212119
    [2022]	valid_0's auc: 0.889627	valid_0's binary_logloss: 0.212023
    [2023]	valid_0's auc: 0.889629	valid_0's binary_logloss: 0.212073
    [2024]	valid_0's auc: 0.889614	valid_0's binary_logloss: 0.211971
    [2025]	valid_0's auc: 0.889642	valid_0's binary_logloss: 0.21174
    [2026]	valid_0's auc: 0.889649	valid_0's binary_logloss: 0.211793
    [2027]	valid_0's auc: 0.889756	valid_0's binary_logloss: 0.211619
    [2028]	valid_0's auc: 0.889759	valid_0's binary_logloss: 0.211714
    [2029]	valid_0's auc: 0.889764	valid_0's binary_logloss: 0.211788
    [2030]	valid_0's auc: 0.88972	valid_0's binary_logloss: 0.211604
    [2031]	valid_0's auc: 0.889724	valid_0's binary_logloss: 0.211675
    [2032]	valid_0's auc: 0.889719	valid_0's binary_logloss: 0.211736
    [2033]	valid_0's auc: 0.889717	valid_0's binary_logloss: 0.211816
    [2034]	valid_0's auc: 0.889721	valid_0's binary_logloss: 0.211869
    [2035]	valid_0's auc: 0.889737	valid_0's binary_logloss: 0.211749
    [2036]	valid_0's auc: 0.88977	valid_0's binary_logloss: 0.211621
    [2037]	valid_0's auc: 0.889742	valid_0's binary_logloss: 0.211502
    [2038]	valid_0's auc: 0.889719	valid_0's binary_logloss: 0.211382
    [2039]	valid_0's auc: 0.889731	valid_0's binary_logloss: 0.211253
    [2040]	valid_0's auc: 0.889737	valid_0's binary_logloss: 0.211343
    [2041]	valid_0's auc: 0.889742	valid_0's binary_logloss: 0.211424
    [2042]	valid_0's auc: 0.889752	valid_0's binary_logloss: 0.211287
    [2043]	valid_0's auc: 0.889763	valid_0's binary_logloss: 0.211161
    [2044]	valid_0's auc: 0.889767	valid_0's binary_logloss: 0.211218
    [2045]	valid_0's auc: 0.889727	valid_0's binary_logloss: 0.211116
    [2046]	valid_0's auc: 0.88973	valid_0's binary_logloss: 0.211174
    [2047]	valid_0's auc: 0.889676	valid_0's binary_logloss: 0.211043
    [2048]	valid_0's auc: 0.889683	valid_0's binary_logloss: 0.21111
    [2049]	valid_0's auc: 0.889687	valid_0's binary_logloss: 0.211174
    [2050]	valid_0's auc: 0.889687	valid_0's binary_logloss: 0.211248
    [2051]	valid_0's auc: 0.889922	valid_0's binary_logloss: 0.210928
    [2052]	valid_0's auc: 0.889927	valid_0's binary_logloss: 0.210813
    [2053]	valid_0's auc: 0.889892	valid_0's binary_logloss: 0.210714
    [2054]	valid_0's auc: 0.889923	valid_0's binary_logloss: 0.210572
    [2055]	valid_0's auc: 0.889844	valid_0's binary_logloss: 0.210399
    [2056]	valid_0's auc: 0.889835	valid_0's binary_logloss: 0.210502
    [2057]	valid_0's auc: 0.889844	valid_0's binary_logloss: 0.210554
    [2058]	valid_0's auc: 0.889848	valid_0's binary_logloss: 0.210621
    [2059]	valid_0's auc: 0.88993	valid_0's binary_logloss: 0.21042
    [2060]	valid_0's auc: 0.88994	valid_0's binary_logloss: 0.210463
    [2061]	valid_0's auc: 0.889946	valid_0's binary_logloss: 0.210515
    [2062]	valid_0's auc: 0.889949	valid_0's binary_logloss: 0.21057
    [2063]	valid_0's auc: 0.889959	valid_0's binary_logloss: 0.21063
    [2064]	valid_0's auc: 0.889962	valid_0's binary_logloss: 0.210698
    [2065]	valid_0's auc: 0.889965	valid_0's binary_logloss: 0.210763
    [2066]	valid_0's auc: 0.889964	valid_0's binary_logloss: 0.210853
    [2067]	valid_0's auc: 0.889912	valid_0's binary_logloss: 0.210782
    [2068]	valid_0's auc: 0.889914	valid_0's binary_logloss: 0.210832
    [2069]	valid_0's auc: 0.889918	valid_0's binary_logloss: 0.210907
    [2070]	valid_0's auc: 0.889921	valid_0's binary_logloss: 0.210982
    [2071]	valid_0's auc: 0.889911	valid_0's binary_logloss: 0.210866
    [2072]	valid_0's auc: 0.889913	valid_0's binary_logloss: 0.210925
    [2073]	valid_0's auc: 0.889894	valid_0's binary_logloss: 0.21081
    [2074]	valid_0's auc: 0.889893	valid_0's binary_logloss: 0.21089
    [2075]	valid_0's auc: 0.889908	valid_0's binary_logloss: 0.210722
    [2076]	valid_0's auc: 0.890049	valid_0's binary_logloss: 0.210515
    [2077]	valid_0's auc: 0.890229	valid_0's binary_logloss: 0.210194
    [2078]	valid_0's auc: 0.89019	valid_0's binary_logloss: 0.210042
    [2079]	valid_0's auc: 0.890194	valid_0's binary_logloss: 0.210105
    [2080]	valid_0's auc: 0.890175	valid_0's binary_logloss: 0.210021
    [2081]	valid_0's auc: 0.890261	valid_0's binary_logloss: 0.209722
    [2082]	valid_0's auc: 0.890196	valid_0's binary_logloss: 0.209634
    [2083]	valid_0's auc: 0.890203	valid_0's binary_logloss: 0.209678
    [2084]	valid_0's auc: 0.890317	valid_0's binary_logloss: 0.209456
    [2085]	valid_0's auc: 0.890533	valid_0's binary_logloss: 0.209202
    [2086]	valid_0's auc: 0.890534	valid_0's binary_logloss: 0.209256
    [2087]	valid_0's auc: 0.890483	valid_0's binary_logloss: 0.209135
    [2088]	valid_0's auc: 0.890724	valid_0's binary_logloss: 0.208885
    [2089]	valid_0's auc: 0.89073	valid_0's binary_logloss: 0.208944
    [2090]	valid_0's auc: 0.890749	valid_0's binary_logloss: 0.208816
    [2091]	valid_0's auc: 0.890743	valid_0's binary_logloss: 0.20892
    [2092]	valid_0's auc: 0.890841	valid_0's binary_logloss: 0.208768
    [2093]	valid_0's auc: 0.890844	valid_0's binary_logloss: 0.208842
    [2094]	valid_0's auc: 0.890997	valid_0's binary_logloss: 0.208562
    [2095]	valid_0's auc: 0.891003	valid_0's binary_logloss: 0.208623
    [2096]	valid_0's auc: 0.890938	valid_0's binary_logloss: 0.208517
    [2097]	valid_0's auc: 0.890937	valid_0's binary_logloss: 0.208422
    [2098]	valid_0's auc: 0.891095	valid_0's binary_logloss: 0.208189
    [2099]	valid_0's auc: 0.891091	valid_0's binary_logloss: 0.208095
    [2100]	valid_0's auc: 0.891088	valid_0's binary_logloss: 0.207998
    [2101]	valid_0's auc: 0.891088	valid_0's binary_logloss: 0.208052
    [2102]	valid_0's auc: 0.891101	valid_0's binary_logloss: 0.207947
    [2103]	valid_0's auc: 0.8911	valid_0's binary_logloss: 0.208014
    [2104]	valid_0's auc: 0.891099	valid_0's binary_logloss: 0.208087
    [2105]	valid_0's auc: 0.891124	valid_0's binary_logloss: 0.207978
    [2106]	valid_0's auc: 0.891204	valid_0's binary_logloss: 0.207703
    [2107]	valid_0's auc: 0.891198	valid_0's binary_logloss: 0.20778
    [2108]	valid_0's auc: 0.891201	valid_0's binary_logloss: 0.207852
    [2109]	valid_0's auc: 0.8912	valid_0's binary_logloss: 0.207923
    [2110]	valid_0's auc: 0.891208	valid_0's binary_logloss: 0.208012
    [2111]	valid_0's auc: 0.891212	valid_0's binary_logloss: 0.208078
    [2112]	valid_0's auc: 0.891212	valid_0's binary_logloss: 0.208153
    [2113]	valid_0's auc: 0.891215	valid_0's binary_logloss: 0.208046
    [2114]	valid_0's auc: 0.89122	valid_0's binary_logloss: 0.208151
    [2115]	valid_0's auc: 0.891229	valid_0's binary_logloss: 0.208226
    [2116]	valid_0's auc: 0.891218	valid_0's binary_logloss: 0.208106
    [2117]	valid_0's auc: 0.891292	valid_0's binary_logloss: 0.207912
    [2118]	valid_0's auc: 0.891301	valid_0's binary_logloss: 0.207968
    [2119]	valid_0's auc: 0.891307	valid_0's binary_logloss: 0.208018
    [2120]	valid_0's auc: 0.891332	valid_0's binary_logloss: 0.207901
    [2121]	valid_0's auc: 0.891335	valid_0's binary_logloss: 0.207791
    [2122]	valid_0's auc: 0.891339	valid_0's binary_logloss: 0.207847
    [2123]	valid_0's auc: 0.891349	valid_0's binary_logloss: 0.207915
    [2124]	valid_0's auc: 0.891355	valid_0's binary_logloss: 0.207993
    [2125]	valid_0's auc: 0.891546	valid_0's binary_logloss: 0.207757
    [2126]	valid_0's auc: 0.891533	valid_0's binary_logloss: 0.207676
    [2127]	valid_0's auc: 0.891539	valid_0's binary_logloss: 0.207759
    [2128]	valid_0's auc: 0.891525	valid_0's binary_logloss: 0.207669
    [2129]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.207721
    [2130]	valid_0's auc: 0.891524	valid_0's binary_logloss: 0.207791
    [2131]	valid_0's auc: 0.891523	valid_0's binary_logloss: 0.207867
    [2132]	valid_0's auc: 0.891486	valid_0's binary_logloss: 0.207786
    [2133]	valid_0's auc: 0.891517	valid_0's binary_logloss: 0.207689
    [2134]	valid_0's auc: 0.891526	valid_0's binary_logloss: 0.207735
    [2135]	valid_0's auc: 0.891533	valid_0's binary_logloss: 0.207782
    [2136]	valid_0's auc: 0.891539	valid_0's binary_logloss: 0.20783
    [2137]	valid_0's auc: 0.891569	valid_0's binary_logloss: 0.207732
    [2138]	valid_0's auc: 0.891528	valid_0's binary_logloss: 0.207645
    [2139]	valid_0's auc: 0.891635	valid_0's binary_logloss: 0.207431
    [2140]	valid_0's auc: 0.891639	valid_0's binary_logloss: 0.207503
    [2141]	valid_0's auc: 0.891644	valid_0's binary_logloss: 0.207581
    [2142]	valid_0's auc: 0.891628	valid_0's binary_logloss: 0.207469
    [2143]	valid_0's auc: 0.891604	valid_0's binary_logloss: 0.207376
    [2144]	valid_0's auc: 0.891606	valid_0's binary_logloss: 0.207425
    [2145]	valid_0's auc: 0.891612	valid_0's binary_logloss: 0.207482
    [2146]	valid_0's auc: 0.891595	valid_0's binary_logloss: 0.20739
    [2147]	valid_0's auc: 0.891591	valid_0's binary_logloss: 0.207471
    [2148]	valid_0's auc: 0.891595	valid_0's binary_logloss: 0.20752
    [2149]	valid_0's auc: 0.891746	valid_0's binary_logloss: 0.207283
    [2150]	valid_0's auc: 0.89177	valid_0's binary_logloss: 0.207163
    [2151]	valid_0's auc: 0.891777	valid_0's binary_logloss: 0.207252
    [2152]	valid_0's auc: 0.89178	valid_0's binary_logloss: 0.207334
    [2153]	valid_0's auc: 0.891716	valid_0's binary_logloss: 0.207251
    [2154]	valid_0's auc: 0.891717	valid_0's binary_logloss: 0.20713
    [2155]	valid_0's auc: 0.891739	valid_0's binary_logloss: 0.207016
    [2156]	valid_0's auc: 0.891739	valid_0's binary_logloss: 0.206899
    [2157]	valid_0's auc: 0.891738	valid_0's binary_logloss: 0.20696
    [2158]	valid_0's auc: 0.891732	valid_0's binary_logloss: 0.206889
    [2159]	valid_0's auc: 0.89174	valid_0's binary_logloss: 0.206945
    [2160]	valid_0's auc: 0.891718	valid_0's binary_logloss: 0.206843
    [2161]	valid_0's auc: 0.891727	valid_0's binary_logloss: 0.206914
    [2162]	valid_0's auc: 0.891729	valid_0's binary_logloss: 0.206976
    [2163]	valid_0's auc: 0.891692	valid_0's binary_logloss: 0.206751
    [2164]	valid_0's auc: 0.891783	valid_0's binary_logloss: 0.206635
    [2165]	valid_0's auc: 0.89177	valid_0's binary_logloss: 0.206539
    [2166]	valid_0's auc: 0.891774	valid_0's binary_logloss: 0.206607
    [2167]	valid_0's auc: 0.891775	valid_0's binary_logloss: 0.206688
    [2168]	valid_0's auc: 0.89176	valid_0's binary_logloss: 0.206589
    [2169]	valid_0's auc: 0.891764	valid_0's binary_logloss: 0.20665
    [2170]	valid_0's auc: 0.891769	valid_0's binary_logloss: 0.206709
    [2171]	valid_0's auc: 0.891774	valid_0's binary_logloss: 0.206654
    [2172]	valid_0's auc: 0.891831	valid_0's binary_logloss: 0.206535
    [2173]	valid_0's auc: 0.891841	valid_0's binary_logloss: 0.206595
    [2174]	valid_0's auc: 0.89181	valid_0's binary_logloss: 0.206512
    [2175]	valid_0's auc: 0.891817	valid_0's binary_logloss: 0.206433
    [2176]	valid_0's auc: 0.891894	valid_0's binary_logloss: 0.206326
    [2177]	valid_0's auc: 0.891904	valid_0's binary_logloss: 0.206388
    [2178]	valid_0's auc: 0.89189	valid_0's binary_logloss: 0.206261
    [2179]	valid_0's auc: 0.891894	valid_0's binary_logloss: 0.206339
    [2180]	valid_0's auc: 0.891915	valid_0's binary_logloss: 0.206229
    [2181]	valid_0's auc: 0.891927	valid_0's binary_logloss: 0.206123
    [2182]	valid_0's auc: 0.892092	valid_0's binary_logloss: 0.205902
    [2183]	valid_0's auc: 0.892105	valid_0's binary_logloss: 0.205971
    [2184]	valid_0's auc: 0.89217	valid_0's binary_logloss: 0.205796
    [2185]	valid_0's auc: 0.892177	valid_0's binary_logloss: 0.205844
    [2186]	valid_0's auc: 0.892182	valid_0's binary_logloss: 0.205896
    [2187]	valid_0's auc: 0.892153	valid_0's binary_logloss: 0.205829
    [2188]	valid_0's auc: 0.892158	valid_0's binary_logloss: 0.205881
    [2189]	valid_0's auc: 0.892128	valid_0's binary_logloss: 0.205787
    [2190]	valid_0's auc: 0.892129	valid_0's binary_logloss: 0.205856
    [2191]	valid_0's auc: 0.892249	valid_0's binary_logloss: 0.205566
    [2192]	valid_0's auc: 0.892249	valid_0's binary_logloss: 0.205622
    [2193]	valid_0's auc: 0.892304	valid_0's binary_logloss: 0.205403
    [2194]	valid_0's auc: 0.892291	valid_0's binary_logloss: 0.205318
    [2195]	valid_0's auc: 0.89226	valid_0's binary_logloss: 0.205251
    [2196]	valid_0's auc: 0.892267	valid_0's binary_logloss: 0.205322
    [2197]	valid_0's auc: 0.892274	valid_0's binary_logloss: 0.205387
    [2198]	valid_0's auc: 0.89226	valid_0's binary_logloss: 0.205294
    [2199]	valid_0's auc: 0.892296	valid_0's binary_logloss: 0.205186
    [2200]	valid_0's auc: 0.89232	valid_0's binary_logloss: 0.205028
    [2201]	valid_0's auc: 0.892322	valid_0's binary_logloss: 0.205071
    [2202]	valid_0's auc: 0.892326	valid_0's binary_logloss: 0.205136
    [2203]	valid_0's auc: 0.892334	valid_0's binary_logloss: 0.20519
    [2204]	valid_0's auc: 0.89234	valid_0's binary_logloss: 0.205241
    [2205]	valid_0's auc: 0.892338	valid_0's binary_logloss: 0.205297
    [2206]	valid_0's auc: 0.892321	valid_0's binary_logloss: 0.205213
    [2207]	valid_0's auc: 0.892334	valid_0's binary_logloss: 0.205272
    [2208]	valid_0's auc: 0.892337	valid_0's binary_logloss: 0.205337
    [2209]	valid_0's auc: 0.89235	valid_0's binary_logloss: 0.20522
    [2210]	valid_0's auc: 0.892356	valid_0's binary_logloss: 0.205276
    [2211]	valid_0's auc: 0.892304	valid_0's binary_logloss: 0.205191
    [2212]	valid_0's auc: 0.892319	valid_0's binary_logloss: 0.205061
    [2213]	valid_0's auc: 0.892322	valid_0's binary_logloss: 0.205115
    [2214]	valid_0's auc: 0.892324	valid_0's binary_logloss: 0.205184
    [2215]	valid_0's auc: 0.892346	valid_0's binary_logloss: 0.205061
    [2216]	valid_0's auc: 0.892357	valid_0's binary_logloss: 0.204899
    [2217]	valid_0's auc: 0.892367	valid_0's binary_logloss: 0.204954
    [2218]	valid_0's auc: 0.892364	valid_0's binary_logloss: 0.205037
    [2219]	valid_0's auc: 0.892426	valid_0's binary_logloss: 0.204806
    [2220]	valid_0's auc: 0.892427	valid_0's binary_logloss: 0.204705
    [2221]	valid_0's auc: 0.89243	valid_0's binary_logloss: 0.204769
    [2222]	valid_0's auc: 0.892419	valid_0's binary_logloss: 0.204672
    [2223]	valid_0's auc: 0.892424	valid_0's binary_logloss: 0.204744
    [2224]	valid_0's auc: 0.892384	valid_0's binary_logloss: 0.204642
    [2225]	valid_0's auc: 0.89239	valid_0's binary_logloss: 0.204465
    [2226]	valid_0's auc: 0.892427	valid_0's binary_logloss: 0.204361
    [2227]	valid_0's auc: 0.892428	valid_0's binary_logloss: 0.204429
    [2228]	valid_0's auc: 0.892431	valid_0's binary_logloss: 0.204486
    [2229]	valid_0's auc: 0.892458	valid_0's binary_logloss: 0.204378
    [2230]	valid_0's auc: 0.892464	valid_0's binary_logloss: 0.204461
    [2231]	valid_0's auc: 0.892447	valid_0's binary_logloss: 0.204376
    [2232]	valid_0's auc: 0.89245	valid_0's binary_logloss: 0.204457
    [2233]	valid_0's auc: 0.892458	valid_0's binary_logloss: 0.204384
    [2234]	valid_0's auc: 0.892466	valid_0's binary_logloss: 0.204431
    [2235]	valid_0's auc: 0.892469	valid_0's binary_logloss: 0.204474
    [2236]	valid_0's auc: 0.892469	valid_0's binary_logloss: 0.204537
    [2237]	valid_0's auc: 0.892438	valid_0's binary_logloss: 0.204448
    [2238]	valid_0's auc: 0.892501	valid_0's binary_logloss: 0.204349
    [2239]	valid_0's auc: 0.892487	valid_0's binary_logloss: 0.204262
    [2240]	valid_0's auc: 0.892486	valid_0's binary_logloss: 0.204316
    [2241]	valid_0's auc: 0.892419	valid_0's binary_logloss: 0.204225
    [2242]	valid_0's auc: 0.892446	valid_0's binary_logloss: 0.204079
    [2243]	valid_0's auc: 0.892452	valid_0's binary_logloss: 0.204128
    [2244]	valid_0's auc: 0.892419	valid_0's binary_logloss: 0.204034
    [2245]	valid_0's auc: 0.892389	valid_0's binary_logloss: 0.203968
    [2246]	valid_0's auc: 0.89239	valid_0's binary_logloss: 0.203917
    [2247]	valid_0's auc: 0.892388	valid_0's binary_logloss: 0.204007
    [2248]	valid_0's auc: 0.892403	valid_0's binary_logloss: 0.204062
    [2249]	valid_0's auc: 0.892555	valid_0's binary_logloss: 0.203752
    [2250]	valid_0's auc: 0.892548	valid_0's binary_logloss: 0.203642
    [2251]	valid_0's auc: 0.892665	valid_0's binary_logloss: 0.203345
    [2252]	valid_0's auc: 0.89274	valid_0's binary_logloss: 0.203188
    [2253]	valid_0's auc: 0.892736	valid_0's binary_logloss: 0.203069
    [2254]	valid_0's auc: 0.892705	valid_0's binary_logloss: 0.20297
    [2255]	valid_0's auc: 0.892826	valid_0's binary_logloss: 0.202751
    [2256]	valid_0's auc: 0.89282	valid_0's binary_logloss: 0.202667
    [2257]	valid_0's auc: 0.892822	valid_0's binary_logloss: 0.202559
    [2258]	valid_0's auc: 0.892833	valid_0's binary_logloss: 0.202488
    [2259]	valid_0's auc: 0.892842	valid_0's binary_logloss: 0.202394
    [2260]	valid_0's auc: 0.892839	valid_0's binary_logloss: 0.202444
    [2261]	valid_0's auc: 0.892842	valid_0's binary_logloss: 0.202357
    [2262]	valid_0's auc: 0.892796	valid_0's binary_logloss: 0.202248
    [2263]	valid_0's auc: 0.892798	valid_0's binary_logloss: 0.202321
    [2264]	valid_0's auc: 0.892836	valid_0's binary_logloss: 0.202227
    [2265]	valid_0's auc: 0.892842	valid_0's binary_logloss: 0.202273
    [2266]	valid_0's auc: 0.892845	valid_0's binary_logloss: 0.202328
    [2267]	valid_0's auc: 0.892851	valid_0's binary_logloss: 0.202379
    [2268]	valid_0's auc: 0.892856	valid_0's binary_logloss: 0.202439
    [2269]	valid_0's auc: 0.892863	valid_0's binary_logloss: 0.202515
    [2270]	valid_0's auc: 0.892879	valid_0's binary_logloss: 0.20242
    [2271]	valid_0's auc: 0.892886	valid_0's binary_logloss: 0.202468
    [2272]	valid_0's auc: 0.892896	valid_0's binary_logloss: 0.202521
    [2273]	valid_0's auc: 0.892897	valid_0's binary_logloss: 0.202593
    [2274]	valid_0's auc: 0.892894	valid_0's binary_logloss: 0.202518
    [2275]	valid_0's auc: 0.89294	valid_0's binary_logloss: 0.20236
    [2276]	valid_0's auc: 0.89294	valid_0's binary_logloss: 0.202412
    [2277]	valid_0's auc: 0.892948	valid_0's binary_logloss: 0.202482
    [2278]	valid_0's auc: 0.892952	valid_0's binary_logloss: 0.202539
    [2279]	valid_0's auc: 0.892909	valid_0's binary_logloss: 0.202445
    [2280]	valid_0's auc: 0.893064	valid_0's binary_logloss: 0.202225
    [2281]	valid_0's auc: 0.893102	valid_0's binary_logloss: 0.202076
    [2282]	valid_0's auc: 0.893101	valid_0's binary_logloss: 0.202137
    [2283]	valid_0's auc: 0.893107	valid_0's binary_logloss: 0.202208
    [2284]	valid_0's auc: 0.893094	valid_0's binary_logloss: 0.202118
    [2285]	valid_0's auc: 0.893088	valid_0's binary_logloss: 0.20205
    [2286]	valid_0's auc: 0.893067	valid_0's binary_logloss: 0.201955
    [2287]	valid_0's auc: 0.893121	valid_0's binary_logloss: 0.201845
    [2288]	valid_0's auc: 0.893147	valid_0's binary_logloss: 0.201727
    [2289]	valid_0's auc: 0.893222	valid_0's binary_logloss: 0.20157
    [2290]	valid_0's auc: 0.893204	valid_0's binary_logloss: 0.201514
    [2291]	valid_0's auc: 0.893211	valid_0's binary_logloss: 0.201582
    [2292]	valid_0's auc: 0.89315	valid_0's binary_logloss: 0.201513
    [2293]	valid_0's auc: 0.893132	valid_0's binary_logloss: 0.201459
    [2294]	valid_0's auc: 0.893143	valid_0's binary_logloss: 0.201505
    [2295]	valid_0's auc: 0.893144	valid_0's binary_logloss: 0.201464
    [2296]	valid_0's auc: 0.89315	valid_0's binary_logloss: 0.201511
    [2297]	valid_0's auc: 0.893152	valid_0's binary_logloss: 0.201552
    [2298]	valid_0's auc: 0.893153	valid_0's binary_logloss: 0.201472
    [2299]	valid_0's auc: 0.893177	valid_0's binary_logloss: 0.201395
    [2300]	valid_0's auc: 0.893183	valid_0's binary_logloss: 0.201433
    [2301]	valid_0's auc: 0.893189	valid_0's binary_logloss: 0.201485
    [2302]	valid_0's auc: 0.893196	valid_0's binary_logloss: 0.201525
    [2303]	valid_0's auc: 0.893203	valid_0's binary_logloss: 0.201591
    [2304]	valid_0's auc: 0.893186	valid_0's binary_logloss: 0.201524
    [2305]	valid_0's auc: 0.893162	valid_0's binary_logloss: 0.201462
    [2306]	valid_0's auc: 0.893285	valid_0's binary_logloss: 0.201265
    [2307]	valid_0's auc: 0.893282	valid_0's binary_logloss: 0.201192
    [2308]	valid_0's auc: 0.893289	valid_0's binary_logloss: 0.201247
    [2309]	valid_0's auc: 0.893291	valid_0's binary_logloss: 0.201311
    [2310]	valid_0's auc: 0.893293	valid_0's binary_logloss: 0.201361
    [2311]	valid_0's auc: 0.893278	valid_0's binary_logloss: 0.201273
    [2312]	valid_0's auc: 0.893277	valid_0's binary_logloss: 0.201352
    [2313]	valid_0's auc: 0.89328	valid_0's binary_logloss: 0.201408
    [2314]	valid_0's auc: 0.893283	valid_0's binary_logloss: 0.201485
    [2315]	valid_0's auc: 0.893388	valid_0's binary_logloss: 0.201201
    [2316]	valid_0's auc: 0.893408	valid_0's binary_logloss: 0.201061
    [2317]	valid_0's auc: 0.893513	valid_0's binary_logloss: 0.200904
    [2318]	valid_0's auc: 0.893478	valid_0's binary_logloss: 0.200836
    [2319]	valid_0's auc: 0.893483	valid_0's binary_logloss: 0.200882
    [2320]	valid_0's auc: 0.893477	valid_0's binary_logloss: 0.200799
    [2321]	valid_0's auc: 0.893489	valid_0's binary_logloss: 0.200698
    [2322]	valid_0's auc: 0.893489	valid_0's binary_logloss: 0.20074
    [2323]	valid_0's auc: 0.893574	valid_0's binary_logloss: 0.200595
    [2324]	valid_0's auc: 0.893506	valid_0's binary_logloss: 0.200529
    [2325]	valid_0's auc: 0.893512	valid_0's binary_logloss: 0.200574
    [2326]	valid_0's auc: 0.893481	valid_0's binary_logloss: 0.200523
    [2327]	valid_0's auc: 0.893474	valid_0's binary_logloss: 0.200358
    [2328]	valid_0's auc: 0.893481	valid_0's binary_logloss: 0.200417
    [2329]	valid_0's auc: 0.893529	valid_0's binary_logloss: 0.200276
    [2330]	valid_0's auc: 0.893534	valid_0's binary_logloss: 0.200339
    [2331]	valid_0's auc: 0.893514	valid_0's binary_logloss: 0.200261
    [2332]	valid_0's auc: 0.893522	valid_0's binary_logloss: 0.200303
    [2333]	valid_0's auc: 0.893661	valid_0's binary_logloss: 0.200097
    [2334]	valid_0's auc: 0.893664	valid_0's binary_logloss: 0.200177
    [2335]	valid_0's auc: 0.89367	valid_0's binary_logloss: 0.200101
    [2336]	valid_0's auc: 0.893703	valid_0's binary_logloss: 0.199968
    [2337]	valid_0's auc: 0.893701	valid_0's binary_logloss: 0.199794
    [2338]	valid_0's auc: 0.893712	valid_0's binary_logloss: 0.199845
    [2339]	valid_0's auc: 0.893717	valid_0's binary_logloss: 0.199891
    [2340]	valid_0's auc: 0.893721	valid_0's binary_logloss: 0.199944
    [2341]	valid_0's auc: 0.893694	valid_0's binary_logloss: 0.199872
    [2342]	valid_0's auc: 0.893696	valid_0's binary_logloss: 0.199928
    [2343]	valid_0's auc: 0.893708	valid_0's binary_logloss: 0.19982
    [2344]	valid_0's auc: 0.893654	valid_0's binary_logloss: 0.199741
    [2345]	valid_0's auc: 0.89365	valid_0's binary_logloss: 0.199652
    [2346]	valid_0's auc: 0.893663	valid_0's binary_logloss: 0.199716
    [2347]	valid_0's auc: 0.893689	valid_0's binary_logloss: 0.199634
    [2348]	valid_0's auc: 0.893689	valid_0's binary_logloss: 0.199522
    [2349]	valid_0's auc: 0.89369	valid_0's binary_logloss: 0.199577
    [2350]	valid_0's auc: 0.893696	valid_0's binary_logloss: 0.199632
    [2351]	valid_0's auc: 0.893816	valid_0's binary_logloss: 0.199458
    [2352]	valid_0's auc: 0.893787	valid_0's binary_logloss: 0.199379
    [2353]	valid_0's auc: 0.893801	valid_0's binary_logloss: 0.199327
    [2354]	valid_0's auc: 0.893808	valid_0's binary_logloss: 0.199357
    [2355]	valid_0's auc: 0.893906	valid_0's binary_logloss: 0.199163
    [2356]	valid_0's auc: 0.893894	valid_0's binary_logloss: 0.199097
    [2357]	valid_0's auc: 0.893897	valid_0's binary_logloss: 0.199151
    [2358]	valid_0's auc: 0.893925	valid_0's binary_logloss: 0.199055
    [2359]	valid_0's auc: 0.89404	valid_0's binary_logloss: 0.198928
    [2360]	valid_0's auc: 0.894026	valid_0's binary_logloss: 0.198842
    [2361]	valid_0's auc: 0.894013	valid_0's binary_logloss: 0.19876
    [2362]	valid_0's auc: 0.894017	valid_0's binary_logloss: 0.198818
    [2363]	valid_0's auc: 0.894016	valid_0's binary_logloss: 0.198869
    [2364]	valid_0's auc: 0.894023	valid_0's binary_logloss: 0.198921
    [2365]	valid_0's auc: 0.894065	valid_0's binary_logloss: 0.198832
    [2366]	valid_0's auc: 0.89408	valid_0's binary_logloss: 0.198728
    [2367]	valid_0's auc: 0.894086	valid_0's binary_logloss: 0.19866
    [2368]	valid_0's auc: 0.894092	valid_0's binary_logloss: 0.198729
    [2369]	valid_0's auc: 0.894099	valid_0's binary_logloss: 0.198782
    [2370]	valid_0's auc: 0.894132	valid_0's binary_logloss: 0.198696
    [2371]	valid_0's auc: 0.894139	valid_0's binary_logloss: 0.198736
    [2372]	valid_0's auc: 0.894122	valid_0's binary_logloss: 0.198669
    [2373]	valid_0's auc: 0.894193	valid_0's binary_logloss: 0.198504
    [2374]	valid_0's auc: 0.89422	valid_0's binary_logloss: 0.198434
    [2375]	valid_0's auc: 0.894206	valid_0's binary_logloss: 0.19837
    [2376]	valid_0's auc: 0.894206	valid_0's binary_logloss: 0.198413
    [2377]	valid_0's auc: 0.894214	valid_0's binary_logloss: 0.19845
    [2378]	valid_0's auc: 0.894237	valid_0's binary_logloss: 0.198378
    [2379]	valid_0's auc: 0.894241	valid_0's binary_logloss: 0.198437
    [2380]	valid_0's auc: 0.894226	valid_0's binary_logloss: 0.198338
    [2381]	valid_0's auc: 0.894208	valid_0's binary_logloss: 0.198193
    [2382]	valid_0's auc: 0.894208	valid_0's binary_logloss: 0.198235
    [2383]	valid_0's auc: 0.894225	valid_0's binary_logloss: 0.198169
    [2384]	valid_0's auc: 0.894232	valid_0's binary_logloss: 0.19822
    [2385]	valid_0's auc: 0.894225	valid_0's binary_logloss: 0.198273
    [2386]	valid_0's auc: 0.894261	valid_0's binary_logloss: 0.198219
    [2387]	valid_0's auc: 0.89426	valid_0's binary_logloss: 0.198157
    [2388]	valid_0's auc: 0.894264	valid_0's binary_logloss: 0.198214
    [2389]	valid_0's auc: 0.894196	valid_0's binary_logloss: 0.198155
    [2390]	valid_0's auc: 0.89429	valid_0's binary_logloss: 0.197973
    [2391]	valid_0's auc: 0.894342	valid_0's binary_logloss: 0.197846
    [2392]	valid_0's auc: 0.894469	valid_0's binary_logloss: 0.197575
    [2393]	valid_0's auc: 0.894509	valid_0's binary_logloss: 0.197434
    [2394]	valid_0's auc: 0.894515	valid_0's binary_logloss: 0.197483
    [2395]	valid_0's auc: 0.89448	valid_0's binary_logloss: 0.197415
    [2396]	valid_0's auc: 0.894484	valid_0's binary_logloss: 0.197475
    [2397]	valid_0's auc: 0.89452	valid_0's binary_logloss: 0.197394
    [2398]	valid_0's auc: 0.894527	valid_0's binary_logloss: 0.19745
    [2399]	valid_0's auc: 0.894535	valid_0's binary_logloss: 0.197497
    [2400]	valid_0's auc: 0.894538	valid_0's binary_logloss: 0.197567
    [2401]	valid_0's auc: 0.894562	valid_0's binary_logloss: 0.197502
    [2402]	valid_0's auc: 0.894566	valid_0's binary_logloss: 0.197564
    [2403]	valid_0's auc: 0.894573	valid_0's binary_logloss: 0.197621
    [2404]	valid_0's auc: 0.894554	valid_0's binary_logloss: 0.197547
    [2405]	valid_0's auc: 0.89456	valid_0's binary_logloss: 0.197585
    [2406]	valid_0's auc: 0.894546	valid_0's binary_logloss: 0.197519
    [2407]	valid_0's auc: 0.894553	valid_0's binary_logloss: 0.197559
    [2408]	valid_0's auc: 0.894541	valid_0's binary_logloss: 0.197511
    [2409]	valid_0's auc: 0.894556	valid_0's binary_logloss: 0.197554
    [2410]	valid_0's auc: 0.894576	valid_0's binary_logloss: 0.197486
    [2411]	valid_0's auc: 0.894577	valid_0's binary_logloss: 0.197538
    [2412]	valid_0's auc: 0.894586	valid_0's binary_logloss: 0.19757
    [2413]	valid_0's auc: 0.89459	valid_0's binary_logloss: 0.197618
    [2414]	valid_0's auc: 0.894586	valid_0's binary_logloss: 0.197576
    [2415]	valid_0's auc: 0.894664	valid_0's binary_logloss: 0.197496
    [2416]	valid_0's auc: 0.894648	valid_0's binary_logloss: 0.197419
    [2417]	valid_0's auc: 0.894653	valid_0's binary_logloss: 0.197475
    [2418]	valid_0's auc: 0.89463	valid_0's binary_logloss: 0.197407
    [2419]	valid_0's auc: 0.894632	valid_0's binary_logloss: 0.197309
    [2420]	valid_0's auc: 0.894637	valid_0's binary_logloss: 0.197355
    [2421]	valid_0's auc: 0.894679	valid_0's binary_logloss: 0.197253
    [2422]	valid_0's auc: 0.894686	valid_0's binary_logloss: 0.197292
    [2423]	valid_0's auc: 0.89469	valid_0's binary_logloss: 0.197341
    [2424]	valid_0's auc: 0.894691	valid_0's binary_logloss: 0.197402
    [2425]	valid_0's auc: 0.894693	valid_0's binary_logloss: 0.197464
    [2426]	valid_0's auc: 0.894697	valid_0's binary_logloss: 0.197512
    [2427]	valid_0's auc: 0.89472	valid_0's binary_logloss: 0.197437
    [2428]	valid_0's auc: 0.894723	valid_0's binary_logloss: 0.197475
    [2429]	valid_0's auc: 0.894727	valid_0's binary_logloss: 0.197517
    [2430]	valid_0's auc: 0.894723	valid_0's binary_logloss: 0.197469
    [2431]	valid_0's auc: 0.894729	valid_0's binary_logloss: 0.197501
    [2432]	valid_0's auc: 0.894723	valid_0's binary_logloss: 0.197444
    [2433]	valid_0's auc: 0.894799	valid_0's binary_logloss: 0.197294
    [2434]	valid_0's auc: 0.8948	valid_0's binary_logloss: 0.197358
    [2435]	valid_0's auc: 0.894805	valid_0's binary_logloss: 0.19741
    [2436]	valid_0's auc: 0.894788	valid_0's binary_logloss: 0.197341
    [2437]	valid_0's auc: 0.894781	valid_0's binary_logloss: 0.197268
    [2438]	valid_0's auc: 0.894782	valid_0's binary_logloss: 0.197179
    [2439]	valid_0's auc: 0.894788	valid_0's binary_logloss: 0.197236
    [2440]	valid_0's auc: 0.894789	valid_0's binary_logloss: 0.197283
    [2441]	valid_0's auc: 0.894775	valid_0's binary_logloss: 0.197236
    [2442]	valid_0's auc: 0.894773	valid_0's binary_logloss: 0.19729
    [2443]	valid_0's auc: 0.894779	valid_0's binary_logloss: 0.197164
    [2444]	valid_0's auc: 0.894853	valid_0's binary_logloss: 0.196911
    [2445]	valid_0's auc: 0.894858	valid_0's binary_logloss: 0.19696
    [2446]	valid_0's auc: 0.894862	valid_0's binary_logloss: 0.197017
    [2447]	valid_0's auc: 0.894859	valid_0's binary_logloss: 0.196965
    [2448]	valid_0's auc: 0.894864	valid_0's binary_logloss: 0.197027
    [2449]	valid_0's auc: 0.894817	valid_0's binary_logloss: 0.196957
    [2450]	valid_0's auc: 0.894816	valid_0's binary_logloss: 0.197008
    [2451]	valid_0's auc: 0.894939	valid_0's binary_logloss: 0.196767
    [2452]	valid_0's auc: 0.895072	valid_0's binary_logloss: 0.196488
    [2453]	valid_0's auc: 0.895076	valid_0's binary_logloss: 0.196533
    [2454]	valid_0's auc: 0.895114	valid_0's binary_logloss: 0.196462
    [2455]	valid_0's auc: 0.895166	valid_0's binary_logloss: 0.196256
    [2456]	valid_0's auc: 0.895165	valid_0's binary_logloss: 0.196325
    [2457]	valid_0's auc: 0.895172	valid_0's binary_logloss: 0.196377
    [2458]	valid_0's auc: 0.895175	valid_0's binary_logloss: 0.196255
    [2459]	valid_0's auc: 0.895287	valid_0's binary_logloss: 0.195989
    [2460]	valid_0's auc: 0.895283	valid_0's binary_logloss: 0.195919
    [2461]	valid_0's auc: 0.895273	valid_0's binary_logloss: 0.195879
    [2462]	valid_0's auc: 0.895268	valid_0's binary_logloss: 0.19595
    [2463]	valid_0's auc: 0.895268	valid_0's binary_logloss: 0.195995
    [2464]	valid_0's auc: 0.895267	valid_0's binary_logloss: 0.196048
    [2465]	valid_0's auc: 0.895234	valid_0's binary_logloss: 0.195996
    [2466]	valid_0's auc: 0.895229	valid_0's binary_logloss: 0.195919
    [2467]	valid_0's auc: 0.895319	valid_0's binary_logloss: 0.195731
    [2468]	valid_0's auc: 0.895323	valid_0's binary_logloss: 0.195759
    [2469]	valid_0's auc: 0.895327	valid_0's binary_logloss: 0.1958
    [2470]	valid_0's auc: 0.895363	valid_0's binary_logloss: 0.1957
    [2471]	valid_0's auc: 0.895362	valid_0's binary_logloss: 0.195629
    [2472]	valid_0's auc: 0.895363	valid_0's binary_logloss: 0.195677
    [2473]	valid_0's auc: 0.89536	valid_0's binary_logloss: 0.195732
    [2474]	valid_0's auc: 0.895379	valid_0's binary_logloss: 0.195626
    [2475]	valid_0's auc: 0.895373	valid_0's binary_logloss: 0.195566
    [2476]	valid_0's auc: 0.895369	valid_0's binary_logloss: 0.195501
    [2477]	valid_0's auc: 0.895358	valid_0's binary_logloss: 0.195448
    [2478]	valid_0's auc: 0.895359	valid_0's binary_logloss: 0.195519
    [2479]	valid_0's auc: 0.895389	valid_0's binary_logloss: 0.195409
    [2480]	valid_0's auc: 0.895384	valid_0's binary_logloss: 0.195446
    [2481]	valid_0's auc: 0.895387	valid_0's binary_logloss: 0.195484
    [2482]	valid_0's auc: 0.895393	valid_0's binary_logloss: 0.195525
    [2483]	valid_0's auc: 0.895402	valid_0's binary_logloss: 0.195567
    [2484]	valid_0's auc: 0.895425	valid_0's binary_logloss: 0.19549
    [2485]	valid_0's auc: 0.895426	valid_0's binary_logloss: 0.195539
    [2486]	valid_0's auc: 0.895424	valid_0's binary_logloss: 0.195583
    [2487]	valid_0's auc: 0.895429	valid_0's binary_logloss: 0.195618
    [2488]	valid_0's auc: 0.895436	valid_0's binary_logloss: 0.195668
    [2489]	valid_0's auc: 0.89544	valid_0's binary_logloss: 0.195704
    [2490]	valid_0's auc: 0.895436	valid_0's binary_logloss: 0.195647
    [2491]	valid_0's auc: 0.89547	valid_0's binary_logloss: 0.195582
    [2492]	valid_0's auc: 0.895468	valid_0's binary_logloss: 0.195624
    [2493]	valid_0's auc: 0.895473	valid_0's binary_logloss: 0.195556
    [2494]	valid_0's auc: 0.895487	valid_0's binary_logloss: 0.195473
    [2495]	valid_0's auc: 0.895482	valid_0's binary_logloss: 0.195415
    [2496]	valid_0's auc: 0.895484	valid_0's binary_logloss: 0.195462
    [2497]	valid_0's auc: 0.895505	valid_0's binary_logloss: 0.195387
    [2498]	valid_0's auc: 0.895506	valid_0's binary_logloss: 0.195429
    [2499]	valid_0's auc: 0.895508	valid_0's binary_logloss: 0.195376
    [2500]	valid_0's auc: 0.895512	valid_0's binary_logloss: 0.195419
    [2501]	valid_0's auc: 0.895496	valid_0's binary_logloss: 0.195353
    [2502]	valid_0's auc: 0.895494	valid_0's binary_logloss: 0.195415
    [2503]	valid_0's auc: 0.895559	valid_0's binary_logloss: 0.195228
    [2504]	valid_0's auc: 0.895537	valid_0's binary_logloss: 0.195178
    [2505]	valid_0's auc: 0.895567	valid_0's binary_logloss: 0.19508
    [2506]	valid_0's auc: 0.895547	valid_0's binary_logloss: 0.19501
    [2507]	valid_0's auc: 0.895583	valid_0's binary_logloss: 0.194932
    [2508]	valid_0's auc: 0.895585	valid_0's binary_logloss: 0.194863
    [2509]	valid_0's auc: 0.89559	valid_0's binary_logloss: 0.194901
    [2510]	valid_0's auc: 0.895593	valid_0's binary_logloss: 0.194941
    [2511]	valid_0's auc: 0.895628	valid_0's binary_logloss: 0.194864
    [2512]	valid_0's auc: 0.895643	valid_0's binary_logloss: 0.194802
    [2513]	valid_0's auc: 0.89564	valid_0's binary_logloss: 0.194836
    [2514]	valid_0's auc: 0.895647	valid_0's binary_logloss: 0.194898
    [2515]	valid_0's auc: 0.895676	valid_0's binary_logloss: 0.194809
    [2516]	valid_0's auc: 0.895675	valid_0's binary_logloss: 0.19472
    [2517]	valid_0's auc: 0.895678	valid_0's binary_logloss: 0.194592
    [2518]	valid_0's auc: 0.895677	valid_0's binary_logloss: 0.19467
    [2519]	valid_0's auc: 0.895688	valid_0's binary_logloss: 0.194714
    [2520]	valid_0's auc: 0.895694	valid_0's binary_logloss: 0.19476
    [2521]	valid_0's auc: 0.895662	valid_0's binary_logloss: 0.194716
    [2522]	valid_0's auc: 0.895665	valid_0's binary_logloss: 0.194755
    [2523]	valid_0's auc: 0.895638	valid_0's binary_logloss: 0.194684
    [2524]	valid_0's auc: 0.895691	valid_0's binary_logloss: 0.194512
    [2525]	valid_0's auc: 0.89573	valid_0's binary_logloss: 0.194398
    [2526]	valid_0's auc: 0.895732	valid_0's binary_logloss: 0.194432
    [2527]	valid_0's auc: 0.895737	valid_0's binary_logloss: 0.19438
    [2528]	valid_0's auc: 0.895744	valid_0's binary_logloss: 0.194424
    [2529]	valid_0's auc: 0.895743	valid_0's binary_logloss: 0.194475
    [2530]	valid_0's auc: 0.895748	valid_0's binary_logloss: 0.194506
    [2531]	valid_0's auc: 0.895722	valid_0's binary_logloss: 0.194435
    [2532]	valid_0's auc: 0.895725	valid_0's binary_logloss: 0.194465
    [2533]	valid_0's auc: 0.895736	valid_0's binary_logloss: 0.194513
    [2534]	valid_0's auc: 0.895741	valid_0's binary_logloss: 0.194557
    [2535]	valid_0's auc: 0.895775	valid_0's binary_logloss: 0.194478
    [2536]	valid_0's auc: 0.895776	valid_0's binary_logloss: 0.194513
    [2537]	valid_0's auc: 0.895776	valid_0's binary_logloss: 0.194561
    [2538]	valid_0's auc: 0.895815	valid_0's binary_logloss: 0.194487
    [2539]	valid_0's auc: 0.895805	valid_0's binary_logloss: 0.194443
    [2540]	valid_0's auc: 0.895811	valid_0's binary_logloss: 0.194494
    [2541]	valid_0's auc: 0.895831	valid_0's binary_logloss: 0.194318
    [2542]	valid_0's auc: 0.895867	valid_0's binary_logloss: 0.194237
    [2543]	valid_0's auc: 0.895867	valid_0's binary_logloss: 0.1943
    [2544]	valid_0's auc: 0.895865	valid_0's binary_logloss: 0.194351
    [2545]	valid_0's auc: 0.895866	valid_0's binary_logloss: 0.194383
    [2546]	valid_0's auc: 0.895871	valid_0's binary_logloss: 0.194436
    [2547]	valid_0's auc: 0.895871	valid_0's binary_logloss: 0.194468
    [2548]	valid_0's auc: 0.895875	valid_0's binary_logloss: 0.194496
    [2549]	valid_0's auc: 0.89588	valid_0's binary_logloss: 0.194535
    [2550]	valid_0's auc: 0.895967	valid_0's binary_logloss: 0.194295
    [2551]	valid_0's auc: 0.895967	valid_0's binary_logloss: 0.194337
    [2552]	valid_0's auc: 0.896032	valid_0's binary_logloss: 0.194171
    [2553]	valid_0's auc: 0.896005	valid_0's binary_logloss: 0.194124
    [2554]	valid_0's auc: 0.89601	valid_0's binary_logloss: 0.194171
    [2555]	valid_0's auc: 0.896012	valid_0's binary_logloss: 0.194199
    [2556]	valid_0's auc: 0.895971	valid_0's binary_logloss: 0.194147
    [2557]	valid_0's auc: 0.895971	valid_0's binary_logloss: 0.1942
    [2558]	valid_0's auc: 0.895976	valid_0's binary_logloss: 0.194243
    [2559]	valid_0's auc: 0.896022	valid_0's binary_logloss: 0.194012
    [2560]	valid_0's auc: 0.896028	valid_0's binary_logloss: 0.193915
    [2561]	valid_0's auc: 0.896066	valid_0's binary_logloss: 0.193825
    [2562]	valid_0's auc: 0.896074	valid_0's binary_logloss: 0.193868
    [2563]	valid_0's auc: 0.896077	valid_0's binary_logloss: 0.193922
    [2564]	valid_0's auc: 0.896078	valid_0's binary_logloss: 0.193978
    [2565]	valid_0's auc: 0.896078	valid_0's binary_logloss: 0.194029
    [2566]	valid_0's auc: 0.896084	valid_0's binary_logloss: 0.19407
    [2567]	valid_0's auc: 0.896092	valid_0's binary_logloss: 0.194005
    [2568]	valid_0's auc: 0.896133	valid_0's binary_logloss: 0.193777
    [2569]	valid_0's auc: 0.896134	valid_0's binary_logloss: 0.19383
    [2570]	valid_0's auc: 0.896147	valid_0's binary_logloss: 0.193695
    [2571]	valid_0's auc: 0.896124	valid_0's binary_logloss: 0.193631
    [2572]	valid_0's auc: 0.896124	valid_0's binary_logloss: 0.193698
    [2573]	valid_0's auc: 0.896124	valid_0's binary_logloss: 0.19363
    [2574]	valid_0's auc: 0.896125	valid_0's binary_logloss: 0.193688
    [2575]	valid_0's auc: 0.896111	valid_0's binary_logloss: 0.193626
    [2576]	valid_0's auc: 0.896106	valid_0's binary_logloss: 0.193559
    [2577]	valid_0's auc: 0.896112	valid_0's binary_logloss: 0.193611
    [2578]	valid_0's auc: 0.896162	valid_0's binary_logloss: 0.193463
    [2579]	valid_0's auc: 0.896164	valid_0's binary_logloss: 0.193502
    [2580]	valid_0's auc: 0.896169	valid_0's binary_logloss: 0.19354
    [2581]	valid_0's auc: 0.896192	valid_0's binary_logloss: 0.1934
    [2582]	valid_0's auc: 0.896246	valid_0's binary_logloss: 0.193281
    [2583]	valid_0's auc: 0.896251	valid_0's binary_logloss: 0.193308
    [2584]	valid_0's auc: 0.896254	valid_0's binary_logloss: 0.193366
    [2585]	valid_0's auc: 0.896302	valid_0's binary_logloss: 0.193299
    [2586]	valid_0's auc: 0.896302	valid_0's binary_logloss: 0.193243
    [2587]	valid_0's auc: 0.896317	valid_0's binary_logloss: 0.193178
    [2588]	valid_0's auc: 0.896318	valid_0's binary_logloss: 0.193211
    [2589]	valid_0's auc: 0.896321	valid_0's binary_logloss: 0.19325
    [2590]	valid_0's auc: 0.89632	valid_0's binary_logloss: 0.193284
    [2591]	valid_0's auc: 0.896326	valid_0's binary_logloss: 0.193236
    [2592]	valid_0's auc: 0.896287	valid_0's binary_logloss: 0.193204
    [2593]	valid_0's auc: 0.896295	valid_0's binary_logloss: 0.193258
    [2594]	valid_0's auc: 0.896239	valid_0's binary_logloss: 0.19321
    [2595]	valid_0's auc: 0.896242	valid_0's binary_logloss: 0.193251
    [2596]	valid_0's auc: 0.896248	valid_0's binary_logloss: 0.193287
    [2597]	valid_0's auc: 0.896273	valid_0's binary_logloss: 0.193205
    [2598]	valid_0's auc: 0.89627	valid_0's binary_logloss: 0.193265
    [2599]	valid_0's auc: 0.896292	valid_0's binary_logloss: 0.193203
    [2600]	valid_0's auc: 0.89629	valid_0's binary_logloss: 0.193131
    [2601]	valid_0's auc: 0.896319	valid_0's binary_logloss: 0.193023
    [2602]	valid_0's auc: 0.896326	valid_0's binary_logloss: 0.193048
    [2603]	valid_0's auc: 0.896327	valid_0's binary_logloss: 0.193097
    [2604]	valid_0's auc: 0.896338	valid_0's binary_logloss: 0.193045
    [2605]	valid_0's auc: 0.89634	valid_0's binary_logloss: 0.193092
    [2606]	valid_0's auc: 0.896337	valid_0's binary_logloss: 0.193149
    [2607]	valid_0's auc: 0.896345	valid_0's binary_logloss: 0.193197
    [2608]	valid_0's auc: 0.896408	valid_0's binary_logloss: 0.193088
    [2609]	valid_0's auc: 0.896475	valid_0's binary_logloss: 0.193004
    [2610]	valid_0's auc: 0.896506	valid_0's binary_logloss: 0.192891
    [2611]	valid_0's auc: 0.896508	valid_0's binary_logloss: 0.192927
    [2612]	valid_0's auc: 0.896511	valid_0's binary_logloss: 0.192973
    [2613]	valid_0's auc: 0.896535	valid_0's binary_logloss: 0.192872
    [2614]	valid_0's auc: 0.896517	valid_0's binary_logloss: 0.192808
    [2615]	valid_0's auc: 0.896519	valid_0's binary_logloss: 0.192862
    [2616]	valid_0's auc: 0.896519	valid_0's binary_logloss: 0.192909
    [2617]	valid_0's auc: 0.89652	valid_0's binary_logloss: 0.192941
    [2618]	valid_0's auc: 0.896512	valid_0's binary_logloss: 0.192881
    [2619]	valid_0's auc: 0.896519	valid_0's binary_logloss: 0.192817
    [2620]	valid_0's auc: 0.89652	valid_0's binary_logloss: 0.192851
    [2621]	valid_0's auc: 0.896566	valid_0's binary_logloss: 0.192709
    [2622]	valid_0's auc: 0.89657	valid_0's binary_logloss: 0.192743
    [2623]	valid_0's auc: 0.896555	valid_0's binary_logloss: 0.192696
    [2624]	valid_0's auc: 0.896558	valid_0's binary_logloss: 0.192734
    [2625]	valid_0's auc: 0.896555	valid_0's binary_logloss: 0.192662
    [2626]	valid_0's auc: 0.896559	valid_0's binary_logloss: 0.192621
    [2627]	valid_0's auc: 0.896575	valid_0's binary_logloss: 0.192519
    [2628]	valid_0's auc: 0.896578	valid_0's binary_logloss: 0.19256
    [2629]	valid_0's auc: 0.89665	valid_0's binary_logloss: 0.192355
    [2630]	valid_0's auc: 0.896655	valid_0's binary_logloss: 0.192413
    [2631]	valid_0's auc: 0.896657	valid_0's binary_logloss: 0.192451
    [2632]	valid_0's auc: 0.896661	valid_0's binary_logloss: 0.192496
    [2633]	valid_0's auc: 0.896653	valid_0's binary_logloss: 0.192421
    [2634]	valid_0's auc: 0.896655	valid_0's binary_logloss: 0.192473
    [2635]	valid_0's auc: 0.896662	valid_0's binary_logloss: 0.192499
    [2636]	valid_0's auc: 0.89675	valid_0's binary_logloss: 0.192276
    [2637]	valid_0's auc: 0.896805	valid_0's binary_logloss: 0.192137
    [2638]	valid_0's auc: 0.896871	valid_0's binary_logloss: 0.191993
    [2639]	valid_0's auc: 0.896871	valid_0's binary_logloss: 0.19204
    [2640]	valid_0's auc: 0.896883	valid_0's binary_logloss: 0.19198
    [2641]	valid_0's auc: 0.896882	valid_0's binary_logloss: 0.192028
    [2642]	valid_0's auc: 0.896892	valid_0's binary_logloss: 0.191969
    [2643]	valid_0's auc: 0.89689	valid_0's binary_logloss: 0.191904
    [2644]	valid_0's auc: 0.896971	valid_0's binary_logloss: 0.191714
    [2645]	valid_0's auc: 0.896975	valid_0's binary_logloss: 0.19174
    [2646]	valid_0's auc: 0.896977	valid_0's binary_logloss: 0.191763
    [2647]	valid_0's auc: 0.896974	valid_0's binary_logloss: 0.191801
    [2648]	valid_0's auc: 0.896977	valid_0's binary_logloss: 0.191852
    [2649]	valid_0's auc: 0.896982	valid_0's binary_logloss: 0.191892
    [2650]	valid_0's auc: 0.896983	valid_0's binary_logloss: 0.191924
    [2651]	valid_0's auc: 0.896986	valid_0's binary_logloss: 0.191961
    [2652]	valid_0's auc: 0.896992	valid_0's binary_logloss: 0.191997
    [2653]	valid_0's auc: 0.897014	valid_0's binary_logloss: 0.191926
    [2654]	valid_0's auc: 0.89701	valid_0's binary_logloss: 0.191864
    [2655]	valid_0's auc: 0.89701	valid_0's binary_logloss: 0.191902
    [2656]	valid_0's auc: 0.897014	valid_0's binary_logloss: 0.191788
    [2657]	valid_0's auc: 0.897017	valid_0's binary_logloss: 0.191833
    [2658]	valid_0's auc: 0.897025	valid_0's binary_logloss: 0.191766
    [2659]	valid_0's auc: 0.897115	valid_0's binary_logloss: 0.191597
    [2660]	valid_0's auc: 0.897154	valid_0's binary_logloss: 0.191401
    [2661]	valid_0's auc: 0.897161	valid_0's binary_logloss: 0.191438
    [2662]	valid_0's auc: 0.897165	valid_0's binary_logloss: 0.191474
    [2663]	valid_0's auc: 0.897177	valid_0's binary_logloss: 0.191402
    [2664]	valid_0's auc: 0.89723	valid_0's binary_logloss: 0.191304
    [2665]	valid_0's auc: 0.897232	valid_0's binary_logloss: 0.191347
    [2666]	valid_0's auc: 0.897238	valid_0's binary_logloss: 0.191388
    [2667]	valid_0's auc: 0.897241	valid_0's binary_logloss: 0.191423
    [2668]	valid_0's auc: 0.897245	valid_0's binary_logloss: 0.191463
    [2669]	valid_0's auc: 0.897243	valid_0's binary_logloss: 0.191535
    [2670]	valid_0's auc: 0.897246	valid_0's binary_logloss: 0.191561
    [2671]	valid_0's auc: 0.897246	valid_0's binary_logloss: 0.191611
    [2672]	valid_0's auc: 0.897172	valid_0's binary_logloss: 0.191574
    [2673]	valid_0's auc: 0.897166	valid_0's binary_logloss: 0.19151
    [2674]	valid_0's auc: 0.897172	valid_0's binary_logloss: 0.191541
    [2675]	valid_0's auc: 0.897219	valid_0's binary_logloss: 0.19145
    [2676]	valid_0's auc: 0.897233	valid_0's binary_logloss: 0.191346
    [2677]	valid_0's auc: 0.89728	valid_0's binary_logloss: 0.191232
    [2678]	valid_0's auc: 0.897292	valid_0's binary_logloss: 0.191137
    [2679]	valid_0's auc: 0.897298	valid_0's binary_logloss: 0.191179
    [2680]	valid_0's auc: 0.8973	valid_0's binary_logloss: 0.19122
    [2681]	valid_0's auc: 0.897299	valid_0's binary_logloss: 0.191255
    [2682]	valid_0's auc: 0.897287	valid_0's binary_logloss: 0.191206
    [2683]	valid_0's auc: 0.897294	valid_0's binary_logloss: 0.19125
    [2684]	valid_0's auc: 0.897293	valid_0's binary_logloss: 0.191277
    [2685]	valid_0's auc: 0.897295	valid_0's binary_logloss: 0.1912
    [2686]	valid_0's auc: 0.897304	valid_0's binary_logloss: 0.191119
    [2687]	valid_0's auc: 0.897306	valid_0's binary_logloss: 0.191171
    [2688]	valid_0's auc: 0.897351	valid_0's binary_logloss: 0.191028
    [2689]	valid_0's auc: 0.897336	valid_0's binary_logloss: 0.190967
    [2690]	valid_0's auc: 0.897329	valid_0's binary_logloss: 0.190933
    [2691]	valid_0's auc: 0.897339	valid_0's binary_logloss: 0.19087
    [2692]	valid_0's auc: 0.897371	valid_0's binary_logloss: 0.19079
    [2693]	valid_0's auc: 0.897387	valid_0's binary_logloss: 0.190696
    [2694]	valid_0's auc: 0.897445	valid_0's binary_logloss: 0.190562
    [2695]	valid_0's auc: 0.897448	valid_0's binary_logloss: 0.19051
    [2696]	valid_0's auc: 0.89745	valid_0's binary_logloss: 0.190541
    [2697]	valid_0's auc: 0.89753	valid_0's binary_logloss: 0.190465
    [2698]	valid_0's auc: 0.897532	valid_0's binary_logloss: 0.190506
    [2699]	valid_0's auc: 0.897527	valid_0's binary_logloss: 0.190551
    [2700]	valid_0's auc: 0.897554	valid_0's binary_logloss: 0.190487
    [2701]	valid_0's auc: 0.897555	valid_0's binary_logloss: 0.190435
    [2702]	valid_0's auc: 0.897592	valid_0's binary_logloss: 0.190355
    [2703]	valid_0's auc: 0.897592	valid_0's binary_logloss: 0.190396
    [2704]	valid_0's auc: 0.897594	valid_0's binary_logloss: 0.190428
    [2705]	valid_0's auc: 0.897618	valid_0's binary_logloss: 0.190382
    [2706]	valid_0's auc: 0.897625	valid_0's binary_logloss: 0.190341
    [2707]	valid_0's auc: 0.897632	valid_0's binary_logloss: 0.190399
    [2708]	valid_0's auc: 0.897672	valid_0's binary_logloss: 0.190341
    [2709]	valid_0's auc: 0.897671	valid_0's binary_logloss: 0.190364
    [2710]	valid_0's auc: 0.897674	valid_0's binary_logloss: 0.190395
    [2711]	valid_0's auc: 0.89768	valid_0's binary_logloss: 0.190429
    [2712]	valid_0's auc: 0.89768	valid_0's binary_logloss: 0.190459
    [2713]	valid_0's auc: 0.897683	valid_0's binary_logloss: 0.1905
    [2714]	valid_0's auc: 0.897735	valid_0's binary_logloss: 0.19037
    [2715]	valid_0's auc: 0.897735	valid_0's binary_logloss: 0.190405
    [2716]	valid_0's auc: 0.897769	valid_0's binary_logloss: 0.190342
    [2717]	valid_0's auc: 0.897761	valid_0's binary_logloss: 0.190282
    [2718]	valid_0's auc: 0.897769	valid_0's binary_logloss: 0.190324
    [2719]	valid_0's auc: 0.897772	valid_0's binary_logloss: 0.190357
    [2720]	valid_0's auc: 0.897772	valid_0's binary_logloss: 0.190388
    [2721]	valid_0's auc: 0.897771	valid_0's binary_logloss: 0.190422
    [2722]	valid_0's auc: 0.897768	valid_0's binary_logloss: 0.190384
    [2723]	valid_0's auc: 0.89777	valid_0's binary_logloss: 0.190423
    [2724]	valid_0's auc: 0.897771	valid_0's binary_logloss: 0.190452
    [2725]	valid_0's auc: 0.897773	valid_0's binary_logloss: 0.1905
    [2726]	valid_0's auc: 0.897775	valid_0's binary_logloss: 0.190531
    [2727]	valid_0's auc: 0.897777	valid_0's binary_logloss: 0.190565
    [2728]	valid_0's auc: 0.897777	valid_0's binary_logloss: 0.190605
    [2729]	valid_0's auc: 0.897779	valid_0's binary_logloss: 0.190539
    [2730]	valid_0's auc: 0.897779	valid_0's binary_logloss: 0.190578
    [2731]	valid_0's auc: 0.897777	valid_0's binary_logloss: 0.190536
    [2732]	valid_0's auc: 0.897791	valid_0's binary_logloss: 0.19048
    [2733]	valid_0's auc: 0.897795	valid_0's binary_logloss: 0.190517
    [2734]	valid_0's auc: 0.897799	valid_0's binary_logloss: 0.190547
    [2735]	valid_0's auc: 0.897817	valid_0's binary_logloss: 0.190422
    [2736]	valid_0's auc: 0.897817	valid_0's binary_logloss: 0.190455
    [2737]	valid_0's auc: 0.897817	valid_0's binary_logloss: 0.190492
    [2738]	valid_0's auc: 0.897822	valid_0's binary_logloss: 0.190523
    [2739]	valid_0's auc: 0.897826	valid_0's binary_logloss: 0.190575
    [2740]	valid_0's auc: 0.897828	valid_0's binary_logloss: 0.190613
    [2741]	valid_0's auc: 0.89783	valid_0's binary_logloss: 0.190637
    [2742]	valid_0's auc: 0.897827	valid_0's binary_logloss: 0.190565
    [2743]	valid_0's auc: 0.897811	valid_0's binary_logloss: 0.190504
    [2744]	valid_0's auc: 0.897813	valid_0's binary_logloss: 0.190532
    [2745]	valid_0's auc: 0.89784	valid_0's binary_logloss: 0.190472
    [2746]	valid_0's auc: 0.897831	valid_0's binary_logloss: 0.190414
    [2747]	valid_0's auc: 0.897801	valid_0's binary_logloss: 0.190367
    [2748]	valid_0's auc: 0.897798	valid_0's binary_logloss: 0.190404
    [2749]	valid_0's auc: 0.8978	valid_0's binary_logloss: 0.190436
    [2750]	valid_0's auc: 0.897799	valid_0's binary_logloss: 0.190477
    [2751]	valid_0's auc: 0.897804	valid_0's binary_logloss: 0.190506
    [2752]	valid_0's auc: 0.897765	valid_0's binary_logloss: 0.190455
    [2753]	valid_0's auc: 0.897765	valid_0's binary_logloss: 0.190511
    [2754]	valid_0's auc: 0.897771	valid_0's binary_logloss: 0.190546
    [2755]	valid_0's auc: 0.897774	valid_0's binary_logloss: 0.190589
    [2756]	valid_0's auc: 0.897776	valid_0's binary_logloss: 0.190626
    [2757]	valid_0's auc: 0.897778	valid_0's binary_logloss: 0.190656
    [2758]	valid_0's auc: 0.897778	valid_0's binary_logloss: 0.19069
    [2759]	valid_0's auc: 0.89778	valid_0's binary_logloss: 0.190742
    [2760]	valid_0's auc: 0.89774	valid_0's binary_logloss: 0.190706
    [2761]	valid_0's auc: 0.89774	valid_0's binary_logloss: 0.190743
    [2762]	valid_0's auc: 0.897743	valid_0's binary_logloss: 0.190778
    [2763]	valid_0's auc: 0.897733	valid_0's binary_logloss: 0.190715
    [2764]	valid_0's auc: 0.897735	valid_0's binary_logloss: 0.190754
    [2765]	valid_0's auc: 0.89774	valid_0's binary_logloss: 0.190785
    [2766]	valid_0's auc: 0.897748	valid_0's binary_logloss: 0.190822
    [2767]	valid_0's auc: 0.897755	valid_0's binary_logloss: 0.190859
    [2768]	valid_0's auc: 0.897763	valid_0's binary_logloss: 0.190901
    [2769]	valid_0's auc: 0.897758	valid_0's binary_logloss: 0.190826
    [2770]	valid_0's auc: 0.89779	valid_0's binary_logloss: 0.190771
    [2771]	valid_0's auc: 0.897797	valid_0's binary_logloss: 0.190669
    [2772]	valid_0's auc: 0.897802	valid_0's binary_logloss: 0.190697
    [2773]	valid_0's auc: 0.897832	valid_0's binary_logloss: 0.190507
    [2774]	valid_0's auc: 0.897839	valid_0's binary_logloss: 0.19054
    [2775]	valid_0's auc: 0.897844	valid_0's binary_logloss: 0.190487
    [2776]	valid_0's auc: 0.897846	valid_0's binary_logloss: 0.190524
    [2777]	valid_0's auc: 0.897836	valid_0's binary_logloss: 0.190477
    [2778]	valid_0's auc: 0.897847	valid_0's binary_logloss: 0.190421
    [2779]	valid_0's auc: 0.897853	valid_0's binary_logloss: 0.190309
    [2780]	valid_0's auc: 0.897856	valid_0's binary_logloss: 0.19034
    [2781]	valid_0's auc: 0.897891	valid_0's binary_logloss: 0.190279
    [2782]	valid_0's auc: 0.897891	valid_0's binary_logloss: 0.190314
    [2783]	valid_0's auc: 0.897897	valid_0's binary_logloss: 0.190351
    [2784]	valid_0's auc: 0.897901	valid_0's binary_logloss: 0.19039
    [2785]	valid_0's auc: 0.897902	valid_0's binary_logloss: 0.190426
    [2786]	valid_0's auc: 0.897901	valid_0's binary_logloss: 0.190463
    [2787]	valid_0's auc: 0.897896	valid_0's binary_logloss: 0.190411
    [2788]	valid_0's auc: 0.897881	valid_0's binary_logloss: 0.190363
    [2789]	valid_0's auc: 0.897883	valid_0's binary_logloss: 0.190415
    [2790]	valid_0's auc: 0.897883	valid_0's binary_logloss: 0.190452
    [2791]	valid_0's auc: 0.897889	valid_0's binary_logloss: 0.190368
    [2792]	valid_0's auc: 0.897876	valid_0's binary_logloss: 0.190329
    [2793]	valid_0's auc: 0.897886	valid_0's binary_logloss: 0.190295
    [2794]	valid_0's auc: 0.897887	valid_0's binary_logloss: 0.190328
    [2795]	valid_0's auc: 0.897875	valid_0's binary_logloss: 0.190283
    [2796]	valid_0's auc: 0.897882	valid_0's binary_logloss: 0.190329
    [2797]	valid_0's auc: 0.897881	valid_0's binary_logloss: 0.190272
    [2798]	valid_0's auc: 0.897883	valid_0's binary_logloss: 0.190319
    [2799]	valid_0's auc: 0.897925	valid_0's binary_logloss: 0.190231
    [2800]	valid_0's auc: 0.897929	valid_0's binary_logloss: 0.190255
    [2801]	valid_0's auc: 0.897917	valid_0's binary_logloss: 0.190218
    [2802]	valid_0's auc: 0.897921	valid_0's binary_logloss: 0.190255
    [2803]	valid_0's auc: 0.897921	valid_0's binary_logloss: 0.190306
    [2804]	valid_0's auc: 0.897922	valid_0's binary_logloss: 0.190345
    [2805]	valid_0's auc: 0.897937	valid_0's binary_logloss: 0.190272
    [2806]	valid_0's auc: 0.898013	valid_0's binary_logloss: 0.190128
    [2807]	valid_0's auc: 0.898016	valid_0's binary_logloss: 0.190165
    [2808]	valid_0's auc: 0.898018	valid_0's binary_logloss: 0.190215
    [2809]	valid_0's auc: 0.898103	valid_0's binary_logloss: 0.189987
    [2810]	valid_0's auc: 0.898105	valid_0's binary_logloss: 0.190026
    [2811]	valid_0's auc: 0.898108	valid_0's binary_logloss: 0.190052
    [2812]	valid_0's auc: 0.898109	valid_0's binary_logloss: 0.190082
    [2813]	valid_0's auc: 0.898121	valid_0's binary_logloss: 0.189998
    [2814]	valid_0's auc: 0.898123	valid_0's binary_logloss: 0.190032
    [2815]	valid_0's auc: 0.898161	valid_0's binary_logloss: 0.189966
    [2816]	valid_0's auc: 0.898215	valid_0's binary_logloss: 0.189864
    [2817]	valid_0's auc: 0.89822	valid_0's binary_logloss: 0.189912
    [2818]	valid_0's auc: 0.89822	valid_0's binary_logloss: 0.189939
    [2819]	valid_0's auc: 0.89823	valid_0's binary_logloss: 0.189842
    [2820]	valid_0's auc: 0.89821	valid_0's binary_logloss: 0.189781
    [2821]	valid_0's auc: 0.898214	valid_0's binary_logloss: 0.189813
    [2822]	valid_0's auc: 0.898217	valid_0's binary_logloss: 0.189856
    [2823]	valid_0's auc: 0.898223	valid_0's binary_logloss: 0.189886
    [2824]	valid_0's auc: 0.898272	valid_0's binary_logloss: 0.189819
    [2825]	valid_0's auc: 0.898273	valid_0's binary_logloss: 0.189864
    [2826]	valid_0's auc: 0.898273	valid_0's binary_logloss: 0.189891
    [2827]	valid_0's auc: 0.898271	valid_0's binary_logloss: 0.189927
    [2828]	valid_0's auc: 0.898261	valid_0's binary_logloss: 0.189867
    [2829]	valid_0's auc: 0.898308	valid_0's binary_logloss: 0.189767
    [2830]	valid_0's auc: 0.898349	valid_0's binary_logloss: 0.189695
    [2831]	valid_0's auc: 0.898346	valid_0's binary_logloss: 0.18965
    [2832]	valid_0's auc: 0.898348	valid_0's binary_logloss: 0.189688
    [2833]	valid_0's auc: 0.898352	valid_0's binary_logloss: 0.189735
    [2834]	valid_0's auc: 0.898354	valid_0's binary_logloss: 0.189754
    [2835]	valid_0's auc: 0.898357	valid_0's binary_logloss: 0.18969
    [2836]	valid_0's auc: 0.898359	valid_0's binary_logloss: 0.189731
    [2837]	valid_0's auc: 0.898363	valid_0's binary_logloss: 0.189768
    [2838]	valid_0's auc: 0.898364	valid_0's binary_logloss: 0.189802
    [2839]	valid_0's auc: 0.898393	valid_0's binary_logloss: 0.189702
    [2840]	valid_0's auc: 0.898398	valid_0's binary_logloss: 0.18964
    [2841]	valid_0's auc: 0.898385	valid_0's binary_logloss: 0.189598
    [2842]	valid_0's auc: 0.898385	valid_0's binary_logloss: 0.189654
    [2843]	valid_0's auc: 0.898395	valid_0's binary_logloss: 0.189593
    [2844]	valid_0's auc: 0.898438	valid_0's binary_logloss: 0.189512
    [2845]	valid_0's auc: 0.898405	valid_0's binary_logloss: 0.18945
    [2846]	valid_0's auc: 0.898411	valid_0's binary_logloss: 0.189481
    [2847]	valid_0's auc: 0.898392	valid_0's binary_logloss: 0.189377
    [2848]	valid_0's auc: 0.898394	valid_0's binary_logloss: 0.189399
    [2849]	valid_0's auc: 0.898397	valid_0's binary_logloss: 0.189442
    [2850]	valid_0's auc: 0.89835	valid_0's binary_logloss: 0.189401
    [2851]	valid_0's auc: 0.89836	valid_0's binary_logloss: 0.189319
    [2852]	valid_0's auc: 0.898362	valid_0's binary_logloss: 0.189267
    [2853]	valid_0's auc: 0.898364	valid_0's binary_logloss: 0.189304
    [2854]	valid_0's auc: 0.898355	valid_0's binary_logloss: 0.189249
    [2855]	valid_0's auc: 0.898348	valid_0's binary_logloss: 0.189204
    [2856]	valid_0's auc: 0.89835	valid_0's binary_logloss: 0.189231
    [2857]	valid_0's auc: 0.898355	valid_0's binary_logloss: 0.189252
    [2858]	valid_0's auc: 0.898419	valid_0's binary_logloss: 0.189179
    [2859]	valid_0's auc: 0.898421	valid_0's binary_logloss: 0.189221
    [2860]	valid_0's auc: 0.898428	valid_0's binary_logloss: 0.189264
    [2861]	valid_0's auc: 0.898432	valid_0's binary_logloss: 0.189309
    [2862]	valid_0's auc: 0.898439	valid_0's binary_logloss: 0.189226
    [2863]	valid_0's auc: 0.898477	valid_0's binary_logloss: 0.189131
    [2864]	valid_0's auc: 0.898514	valid_0's binary_logloss: 0.189034
    [2865]	valid_0's auc: 0.898539	valid_0's binary_logloss: 0.188835
    [2866]	valid_0's auc: 0.898534	valid_0's binary_logloss: 0.188765
    [2867]	valid_0's auc: 0.898534	valid_0's binary_logloss: 0.188726
    [2868]	valid_0's auc: 0.89854	valid_0's binary_logloss: 0.188755
    [2869]	valid_0's auc: 0.898542	valid_0's binary_logloss: 0.188785
    [2870]	valid_0's auc: 0.898544	valid_0's binary_logloss: 0.188809
    [2871]	valid_0's auc: 0.898562	valid_0's binary_logloss: 0.188623
    [2872]	valid_0's auc: 0.898586	valid_0's binary_logloss: 0.188576
    [2873]	valid_0's auc: 0.898625	valid_0's binary_logloss: 0.188502
    [2874]	valid_0's auc: 0.89862	valid_0's binary_logloss: 0.188463
    [2875]	valid_0's auc: 0.898617	valid_0's binary_logloss: 0.188511
    [2876]	valid_0's auc: 0.89862	valid_0's binary_logloss: 0.188534
    [2877]	valid_0's auc: 0.898654	valid_0's binary_logloss: 0.188338
    [2878]	valid_0's auc: 0.898658	valid_0's binary_logloss: 0.188383
    [2879]	valid_0's auc: 0.898669	valid_0's binary_logloss: 0.18832
    [2880]	valid_0's auc: 0.898704	valid_0's binary_logloss: 0.188252
    [2881]	valid_0's auc: 0.89871	valid_0's binary_logloss: 0.188075
    [2882]	valid_0's auc: 0.898689	valid_0's binary_logloss: 0.188024
    [2883]	valid_0's auc: 0.89869	valid_0's binary_logloss: 0.187974
    [2884]	valid_0's auc: 0.898692	valid_0's binary_logloss: 0.188006
    [2885]	valid_0's auc: 0.898713	valid_0's binary_logloss: 0.187961
    [2886]	valid_0's auc: 0.898718	valid_0's binary_logloss: 0.188004
    [2887]	valid_0's auc: 0.898722	valid_0's binary_logloss: 0.188037
    [2888]	valid_0's auc: 0.898722	valid_0's binary_logloss: 0.188078
    [2889]	valid_0's auc: 0.898711	valid_0's binary_logloss: 0.188039
    [2890]	valid_0's auc: 0.898745	valid_0's binary_logloss: 0.187917
    [2891]	valid_0's auc: 0.898747	valid_0's binary_logloss: 0.187949
    [2892]	valid_0's auc: 0.89875	valid_0's binary_logloss: 0.187982
    [2893]	valid_0's auc: 0.898784	valid_0's binary_logloss: 0.187924
    [2894]	valid_0's auc: 0.898786	valid_0's binary_logloss: 0.187948
    [2895]	valid_0's auc: 0.898787	valid_0's binary_logloss: 0.18798
    [2896]	valid_0's auc: 0.89881	valid_0's binary_logloss: 0.18792
    [2897]	valid_0's auc: 0.898812	valid_0's binary_logloss: 0.187946
    [2898]	valid_0's auc: 0.898819	valid_0's binary_logloss: 0.187976
    [2899]	valid_0's auc: 0.898839	valid_0's binary_logloss: 0.187882
    [2900]	valid_0's auc: 0.89884	valid_0's binary_logloss: 0.187824
    [2901]	valid_0's auc: 0.898824	valid_0's binary_logloss: 0.187783
    [2902]	valid_0's auc: 0.898829	valid_0's binary_logloss: 0.18782
    [2903]	valid_0's auc: 0.898831	valid_0's binary_logloss: 0.18785
    [2904]	valid_0's auc: 0.898798	valid_0's binary_logloss: 0.187767
    [2905]	valid_0's auc: 0.89881	valid_0's binary_logloss: 0.187739
    [2906]	valid_0's auc: 0.898806	valid_0's binary_logloss: 0.187786
    [2907]	valid_0's auc: 0.898806	valid_0's binary_logloss: 0.187816
    [2908]	valid_0's auc: 0.898807	valid_0's binary_logloss: 0.187857
    [2909]	valid_0's auc: 0.898813	valid_0's binary_logloss: 0.187905
    [2910]	valid_0's auc: 0.898799	valid_0's binary_logloss: 0.187812
    [2911]	valid_0's auc: 0.898777	valid_0's binary_logloss: 0.18771
    [2912]	valid_0's auc: 0.898842	valid_0's binary_logloss: 0.187629
    [2913]	valid_0's auc: 0.898844	valid_0's binary_logloss: 0.187667
    [2914]	valid_0's auc: 0.898846	valid_0's binary_logloss: 0.187615
    [2915]	valid_0's auc: 0.898853	valid_0's binary_logloss: 0.187634
    [2916]	valid_0's auc: 0.898851	valid_0's binary_logloss: 0.187578
    [2917]	valid_0's auc: 0.898852	valid_0's binary_logloss: 0.187618
    [2918]	valid_0's auc: 0.898848	valid_0's binary_logloss: 0.187474
    [2919]	valid_0's auc: 0.898849	valid_0's binary_logloss: 0.187502
    [2920]	valid_0's auc: 0.898851	valid_0's binary_logloss: 0.187535
    [2921]	valid_0's auc: 0.898861	valid_0's binary_logloss: 0.187471
    [2922]	valid_0's auc: 0.898869	valid_0's binary_logloss: 0.187393
    [2923]	valid_0's auc: 0.898874	valid_0's binary_logloss: 0.187427
    [2924]	valid_0's auc: 0.898853	valid_0's binary_logloss: 0.187361
    [2925]	valid_0's auc: 0.898857	valid_0's binary_logloss: 0.187397
    [2926]	valid_0's auc: 0.898855	valid_0's binary_logloss: 0.187435
    [2927]	valid_0's auc: 0.898865	valid_0's binary_logloss: 0.187473
    [2928]	valid_0's auc: 0.898887	valid_0's binary_logloss: 0.187417
    [2929]	valid_0's auc: 0.898893	valid_0's binary_logloss: 0.187455
    [2930]	valid_0's auc: 0.898923	valid_0's binary_logloss: 0.187363
    [2931]	valid_0's auc: 0.898898	valid_0's binary_logloss: 0.187306
    [2932]	valid_0's auc: 0.898899	valid_0's binary_logloss: 0.187345
    [2933]	valid_0's auc: 0.898903	valid_0's binary_logloss: 0.187368
    [2934]	valid_0's auc: 0.898939	valid_0's binary_logloss: 0.187247
    [2935]	valid_0's auc: 0.898963	valid_0's binary_logloss: 0.18719
    [2936]	valid_0's auc: 0.898983	valid_0's binary_logloss: 0.187113
    [2937]	valid_0's auc: 0.898986	valid_0's binary_logloss: 0.18716
    [2938]	valid_0's auc: 0.89899	valid_0's binary_logloss: 0.187192
    [2939]	valid_0's auc: 0.898995	valid_0's binary_logloss: 0.18715
    [2940]	valid_0's auc: 0.89901	valid_0's binary_logloss: 0.186991
    [2941]	valid_0's auc: 0.898983	valid_0's binary_logloss: 0.186946
    [2942]	valid_0's auc: 0.898987	valid_0's binary_logloss: 0.186968
    [2943]	valid_0's auc: 0.899008	valid_0's binary_logloss: 0.186805
    [2944]	valid_0's auc: 0.899006	valid_0's binary_logloss: 0.186751
    [2945]	valid_0's auc: 0.899004	valid_0's binary_logloss: 0.186782
    [2946]	valid_0's auc: 0.899004	valid_0's binary_logloss: 0.186809
    [2947]	valid_0's auc: 0.898972	valid_0's binary_logloss: 0.186771
    [2948]	valid_0's auc: 0.89898	valid_0's binary_logloss: 0.186806
    [2949]	valid_0's auc: 0.898982	valid_0's binary_logloss: 0.186835
    [2950]	valid_0's auc: 0.898993	valid_0's binary_logloss: 0.186737
    [2951]	valid_0's auc: 0.899001	valid_0's binary_logloss: 0.18665
    [2952]	valid_0's auc: 0.899002	valid_0's binary_logloss: 0.186676
    [2953]	valid_0's auc: 0.899006	valid_0's binary_logloss: 0.1867
    [2954]	valid_0's auc: 0.899012	valid_0's binary_logloss: 0.186651
    [2955]	valid_0's auc: 0.899016	valid_0's binary_logloss: 0.186695
    [2956]	valid_0's auc: 0.899029	valid_0's binary_logloss: 0.186648
    [2957]	valid_0's auc: 0.899032	valid_0's binary_logloss: 0.186679
    [2958]	valid_0's auc: 0.899023	valid_0's binary_logloss: 0.186604
    [2959]	valid_0's auc: 0.899024	valid_0's binary_logloss: 0.186632
    [2960]	valid_0's auc: 0.899022	valid_0's binary_logloss: 0.18667
    [2961]	valid_0's auc: 0.899031	valid_0's binary_logloss: 0.186699
    [2962]	valid_0's auc: 0.898996	valid_0's binary_logloss: 0.186668
    [2963]	valid_0's auc: 0.899	valid_0's binary_logloss: 0.186619
    [2964]	valid_0's auc: 0.899001	valid_0's binary_logloss: 0.186644
    [2965]	valid_0's auc: 0.898975	valid_0's binary_logloss: 0.186601
    [2966]	valid_0's auc: 0.89896	valid_0's binary_logloss: 0.186554
    [2967]	valid_0's auc: 0.898944	valid_0's binary_logloss: 0.186518
    [2968]	valid_0's auc: 0.898971	valid_0's binary_logloss: 0.186466
    [2969]	valid_0's auc: 0.898963	valid_0's binary_logloss: 0.186426
    [2970]	valid_0's auc: 0.898999	valid_0's binary_logloss: 0.18637
    [2971]	valid_0's auc: 0.899088	valid_0's binary_logloss: 0.186171
    [2972]	valid_0's auc: 0.899096	valid_0's binary_logloss: 0.186126
    [2973]	valid_0's auc: 0.8991	valid_0's binary_logloss: 0.185961
    [2974]	valid_0's auc: 0.899152	valid_0's binary_logloss: 0.1859
    [2975]	valid_0's auc: 0.899157	valid_0's binary_logloss: 0.185926
    [2976]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.185956
    [2977]	valid_0's auc: 0.899162	valid_0's binary_logloss: 0.185993
    [2978]	valid_0's auc: 0.89915	valid_0's binary_logloss: 0.185942
    [2979]	valid_0's auc: 0.899142	valid_0's binary_logloss: 0.185792
    [2980]	valid_0's auc: 0.899143	valid_0's binary_logloss: 0.185838
    [2981]	valid_0's auc: 0.899195	valid_0's binary_logloss: 0.185759
    [2982]	valid_0's auc: 0.899196	valid_0's binary_logloss: 0.185717
    [2983]	valid_0's auc: 0.899202	valid_0's binary_logloss: 0.18563
    [2984]	valid_0's auc: 0.89918	valid_0's binary_logloss: 0.185598
    [2985]	valid_0's auc: 0.899185	valid_0's binary_logloss: 0.185631
    [2986]	valid_0's auc: 0.899187	valid_0's binary_logloss: 0.18566
    [2987]	valid_0's auc: 0.899189	valid_0's binary_logloss: 0.185685
    [2988]	valid_0's auc: 0.899209	valid_0's binary_logloss: 0.185631
    [2989]	valid_0's auc: 0.899203	valid_0's binary_logloss: 0.185586
    [2990]	valid_0's auc: 0.899205	valid_0's binary_logloss: 0.185607
    [2991]	valid_0's auc: 0.899193	valid_0's binary_logloss: 0.185576
    [2992]	valid_0's auc: 0.899196	valid_0's binary_logloss: 0.185598
    [2993]	valid_0's auc: 0.899208	valid_0's binary_logloss: 0.185555
    [2994]	valid_0's auc: 0.899217	valid_0's binary_logloss: 0.185585
    [2995]	valid_0's auc: 0.899217	valid_0's binary_logloss: 0.185548
    [2996]	valid_0's auc: 0.89922	valid_0's binary_logloss: 0.185591
    [2997]	valid_0's auc: 0.89922	valid_0's binary_logloss: 0.185621
    [2998]	valid_0's auc: 0.899221	valid_0's binary_logloss: 0.185646
    [2999]	valid_0's auc: 0.899272	valid_0's binary_logloss: 0.185595
    [3000]	valid_0's auc: 0.899293	valid_0's binary_logloss: 0.185485
    [3001]	valid_0's auc: 0.899298	valid_0's binary_logloss: 0.185511
    [3002]	valid_0's auc: 0.899302	valid_0's binary_logloss: 0.185546
    [3003]	valid_0's auc: 0.899299	valid_0's binary_logloss: 0.185497
    [3004]	valid_0's auc: 0.899297	valid_0's binary_logloss: 0.185531
    [3005]	valid_0's auc: 0.899323	valid_0's binary_logloss: 0.185446
    [3006]	valid_0's auc: 0.899332	valid_0's binary_logloss: 0.185466
    [3007]	valid_0's auc: 0.899324	valid_0's binary_logloss: 0.185416
    [3008]	valid_0's auc: 0.899328	valid_0's binary_logloss: 0.185445
    [3009]	valid_0's auc: 0.899347	valid_0's binary_logloss: 0.18541
    [3010]	valid_0's auc: 0.899348	valid_0's binary_logloss: 0.185434
    [3011]	valid_0's auc: 0.899323	valid_0's binary_logloss: 0.18529
    [3012]	valid_0's auc: 0.899321	valid_0's binary_logloss: 0.185324
    [3013]	valid_0's auc: 0.899324	valid_0's binary_logloss: 0.185351
    [3014]	valid_0's auc: 0.89938	valid_0's binary_logloss: 0.18526
    [3015]	valid_0's auc: 0.899381	valid_0's binary_logloss: 0.185186
    [3016]	valid_0's auc: 0.89938	valid_0's binary_logloss: 0.185227
    [3017]	valid_0's auc: 0.89938	valid_0's binary_logloss: 0.185194
    [3018]	valid_0's auc: 0.899379	valid_0's binary_logloss: 0.185154
    [3019]	valid_0's auc: 0.899383	valid_0's binary_logloss: 0.185175
    [3020]	valid_0's auc: 0.899409	valid_0's binary_logloss: 0.185119
    [3021]	valid_0's auc: 0.899413	valid_0's binary_logloss: 0.185149
    [3022]	valid_0's auc: 0.899416	valid_0's binary_logloss: 0.185173
    [3023]	valid_0's auc: 0.899429	valid_0's binary_logloss: 0.185139
    [3024]	valid_0's auc: 0.899435	valid_0's binary_logloss: 0.185093
    [3025]	valid_0's auc: 0.899437	valid_0's binary_logloss: 0.185123
    [3026]	valid_0's auc: 0.89941	valid_0's binary_logloss: 0.18509
    [3027]	valid_0's auc: 0.899397	valid_0's binary_logloss: 0.185046
    [3028]	valid_0's auc: 0.899443	valid_0's binary_logloss: 0.184959
    [3029]	valid_0's auc: 0.899447	valid_0's binary_logloss: 0.185005
    [3030]	valid_0's auc: 0.899447	valid_0's binary_logloss: 0.185047
    [3031]	valid_0's auc: 0.899447	valid_0's binary_logloss: 0.18499
    [3032]	valid_0's auc: 0.89945	valid_0's binary_logloss: 0.185016
    [3033]	valid_0's auc: 0.899451	valid_0's binary_logloss: 0.184972
    [3034]	valid_0's auc: 0.899474	valid_0's binary_logloss: 0.184923
    [3035]	valid_0's auc: 0.899487	valid_0's binary_logloss: 0.184878
    [3036]	valid_0's auc: 0.899489	valid_0's binary_logloss: 0.184907
    [3037]	valid_0's auc: 0.899494	valid_0's binary_logloss: 0.184932
    [3038]	valid_0's auc: 0.899489	valid_0's binary_logloss: 0.184895
    [3039]	valid_0's auc: 0.899495	valid_0's binary_logloss: 0.184917
    [3040]	valid_0's auc: 0.899482	valid_0's binary_logloss: 0.184882
    [3041]	valid_0's auc: 0.899483	valid_0's binary_logloss: 0.18491
    [3042]	valid_0's auc: 0.899487	valid_0's binary_logloss: 0.184936
    [3043]	valid_0's auc: 0.899503	valid_0's binary_logloss: 0.184876
    [3044]	valid_0's auc: 0.899503	valid_0's binary_logloss: 0.1849
    [3045]	valid_0's auc: 0.899505	valid_0's binary_logloss: 0.18494
    [3046]	valid_0's auc: 0.899499	valid_0's binary_logloss: 0.184897
    [3047]	valid_0's auc: 0.89949	valid_0's binary_logloss: 0.184862
    [3048]	valid_0's auc: 0.899526	valid_0's binary_logloss: 0.184688
    [3049]	valid_0's auc: 0.899531	valid_0's binary_logloss: 0.184711
    [3050]	valid_0's auc: 0.899524	valid_0's binary_logloss: 0.184661
    [3051]	valid_0's auc: 0.899554	valid_0's binary_logloss: 0.184604
    [3052]	valid_0's auc: 0.899557	valid_0's binary_logloss: 0.184568
    [3053]	valid_0's auc: 0.899557	valid_0's binary_logloss: 0.184587
    [3054]	valid_0's auc: 0.89955	valid_0's binary_logloss: 0.184542
    [3055]	valid_0's auc: 0.899545	valid_0's binary_logloss: 0.184515
    [3056]	valid_0's auc: 0.899549	valid_0's binary_logloss: 0.184481
    [3057]	valid_0's auc: 0.899557	valid_0's binary_logloss: 0.184513
    [3058]	valid_0's auc: 0.89956	valid_0's binary_logloss: 0.184536
    [3059]	valid_0's auc: 0.899558	valid_0's binary_logloss: 0.18457
    [3060]	valid_0's auc: 0.899562	valid_0's binary_logloss: 0.184599
    [3061]	valid_0's auc: 0.8996	valid_0's binary_logloss: 0.18454
    [3062]	valid_0's auc: 0.899604	valid_0's binary_logloss: 0.184561
    [3063]	valid_0's auc: 0.899597	valid_0's binary_logloss: 0.184527
    [3064]	valid_0's auc: 0.899595	valid_0's binary_logloss: 0.184467
    [3065]	valid_0's auc: 0.899595	valid_0's binary_logloss: 0.184511
    [3066]	valid_0's auc: 0.899598	valid_0's binary_logloss: 0.184534
    [3067]	valid_0's auc: 0.899602	valid_0's binary_logloss: 0.184564
    [3068]	valid_0's auc: 0.89961	valid_0's binary_logloss: 0.184486
    [3069]	valid_0's auc: 0.899614	valid_0's binary_logloss: 0.184517
    [3070]	valid_0's auc: 0.899642	valid_0's binary_logloss: 0.184338
    [3071]	valid_0's auc: 0.899646	valid_0's binary_logloss: 0.184367
    [3072]	valid_0's auc: 0.89965	valid_0's binary_logloss: 0.18439
    [3073]	valid_0's auc: 0.899657	valid_0's binary_logloss: 0.184421
    [3074]	valid_0's auc: 0.899715	valid_0's binary_logloss: 0.184309
    [3075]	valid_0's auc: 0.899716	valid_0's binary_logloss: 0.184337
    [3076]	valid_0's auc: 0.89974	valid_0's binary_logloss: 0.184172
    [3077]	valid_0's auc: 0.899743	valid_0's binary_logloss: 0.184213
    [3078]	valid_0's auc: 0.899744	valid_0's binary_logloss: 0.184251
    [3079]	valid_0's auc: 0.899739	valid_0's binary_logloss: 0.184221
    [3080]	valid_0's auc: 0.899742	valid_0's binary_logloss: 0.18425
    [3081]	valid_0's auc: 0.899796	valid_0's binary_logloss: 0.184166
    [3082]	valid_0's auc: 0.899799	valid_0's binary_logloss: 0.184196
    [3083]	valid_0's auc: 0.8998	valid_0's binary_logloss: 0.184155
    [3084]	valid_0's auc: 0.899816	valid_0's binary_logloss: 0.184102
    [3085]	valid_0's auc: 0.899841	valid_0's binary_logloss: 0.184068
    [3086]	valid_0's auc: 0.899832	valid_0's binary_logloss: 0.184006
    [3087]	valid_0's auc: 0.899832	valid_0's binary_logloss: 0.184028
    [3088]	valid_0's auc: 0.899838	valid_0's binary_logloss: 0.184043
    [3089]	valid_0's auc: 0.899838	valid_0's binary_logloss: 0.184083
    [3090]	valid_0's auc: 0.899825	valid_0's binary_logloss: 0.184048
    [3091]	valid_0's auc: 0.899828	valid_0's binary_logloss: 0.184072
    [3092]	valid_0's auc: 0.899826	valid_0's binary_logloss: 0.184098
    [3093]	valid_0's auc: 0.89982	valid_0's binary_logloss: 0.184068
    [3094]	valid_0's auc: 0.899835	valid_0's binary_logloss: 0.183999
    [3095]	valid_0's auc: 0.899828	valid_0's binary_logloss: 0.18396
    [3096]	valid_0's auc: 0.899868	valid_0's binary_logloss: 0.183906
    [3097]	valid_0's auc: 0.899869	valid_0's binary_logloss: 0.183929
    [3098]	valid_0's auc: 0.899871	valid_0's binary_logloss: 0.183955
    [3099]	valid_0's auc: 0.899887	valid_0's binary_logloss: 0.183907
    [3100]	valid_0's auc: 0.899927	valid_0's binary_logloss: 0.18379
    [3101]	valid_0's auc: 0.899933	valid_0's binary_logloss: 0.183715
    [3102]	valid_0's auc: 0.899952	valid_0's binary_logloss: 0.183675
    [3103]	valid_0's auc: 0.899951	valid_0's binary_logloss: 0.183697
    [3104]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.183648
    [3105]	valid_0's auc: 0.899966	valid_0's binary_logloss: 0.183623
    [3106]	valid_0's auc: 0.89994	valid_0's binary_logloss: 0.183605
    [3107]	valid_0's auc: 0.899945	valid_0's binary_logloss: 0.183637
    [3108]	valid_0's auc: 0.899973	valid_0's binary_logloss: 0.183477
    [3109]	valid_0's auc: 0.899977	valid_0's binary_logloss: 0.183505
    [3110]	valid_0's auc: 0.899982	valid_0's binary_logloss: 0.183529
    [3111]	valid_0's auc: 0.899981	valid_0's binary_logloss: 0.183554
    [3112]	valid_0's auc: 0.899979	valid_0's binary_logloss: 0.183576
    [3113]	valid_0's auc: 0.899987	valid_0's binary_logloss: 0.183618
    [3114]	valid_0's auc: 0.899979	valid_0's binary_logloss: 0.183581
    [3115]	valid_0's auc: 0.899982	valid_0's binary_logloss: 0.18352
    [3116]	valid_0's auc: 0.899989	valid_0's binary_logloss: 0.183556
    [3117]	valid_0's auc: 0.89999	valid_0's binary_logloss: 0.183575
    [3118]	valid_0's auc: 0.900001	valid_0's binary_logloss: 0.183524
    [3119]	valid_0's auc: 0.900007	valid_0's binary_logloss: 0.183545
    [3120]	valid_0's auc: 0.900002	valid_0's binary_logloss: 0.183463
    [3121]	valid_0's auc: 0.900005	valid_0's binary_logloss: 0.183486
    [3122]	valid_0's auc: 0.900005	valid_0's binary_logloss: 0.183417
    [3123]	valid_0's auc: 0.900021	valid_0's binary_logloss: 0.183378
    [3124]	valid_0's auc: 0.900021	valid_0's binary_logloss: 0.183411
    [3125]	valid_0's auc: 0.900031	valid_0's binary_logloss: 0.183371
    [3126]	valid_0's auc: 0.90003	valid_0's binary_logloss: 0.183397
    [3127]	valid_0's auc: 0.900037	valid_0's binary_logloss: 0.183439
    [3128]	valid_0's auc: 0.90004	valid_0's binary_logloss: 0.183475
    [3129]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.183498
    [3130]	valid_0's auc: 0.900028	valid_0's binary_logloss: 0.183445
    [3131]	valid_0's auc: 0.900055	valid_0's binary_logloss: 0.183377
    [3132]	valid_0's auc: 0.900056	valid_0's binary_logloss: 0.183399
    [3133]	valid_0's auc: 0.900057	valid_0's binary_logloss: 0.183357
    [3134]	valid_0's auc: 0.900063	valid_0's binary_logloss: 0.183389
    [3135]	valid_0's auc: 0.900066	valid_0's binary_logloss: 0.183426
    [3136]	valid_0's auc: 0.90007	valid_0's binary_logloss: 0.183392
    [3137]	valid_0's auc: 0.900048	valid_0's binary_logloss: 0.183368
    [3138]	valid_0's auc: 0.90007	valid_0's binary_logloss: 0.183254
    [3139]	valid_0's auc: 0.900112	valid_0's binary_logloss: 0.183198
    [3140]	valid_0's auc: 0.900112	valid_0's binary_logloss: 0.18323
    [3141]	valid_0's auc: 0.900102	valid_0's binary_logloss: 0.183191
    [3142]	valid_0's auc: 0.900065	valid_0's binary_logloss: 0.18314
    [3143]	valid_0's auc: 0.900065	valid_0's binary_logloss: 0.183167
    [3144]	valid_0's auc: 0.900065	valid_0's binary_logloss: 0.1832
    [3145]	valid_0's auc: 0.900087	valid_0's binary_logloss: 0.183125
    [3146]	valid_0's auc: 0.900089	valid_0's binary_logloss: 0.183141
    [3147]	valid_0's auc: 0.90007	valid_0's binary_logloss: 0.183096
    [3148]	valid_0's auc: 0.900069	valid_0's binary_logloss: 0.183032
    [3149]	valid_0's auc: 0.900049	valid_0's binary_logloss: 0.183006
    [3150]	valid_0's auc: 0.900033	valid_0's binary_logloss: 0.182974
    [3151]	valid_0's auc: 0.900037	valid_0's binary_logloss: 0.183
    [3152]	valid_0's auc: 0.90004	valid_0's binary_logloss: 0.18302
    [3153]	valid_0's auc: 0.900047	valid_0's binary_logloss: 0.183049
    [3154]	valid_0's auc: 0.900032	valid_0's binary_logloss: 0.183022
    [3155]	valid_0's auc: 0.900036	valid_0's binary_logloss: 0.182988
    [3156]	valid_0's auc: 0.900012	valid_0's binary_logloss: 0.182958
    [3157]	valid_0's auc: 0.900016	valid_0's binary_logloss: 0.18298
    [3158]	valid_0's auc: 0.90003	valid_0's binary_logloss: 0.182936
    [3159]	valid_0's auc: 0.900033	valid_0's binary_logloss: 0.18296
    [3160]	valid_0's auc: 0.900038	valid_0's binary_logloss: 0.182986
    [3161]	valid_0's auc: 0.900042	valid_0's binary_logloss: 0.183024
    [3162]	valid_0's auc: 0.900064	valid_0's binary_logloss: 0.182974
    [3163]	valid_0's auc: 0.90007	valid_0's binary_logloss: 0.183003
    [3164]	valid_0's auc: 0.900074	valid_0's binary_logloss: 0.182969
    [3165]	valid_0's auc: 0.900088	valid_0's binary_logloss: 0.182931
    [3166]	valid_0's auc: 0.900091	valid_0's binary_logloss: 0.182967
    [3167]	valid_0's auc: 0.900098	valid_0's binary_logloss: 0.182992
    [3168]	valid_0's auc: 0.900097	valid_0's binary_logloss: 0.183028
    [3169]	valid_0's auc: 0.900099	valid_0's binary_logloss: 0.18306
    [3170]	valid_0's auc: 0.9001	valid_0's binary_logloss: 0.183079
    [3171]	valid_0's auc: 0.900104	valid_0's binary_logloss: 0.183109
    [3172]	valid_0's auc: 0.900104	valid_0's binary_logloss: 0.183076
    [3173]	valid_0's auc: 0.900105	valid_0's binary_logloss: 0.183097
    [3174]	valid_0's auc: 0.900125	valid_0's binary_logloss: 0.183046
    [3175]	valid_0's auc: 0.900123	valid_0's binary_logloss: 0.182991
    [3176]	valid_0's auc: 0.900123	valid_0's binary_logloss: 0.183018
    [3177]	valid_0's auc: 0.90011	valid_0's binary_logloss: 0.182974
    [3178]	valid_0's auc: 0.900111	valid_0's binary_logloss: 0.182954
    [3179]	valid_0's auc: 0.900114	valid_0's binary_logloss: 0.182981
    [3180]	valid_0's auc: 0.900113	valid_0's binary_logloss: 0.183012
    [3181]	valid_0's auc: 0.900114	valid_0's binary_logloss: 0.183042
    [3182]	valid_0's auc: 0.900114	valid_0's binary_logloss: 0.183001
    [3183]	valid_0's auc: 0.900142	valid_0's binary_logloss: 0.182871
    [3184]	valid_0's auc: 0.900145	valid_0's binary_logloss: 0.182896
    [3185]	valid_0's auc: 0.900148	valid_0's binary_logloss: 0.18293
    [3186]	valid_0's auc: 0.900146	valid_0's binary_logloss: 0.182958
    [3187]	valid_0's auc: 0.900143	valid_0's binary_logloss: 0.182925
    [3188]	valid_0's auc: 0.900143	valid_0's binary_logloss: 0.182867
    [3189]	valid_0's auc: 0.900147	valid_0's binary_logloss: 0.18282
    [3190]	valid_0's auc: 0.900149	valid_0's binary_logloss: 0.182858
    [3191]	valid_0's auc: 0.900148	valid_0's binary_logloss: 0.182879
    [3192]	valid_0's auc: 0.90015	valid_0's binary_logloss: 0.182909
    [3193]	valid_0's auc: 0.900155	valid_0's binary_logloss: 0.182942
    [3194]	valid_0's auc: 0.900157	valid_0's binary_logloss: 0.182973
    [3195]	valid_0's auc: 0.900153	valid_0's binary_logloss: 0.182852
    [3196]	valid_0's auc: 0.900152	valid_0's binary_logloss: 0.182884
    [3197]	valid_0's auc: 0.900154	valid_0's binary_logloss: 0.182906
    [3198]	valid_0's auc: 0.90017	valid_0's binary_logloss: 0.182794
    [3199]	valid_0's auc: 0.900193	valid_0's binary_logloss: 0.182748
    [3200]	valid_0's auc: 0.900195	valid_0's binary_logloss: 0.182783
    [3201]	valid_0's auc: 0.900185	valid_0's binary_logloss: 0.182749
    [3202]	valid_0's auc: 0.900189	valid_0's binary_logloss: 0.182771
    [3203]	valid_0's auc: 0.900219	valid_0's binary_logloss: 0.182673
    [3204]	valid_0's auc: 0.900218	valid_0's binary_logloss: 0.182635
    [3205]	valid_0's auc: 0.900219	valid_0's binary_logloss: 0.182669
    [3206]	valid_0's auc: 0.900213	valid_0's binary_logloss: 0.182649
    [3207]	valid_0's auc: 0.900215	valid_0's binary_logloss: 0.182675
    [3208]	valid_0's auc: 0.900221	valid_0's binary_logloss: 0.182709
    [3209]	valid_0's auc: 0.900214	valid_0's binary_logloss: 0.18268
    [3210]	valid_0's auc: 0.900225	valid_0's binary_logloss: 0.182657
    [3211]	valid_0's auc: 0.900231	valid_0's binary_logloss: 0.182588
    [3212]	valid_0's auc: 0.90023	valid_0's binary_logloss: 0.182622
    [3213]	valid_0's auc: 0.90023	valid_0's binary_logloss: 0.182641
    [3214]	valid_0's auc: 0.900235	valid_0's binary_logloss: 0.182665
    [3215]	valid_0's auc: 0.900228	valid_0's binary_logloss: 0.182632
    [3216]	valid_0's auc: 0.900268	valid_0's binary_logloss: 0.182554
    [3217]	valid_0's auc: 0.900271	valid_0's binary_logloss: 0.182578
    [3218]	valid_0's auc: 0.900275	valid_0's binary_logloss: 0.182603
    [3219]	valid_0's auc: 0.900278	valid_0's binary_logloss: 0.18264
    [3220]	valid_0's auc: 0.900285	valid_0's binary_logloss: 0.182601
    [3221]	valid_0's auc: 0.900273	valid_0's binary_logloss: 0.182561
    [3222]	valid_0's auc: 0.90028	valid_0's binary_logloss: 0.182602
    [3223]	valid_0's auc: 0.900271	valid_0's binary_logloss: 0.182564
    [3224]	valid_0's auc: 0.900261	valid_0's binary_logloss: 0.182527
    [3225]	valid_0's auc: 0.900261	valid_0's binary_logloss: 0.182492
    [3226]	valid_0's auc: 0.90026	valid_0's binary_logloss: 0.182514
    [3227]	valid_0's auc: 0.900238	valid_0's binary_logloss: 0.182489
    [3228]	valid_0's auc: 0.900235	valid_0's binary_logloss: 0.182516
    [3229]	valid_0's auc: 0.900229	valid_0's binary_logloss: 0.182551
    [3230]	valid_0's auc: 0.900226	valid_0's binary_logloss: 0.182596
    [3231]	valid_0's auc: 0.900226	valid_0's binary_logloss: 0.182621
    [3232]	valid_0's auc: 0.900239	valid_0's binary_logloss: 0.182594
    [3233]	valid_0's auc: 0.900228	valid_0's binary_logloss: 0.182562
    [3234]	valid_0's auc: 0.900197	valid_0's binary_logloss: 0.182539
    [3235]	valid_0's auc: 0.900167	valid_0's binary_logloss: 0.182511
    [3236]	valid_0's auc: 0.900165	valid_0's binary_logloss: 0.182547
    [3237]	valid_0's auc: 0.900164	valid_0's binary_logloss: 0.182576
    [3238]	valid_0's auc: 0.900166	valid_0's binary_logloss: 0.182605
    [3239]	valid_0's auc: 0.900171	valid_0's binary_logloss: 0.182622
    [3240]	valid_0's auc: 0.900173	valid_0's binary_logloss: 0.182531
    [3241]	valid_0's auc: 0.900171	valid_0's binary_logloss: 0.18257
    [3242]	valid_0's auc: 0.900166	valid_0's binary_logloss: 0.182539
    [3243]	valid_0's auc: 0.900193	valid_0's binary_logloss: 0.182501
    [3244]	valid_0's auc: 0.900176	valid_0's binary_logloss: 0.182475
    [3245]	valid_0's auc: 0.900224	valid_0's binary_logloss: 0.182424
    [3246]	valid_0's auc: 0.900227	valid_0's binary_logloss: 0.182441
    [3247]	valid_0's auc: 0.90023	valid_0's binary_logloss: 0.18247
    [3248]	valid_0's auc: 0.900234	valid_0's binary_logloss: 0.182444
    [3249]	valid_0's auc: 0.900233	valid_0's binary_logloss: 0.182415
    [3250]	valid_0's auc: 0.900293	valid_0's binary_logloss: 0.182318
    [3251]	valid_0's auc: 0.900294	valid_0's binary_logloss: 0.182342
    [3252]	valid_0's auc: 0.900284	valid_0's binary_logloss: 0.182223
    [3253]	valid_0's auc: 0.900286	valid_0's binary_logloss: 0.182264
    [3254]	valid_0's auc: 0.900295	valid_0's binary_logloss: 0.182203
    [3255]	valid_0's auc: 0.900275	valid_0's binary_logloss: 0.182179
    [3256]	valid_0's auc: 0.900274	valid_0's binary_logloss: 0.182214
    [3257]	valid_0's auc: 0.900296	valid_0's binary_logloss: 0.182175
    [3258]	valid_0's auc: 0.900304	valid_0's binary_logloss: 0.182144
    [3259]	valid_0's auc: 0.900316	valid_0's binary_logloss: 0.182101
    [3260]	valid_0's auc: 0.900321	valid_0's binary_logloss: 0.182114
    [3261]	valid_0's auc: 0.900325	valid_0's binary_logloss: 0.18214
    [3262]	valid_0's auc: 0.900314	valid_0's binary_logloss: 0.182064
    [3263]	valid_0's auc: 0.900322	valid_0's binary_logloss: 0.182034
    [3264]	valid_0's auc: 0.900325	valid_0's binary_logloss: 0.182058
    [3265]	valid_0's auc: 0.900326	valid_0's binary_logloss: 0.181966
    [3266]	valid_0's auc: 0.90033	valid_0's binary_logloss: 0.181988
    [3267]	valid_0's auc: 0.900333	valid_0's binary_logloss: 0.182015
    [3268]	valid_0's auc: 0.900337	valid_0's binary_logloss: 0.182033
    [3269]	valid_0's auc: 0.900337	valid_0's binary_logloss: 0.182062
    [3270]	valid_0's auc: 0.900335	valid_0's binary_logloss: 0.182085
    [3271]	valid_0's auc: 0.900365	valid_0's binary_logloss: 0.181945
    [3272]	valid_0's auc: 0.900369	valid_0's binary_logloss: 0.181907
    [3273]	valid_0's auc: 0.900401	valid_0's binary_logloss: 0.181809
    [3274]	valid_0's auc: 0.900404	valid_0's binary_logloss: 0.181835
    [3275]	valid_0's auc: 0.900409	valid_0's binary_logloss: 0.181796
    [3276]	valid_0's auc: 0.900415	valid_0's binary_logloss: 0.181757
    [3277]	valid_0's auc: 0.900461	valid_0's binary_logloss: 0.181692
    [3278]	valid_0's auc: 0.900461	valid_0's binary_logloss: 0.181712
    [3279]	valid_0's auc: 0.900462	valid_0's binary_logloss: 0.181738
    [3280]	valid_0's auc: 0.900502	valid_0's binary_logloss: 0.181638
    [3281]	valid_0's auc: 0.900509	valid_0's binary_logloss: 0.181603
    [3282]	valid_0's auc: 0.900498	valid_0's binary_logloss: 0.181545
    [3283]	valid_0's auc: 0.900502	valid_0's binary_logloss: 0.181567
    [3284]	valid_0's auc: 0.900504	valid_0's binary_logloss: 0.181598
    [3285]	valid_0's auc: 0.90052	valid_0's binary_logloss: 0.181566
    [3286]	valid_0's auc: 0.900519	valid_0's binary_logloss: 0.181594
    [3287]	valid_0's auc: 0.900518	valid_0's binary_logloss: 0.181626
    [3288]	valid_0's auc: 0.900523	valid_0's binary_logloss: 0.181648
    [3289]	valid_0's auc: 0.900534	valid_0's binary_logloss: 0.18161
    [3290]	valid_0's auc: 0.900543	valid_0's binary_logloss: 0.181534
    [3291]	valid_0's auc: 0.900545	valid_0's binary_logloss: 0.181575
    [3292]	valid_0's auc: 0.900515	valid_0's binary_logloss: 0.181539
    [3293]	valid_0's auc: 0.90051	valid_0's binary_logloss: 0.181563
    [3294]	valid_0's auc: 0.900483	valid_0's binary_logloss: 0.181535
    [3295]	valid_0's auc: 0.900485	valid_0's binary_logloss: 0.181568
    [3296]	valid_0's auc: 0.900488	valid_0's binary_logloss: 0.181584
    [3297]	valid_0's auc: 0.90048	valid_0's binary_logloss: 0.181554
    [3298]	valid_0's auc: 0.900481	valid_0's binary_logloss: 0.181588
    [3299]	valid_0's auc: 0.900526	valid_0's binary_logloss: 0.181426
    [3300]	valid_0's auc: 0.900569	valid_0's binary_logloss: 0.181358
    [3301]	valid_0's auc: 0.900558	valid_0's binary_logloss: 0.181287
    [3302]	valid_0's auc: 0.90056	valid_0's binary_logloss: 0.181308
    [3303]	valid_0's auc: 0.900559	valid_0's binary_logloss: 0.181345
    [3304]	valid_0's auc: 0.900559	valid_0's binary_logloss: 0.181319
    [3305]	valid_0's auc: 0.900599	valid_0's binary_logloss: 0.181217
    [3306]	valid_0's auc: 0.900603	valid_0's binary_logloss: 0.18124
    [3307]	valid_0's auc: 0.900606	valid_0's binary_logloss: 0.181269
    [3308]	valid_0's auc: 0.900608	valid_0's binary_logloss: 0.181304
    [3309]	valid_0's auc: 0.900632	valid_0's binary_logloss: 0.181226
    [3310]	valid_0's auc: 0.900646	valid_0's binary_logloss: 0.181185
    [3311]	valid_0's auc: 0.900646	valid_0's binary_logloss: 0.181202
    [3312]	valid_0's auc: 0.900623	valid_0's binary_logloss: 0.181174
    [3313]	valid_0's auc: 0.900644	valid_0's binary_logloss: 0.181051
    [3314]	valid_0's auc: 0.900661	valid_0's binary_logloss: 0.180985
    [3315]	valid_0's auc: 0.900663	valid_0's binary_logloss: 0.18101
    [3316]	valid_0's auc: 0.900666	valid_0's binary_logloss: 0.181033
    [3317]	valid_0's auc: 0.900645	valid_0's binary_logloss: 0.180994
    [3318]	valid_0's auc: 0.900647	valid_0's binary_logloss: 0.181013
    [3319]	valid_0's auc: 0.900647	valid_0's binary_logloss: 0.180984
    [3320]	valid_0's auc: 0.900646	valid_0's binary_logloss: 0.181014
    [3321]	valid_0's auc: 0.900643	valid_0's binary_logloss: 0.181041
    [3322]	valid_0's auc: 0.900651	valid_0's binary_logloss: 0.180955
    [3323]	valid_0's auc: 0.900649	valid_0's binary_logloss: 0.180921
    [3324]	valid_0's auc: 0.900653	valid_0's binary_logloss: 0.180939
    [3325]	valid_0's auc: 0.900652	valid_0's binary_logloss: 0.180971
    [3326]	valid_0's auc: 0.900654	valid_0's binary_logloss: 0.180942
    [3327]	valid_0's auc: 0.900664	valid_0's binary_logloss: 0.180809
    [3328]	valid_0's auc: 0.900674	valid_0's binary_logloss: 0.180745
    [3329]	valid_0's auc: 0.900677	valid_0's binary_logloss: 0.180765
    [3330]	valid_0's auc: 0.900708	valid_0's binary_logloss: 0.180673
    [3331]	valid_0's auc: 0.900711	valid_0's binary_logloss: 0.180585
    [3332]	valid_0's auc: 0.900697	valid_0's binary_logloss: 0.180558
    [3333]	valid_0's auc: 0.900731	valid_0's binary_logloss: 0.180494
    [3334]	valid_0's auc: 0.90072	valid_0's binary_logloss: 0.180467
    [3335]	valid_0's auc: 0.90078	valid_0's binary_logloss: 0.180377
    [3336]	valid_0's auc: 0.900776	valid_0's binary_logloss: 0.18041
    [3337]	valid_0's auc: 0.900753	valid_0's binary_logloss: 0.180385
    [3338]	valid_0's auc: 0.900745	valid_0's binary_logloss: 0.180353
    [3339]	valid_0's auc: 0.900754	valid_0's binary_logloss: 0.180266
    [3340]	valid_0's auc: 0.900746	valid_0's binary_logloss: 0.180235
    [3341]	valid_0's auc: 0.900752	valid_0's binary_logloss: 0.180255
    [3342]	valid_0's auc: 0.900747	valid_0's binary_logloss: 0.180222
    [3343]	valid_0's auc: 0.900749	valid_0's binary_logloss: 0.180246
    [3344]	valid_0's auc: 0.900758	valid_0's binary_logloss: 0.180223
    [3345]	valid_0's auc: 0.900767	valid_0's binary_logloss: 0.180182
    [3346]	valid_0's auc: 0.900765	valid_0's binary_logloss: 0.180148
    [3347]	valid_0's auc: 0.90077	valid_0's binary_logloss: 0.180164
    [3348]	valid_0's auc: 0.900801	valid_0's binary_logloss: 0.180126
    [3349]	valid_0's auc: 0.900794	valid_0's binary_logloss: 0.1801
    [3350]	valid_0's auc: 0.900823	valid_0's binary_logloss: 0.180052
    [3351]	valid_0's auc: 0.90083	valid_0's binary_logloss: 0.180021
    [3352]	valid_0's auc: 0.900868	valid_0's binary_logloss: 0.179926
    [3353]	valid_0's auc: 0.900874	valid_0's binary_logloss: 0.179953
    [3354]	valid_0's auc: 0.900873	valid_0's binary_logloss: 0.179971
    [3355]	valid_0's auc: 0.90093	valid_0's binary_logloss: 0.179925
    [3356]	valid_0's auc: 0.900959	valid_0's binary_logloss: 0.179782
    [3357]	valid_0's auc: 0.900951	valid_0's binary_logloss: 0.179758
    [3358]	valid_0's auc: 0.900954	valid_0's binary_logloss: 0.179785
    [3359]	valid_0's auc: 0.900958	valid_0's binary_logloss: 0.179824
    [3360]	valid_0's auc: 0.90096	valid_0's binary_logloss: 0.17984
    [3361]	valid_0's auc: 0.900969	valid_0's binary_logloss: 0.179818
    [3362]	valid_0's auc: 0.90097	valid_0's binary_logloss: 0.17979
    [3363]	valid_0's auc: 0.900949	valid_0's binary_logloss: 0.179772
    [3364]	valid_0's auc: 0.900933	valid_0's binary_logloss: 0.179751
    [3365]	valid_0's auc: 0.900934	valid_0's binary_logloss: 0.179788
    [3366]	valid_0's auc: 0.900939	valid_0's binary_logloss: 0.179759
    [3367]	valid_0's auc: 0.900941	valid_0's binary_logloss: 0.179774
    [3368]	valid_0's auc: 0.900942	valid_0's binary_logloss: 0.179796
    [3369]	valid_0's auc: 0.900914	valid_0's binary_logloss: 0.179779
    [3370]	valid_0's auc: 0.900925	valid_0's binary_logloss: 0.17974
    [3371]	valid_0's auc: 0.900933	valid_0's binary_logloss: 0.179696
    [3372]	valid_0's auc: 0.900914	valid_0's binary_logloss: 0.179676
    [3373]	valid_0's auc: 0.900928	valid_0's binary_logloss: 0.179635
    [3374]	valid_0's auc: 0.900951	valid_0's binary_logloss: 0.179559
    [3375]	valid_0's auc: 0.900948	valid_0's binary_logloss: 0.179582
    [3376]	valid_0's auc: 0.900936	valid_0's binary_logloss: 0.179551
    [3377]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.179574
    [3378]	valid_0's auc: 0.900937	valid_0's binary_logloss: 0.179595
    [3379]	valid_0's auc: 0.90091	valid_0's binary_logloss: 0.179574
    [3380]	valid_0's auc: 0.900907	valid_0's binary_logloss: 0.179546
    [3381]	valid_0's auc: 0.900919	valid_0's binary_logloss: 0.179473
    [3382]	valid_0's auc: 0.900942	valid_0's binary_logloss: 0.179353
    [3383]	valid_0's auc: 0.90094	valid_0's binary_logloss: 0.179375
    [3384]	valid_0's auc: 0.900939	valid_0's binary_logloss: 0.179394
    [3385]	valid_0's auc: 0.900935	valid_0's binary_logloss: 0.179424
    [3386]	valid_0's auc: 0.900952	valid_0's binary_logloss: 0.179385
    [3387]	valid_0's auc: 0.901001	valid_0's binary_logloss: 0.179325
    [3388]	valid_0's auc: 0.90101	valid_0's binary_logloss: 0.179298
    [3389]	valid_0's auc: 0.900983	valid_0's binary_logloss: 0.179282
    [3390]	valid_0's auc: 0.900986	valid_0's binary_logloss: 0.179303
    [3391]	valid_0's auc: 0.900989	valid_0's binary_logloss: 0.179324
    [3392]	valid_0's auc: 0.90099	valid_0's binary_logloss: 0.179346
    [3393]	valid_0's auc: 0.900996	valid_0's binary_logloss: 0.179294
    [3394]	valid_0's auc: 0.900998	valid_0's binary_logloss: 0.179313
    [3395]	valid_0's auc: 0.901001	valid_0's binary_logloss: 0.179341
    [3396]	valid_0's auc: 0.901036	valid_0's binary_logloss: 0.179279
    [3397]	valid_0's auc: 0.901038	valid_0's binary_logloss: 0.179295
    [3398]	valid_0's auc: 0.901048	valid_0's binary_logloss: 0.179263
    [3399]	valid_0's auc: 0.901038	valid_0's binary_logloss: 0.179243
    [3400]	valid_0's auc: 0.901015	valid_0's binary_logloss: 0.179217
    [3401]	valid_0's auc: 0.901018	valid_0's binary_logloss: 0.179248
    [3402]	valid_0's auc: 0.901009	valid_0's binary_logloss: 0.17922
    [3403]	valid_0's auc: 0.901009	valid_0's binary_logloss: 0.179249
    [3404]	valid_0's auc: 0.901018	valid_0's binary_logloss: 0.17921
    [3405]	valid_0's auc: 0.901032	valid_0's binary_logloss: 0.179165
    [3406]	valid_0's auc: 0.901034	valid_0's binary_logloss: 0.179136
    [3407]	valid_0's auc: 0.901033	valid_0's binary_logloss: 0.179121
    [3408]	valid_0's auc: 0.901032	valid_0's binary_logloss: 0.179143
    [3409]	valid_0's auc: 0.901034	valid_0's binary_logloss: 0.179159
    [3410]	valid_0's auc: 0.901046	valid_0's binary_logloss: 0.179124
    [3411]	valid_0's auc: 0.90104	valid_0's binary_logloss: 0.179082
    [3412]	valid_0's auc: 0.901037	valid_0's binary_logloss: 0.179112
    [3413]	valid_0's auc: 0.901085	valid_0's binary_logloss: 0.179056
    [3414]	valid_0's auc: 0.901083	valid_0's binary_logloss: 0.179032
    [3415]	valid_0's auc: 0.901078	valid_0's binary_logloss: 0.178998
    [3416]	valid_0's auc: 0.901078	valid_0's binary_logloss: 0.179019
    [3417]	valid_0's auc: 0.901086	valid_0's binary_logloss: 0.178995
    [3418]	valid_0's auc: 0.901087	valid_0's binary_logloss: 0.17902
    [3419]	valid_0's auc: 0.901093	valid_0's binary_logloss: 0.179037
    [3420]	valid_0's auc: 0.901097	valid_0's binary_logloss: 0.179055
    [3421]	valid_0's auc: 0.901094	valid_0's binary_logloss: 0.179019
    [3422]	valid_0's auc: 0.901094	valid_0's binary_logloss: 0.179049
    [3423]	valid_0's auc: 0.901117	valid_0's binary_logloss: 0.178918
    [3424]	valid_0's auc: 0.901155	valid_0's binary_logloss: 0.178833
    [3425]	valid_0's auc: 0.901175	valid_0's binary_logloss: 0.178776
    [3426]	valid_0's auc: 0.901176	valid_0's binary_logloss: 0.178786
    [3427]	valid_0's auc: 0.90118	valid_0's binary_logloss: 0.178805
    [3428]	valid_0's auc: 0.901165	valid_0's binary_logloss: 0.178786
    [3429]	valid_0's auc: 0.901169	valid_0's binary_logloss: 0.178807
    [3430]	valid_0's auc: 0.901172	valid_0's binary_logloss: 0.178825
    [3431]	valid_0's auc: 0.901176	valid_0's binary_logloss: 0.178855
    [3432]	valid_0's auc: 0.901177	valid_0's binary_logloss: 0.178875
    [3433]	valid_0's auc: 0.90118	valid_0's binary_logloss: 0.178892
    [3434]	valid_0's auc: 0.901177	valid_0's binary_logloss: 0.178915
    [3435]	valid_0's auc: 0.901166	valid_0's binary_logloss: 0.178898
    [3436]	valid_0's auc: 0.90116	valid_0's binary_logloss: 0.178865
    [3437]	valid_0's auc: 0.901163	valid_0's binary_logloss: 0.178884
    [3438]	valid_0's auc: 0.901163	valid_0's binary_logloss: 0.178908
    [3439]	valid_0's auc: 0.901183	valid_0's binary_logloss: 0.17882
    [3440]	valid_0's auc: 0.901183	valid_0's binary_logloss: 0.178842
    [3441]	valid_0's auc: 0.901183	valid_0's binary_logloss: 0.178828
    [3442]	valid_0's auc: 0.901185	valid_0's binary_logloss: 0.178845
    [3443]	valid_0's auc: 0.901189	valid_0's binary_logloss: 0.178873
    [3444]	valid_0's auc: 0.901206	valid_0's binary_logloss: 0.17885
    [3445]	valid_0's auc: 0.901224	valid_0's binary_logloss: 0.178763
    [3446]	valid_0's auc: 0.901249	valid_0's binary_logloss: 0.178731
    [3447]	valid_0's auc: 0.901249	valid_0's binary_logloss: 0.178755
    [3448]	valid_0's auc: 0.901253	valid_0's binary_logloss: 0.17877
    [3449]	valid_0's auc: 0.901251	valid_0's binary_logloss: 0.178791
    [3450]	valid_0's auc: 0.901253	valid_0's binary_logloss: 0.178808
    [3451]	valid_0's auc: 0.901268	valid_0's binary_logloss: 0.178776
    [3452]	valid_0's auc: 0.901267	valid_0's binary_logloss: 0.178807
    [3453]	valid_0's auc: 0.901269	valid_0's binary_logloss: 0.178835
    [3454]	valid_0's auc: 0.901271	valid_0's binary_logloss: 0.178872
    [3455]	valid_0's auc: 0.901277	valid_0's binary_logloss: 0.178892
    [3456]	valid_0's auc: 0.901251	valid_0's binary_logloss: 0.178868
    [3457]	valid_0's auc: 0.901282	valid_0's binary_logloss: 0.178836
    [3458]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.178807
    [3459]	valid_0's auc: 0.901273	valid_0's binary_logloss: 0.178824
    [3460]	valid_0's auc: 0.901274	valid_0's binary_logloss: 0.178852
    [3461]	valid_0's auc: 0.901276	valid_0's binary_logloss: 0.178877
    [3462]	valid_0's auc: 0.901275	valid_0's binary_logloss: 0.178845
    [3463]	valid_0's auc: 0.901267	valid_0's binary_logloss: 0.17882
    [3464]	valid_0's auc: 0.901257	valid_0's binary_logloss: 0.178798
    [3465]	valid_0's auc: 0.901262	valid_0's binary_logloss: 0.178825
    [3466]	valid_0's auc: 0.901233	valid_0's binary_logloss: 0.17881
    [3467]	valid_0's auc: 0.901206	valid_0's binary_logloss: 0.178786
    [3468]	valid_0's auc: 0.901209	valid_0's binary_logloss: 0.178809
    [3469]	valid_0's auc: 0.901212	valid_0's binary_logloss: 0.17883
    [3470]	valid_0's auc: 0.901216	valid_0's binary_logloss: 0.178854
    [3471]	valid_0's auc: 0.901221	valid_0's binary_logloss: 0.178822
    [3472]	valid_0's auc: 0.90121	valid_0's binary_logloss: 0.178787
    [3473]	valid_0's auc: 0.901208	valid_0's binary_logloss: 0.178813
    [3474]	valid_0's auc: 0.901221	valid_0's binary_logloss: 0.178782
    [3475]	valid_0's auc: 0.901239	valid_0's binary_logloss: 0.178741
    [3476]	valid_0's auc: 0.90124	valid_0's binary_logloss: 0.178773
    [3477]	valid_0's auc: 0.901263	valid_0's binary_logloss: 0.178739
    [3478]	valid_0's auc: 0.901243	valid_0's binary_logloss: 0.178725
    [3479]	valid_0's auc: 0.901249	valid_0's binary_logloss: 0.178747
    [3480]	valid_0's auc: 0.901251	valid_0's binary_logloss: 0.178759
    [3481]	valid_0's auc: 0.901256	valid_0's binary_logloss: 0.178775
    [3482]	valid_0's auc: 0.901273	valid_0's binary_logloss: 0.178658
    [3483]	valid_0's auc: 0.901274	valid_0's binary_logloss: 0.178675
    [3484]	valid_0's auc: 0.901262	valid_0's binary_logloss: 0.178654
    [3485]	valid_0's auc: 0.901241	valid_0's binary_logloss: 0.178638
    [3486]	valid_0's auc: 0.901243	valid_0's binary_logloss: 0.178664
    [3487]	valid_0's auc: 0.90124	valid_0's binary_logloss: 0.178687
    [3488]	valid_0's auc: 0.901231	valid_0's binary_logloss: 0.178661
    [3489]	valid_0's auc: 0.901231	valid_0's binary_logloss: 0.178687
    [3490]	valid_0's auc: 0.901212	valid_0's binary_logloss: 0.178655
    [3491]	valid_0's auc: 0.901211	valid_0's binary_logloss: 0.17868
    [3492]	valid_0's auc: 0.901212	valid_0's binary_logloss: 0.178707
    [3493]	valid_0's auc: 0.901239	valid_0's binary_logloss: 0.178664
    [3494]	valid_0's auc: 0.901239	valid_0's binary_logloss: 0.178687
    [3495]	valid_0's auc: 0.90124	valid_0's binary_logloss: 0.178708
    [3496]	valid_0's auc: 0.901246	valid_0's binary_logloss: 0.178731
    [3497]	valid_0's auc: 0.90123	valid_0's binary_logloss: 0.178713
    [3498]	valid_0's auc: 0.901249	valid_0's binary_logloss: 0.178659
    [3499]	valid_0's auc: 0.901253	valid_0's binary_logloss: 0.178679
    [3500]	valid_0's auc: 0.901255	valid_0's binary_logloss: 0.178696
    [3501]	valid_0's auc: 0.901255	valid_0's binary_logloss: 0.178726
    [3502]	valid_0's auc: 0.901254	valid_0's binary_logloss: 0.178751
    [3503]	valid_0's auc: 0.901253	valid_0's binary_logloss: 0.178777
    [3504]	valid_0's auc: 0.901243	valid_0's binary_logloss: 0.178753
    [3505]	valid_0's auc: 0.901246	valid_0's binary_logloss: 0.178775
    [3506]	valid_0's auc: 0.901246	valid_0's binary_logloss: 0.178717
    [3507]	valid_0's auc: 0.901244	valid_0's binary_logloss: 0.178734
    [3508]	valid_0's auc: 0.901262	valid_0's binary_logloss: 0.178703
    [3509]	valid_0's auc: 0.901229	valid_0's binary_logloss: 0.178671
    [3510]	valid_0's auc: 0.901229	valid_0's binary_logloss: 0.178701
    [3511]	valid_0's auc: 0.901228	valid_0's binary_logloss: 0.178719
    [3512]	valid_0's auc: 0.90126	valid_0's binary_logloss: 0.178677
    [3513]	valid_0's auc: 0.901257	valid_0's binary_logloss: 0.178568
    [3514]	valid_0's auc: 0.90126	valid_0's binary_logloss: 0.178515
    [3515]	valid_0's auc: 0.901262	valid_0's binary_logloss: 0.178539
    [3516]	valid_0's auc: 0.901265	valid_0's binary_logloss: 0.178561
    [3517]	valid_0's auc: 0.901247	valid_0's binary_logloss: 0.178541
    [3518]	valid_0's auc: 0.901258	valid_0's binary_logloss: 0.178486
    [3519]	valid_0's auc: 0.90126	valid_0's binary_logloss: 0.178506
    [3520]	valid_0's auc: 0.901281	valid_0's binary_logloss: 0.178445
    [3521]	valid_0's auc: 0.901277	valid_0's binary_logloss: 0.178407
    [3522]	valid_0's auc: 0.901277	valid_0's binary_logloss: 0.178426
    [3523]	valid_0's auc: 0.901325	valid_0's binary_logloss: 0.178358
    [3524]	valid_0's auc: 0.901355	valid_0's binary_logloss: 0.178331
    [3525]	valid_0's auc: 0.901355	valid_0's binary_logloss: 0.178355
    [3526]	valid_0's auc: 0.901355	valid_0's binary_logloss: 0.178375
    [3527]	valid_0's auc: 0.901388	valid_0's binary_logloss: 0.178258
    [3528]	valid_0's auc: 0.90139	valid_0's binary_logloss: 0.178275
    [3529]	valid_0's auc: 0.901385	valid_0's binary_logloss: 0.178261
    [3530]	valid_0's auc: 0.901358	valid_0's binary_logloss: 0.178239
    [3531]	valid_0's auc: 0.901348	valid_0's binary_logloss: 0.178213
    [3532]	valid_0's auc: 0.901375	valid_0's binary_logloss: 0.178161
    [3533]	valid_0's auc: 0.901377	valid_0's binary_logloss: 0.178176
    [3534]	valid_0's auc: 0.901376	valid_0's binary_logloss: 0.1782
    [3535]	valid_0's auc: 0.901374	valid_0's binary_logloss: 0.17817
    [3536]	valid_0's auc: 0.901378	valid_0's binary_logloss: 0.178193
    [3537]	valid_0's auc: 0.901368	valid_0's binary_logloss: 0.178165
    [3538]	valid_0's auc: 0.90136	valid_0's binary_logloss: 0.178129
    [3539]	valid_0's auc: 0.901349	valid_0's binary_logloss: 0.178048
    [3540]	valid_0's auc: 0.901356	valid_0's binary_logloss: 0.178024
    [3541]	valid_0's auc: 0.901348	valid_0's binary_logloss: 0.178001
    [3542]	valid_0's auc: 0.901363	valid_0's binary_logloss: 0.177948
    [3543]	valid_0's auc: 0.901386	valid_0's binary_logloss: 0.177916
    [3544]	valid_0's auc: 0.901418	valid_0's binary_logloss: 0.177834
    [3545]	valid_0's auc: 0.901422	valid_0's binary_logloss: 0.177855
    [3546]	valid_0's auc: 0.90142	valid_0's binary_logloss: 0.17788
    [3547]	valid_0's auc: 0.901431	valid_0's binary_logloss: 0.177844
    [3548]	valid_0's auc: 0.901432	valid_0's binary_logloss: 0.177826
    [3549]	valid_0's auc: 0.901434	valid_0's binary_logloss: 0.177851
    [3550]	valid_0's auc: 0.901458	valid_0's binary_logloss: 0.177797
    [3551]	valid_0's auc: 0.901459	valid_0's binary_logloss: 0.177815
    [3552]	valid_0's auc: 0.901446	valid_0's binary_logloss: 0.17778
    [3553]	valid_0's auc: 0.901468	valid_0's binary_logloss: 0.177695
    [3554]	valid_0's auc: 0.901465	valid_0's binary_logloss: 0.177674
    [3555]	valid_0's auc: 0.901467	valid_0's binary_logloss: 0.177664
    [3556]	valid_0's auc: 0.901464	valid_0's binary_logloss: 0.177681
    [3557]	valid_0's auc: 0.901461	valid_0's binary_logloss: 0.177703
    [3558]	valid_0's auc: 0.901497	valid_0's binary_logloss: 0.177626
    [3559]	valid_0's auc: 0.901503	valid_0's binary_logloss: 0.177595
    [3560]	valid_0's auc: 0.901509	valid_0's binary_logloss: 0.177577
    [3561]	valid_0's auc: 0.901494	valid_0's binary_logloss: 0.177561
    [3562]	valid_0's auc: 0.901507	valid_0's binary_logloss: 0.177451
    [3563]	valid_0's auc: 0.901511	valid_0's binary_logloss: 0.177372
    [3564]	valid_0's auc: 0.901516	valid_0's binary_logloss: 0.177396
    [3565]	valid_0's auc: 0.901521	valid_0's binary_logloss: 0.177418
    [3566]	valid_0's auc: 0.901528	valid_0's binary_logloss: 0.177389
    [3567]	valid_0's auc: 0.901528	valid_0's binary_logloss: 0.177367
    [3568]	valid_0's auc: 0.901528	valid_0's binary_logloss: 0.177385
    [3569]	valid_0's auc: 0.901521	valid_0's binary_logloss: 0.17736
    [3570]	valid_0's auc: 0.901525	valid_0's binary_logloss: 0.177378
    [3571]	valid_0's auc: 0.901523	valid_0's binary_logloss: 0.177357
    [3572]	valid_0's auc: 0.901523	valid_0's binary_logloss: 0.177335
    [3573]	valid_0's auc: 0.901523	valid_0's binary_logloss: 0.177356
    [3574]	valid_0's auc: 0.901509	valid_0's binary_logloss: 0.177327
    [3575]	valid_0's auc: 0.901533	valid_0's binary_logloss: 0.177254
    [3576]	valid_0's auc: 0.901512	valid_0's binary_logloss: 0.177239
    [3577]	valid_0's auc: 0.901517	valid_0's binary_logloss: 0.177261
    [3578]	valid_0's auc: 0.901519	valid_0's binary_logloss: 0.177249
    [3579]	valid_0's auc: 0.901524	valid_0's binary_logloss: 0.177272
    [3580]	valid_0's auc: 0.901525	valid_0's binary_logloss: 0.177287
    [3581]	valid_0's auc: 0.901518	valid_0's binary_logloss: 0.177264
    [3582]	valid_0's auc: 0.901521	valid_0's binary_logloss: 0.177283
    [3583]	valid_0's auc: 0.9015	valid_0's binary_logloss: 0.177249
    [3584]	valid_0's auc: 0.9015	valid_0's binary_logloss: 0.177272
    [3585]	valid_0's auc: 0.901488	valid_0's binary_logloss: 0.177262
    [3586]	valid_0's auc: 0.901488	valid_0's binary_logloss: 0.177279
    [3587]	valid_0's auc: 0.901485	valid_0's binary_logloss: 0.177307
    [3588]	valid_0's auc: 0.901481	valid_0's binary_logloss: 0.177295
    [3589]	valid_0's auc: 0.901485	valid_0's binary_logloss: 0.177272
    [3590]	valid_0's auc: 0.90149	valid_0's binary_logloss: 0.1773
    [3591]	valid_0's auc: 0.901501	valid_0's binary_logloss: 0.177264
    [3592]	valid_0's auc: 0.901497	valid_0's binary_logloss: 0.177286
    [3593]	valid_0's auc: 0.901495	valid_0's binary_logloss: 0.177299
    [3594]	valid_0's auc: 0.901487	valid_0's binary_logloss: 0.177281
    [3595]	valid_0's auc: 0.901491	valid_0's binary_logloss: 0.177304
    [3596]	valid_0's auc: 0.901491	valid_0's binary_logloss: 0.177329
    [3597]	valid_0's auc: 0.901494	valid_0's binary_logloss: 0.177345
    [3598]	valid_0's auc: 0.901493	valid_0's binary_logloss: 0.177302
    [3599]	valid_0's auc: 0.901484	valid_0's binary_logloss: 0.177282
    [3600]	valid_0's auc: 0.901487	valid_0's binary_logloss: 0.177264
    [3601]	valid_0's auc: 0.901486	valid_0's binary_logloss: 0.177234
    [3602]	valid_0's auc: 0.901517	valid_0's binary_logloss: 0.177175
    [3603]	valid_0's auc: 0.901521	valid_0's binary_logloss: 0.177197
    [3604]	valid_0's auc: 0.901521	valid_0's binary_logloss: 0.177217
    [3605]	valid_0's auc: 0.901556	valid_0's binary_logloss: 0.177161
    [3606]	valid_0's auc: 0.90156	valid_0's binary_logloss: 0.177176
    [3607]	valid_0's auc: 0.901564	valid_0's binary_logloss: 0.177193
    [3608]	valid_0's auc: 0.901566	valid_0's binary_logloss: 0.177219
    [3609]	valid_0's auc: 0.901541	valid_0's binary_logloss: 0.177207
    [3610]	valid_0's auc: 0.901546	valid_0's binary_logloss: 0.177188
    [3611]	valid_0's auc: 0.901561	valid_0's binary_logloss: 0.177155
    [3612]	valid_0's auc: 0.901562	valid_0's binary_logloss: 0.177172
    [3613]	valid_0's auc: 0.901563	valid_0's binary_logloss: 0.177193
    [3614]	valid_0's auc: 0.901561	valid_0's binary_logloss: 0.177209
    [3615]	valid_0's auc: 0.901566	valid_0's binary_logloss: 0.177186
    [3616]	valid_0's auc: 0.90156	valid_0's binary_logloss: 0.17717
    [3617]	valid_0's auc: 0.901562	valid_0's binary_logloss: 0.177191
    [3618]	valid_0's auc: 0.901576	valid_0's binary_logloss: 0.177166
    [3619]	valid_0's auc: 0.901585	valid_0's binary_logloss: 0.177137
    [3620]	valid_0's auc: 0.901587	valid_0's binary_logloss: 0.177148
    [3621]	valid_0's auc: 0.901579	valid_0's binary_logloss: 0.17713
    [3622]	valid_0's auc: 0.901582	valid_0's binary_logloss: 0.177156
    [3623]	valid_0's auc: 0.901585	valid_0's binary_logloss: 0.177175
    [3624]	valid_0's auc: 0.901631	valid_0's binary_logloss: 0.177042
    [3625]	valid_0's auc: 0.901622	valid_0's binary_logloss: 0.177022
    [3626]	valid_0's auc: 0.901629	valid_0's binary_logloss: 0.17699
    [3627]	valid_0's auc: 0.901684	valid_0's binary_logloss: 0.176943
    [3628]	valid_0's auc: 0.901678	valid_0's binary_logloss: 0.176875
    [3629]	valid_0's auc: 0.901683	valid_0's binary_logloss: 0.176852
    [3630]	valid_0's auc: 0.901682	valid_0's binary_logloss: 0.176868
    [3631]	valid_0's auc: 0.901699	valid_0's binary_logloss: 0.176849
    [3632]	valid_0's auc: 0.901703	valid_0's binary_logloss: 0.176865
    [3633]	valid_0's auc: 0.901723	valid_0's binary_logloss: 0.17681
    [3634]	valid_0's auc: 0.901727	valid_0's binary_logloss: 0.176832
    [3635]	valid_0's auc: 0.901728	valid_0's binary_logloss: 0.176847
    [3636]	valid_0's auc: 0.901729	valid_0's binary_logloss: 0.176865
    [3637]	valid_0's auc: 0.901724	valid_0's binary_logloss: 0.176853
    [3638]	valid_0's auc: 0.901724	valid_0's binary_logloss: 0.176871
    [3639]	valid_0's auc: 0.901724	valid_0's binary_logloss: 0.176885
    [3640]	valid_0's auc: 0.90172	valid_0's binary_logloss: 0.176906
    [3641]	valid_0's auc: 0.901721	valid_0's binary_logloss: 0.176924
    [3642]	valid_0's auc: 0.901706	valid_0's binary_logloss: 0.176903
    [3643]	valid_0's auc: 0.901708	valid_0's binary_logloss: 0.17692
    [3644]	valid_0's auc: 0.901707	valid_0's binary_logloss: 0.176941
    [3645]	valid_0's auc: 0.901706	valid_0's binary_logloss: 0.17696
    [3646]	valid_0's auc: 0.901702	valid_0's binary_logloss: 0.17694
    [3647]	valid_0's auc: 0.901746	valid_0's binary_logloss: 0.176899
    [3648]	valid_0's auc: 0.901751	valid_0's binary_logloss: 0.176885
    [3649]	valid_0's auc: 0.90175	valid_0's binary_logloss: 0.176906
    [3650]	valid_0's auc: 0.901751	valid_0's binary_logloss: 0.176923
    [3651]	valid_0's auc: 0.901754	valid_0's binary_logloss: 0.176939
    [3652]	valid_0's auc: 0.901752	valid_0's binary_logloss: 0.176964
    [3653]	valid_0's auc: 0.901805	valid_0's binary_logloss: 0.176886
    [3654]	valid_0's auc: 0.90181	valid_0's binary_logloss: 0.176864
    [3655]	valid_0's auc: 0.901815	valid_0's binary_logloss: 0.176877
    [3656]	valid_0's auc: 0.901815	valid_0's binary_logloss: 0.176898
    [3657]	valid_0's auc: 0.901821	valid_0's binary_logloss: 0.176923
    [3658]	valid_0's auc: 0.901827	valid_0's binary_logloss: 0.176937
    [3659]	valid_0's auc: 0.901819	valid_0's binary_logloss: 0.176954
    [3660]	valid_0's auc: 0.901847	valid_0's binary_logloss: 0.176855
    [3661]	valid_0's auc: 0.901871	valid_0's binary_logloss: 0.176797
    [3662]	valid_0's auc: 0.901859	valid_0's binary_logloss: 0.176769
    [3663]	valid_0's auc: 0.901861	valid_0's binary_logloss: 0.176787
    [3664]	valid_0's auc: 0.901866	valid_0's binary_logloss: 0.176731
    [3665]	valid_0's auc: 0.901871	valid_0's binary_logloss: 0.176751
    [3666]	valid_0's auc: 0.901871	valid_0's binary_logloss: 0.176769
    [3667]	valid_0's auc: 0.901889	valid_0's binary_logloss: 0.17669
    [3668]	valid_0's auc: 0.901897	valid_0's binary_logloss: 0.176588
    [3669]	valid_0's auc: 0.901902	valid_0's binary_logloss: 0.176603
    [3670]	valid_0's auc: 0.901894	valid_0's binary_logloss: 0.176587
    [3671]	valid_0's auc: 0.901901	valid_0's binary_logloss: 0.17655
    [3672]	valid_0's auc: 0.901901	valid_0's binary_logloss: 0.176567
    [3673]	valid_0's auc: 0.901898	valid_0's binary_logloss: 0.176588
    [3674]	valid_0's auc: 0.901912	valid_0's binary_logloss: 0.176543
    [3675]	valid_0's auc: 0.901913	valid_0's binary_logloss: 0.176562
    [3676]	valid_0's auc: 0.901917	valid_0's binary_logloss: 0.176586
    [3677]	valid_0's auc: 0.901914	valid_0's binary_logloss: 0.176577
    [3678]	valid_0's auc: 0.901918	valid_0's binary_logloss: 0.176608
    [3679]	valid_0's auc: 0.901903	valid_0's binary_logloss: 0.176599
    [3680]	valid_0's auc: 0.901903	valid_0's binary_logloss: 0.176565
    [3681]	valid_0's auc: 0.901915	valid_0's binary_logloss: 0.176499
    [3682]	valid_0's auc: 0.901928	valid_0's binary_logloss: 0.176421
    [3683]	valid_0's auc: 0.901908	valid_0's binary_logloss: 0.176403
    [3684]	valid_0's auc: 0.901909	valid_0's binary_logloss: 0.176425
    [3685]	valid_0's auc: 0.901909	valid_0's binary_logloss: 0.176396
    [3686]	valid_0's auc: 0.901919	valid_0's binary_logloss: 0.17633
    [3687]	valid_0's auc: 0.901918	valid_0's binary_logloss: 0.176349
    [3688]	valid_0's auc: 0.90191	valid_0's binary_logloss: 0.176322
    [3689]	valid_0's auc: 0.901914	valid_0's binary_logloss: 0.176343
    [3690]	valid_0's auc: 0.901917	valid_0's binary_logloss: 0.176276
    [3691]	valid_0's auc: 0.901917	valid_0's binary_logloss: 0.176252
    [3692]	valid_0's auc: 0.901922	valid_0's binary_logloss: 0.176272
    [3693]	valid_0's auc: 0.90193	valid_0's binary_logloss: 0.176237
    [3694]	valid_0's auc: 0.901934	valid_0's binary_logloss: 0.176259
    [3695]	valid_0's auc: 0.901936	valid_0's binary_logloss: 0.176278
    [3696]	valid_0's auc: 0.901962	valid_0's binary_logloss: 0.176204
    [3697]	valid_0's auc: 0.901963	valid_0's binary_logloss: 0.176223
    [3698]	valid_0's auc: 0.902005	valid_0's binary_logloss: 0.176167
    [3699]	valid_0's auc: 0.902014	valid_0's binary_logloss: 0.176103
    [3700]	valid_0's auc: 0.902015	valid_0's binary_logloss: 0.176124
    [3701]	valid_0's auc: 0.902017	valid_0's binary_logloss: 0.176051
    [3702]	valid_0's auc: 0.902039	valid_0's binary_logloss: 0.175993
    [3703]	valid_0's auc: 0.902063	valid_0's binary_logloss: 0.175967
    [3704]	valid_0's auc: 0.902059	valid_0's binary_logloss: 0.175946
    [3705]	valid_0's auc: 0.902041	valid_0's binary_logloss: 0.175904
    [3706]	valid_0's auc: 0.902057	valid_0's binary_logloss: 0.175834
    [3707]	valid_0's auc: 0.902063	valid_0's binary_logloss: 0.175789
    [3708]	valid_0's auc: 0.902038	valid_0's binary_logloss: 0.175777
    [3709]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.175793
    [3710]	valid_0's auc: 0.902033	valid_0's binary_logloss: 0.175687
    [3711]	valid_0's auc: 0.902036	valid_0's binary_logloss: 0.175666
    [3712]	valid_0's auc: 0.902052	valid_0's binary_logloss: 0.175628
    [3713]	valid_0's auc: 0.902045	valid_0's binary_logloss: 0.175609
    [3714]	valid_0's auc: 0.902044	valid_0's binary_logloss: 0.175588
    [3715]	valid_0's auc: 0.902043	valid_0's binary_logloss: 0.175606
    [3716]	valid_0's auc: 0.902054	valid_0's binary_logloss: 0.175578
    [3717]	valid_0's auc: 0.902056	valid_0's binary_logloss: 0.175591
    [3718]	valid_0's auc: 0.90206	valid_0's binary_logloss: 0.175607
    [3719]	valid_0's auc: 0.902063	valid_0's binary_logloss: 0.175626
    [3720]	valid_0's auc: 0.902065	valid_0's binary_logloss: 0.175594
    [3721]	valid_0's auc: 0.902066	valid_0's binary_logloss: 0.175576
    [3722]	valid_0's auc: 0.902052	valid_0's binary_logloss: 0.175561
    [3723]	valid_0's auc: 0.902054	valid_0's binary_logloss: 0.17558
    [3724]	valid_0's auc: 0.90203	valid_0's binary_logloss: 0.175562
    [3725]	valid_0's auc: 0.902032	valid_0's binary_logloss: 0.175577
    [3726]	valid_0's auc: 0.902018	valid_0's binary_logloss: 0.175551
    [3727]	valid_0's auc: 0.902021	valid_0's binary_logloss: 0.175576
    [3728]	valid_0's auc: 0.902022	valid_0's binary_logloss: 0.175556
    [3729]	valid_0's auc: 0.902023	valid_0's binary_logloss: 0.175571
    [3730]	valid_0's auc: 0.902026	valid_0's binary_logloss: 0.175587
    [3731]	valid_0's auc: 0.90203	valid_0's binary_logloss: 0.175557
    [3732]	valid_0's auc: 0.902023	valid_0's binary_logloss: 0.175527
    [3733]	valid_0's auc: 0.90203	valid_0's binary_logloss: 0.175507
    [3734]	valid_0's auc: 0.902031	valid_0's binary_logloss: 0.175523
    [3735]	valid_0's auc: 0.902051	valid_0's binary_logloss: 0.175415
    [3736]	valid_0's auc: 0.90205	valid_0's binary_logloss: 0.17543
    [3737]	valid_0's auc: 0.902069	valid_0's binary_logloss: 0.17537
    [3738]	valid_0's auc: 0.902068	valid_0's binary_logloss: 0.175395
    [3739]	valid_0's auc: 0.90207	valid_0's binary_logloss: 0.175411
    [3740]	valid_0's auc: 0.902082	valid_0's binary_logloss: 0.175389
    [3741]	valid_0's auc: 0.902101	valid_0's binary_logloss: 0.175356
    [3742]	valid_0's auc: 0.902095	valid_0's binary_logloss: 0.175342
    [3743]	valid_0's auc: 0.902095	valid_0's binary_logloss: 0.175354
    [3744]	valid_0's auc: 0.902102	valid_0's binary_logloss: 0.17537
    [3745]	valid_0's auc: 0.902107	valid_0's binary_logloss: 0.175384
    [3746]	valid_0's auc: 0.902111	valid_0's binary_logloss: 0.175405
    [3747]	valid_0's auc: 0.902101	valid_0's binary_logloss: 0.175384
    [3748]	valid_0's auc: 0.9021	valid_0's binary_logloss: 0.175404
    [3749]	valid_0's auc: 0.902102	valid_0's binary_logloss: 0.175422
    [3750]	valid_0's auc: 0.902107	valid_0's binary_logloss: 0.175399
    [3751]	valid_0's auc: 0.902086	valid_0's binary_logloss: 0.175385
    [3752]	valid_0's auc: 0.902097	valid_0's binary_logloss: 0.175355
    [3753]	valid_0's auc: 0.902099	valid_0's binary_logloss: 0.175371
    [3754]	valid_0's auc: 0.902102	valid_0's binary_logloss: 0.17536
    [3755]	valid_0's auc: 0.902102	valid_0's binary_logloss: 0.175379
    [3756]	valid_0's auc: 0.902117	valid_0's binary_logloss: 0.175349
    [3757]	valid_0's auc: 0.902101	valid_0's binary_logloss: 0.175325
    [3758]	valid_0's auc: 0.9021	valid_0's binary_logloss: 0.175345
    [3759]	valid_0's auc: 0.902105	valid_0's binary_logloss: 0.175363
    [3760]	valid_0's auc: 0.902105	valid_0's binary_logloss: 0.175379
    [3761]	valid_0's auc: 0.902106	valid_0's binary_logloss: 0.175392
    [3762]	valid_0's auc: 0.90212	valid_0's binary_logloss: 0.175372
    [3763]	valid_0's auc: 0.902117	valid_0's binary_logloss: 0.175391
    [3764]	valid_0's auc: 0.902119	valid_0's binary_logloss: 0.175405
    [3765]	valid_0's auc: 0.902107	valid_0's binary_logloss: 0.175314
    [3766]	valid_0's auc: 0.902105	valid_0's binary_logloss: 0.175328
    [3767]	valid_0's auc: 0.902121	valid_0's binary_logloss: 0.175284
    [3768]	valid_0's auc: 0.902137	valid_0's binary_logloss: 0.175272
    [3769]	valid_0's auc: 0.902146	valid_0's binary_logloss: 0.175207
    [3770]	valid_0's auc: 0.902146	valid_0's binary_logloss: 0.175223
    [3771]	valid_0's auc: 0.902146	valid_0's binary_logloss: 0.175243
    [3772]	valid_0's auc: 0.902141	valid_0's binary_logloss: 0.175264
    [3773]	valid_0's auc: 0.902141	valid_0's binary_logloss: 0.175282
    [3774]	valid_0's auc: 0.902144	valid_0's binary_logloss: 0.175266
    [3775]	valid_0's auc: 0.902156	valid_0's binary_logloss: 0.1752
    [3776]	valid_0's auc: 0.902156	valid_0's binary_logloss: 0.175226
    [3777]	valid_0's auc: 0.902158	valid_0's binary_logloss: 0.175235
    [3778]	valid_0's auc: 0.902158	valid_0's binary_logloss: 0.17525
    [3779]	valid_0's auc: 0.90216	valid_0's binary_logloss: 0.175231
    [3780]	valid_0's auc: 0.902164	valid_0's binary_logloss: 0.175208
    [3781]	valid_0's auc: 0.902181	valid_0's binary_logloss: 0.175167
    [3782]	valid_0's auc: 0.902146	valid_0's binary_logloss: 0.175162
    [3783]	valid_0's auc: 0.902131	valid_0's binary_logloss: 0.175141
    [3784]	valid_0's auc: 0.902132	valid_0's binary_logloss: 0.175158
    [3785]	valid_0's auc: 0.902137	valid_0's binary_logloss: 0.175144
    [3786]	valid_0's auc: 0.90216	valid_0's binary_logloss: 0.175076
    [3787]	valid_0's auc: 0.902164	valid_0's binary_logloss: 0.17509
    [3788]	valid_0's auc: 0.902166	valid_0's binary_logloss: 0.175109
    [3789]	valid_0's auc: 0.902168	valid_0's binary_logloss: 0.175128
    [3790]	valid_0's auc: 0.902168	valid_0's binary_logloss: 0.175147
    [3791]	valid_0's auc: 0.902162	valid_0's binary_logloss: 0.175131
    [3792]	valid_0's auc: 0.902144	valid_0's binary_logloss: 0.175119
    [3793]	valid_0's auc: 0.902162	valid_0's binary_logloss: 0.175093
    [3794]	valid_0's auc: 0.902155	valid_0's binary_logloss: 0.175081
    [3795]	valid_0's auc: 0.902153	valid_0's binary_logloss: 0.175095
    [3796]	valid_0's auc: 0.902155	valid_0's binary_logloss: 0.175073
    [3797]	valid_0's auc: 0.902145	valid_0's binary_logloss: 0.175056
    [3798]	valid_0's auc: 0.902149	valid_0's binary_logloss: 0.175068
    [3799]	valid_0's auc: 0.902157	valid_0's binary_logloss: 0.175033
    [3800]	valid_0's auc: 0.90216	valid_0's binary_logloss: 0.175012
    [3801]	valid_0's auc: 0.90219	valid_0's binary_logloss: 0.174974
    [3802]	valid_0's auc: 0.90219	valid_0's binary_logloss: 0.174991
    [3803]	valid_0's auc: 0.902202	valid_0's binary_logloss: 0.174917
    [3804]	valid_0's auc: 0.902189	valid_0's binary_logloss: 0.174903
    [3805]	valid_0's auc: 0.902183	valid_0's binary_logloss: 0.174882
    [3806]	valid_0's auc: 0.90219	valid_0's binary_logloss: 0.174852
    [3807]	valid_0's auc: 0.902187	valid_0's binary_logloss: 0.17487
    [3808]	valid_0's auc: 0.902191	valid_0's binary_logloss: 0.174889
    [3809]	valid_0's auc: 0.902191	valid_0's binary_logloss: 0.174903
    [3810]	valid_0's auc: 0.902229	valid_0's binary_logloss: 0.174864
    [3811]	valid_0's auc: 0.902223	valid_0's binary_logloss: 0.174838
    [3812]	valid_0's auc: 0.902245	valid_0's binary_logloss: 0.174773
    [3813]	valid_0's auc: 0.902249	valid_0's binary_logloss: 0.174685
    [3814]	valid_0's auc: 0.902251	valid_0's binary_logloss: 0.174697
    [3815]	valid_0's auc: 0.902255	valid_0's binary_logloss: 0.174709
    [3816]	valid_0's auc: 0.90224	valid_0's binary_logloss: 0.174689
    [3817]	valid_0's auc: 0.902236	valid_0's binary_logloss: 0.174701
    [3818]	valid_0's auc: 0.90224	valid_0's binary_logloss: 0.174673
    [3819]	valid_0's auc: 0.902239	valid_0's binary_logloss: 0.174656
    [3820]	valid_0's auc: 0.90224	valid_0's binary_logloss: 0.174674
    [3821]	valid_0's auc: 0.902238	valid_0's binary_logloss: 0.174688
    [3822]	valid_0's auc: 0.902243	valid_0's binary_logloss: 0.17471
    [3823]	valid_0's auc: 0.902244	valid_0's binary_logloss: 0.174723
    [3824]	valid_0's auc: 0.902239	valid_0's binary_logloss: 0.174704
    [3825]	valid_0's auc: 0.902242	valid_0's binary_logloss: 0.174724
    [3826]	valid_0's auc: 0.90225	valid_0's binary_logloss: 0.174706
    [3827]	valid_0's auc: 0.902286	valid_0's binary_logloss: 0.174664
    [3828]	valid_0's auc: 0.902304	valid_0's binary_logloss: 0.17462
    [3829]	valid_0's auc: 0.902304	valid_0's binary_logloss: 0.174639
    [3830]	valid_0's auc: 0.902305	valid_0's binary_logloss: 0.174649
    [3831]	valid_0's auc: 0.90227	valid_0's binary_logloss: 0.17464
    [3832]	valid_0's auc: 0.902268	valid_0's binary_logloss: 0.174588
    [3833]	valid_0's auc: 0.902272	valid_0's binary_logloss: 0.174604
    [3834]	valid_0's auc: 0.902272	valid_0's binary_logloss: 0.174618
    [3835]	valid_0's auc: 0.902268	valid_0's binary_logloss: 0.174605
    [3836]	valid_0's auc: 0.902272	valid_0's binary_logloss: 0.174619
    [3837]	valid_0's auc: 0.902281	valid_0's binary_logloss: 0.174591
    [3838]	valid_0's auc: 0.902284	valid_0's binary_logloss: 0.174608
    [3839]	valid_0's auc: 0.902288	valid_0's binary_logloss: 0.174588
    [3840]	valid_0's auc: 0.90229	valid_0's binary_logloss: 0.17461
    [3841]	valid_0's auc: 0.902277	valid_0's binary_logloss: 0.174592
    [3842]	valid_0's auc: 0.902282	valid_0's binary_logloss: 0.174615
    [3843]	valid_0's auc: 0.902346	valid_0's binary_logloss: 0.174565
    [3844]	valid_0's auc: 0.902377	valid_0's binary_logloss: 0.17453
    [3845]	valid_0's auc: 0.902375	valid_0's binary_logloss: 0.174514
    [3846]	valid_0's auc: 0.902378	valid_0's binary_logloss: 0.174533
    [3847]	valid_0's auc: 0.902371	valid_0's binary_logloss: 0.174522
    [3848]	valid_0's auc: 0.902385	valid_0's binary_logloss: 0.174502
    [3849]	valid_0's auc: 0.902394	valid_0's binary_logloss: 0.174456
    [3850]	valid_0's auc: 0.902393	valid_0's binary_logloss: 0.174472
    [3851]	valid_0's auc: 0.902402	valid_0's binary_logloss: 0.174491
    [3852]	valid_0's auc: 0.902403	valid_0's binary_logloss: 0.174514
    [3853]	valid_0's auc: 0.902403	valid_0's binary_logloss: 0.174531
    [3854]	valid_0's auc: 0.902406	valid_0's binary_logloss: 0.174548
    [3855]	valid_0's auc: 0.902393	valid_0's binary_logloss: 0.174518
    [3856]	valid_0's auc: 0.902398	valid_0's binary_logloss: 0.174536
    [3857]	valid_0's auc: 0.902399	valid_0's binary_logloss: 0.174551
    [3858]	valid_0's auc: 0.902396	valid_0's binary_logloss: 0.174531
    [3859]	valid_0's auc: 0.902398	valid_0's binary_logloss: 0.174551
    [3860]	valid_0's auc: 0.90239	valid_0's binary_logloss: 0.174529
    [3861]	valid_0's auc: 0.902413	valid_0's binary_logloss: 0.174489
    [3862]	valid_0's auc: 0.902414	valid_0's binary_logloss: 0.1745
    [3863]	valid_0's auc: 0.902423	valid_0's binary_logloss: 0.174523
    [3864]	valid_0's auc: 0.902438	valid_0's binary_logloss: 0.174506
    [3865]	valid_0's auc: 0.902438	valid_0's binary_logloss: 0.174523
    [3866]	valid_0's auc: 0.902436	valid_0's binary_logloss: 0.174539
    [3867]	valid_0's auc: 0.902446	valid_0's binary_logloss: 0.174567
    [3868]	valid_0's auc: 0.90248	valid_0's binary_logloss: 0.174523
    [3869]	valid_0's auc: 0.902475	valid_0's binary_logloss: 0.17454
    [3870]	valid_0's auc: 0.902466	valid_0's binary_logloss: 0.174521
    [3871]	valid_0's auc: 0.902467	valid_0's binary_logloss: 0.17454
    [3872]	valid_0's auc: 0.902484	valid_0's binary_logloss: 0.174523
    [3873]	valid_0's auc: 0.902485	valid_0's binary_logloss: 0.174545
    [3874]	valid_0's auc: 0.902483	valid_0's binary_logloss: 0.174529
    [3875]	valid_0's auc: 0.90249	valid_0's binary_logloss: 0.174549
    [3876]	valid_0's auc: 0.902516	valid_0's binary_logloss: 0.174496
    [3877]	valid_0's auc: 0.902522	valid_0's binary_logloss: 0.174412
    [3878]	valid_0's auc: 0.902535	valid_0's binary_logloss: 0.174386
    [3879]	valid_0's auc: 0.902526	valid_0's binary_logloss: 0.174368
    [3880]	valid_0's auc: 0.902527	valid_0's binary_logloss: 0.174381
    [3881]	valid_0's auc: 0.902528	valid_0's binary_logloss: 0.174395
    [3882]	valid_0's auc: 0.902539	valid_0's binary_logloss: 0.174305
    [3883]	valid_0's auc: 0.902546	valid_0's binary_logloss: 0.174288
    [3884]	valid_0's auc: 0.90255	valid_0's binary_logloss: 0.174306
    [3885]	valid_0's auc: 0.902543	valid_0's binary_logloss: 0.17429
    [3886]	valid_0's auc: 0.902548	valid_0's binary_logloss: 0.174257
    [3887]	valid_0's auc: 0.902546	valid_0's binary_logloss: 0.174236
    [3888]	valid_0's auc: 0.902545	valid_0's binary_logloss: 0.174255
    [3889]	valid_0's auc: 0.902524	valid_0's binary_logloss: 0.174235
    [3890]	valid_0's auc: 0.902525	valid_0's binary_logloss: 0.174249
    [3891]	valid_0's auc: 0.90251	valid_0's binary_logloss: 0.17423
    [3892]	valid_0's auc: 0.902508	valid_0's binary_logloss: 0.174248
    [3893]	valid_0's auc: 0.902526	valid_0's binary_logloss: 0.17419
    [3894]	valid_0's auc: 0.902555	valid_0's binary_logloss: 0.174098
    [3895]	valid_0's auc: 0.902558	valid_0's binary_logloss: 0.174115
    [3896]	valid_0's auc: 0.902531	valid_0's binary_logloss: 0.174106
    [3897]	valid_0's auc: 0.902523	valid_0's binary_logloss: 0.174089
    [3898]	valid_0's auc: 0.90252	valid_0's binary_logloss: 0.174111
    [3899]	valid_0's auc: 0.902524	valid_0's binary_logloss: 0.174132
    [3900]	valid_0's auc: 0.902529	valid_0's binary_logloss: 0.174152
    [3901]	valid_0's auc: 0.90253	valid_0's binary_logloss: 0.174136
    [3902]	valid_0's auc: 0.902527	valid_0's binary_logloss: 0.174126
    [3903]	valid_0's auc: 0.902539	valid_0's binary_logloss: 0.174104
    [3904]	valid_0's auc: 0.902548	valid_0's binary_logloss: 0.174093
    [3905]	valid_0's auc: 0.902548	valid_0's binary_logloss: 0.174109
    [3906]	valid_0's auc: 0.90255	valid_0's binary_logloss: 0.17412
    [3907]	valid_0's auc: 0.902548	valid_0's binary_logloss: 0.1741
    [3908]	valid_0's auc: 0.902547	valid_0's binary_logloss: 0.174116
    [3909]	valid_0's auc: 0.902549	valid_0's binary_logloss: 0.174096
    [3910]	valid_0's auc: 0.902536	valid_0's binary_logloss: 0.174082
    [3911]	valid_0's auc: 0.902537	valid_0's binary_logloss: 0.174105
    [3912]	valid_0's auc: 0.902541	valid_0's binary_logloss: 0.174069
    [3913]	valid_0's auc: 0.902536	valid_0's binary_logloss: 0.174052
    [3914]	valid_0's auc: 0.902547	valid_0's binary_logloss: 0.17402
    [3915]	valid_0's auc: 0.90253	valid_0's binary_logloss: 0.174007
    [3916]	valid_0's auc: 0.902524	valid_0's binary_logloss: 0.173994
    [3917]	valid_0's auc: 0.902526	valid_0's binary_logloss: 0.174009
    [3918]	valid_0's auc: 0.902527	valid_0's binary_logloss: 0.174026
    [3919]	valid_0's auc: 0.902527	valid_0's binary_logloss: 0.174043
    [3920]	valid_0's auc: 0.902547	valid_0's binary_logloss: 0.174
    [3921]	valid_0's auc: 0.902549	valid_0's binary_logloss: 0.174015
    [3922]	valid_0's auc: 0.902557	valid_0's binary_logloss: 0.173968
    [3923]	valid_0's auc: 0.902563	valid_0's binary_logloss: 0.173943
    [3924]	valid_0's auc: 0.902595	valid_0's binary_logloss: 0.173924
    [3925]	valid_0's auc: 0.902595	valid_0's binary_logloss: 0.173937
    [3926]	valid_0's auc: 0.902594	valid_0's binary_logloss: 0.173953
    [3927]	valid_0's auc: 0.902608	valid_0's binary_logloss: 0.173905
    [3928]	valid_0's auc: 0.902609	valid_0's binary_logloss: 0.173918
    [3929]	valid_0's auc: 0.902609	valid_0's binary_logloss: 0.173899
    [3930]	valid_0's auc: 0.90261	valid_0's binary_logloss: 0.173912
    [3931]	valid_0's auc: 0.902616	valid_0's binary_logloss: 0.1739
    [3932]	valid_0's auc: 0.902612	valid_0's binary_logloss: 0.173885
    [3933]	valid_0's auc: 0.902622	valid_0's binary_logloss: 0.173874
    [3934]	valid_0's auc: 0.902621	valid_0's binary_logloss: 0.173897
    [3935]	valid_0's auc: 0.902656	valid_0's binary_logloss: 0.173875
    [3936]	valid_0's auc: 0.902662	valid_0's binary_logloss: 0.173857
    [3937]	valid_0's auc: 0.902665	valid_0's binary_logloss: 0.173871
    [3938]	valid_0's auc: 0.902682	valid_0's binary_logloss: 0.17384
    [3939]	valid_0's auc: 0.902672	valid_0's binary_logloss: 0.173828
    [3940]	valid_0's auc: 0.902663	valid_0's binary_logloss: 0.173817
    [3941]	valid_0's auc: 0.902648	valid_0's binary_logloss: 0.173806
    [3942]	valid_0's auc: 0.902644	valid_0's binary_logloss: 0.173825
    [3943]	valid_0's auc: 0.902645	valid_0's binary_logloss: 0.173809
    [3944]	valid_0's auc: 0.902656	valid_0's binary_logloss: 0.173761
    [3945]	valid_0's auc: 0.902649	valid_0's binary_logloss: 0.173735
    [3946]	valid_0's auc: 0.902635	valid_0's binary_logloss: 0.173725
    [3947]	valid_0's auc: 0.90264	valid_0's binary_logloss: 0.173739
    [3948]	valid_0's auc: 0.90263	valid_0's binary_logloss: 0.173727
    [3949]	valid_0's auc: 0.902646	valid_0's binary_logloss: 0.173704
    [3950]	valid_0's auc: 0.902651	valid_0's binary_logloss: 0.173726
    [3951]	valid_0's auc: 0.902675	valid_0's binary_logloss: 0.17368
    [3952]	valid_0's auc: 0.90268	valid_0's binary_logloss: 0.1737
    [3953]	valid_0's auc: 0.902685	valid_0's binary_logloss: 0.173681
    [3954]	valid_0's auc: 0.902687	valid_0's binary_logloss: 0.173659
    [3955]	valid_0's auc: 0.902726	valid_0's binary_logloss: 0.173619
    [3956]	valid_0's auc: 0.902731	valid_0's binary_logloss: 0.173608
    [3957]	valid_0's auc: 0.902734	valid_0's binary_logloss: 0.173625
    [3958]	valid_0's auc: 0.902739	valid_0's binary_logloss: 0.173638
    [3959]	valid_0's auc: 0.902741	valid_0's binary_logloss: 0.173649
    [3960]	valid_0's auc: 0.902724	valid_0's binary_logloss: 0.173636
    [3961]	valid_0's auc: 0.902725	valid_0's binary_logloss: 0.173643
    [3962]	valid_0's auc: 0.902743	valid_0's binary_logloss: 0.17357
    [3963]	valid_0's auc: 0.902743	valid_0's binary_logloss: 0.173529
    [3964]	valid_0's auc: 0.902745	valid_0's binary_logloss: 0.173541
    [3965]	valid_0's auc: 0.902747	valid_0's binary_logloss: 0.173554
    [3966]	valid_0's auc: 0.902737	valid_0's binary_logloss: 0.173544
    [3967]	valid_0's auc: 0.902717	valid_0's binary_logloss: 0.173531
    [3968]	valid_0's auc: 0.90272	valid_0's binary_logloss: 0.173544
    [3969]	valid_0's auc: 0.902724	valid_0's binary_logloss: 0.173557
    [3970]	valid_0's auc: 0.902764	valid_0's binary_logloss: 0.173481
    [3971]	valid_0's auc: 0.902764	valid_0's binary_logloss: 0.173496
    [3972]	valid_0's auc: 0.902765	valid_0's binary_logloss: 0.17351
    [3973]	valid_0's auc: 0.902769	valid_0's binary_logloss: 0.173533
    [3974]	valid_0's auc: 0.902768	valid_0's binary_logloss: 0.173547
    [3975]	valid_0's auc: 0.902769	valid_0's binary_logloss: 0.173562
    [3976]	valid_0's auc: 0.902769	valid_0's binary_logloss: 0.173579
    [3977]	valid_0's auc: 0.902771	valid_0's binary_logloss: 0.173596
    [3978]	valid_0's auc: 0.902771	valid_0's binary_logloss: 0.17361
    [3979]	valid_0's auc: 0.902775	valid_0's binary_logloss: 0.173629
    [3980]	valid_0's auc: 0.902775	valid_0's binary_logloss: 0.173644
    [3981]	valid_0's auc: 0.902817	valid_0's binary_logloss: 0.173602
    [3982]	valid_0's auc: 0.902821	valid_0's binary_logloss: 0.173624
    [3983]	valid_0's auc: 0.90279	valid_0's binary_logloss: 0.17361
    [3984]	valid_0's auc: 0.902788	valid_0's binary_logloss: 0.173629
    [3985]	valid_0's auc: 0.902815	valid_0's binary_logloss: 0.1736
    [3986]	valid_0's auc: 0.902817	valid_0's binary_logloss: 0.173621
    [3987]	valid_0's auc: 0.902819	valid_0's binary_logloss: 0.173632
    [3988]	valid_0's auc: 0.902821	valid_0's binary_logloss: 0.173652
    [3989]	valid_0's auc: 0.902821	valid_0's binary_logloss: 0.173672
    [3990]	valid_0's auc: 0.902857	valid_0's binary_logloss: 0.173633
    [3991]	valid_0's auc: 0.902858	valid_0's binary_logloss: 0.173643
    [3992]	valid_0's auc: 0.902869	valid_0's binary_logloss: 0.17362
    [3993]	valid_0's auc: 0.90287	valid_0's binary_logloss: 0.173632
    [3994]	valid_0's auc: 0.902871	valid_0's binary_logloss: 0.173617
    [3995]	valid_0's auc: 0.902872	valid_0's binary_logloss: 0.173603
    [3996]	valid_0's auc: 0.902874	valid_0's binary_logloss: 0.17362
    [3997]	valid_0's auc: 0.902871	valid_0's binary_logloss: 0.173599
    [3998]	valid_0's auc: 0.902871	valid_0's binary_logloss: 0.173617
    [3999]	valid_0's auc: 0.902876	valid_0's binary_logloss: 0.173638
    [4000]	valid_0's auc: 0.902865	valid_0's binary_logloss: 0.173626
    [4001]	valid_0's auc: 0.902882	valid_0's binary_logloss: 0.173606
    [4002]	valid_0's auc: 0.90288	valid_0's binary_logloss: 0.173588
    [4003]	valid_0's auc: 0.90288	valid_0's binary_logloss: 0.173604
    [4004]	valid_0's auc: 0.902854	valid_0's binary_logloss: 0.173595
    [4005]	valid_0's auc: 0.902861	valid_0's binary_logloss: 0.173609
    [4006]	valid_0's auc: 0.902862	valid_0's binary_logloss: 0.173621
    [4007]	valid_0's auc: 0.902862	valid_0's binary_logloss: 0.173639
    [4008]	valid_0's auc: 0.902854	valid_0's binary_logloss: 0.173621
    [4009]	valid_0's auc: 0.902855	valid_0's binary_logloss: 0.173635
    [4010]	valid_0's auc: 0.902856	valid_0's binary_logloss: 0.173646
    [4011]	valid_0's auc: 0.902853	valid_0's binary_logloss: 0.173604
    [4012]	valid_0's auc: 0.902869	valid_0's binary_logloss: 0.173563
    [4013]	valid_0's auc: 0.902869	valid_0's binary_logloss: 0.173583
    [4014]	valid_0's auc: 0.902904	valid_0's binary_logloss: 0.173496
    [4015]	valid_0's auc: 0.902905	valid_0's binary_logloss: 0.173421
    [4016]	valid_0's auc: 0.902943	valid_0's binary_logloss: 0.173396
    [4017]	valid_0's auc: 0.902966	valid_0's binary_logloss: 0.173369
    [4018]	valid_0's auc: 0.902975	valid_0's binary_logloss: 0.173348
    [4019]	valid_0's auc: 0.902974	valid_0's binary_logloss: 0.173363
    [4020]	valid_0's auc: 0.902977	valid_0's binary_logloss: 0.17338
    [4021]	valid_0's auc: 0.902981	valid_0's binary_logloss: 0.173403
    [4022]	valid_0's auc: 0.902987	valid_0's binary_logloss: 0.173391
    [4023]	valid_0's auc: 0.902986	valid_0's binary_logloss: 0.17341
    [4024]	valid_0's auc: 0.902976	valid_0's binary_logloss: 0.173384
    [4025]	valid_0's auc: 0.902991	valid_0's binary_logloss: 0.173331
    [4026]	valid_0's auc: 0.902994	valid_0's binary_logloss: 0.173341
    [4027]	valid_0's auc: 0.903005	valid_0's binary_logloss: 0.173318
    [4028]	valid_0's auc: 0.90301	valid_0's binary_logloss: 0.173329
    [4029]	valid_0's auc: 0.903012	valid_0's binary_logloss: 0.173347
    [4030]	valid_0's auc: 0.903012	valid_0's binary_logloss: 0.173363
    [4031]	valid_0's auc: 0.903013	valid_0's binary_logloss: 0.173377
    [4032]	valid_0's auc: 0.903012	valid_0's binary_logloss: 0.17339
    [4033]	valid_0's auc: 0.903017	valid_0's binary_logloss: 0.173405
    [4034]	valid_0's auc: 0.903	valid_0's binary_logloss: 0.173396
    [4035]	valid_0's auc: 0.903004	valid_0's binary_logloss: 0.173413
    [4036]	valid_0's auc: 0.902994	valid_0's binary_logloss: 0.173393
    [4037]	valid_0's auc: 0.902995	valid_0's binary_logloss: 0.173406
    [4038]	valid_0's auc: 0.902996	valid_0's binary_logloss: 0.173419
    [4039]	valid_0's auc: 0.90301	valid_0's binary_logloss: 0.173398
    [4040]	valid_0's auc: 0.903	valid_0's binary_logloss: 0.173376
    [4041]	valid_0's auc: 0.903004	valid_0's binary_logloss: 0.173355
    [4042]	valid_0's auc: 0.903004	valid_0's binary_logloss: 0.173335
    [4043]	valid_0's auc: 0.903002	valid_0's binary_logloss: 0.173344
    [4044]	valid_0's auc: 0.902994	valid_0's binary_logloss: 0.173328
    [4045]	valid_0's auc: 0.902999	valid_0's binary_logloss: 0.173345
    [4046]	valid_0's auc: 0.902981	valid_0's binary_logloss: 0.173331
    [4047]	valid_0's auc: 0.902985	valid_0's binary_logloss: 0.173342
    [4048]	valid_0's auc: 0.902978	valid_0's binary_logloss: 0.173326
    [4049]	valid_0's auc: 0.90298	valid_0's binary_logloss: 0.173339
    [4050]	valid_0's auc: 0.902982	valid_0's binary_logloss: 0.173356
    [4051]	valid_0's auc: 0.902985	valid_0's binary_logloss: 0.173375
    [4052]	valid_0's auc: 0.902999	valid_0's binary_logloss: 0.173303
    [4053]	valid_0's auc: 0.902998	valid_0's binary_logloss: 0.173289
    [4054]	valid_0's auc: 0.903001	valid_0's binary_logloss: 0.173301
    [4055]	valid_0's auc: 0.902996	valid_0's binary_logloss: 0.173326
    [4056]	valid_0's auc: 0.903	valid_0's binary_logloss: 0.173343
    [4057]	valid_0's auc: 0.902987	valid_0's binary_logloss: 0.173331
    [4058]	valid_0's auc: 0.902988	valid_0's binary_logloss: 0.173345
    [4059]	valid_0's auc: 0.902979	valid_0's binary_logloss: 0.173334
    [4060]	valid_0's auc: 0.902988	valid_0's binary_logloss: 0.173307
    [4061]	valid_0's auc: 0.902995	valid_0's binary_logloss: 0.173272
    [4062]	valid_0's auc: 0.902995	valid_0's binary_logloss: 0.173286
    [4063]	valid_0's auc: 0.902997	valid_0's binary_logloss: 0.1733
    [4064]	valid_0's auc: 0.902998	valid_0's binary_logloss: 0.173318
    [4065]	valid_0's auc: 0.902998	valid_0's binary_logloss: 0.173329
    [4066]	valid_0's auc: 0.903	valid_0's binary_logloss: 0.173341
    [4067]	valid_0's auc: 0.902998	valid_0's binary_logloss: 0.173357
    [4068]	valid_0's auc: 0.903005	valid_0's binary_logloss: 0.173378
    [4069]	valid_0's auc: 0.903005	valid_0's binary_logloss: 0.173399
    [4070]	valid_0's auc: 0.903001	valid_0's binary_logloss: 0.17339
    [4071]	valid_0's auc: 0.902992	valid_0's binary_logloss: 0.173359
    [4072]	valid_0's auc: 0.902971	valid_0's binary_logloss: 0.173347
    [4073]	valid_0's auc: 0.902986	valid_0's binary_logloss: 0.173309
    [4074]	valid_0's auc: 0.90301	valid_0's binary_logloss: 0.173249
    [4075]	valid_0's auc: 0.903012	valid_0's binary_logloss: 0.173262
    [4076]	valid_0's auc: 0.903021	valid_0's binary_logloss: 0.173231
    [4077]	valid_0's auc: 0.90304	valid_0's binary_logloss: 0.173195
    [4078]	valid_0's auc: 0.903034	valid_0's binary_logloss: 0.173183
    [4079]	valid_0's auc: 0.903034	valid_0's binary_logloss: 0.173137
    [4080]	valid_0's auc: 0.903035	valid_0's binary_logloss: 0.173151
    [4081]	valid_0's auc: 0.903026	valid_0's binary_logloss: 0.17313
    [4082]	valid_0's auc: 0.903025	valid_0's binary_logloss: 0.173151
    [4083]	valid_0's auc: 0.903031	valid_0's binary_logloss: 0.173165
    [4084]	valid_0's auc: 0.903025	valid_0's binary_logloss: 0.17315
    [4085]	valid_0's auc: 0.903023	valid_0's binary_logloss: 0.173164
    [4086]	valid_0's auc: 0.903028	valid_0's binary_logloss: 0.173183
    [4087]	valid_0's auc: 0.903032	valid_0's binary_logloss: 0.173193
    [4088]	valid_0's auc: 0.903034	valid_0's binary_logloss: 0.173204
    [4089]	valid_0's auc: 0.903036	valid_0's binary_logloss: 0.173178
    [4090]	valid_0's auc: 0.903039	valid_0's binary_logloss: 0.173197
    [4091]	valid_0's auc: 0.903043	valid_0's binary_logloss: 0.173207
    [4092]	valid_0's auc: 0.903058	valid_0's binary_logloss: 0.173182
    [4093]	valid_0's auc: 0.903079	valid_0's binary_logloss: 0.17316
    [4094]	valid_0's auc: 0.903109	valid_0's binary_logloss: 0.173114
    [4095]	valid_0's auc: 0.90311	valid_0's binary_logloss: 0.173126
    [4096]	valid_0's auc: 0.903107	valid_0's binary_logloss: 0.173138
    [4097]	valid_0's auc: 0.903112	valid_0's binary_logloss: 0.173162
    [4098]	valid_0's auc: 0.903114	valid_0's binary_logloss: 0.173173
    [4099]	valid_0's auc: 0.903151	valid_0's binary_logloss: 0.173116
    [4100]	valid_0's auc: 0.903151	valid_0's binary_logloss: 0.173132
    [4101]	valid_0's auc: 0.903155	valid_0's binary_logloss: 0.173155
    [4102]	valid_0's auc: 0.903149	valid_0's binary_logloss: 0.173139
    [4103]	valid_0's auc: 0.903149	valid_0's binary_logloss: 0.173131
    [4104]	valid_0's auc: 0.903135	valid_0's binary_logloss: 0.173118
    [4105]	valid_0's auc: 0.90314	valid_0's binary_logloss: 0.173135
    [4106]	valid_0's auc: 0.903157	valid_0's binary_logloss: 0.173109
    [4107]	valid_0's auc: 0.903166	valid_0's binary_logloss: 0.173086
    [4108]	valid_0's auc: 0.903149	valid_0's binary_logloss: 0.173071
    [4109]	valid_0's auc: 0.903153	valid_0's binary_logloss: 0.173082
    [4110]	valid_0's auc: 0.903157	valid_0's binary_logloss: 0.173095
    [4111]	valid_0's auc: 0.903153	valid_0's binary_logloss: 0.173088
    [4112]	valid_0's auc: 0.903153	valid_0's binary_logloss: 0.173103
    [4113]	valid_0's auc: 0.903153	valid_0's binary_logloss: 0.17313
    [4114]	valid_0's auc: 0.903148	valid_0's binary_logloss: 0.17312
    [4115]	valid_0's auc: 0.903127	valid_0's binary_logloss: 0.173114
    [4116]	valid_0's auc: 0.90314	valid_0's binary_logloss: 0.173055
    [4117]	valid_0's auc: 0.903144	valid_0's binary_logloss: 0.173038
    [4118]	valid_0's auc: 0.903144	valid_0's binary_logloss: 0.173027
    [4119]	valid_0's auc: 0.903147	valid_0's binary_logloss: 0.17304
    [4120]	valid_0's auc: 0.903147	valid_0's binary_logloss: 0.173061
    [4121]	valid_0's auc: 0.903148	valid_0's binary_logloss: 0.173025
    [4122]	valid_0's auc: 0.903144	valid_0's binary_logloss: 0.173013
    [4123]	valid_0's auc: 0.903149	valid_0's binary_logloss: 0.173038
    [4124]	valid_0's auc: 0.90315	valid_0's binary_logloss: 0.173057
    [4125]	valid_0's auc: 0.903169	valid_0's binary_logloss: 0.172994
    [4126]	valid_0's auc: 0.903189	valid_0's binary_logloss: 0.172968
    [4127]	valid_0's auc: 0.903187	valid_0's binary_logloss: 0.172979
    [4128]	valid_0's auc: 0.903218	valid_0's binary_logloss: 0.172956
    [4129]	valid_0's auc: 0.903222	valid_0's binary_logloss: 0.17297
    [4130]	valid_0's auc: 0.903256	valid_0's binary_logloss: 0.172935
    [4131]	valid_0's auc: 0.903252	valid_0's binary_logloss: 0.172916
    [4132]	valid_0's auc: 0.903256	valid_0's binary_logloss: 0.172902
    [4133]	valid_0's auc: 0.903228	valid_0's binary_logloss: 0.172893
    [4134]	valid_0's auc: 0.903232	valid_0's binary_logloss: 0.172909
    [4135]	valid_0's auc: 0.903219	valid_0's binary_logloss: 0.172889
    [4136]	valid_0's auc: 0.903221	valid_0's binary_logloss: 0.172904
    [4137]	valid_0's auc: 0.903222	valid_0's binary_logloss: 0.172919
    [4138]	valid_0's auc: 0.903222	valid_0's binary_logloss: 0.172931
    [4139]	valid_0's auc: 0.903222	valid_0's binary_logloss: 0.172942
    [4140]	valid_0's auc: 0.903225	valid_0's binary_logloss: 0.172959
    [4141]	valid_0's auc: 0.903213	valid_0's binary_logloss: 0.172946
    [4142]	valid_0's auc: 0.903213	valid_0's binary_logloss: 0.172958
    [4143]	valid_0's auc: 0.903215	valid_0's binary_logloss: 0.172971
    [4144]	valid_0's auc: 0.903209	valid_0's binary_logloss: 0.172962
    [4145]	valid_0's auc: 0.90321	valid_0's binary_logloss: 0.172973
    [4146]	valid_0's auc: 0.90321	valid_0's binary_logloss: 0.17299
    [4147]	valid_0's auc: 0.903211	valid_0's binary_logloss: 0.172999
    [4148]	valid_0's auc: 0.903222	valid_0's binary_logloss: 0.172954
    [4149]	valid_0's auc: 0.903261	valid_0's binary_logloss: 0.172864
    [4150]	valid_0's auc: 0.903263	valid_0's binary_logloss: 0.172878
    [4151]	valid_0's auc: 0.903263	valid_0's binary_logloss: 0.172894
    [4152]	valid_0's auc: 0.903263	valid_0's binary_logloss: 0.172908
    [4153]	valid_0's auc: 0.903262	valid_0's binary_logloss: 0.172927
    [4154]	valid_0's auc: 0.903261	valid_0's binary_logloss: 0.172901
    [4155]	valid_0's auc: 0.903263	valid_0's binary_logloss: 0.172886
    [4156]	valid_0's auc: 0.903262	valid_0's binary_logloss: 0.172903
    [4157]	valid_0's auc: 0.903261	valid_0's binary_logloss: 0.172917
    [4158]	valid_0's auc: 0.903258	valid_0's binary_logloss: 0.172931
    [4159]	valid_0's auc: 0.903287	valid_0's binary_logloss: 0.172895
    [4160]	valid_0's auc: 0.903268	valid_0's binary_logloss: 0.172874
    [4161]	valid_0's auc: 0.903266	valid_0's binary_logloss: 0.172853
    [4162]	valid_0's auc: 0.903267	valid_0's binary_logloss: 0.172868
    [4163]	valid_0's auc: 0.903273	valid_0's binary_logloss: 0.172852
    [4164]	valid_0's auc: 0.903275	valid_0's binary_logloss: 0.172865
    [4165]	valid_0's auc: 0.903275	valid_0's binary_logloss: 0.172834
    [4166]	valid_0's auc: 0.903276	valid_0's binary_logloss: 0.172846
    [4167]	valid_0's auc: 0.903272	valid_0's binary_logloss: 0.172829
    [4168]	valid_0's auc: 0.903273	valid_0's binary_logloss: 0.172843
    [4169]	valid_0's auc: 0.903274	valid_0's binary_logloss: 0.172863
    [4170]	valid_0's auc: 0.90326	valid_0's binary_logloss: 0.172845
    [4171]	valid_0's auc: 0.90326	valid_0's binary_logloss: 0.172859
    [4172]	valid_0's auc: 0.903265	valid_0's binary_logloss: 0.172875
    [4173]	valid_0's auc: 0.903269	valid_0's binary_logloss: 0.172838
    [4174]	valid_0's auc: 0.903271	valid_0's binary_logloss: 0.172851
    [4175]	valid_0's auc: 0.903273	valid_0's binary_logloss: 0.172865
    [4176]	valid_0's auc: 0.903272	valid_0's binary_logloss: 0.17288
    [4177]	valid_0's auc: 0.903274	valid_0's binary_logloss: 0.172889
    [4178]	valid_0's auc: 0.903277	valid_0's binary_logloss: 0.172898
    [4179]	valid_0's auc: 0.903281	valid_0's binary_logloss: 0.172883
    [4180]	valid_0's auc: 0.903287	valid_0's binary_logloss: 0.172856
    [4181]	valid_0's auc: 0.903287	valid_0's binary_logloss: 0.172871
    [4182]	valid_0's auc: 0.903288	valid_0's binary_logloss: 0.172883
    [4183]	valid_0's auc: 0.903287	valid_0's binary_logloss: 0.172893
    [4184]	valid_0's auc: 0.903298	valid_0's binary_logloss: 0.172875
    [4185]	valid_0's auc: 0.903291	valid_0's binary_logloss: 0.172799
    [4186]	valid_0's auc: 0.903294	valid_0's binary_logloss: 0.172771
    [4187]	valid_0's auc: 0.9033	valid_0's binary_logloss: 0.17279
    [4188]	valid_0's auc: 0.903301	valid_0's binary_logloss: 0.172801
    [4189]	valid_0's auc: 0.903302	valid_0's binary_logloss: 0.172812
    [4190]	valid_0's auc: 0.903322	valid_0's binary_logloss: 0.172752
    [4191]	valid_0's auc: 0.903322	valid_0's binary_logloss: 0.172735
    [4192]	valid_0's auc: 0.903353	valid_0's binary_logloss: 0.172676
    [4193]	valid_0's auc: 0.903374	valid_0's binary_logloss: 0.172649
    [4194]	valid_0's auc: 0.903377	valid_0's binary_logloss: 0.172663
    [4195]	valid_0's auc: 0.90338	valid_0's binary_logloss: 0.17261
    [4196]	valid_0's auc: 0.903386	valid_0's binary_logloss: 0.172637
    [4197]	valid_0's auc: 0.903404	valid_0's binary_logloss: 0.172612
    [4198]	valid_0's auc: 0.903426	valid_0's binary_logloss: 0.17255
    [4199]	valid_0's auc: 0.903431	valid_0's binary_logloss: 0.172564
    [4200]	valid_0's auc: 0.903427	valid_0's binary_logloss: 0.172578
    [4201]	valid_0's auc: 0.90343	valid_0's binary_logloss: 0.172586
    [4202]	valid_0's auc: 0.903433	valid_0's binary_logloss: 0.172606
    [4203]	valid_0's auc: 0.903433	valid_0's binary_logloss: 0.172593
    [4204]	valid_0's auc: 0.903433	valid_0's binary_logloss: 0.172604
    [4205]	valid_0's auc: 0.903419	valid_0's binary_logloss: 0.172583
    [4206]	valid_0's auc: 0.903422	valid_0's binary_logloss: 0.172597
    [4207]	valid_0's auc: 0.903424	valid_0's binary_logloss: 0.17261
    [4208]	valid_0's auc: 0.903425	valid_0's binary_logloss: 0.172624
    [4209]	valid_0's auc: 0.903427	valid_0's binary_logloss: 0.172643
    [4210]	valid_0's auc: 0.903408	valid_0's binary_logloss: 0.172632
    [4211]	valid_0's auc: 0.903409	valid_0's binary_logloss: 0.172654
    [4212]	valid_0's auc: 0.903408	valid_0's binary_logloss: 0.172619
    [4213]	valid_0's auc: 0.9034	valid_0's binary_logloss: 0.172598
    [4214]	valid_0's auc: 0.903396	valid_0's binary_logloss: 0.172585
    [4215]	valid_0's auc: 0.903401	valid_0's binary_logloss: 0.172547
    [4216]	valid_0's auc: 0.903402	valid_0's binary_logloss: 0.172558
    [4217]	valid_0's auc: 0.903404	valid_0's binary_logloss: 0.17257
    [4218]	valid_0's auc: 0.903407	valid_0's binary_logloss: 0.172581
    [4219]	valid_0's auc: 0.9034	valid_0's binary_logloss: 0.172545
    [4220]	valid_0's auc: 0.903409	valid_0's binary_logloss: 0.17253
    [4221]	valid_0's auc: 0.903408	valid_0's binary_logloss: 0.172514
    [4222]	valid_0's auc: 0.903409	valid_0's binary_logloss: 0.172531
    [4223]	valid_0's auc: 0.903426	valid_0's binary_logloss: 0.172504
    [4224]	valid_0's auc: 0.903426	valid_0's binary_logloss: 0.172491
    [4225]	valid_0's auc: 0.903434	valid_0's binary_logloss: 0.172454
    [4226]	valid_0's auc: 0.903437	valid_0's binary_logloss: 0.172467
    [4227]	valid_0's auc: 0.903445	valid_0's binary_logloss: 0.172423
    [4228]	valid_0's auc: 0.903447	valid_0's binary_logloss: 0.172437
    [4229]	valid_0's auc: 0.903452	valid_0's binary_logloss: 0.172458
    [4230]	valid_0's auc: 0.903449	valid_0's binary_logloss: 0.172438
    [4231]	valid_0's auc: 0.903454	valid_0's binary_logloss: 0.172459
    [4232]	valid_0's auc: 0.903468	valid_0's binary_logloss: 0.172437
    [4233]	valid_0's auc: 0.90346	valid_0's binary_logloss: 0.172416
    [4234]	valid_0's auc: 0.90346	valid_0's binary_logloss: 0.172401
    [4235]	valid_0's auc: 0.903466	valid_0's binary_logloss: 0.172414
    [4236]	valid_0's auc: 0.903492	valid_0's binary_logloss: 0.172384
    [4237]	valid_0's auc: 0.903477	valid_0's binary_logloss: 0.172367
    [4238]	valid_0's auc: 0.903478	valid_0's binary_logloss: 0.172382
    [4239]	valid_0's auc: 0.903481	valid_0's binary_logloss: 0.172398
    [4240]	valid_0's auc: 0.903474	valid_0's binary_logloss: 0.172382
    [4241]	valid_0's auc: 0.903475	valid_0's binary_logloss: 0.172394
    [4242]	valid_0's auc: 0.903476	valid_0's binary_logloss: 0.172376
    [4243]	valid_0's auc: 0.903479	valid_0's binary_logloss: 0.172396
    [4244]	valid_0's auc: 0.903477	valid_0's binary_logloss: 0.172379
    [4245]	valid_0's auc: 0.903482	valid_0's binary_logloss: 0.172402
    [4246]	valid_0's auc: 0.903485	valid_0's binary_logloss: 0.172375
    [4247]	valid_0's auc: 0.903473	valid_0's binary_logloss: 0.172364
    [4248]	valid_0's auc: 0.903475	valid_0's binary_logloss: 0.172378
    [4249]	valid_0's auc: 0.903479	valid_0's binary_logloss: 0.172395
    [4250]	valid_0's auc: 0.903481	valid_0's binary_logloss: 0.172406
    [4251]	valid_0's auc: 0.903491	valid_0's binary_logloss: 0.172382
    [4252]	valid_0's auc: 0.903509	valid_0's binary_logloss: 0.172359
    [4253]	valid_0's auc: 0.903512	valid_0's binary_logloss: 0.17237
    [4254]	valid_0's auc: 0.90351	valid_0's binary_logloss: 0.172377
    [4255]	valid_0's auc: 0.903515	valid_0's binary_logloss: 0.172397
    [4256]	valid_0's auc: 0.903506	valid_0's binary_logloss: 0.172342
    [4257]	valid_0's auc: 0.903518	valid_0's binary_logloss: 0.172321
    [4258]	valid_0's auc: 0.903515	valid_0's binary_logloss: 0.172309
    [4259]	valid_0's auc: 0.903499	valid_0's binary_logloss: 0.172301
    [4260]	valid_0's auc: 0.903481	valid_0's binary_logloss: 0.172251
    [4261]	valid_0's auc: 0.903482	valid_0's binary_logloss: 0.172271
    [4262]	valid_0's auc: 0.903482	valid_0's binary_logloss: 0.172254
    [4263]	valid_0's auc: 0.903487	valid_0's binary_logloss: 0.172273
    [4264]	valid_0's auc: 0.903513	valid_0's binary_logloss: 0.172227
    [4265]	valid_0's auc: 0.903518	valid_0's binary_logloss: 0.172243
    [4266]	valid_0's auc: 0.903521	valid_0's binary_logloss: 0.172253
    [4267]	valid_0's auc: 0.903522	valid_0's binary_logloss: 0.172269
    [4268]	valid_0's auc: 0.903522	valid_0's binary_logloss: 0.172284
    [4269]	valid_0's auc: 0.903523	valid_0's binary_logloss: 0.172299
    [4270]	valid_0's auc: 0.903535	valid_0's binary_logloss: 0.172221
    [4271]	valid_0's auc: 0.903535	valid_0's binary_logloss: 0.172229
    [4272]	valid_0's auc: 0.90354	valid_0's binary_logloss: 0.172242
    [4273]	valid_0's auc: 0.903542	valid_0's binary_logloss: 0.172254
    [4274]	valid_0's auc: 0.903541	valid_0's binary_logloss: 0.172265
    [4275]	valid_0's auc: 0.903545	valid_0's binary_logloss: 0.172285
    [4276]	valid_0's auc: 0.903544	valid_0's binary_logloss: 0.172265
    [4277]	valid_0's auc: 0.903545	valid_0's binary_logloss: 0.172284
    [4278]	valid_0's auc: 0.903546	valid_0's binary_logloss: 0.172293
    [4279]	valid_0's auc: 0.903547	valid_0's binary_logloss: 0.172274
    [4280]	valid_0's auc: 0.903547	valid_0's binary_logloss: 0.172243
    [4281]	valid_0's auc: 0.903552	valid_0's binary_logloss: 0.172264
    [4282]	valid_0's auc: 0.903557	valid_0's binary_logloss: 0.172203
    [4283]	valid_0's auc: 0.903571	valid_0's binary_logloss: 0.172185
    [4284]	valid_0's auc: 0.903581	valid_0's binary_logloss: 0.172156
    [4285]	valid_0's auc: 0.903579	valid_0's binary_logloss: 0.172144
    [4286]	valid_0's auc: 0.903574	valid_0's binary_logloss: 0.172125
    [4287]	valid_0's auc: 0.903577	valid_0's binary_logloss: 0.172137
    [4288]	valid_0's auc: 0.90358	valid_0's binary_logloss: 0.172155
    [4289]	valid_0's auc: 0.903582	valid_0's binary_logloss: 0.172173
    [4290]	valid_0's auc: 0.903583	valid_0's binary_logloss: 0.172185
    [4291]	valid_0's auc: 0.903573	valid_0's binary_logloss: 0.172174
    [4292]	valid_0's auc: 0.903573	valid_0's binary_logloss: 0.172189
    [4293]	valid_0's auc: 0.903574	valid_0's binary_logloss: 0.172201
    [4294]	valid_0's auc: 0.903576	valid_0's binary_logloss: 0.172212
    [4295]	valid_0's auc: 0.903577	valid_0's binary_logloss: 0.172228
    [4296]	valid_0's auc: 0.903579	valid_0's binary_logloss: 0.17224
    [4297]	valid_0's auc: 0.90358	valid_0's binary_logloss: 0.17225
    [4298]	valid_0's auc: 0.903552	valid_0's binary_logloss: 0.172221
    [4299]	valid_0's auc: 0.903552	valid_0's binary_logloss: 0.172231
    [4300]	valid_0's auc: 0.903525	valid_0's binary_logloss: 0.172223
    [4301]	valid_0's auc: 0.903527	valid_0's binary_logloss: 0.17224
    [4302]	valid_0's auc: 0.903548	valid_0's binary_logloss: 0.172194
    [4303]	valid_0's auc: 0.903539	valid_0's binary_logloss: 0.172178
    [4304]	valid_0's auc: 0.90354	valid_0's binary_logloss: 0.172156
    [4305]	valid_0's auc: 0.903545	valid_0's binary_logloss: 0.17218
    [4306]	valid_0's auc: 0.903538	valid_0's binary_logloss: 0.17212
    [4307]	valid_0's auc: 0.903546	valid_0's binary_logloss: 0.172103
    [4308]	valid_0's auc: 0.903548	valid_0's binary_logloss: 0.172115
    [4309]	valid_0's auc: 0.903548	valid_0's binary_logloss: 0.172064
    [4310]	valid_0's auc: 0.903548	valid_0's binary_logloss: 0.17208
    [4311]	valid_0's auc: 0.903529	valid_0's binary_logloss: 0.172066
    [4312]	valid_0's auc: 0.903523	valid_0's binary_logloss: 0.17205
    [4313]	valid_0's auc: 0.903527	valid_0's binary_logloss: 0.172036
    [4314]	valid_0's auc: 0.903529	valid_0's binary_logloss: 0.172056
    [4315]	valid_0's auc: 0.90353	valid_0's binary_logloss: 0.172069
    [4316]	valid_0's auc: 0.903533	valid_0's binary_logloss: 0.172086
    [4317]	valid_0's auc: 0.903536	valid_0's binary_logloss: 0.172068
    [4318]	valid_0's auc: 0.903535	valid_0's binary_logloss: 0.172051
    [4319]	valid_0's auc: 0.903535	valid_0's binary_logloss: 0.172037
    [4320]	valid_0's auc: 0.903557	valid_0's binary_logloss: 0.171963
    [4321]	valid_0's auc: 0.903557	valid_0's binary_logloss: 0.17198
    [4322]	valid_0's auc: 0.903556	valid_0's binary_logloss: 0.171994
    [4323]	valid_0's auc: 0.903558	valid_0's binary_logloss: 0.172005
    [4324]	valid_0's auc: 0.903559	valid_0's binary_logloss: 0.172019
    [4325]	valid_0's auc: 0.903561	valid_0's binary_logloss: 0.172033
    [4326]	valid_0's auc: 0.903564	valid_0's binary_logloss: 0.172044
    [4327]	valid_0's auc: 0.903569	valid_0's binary_logloss: 0.172066
    [4328]	valid_0's auc: 0.903571	valid_0's binary_logloss: 0.172078
    [4329]	valid_0's auc: 0.903583	valid_0's binary_logloss: 0.172029
    [4330]	valid_0's auc: 0.903585	valid_0's binary_logloss: 0.172043
    [4331]	valid_0's auc: 0.903585	valid_0's binary_logloss: 0.172022
    [4332]	valid_0's auc: 0.903588	valid_0's binary_logloss: 0.172039
    [4333]	valid_0's auc: 0.903587	valid_0's binary_logloss: 0.172051
    [4334]	valid_0's auc: 0.903588	valid_0's binary_logloss: 0.172065
    [4335]	valid_0's auc: 0.903603	valid_0's binary_logloss: 0.171974
    [4336]	valid_0's auc: 0.903602	valid_0's binary_logloss: 0.171955
    [4337]	valid_0's auc: 0.90359	valid_0's binary_logloss: 0.171937
    [4338]	valid_0's auc: 0.903595	valid_0's binary_logloss: 0.171959
    [4339]	valid_0's auc: 0.903594	valid_0's binary_logloss: 0.171971
    [4340]	valid_0's auc: 0.903608	valid_0's binary_logloss: 0.171945
    [4341]	valid_0's auc: 0.903611	valid_0's binary_logloss: 0.171966
    [4342]	valid_0's auc: 0.903609	valid_0's binary_logloss: 0.17198
    [4343]	valid_0's auc: 0.90361	valid_0's binary_logloss: 0.171962
    [4344]	valid_0's auc: 0.903626	valid_0's binary_logloss: 0.171933
    [4345]	valid_0's auc: 0.903641	valid_0's binary_logloss: 0.17191
    [4346]	valid_0's auc: 0.903643	valid_0's binary_logloss: 0.171896
    [4347]	valid_0's auc: 0.903642	valid_0's binary_logloss: 0.171875
    [4348]	valid_0's auc: 0.903646	valid_0's binary_logloss: 0.171864
    [4349]	valid_0's auc: 0.903646	valid_0's binary_logloss: 0.171874
    [4350]	valid_0's auc: 0.903654	valid_0's binary_logloss: 0.171855
    [4351]	valid_0's auc: 0.903655	valid_0's binary_logloss: 0.171865
    [4352]	valid_0's auc: 0.903677	valid_0's binary_logloss: 0.171824
    [4353]	valid_0's auc: 0.903677	valid_0's binary_logloss: 0.171809
    [4354]	valid_0's auc: 0.903682	valid_0's binary_logloss: 0.171817
    [4355]	valid_0's auc: 0.903682	valid_0's binary_logloss: 0.171831
    [4356]	valid_0's auc: 0.903683	valid_0's binary_logloss: 0.171839
    [4357]	valid_0's auc: 0.903684	valid_0's binary_logloss: 0.171852
    [4358]	valid_0's auc: 0.903685	valid_0's binary_logloss: 0.171861
    [4359]	valid_0's auc: 0.903681	valid_0's binary_logloss: 0.17185
    [4360]	valid_0's auc: 0.903706	valid_0's binary_logloss: 0.171819
    [4361]	valid_0's auc: 0.903706	valid_0's binary_logloss: 0.171833
    [4362]	valid_0's auc: 0.90371	valid_0's binary_logloss: 0.171852
    [4363]	valid_0's auc: 0.903715	valid_0's binary_logloss: 0.171826
    [4364]	valid_0's auc: 0.903716	valid_0's binary_logloss: 0.171835
    [4365]	valid_0's auc: 0.903724	valid_0's binary_logloss: 0.171816
    [4366]	valid_0's auc: 0.903724	valid_0's binary_logloss: 0.171838
    [4367]	valid_0's auc: 0.903728	valid_0's binary_logloss: 0.171856
    [4368]	valid_0's auc: 0.903723	valid_0's binary_logloss: 0.17184
    [4369]	valid_0's auc: 0.903727	valid_0's binary_logloss: 0.171854
    [4370]	valid_0's auc: 0.903726	valid_0's binary_logloss: 0.171869
    [4371]	valid_0's auc: 0.903735	valid_0's binary_logloss: 0.171853
    [4372]	valid_0's auc: 0.903738	valid_0's binary_logloss: 0.171865
    [4373]	valid_0's auc: 0.903738	valid_0's binary_logloss: 0.171876
    [4374]	valid_0's auc: 0.903741	valid_0's binary_logloss: 0.171893
    [4375]	valid_0's auc: 0.90374	valid_0's binary_logloss: 0.171902
    [4376]	valid_0's auc: 0.903737	valid_0's binary_logloss: 0.171885
    [4377]	valid_0's auc: 0.903727	valid_0's binary_logloss: 0.171859
    [4378]	valid_0's auc: 0.903726	valid_0's binary_logloss: 0.171871
    [4379]	valid_0's auc: 0.903714	valid_0's binary_logloss: 0.171854
    [4380]	valid_0's auc: 0.903743	valid_0's binary_logloss: 0.171799
    [4381]	valid_0's auc: 0.903743	valid_0's binary_logloss: 0.171815
    [4382]	valid_0's auc: 0.903734	valid_0's binary_logloss: 0.171799
    [4383]	valid_0's auc: 0.903737	valid_0's binary_logloss: 0.171811
    [4384]	valid_0's auc: 0.903741	valid_0's binary_logloss: 0.171828
    [4385]	valid_0's auc: 0.903743	valid_0's binary_logloss: 0.171843
    [4386]	valid_0's auc: 0.903732	valid_0's binary_logloss: 0.171822
    [4387]	valid_0's auc: 0.903757	valid_0's binary_logloss: 0.171773
    [4388]	valid_0's auc: 0.903762	valid_0's binary_logloss: 0.171755
    [4389]	valid_0's auc: 0.903765	valid_0's binary_logloss: 0.171765
    [4390]	valid_0's auc: 0.903755	valid_0's binary_logloss: 0.171752
    [4391]	valid_0's auc: 0.903757	valid_0's binary_logloss: 0.171762
    [4392]	valid_0's auc: 0.903762	valid_0's binary_logloss: 0.171738
    [4393]	valid_0's auc: 0.90376	valid_0's binary_logloss: 0.17176
    [4394]	valid_0's auc: 0.903764	valid_0's binary_logloss: 0.171775
    [4395]	valid_0's auc: 0.903766	valid_0's binary_logloss: 0.171784
    [4396]	valid_0's auc: 0.90376	valid_0's binary_logloss: 0.17177
    [4397]	valid_0's auc: 0.90376	valid_0's binary_logloss: 0.171782
    [4398]	valid_0's auc: 0.903764	valid_0's binary_logloss: 0.171797
    [4399]	valid_0's auc: 0.903811	valid_0's binary_logloss: 0.171771
    [4400]	valid_0's auc: 0.903812	valid_0's binary_logloss: 0.171781
    [4401]	valid_0's auc: 0.903813	valid_0's binary_logloss: 0.171792
    [4402]	valid_0's auc: 0.903825	valid_0's binary_logloss: 0.171749
    [4403]	valid_0's auc: 0.903812	valid_0's binary_logloss: 0.171739
    [4404]	valid_0's auc: 0.903814	valid_0's binary_logloss: 0.171749
    [4405]	valid_0's auc: 0.903799	valid_0's binary_logloss: 0.171739
    [4406]	valid_0's auc: 0.903803	valid_0's binary_logloss: 0.171748
    [4407]	valid_0's auc: 0.903809	valid_0's binary_logloss: 0.17176
    [4408]	valid_0's auc: 0.903809	valid_0's binary_logloss: 0.17177
    [4409]	valid_0's auc: 0.90381	valid_0's binary_logloss: 0.171778
    [4410]	valid_0's auc: 0.90383	valid_0's binary_logloss: 0.171753
    [4411]	valid_0's auc: 0.903834	valid_0's binary_logloss: 0.171767
    [4412]	valid_0's auc: 0.903839	valid_0's binary_logloss: 0.171784
    [4413]	valid_0's auc: 0.903845	valid_0's binary_logloss: 0.171765
    [4414]	valid_0's auc: 0.90384	valid_0's binary_logloss: 0.171749
    [4415]	valid_0's auc: 0.903834	valid_0's binary_logloss: 0.171733
    [4416]	valid_0's auc: 0.903837	valid_0's binary_logloss: 0.171744
    [4417]	valid_0's auc: 0.903835	valid_0's binary_logloss: 0.171691
    [4418]	valid_0's auc: 0.903822	valid_0's binary_logloss: 0.171683
    [4419]	valid_0's auc: 0.903825	valid_0's binary_logloss: 0.171703
    [4420]	valid_0's auc: 0.903832	valid_0's binary_logloss: 0.171719
    [4421]	valid_0's auc: 0.903837	valid_0's binary_logloss: 0.171738
    [4422]	valid_0's auc: 0.903836	valid_0's binary_logloss: 0.171705
    [4423]	valid_0's auc: 0.903838	valid_0's binary_logloss: 0.171717
    [4424]	valid_0's auc: 0.903826	valid_0's binary_logloss: 0.171702
    [4425]	valid_0's auc: 0.903828	valid_0's binary_logloss: 0.171715
    [4426]	valid_0's auc: 0.90383	valid_0's binary_logloss: 0.171698
    [4427]	valid_0's auc: 0.903832	valid_0's binary_logloss: 0.171713
    [4428]	valid_0's auc: 0.903842	valid_0's binary_logloss: 0.171692
    [4429]	valid_0's auc: 0.903842	valid_0's binary_logloss: 0.171705
    [4430]	valid_0's auc: 0.903844	valid_0's binary_logloss: 0.171713
    [4431]	valid_0's auc: 0.903849	valid_0's binary_logloss: 0.171696
    [4432]	valid_0's auc: 0.903852	valid_0's binary_logloss: 0.171714
    [4433]	valid_0's auc: 0.903854	valid_0's binary_logloss: 0.171724
    [4434]	valid_0's auc: 0.903863	valid_0's binary_logloss: 0.171689
    [4435]	valid_0's auc: 0.903858	valid_0's binary_logloss: 0.171671
    [4436]	valid_0's auc: 0.90386	valid_0's binary_logloss: 0.171682
    [4437]	valid_0's auc: 0.903865	valid_0's binary_logloss: 0.171698
    [4438]	valid_0's auc: 0.903868	valid_0's binary_logloss: 0.171714
    [4439]	valid_0's auc: 0.903887	valid_0's binary_logloss: 0.17164
    [4440]	valid_0's auc: 0.903889	valid_0's binary_logloss: 0.171648
    [4441]	valid_0's auc: 0.903886	valid_0's binary_logloss: 0.171628
    [4442]	valid_0's auc: 0.903888	valid_0's binary_logloss: 0.171642
    [4443]	valid_0's auc: 0.90389	valid_0's binary_logloss: 0.171652
    [4444]	valid_0's auc: 0.903894	valid_0's binary_logloss: 0.171634
    [4445]	valid_0's auc: 0.903888	valid_0's binary_logloss: 0.17155
    [4446]	valid_0's auc: 0.90389	valid_0's binary_logloss: 0.171557
    [4447]	valid_0's auc: 0.903889	valid_0's binary_logloss: 0.171567
    [4448]	valid_0's auc: 0.903893	valid_0's binary_logloss: 0.171579
    [4449]	valid_0's auc: 0.903893	valid_0's binary_logloss: 0.171592
    [4450]	valid_0's auc: 0.903912	valid_0's binary_logloss: 0.171515
    [4451]	valid_0's auc: 0.903913	valid_0's binary_logloss: 0.171524
    [4452]	valid_0's auc: 0.903915	valid_0's binary_logloss: 0.171538
    [4453]	valid_0's auc: 0.903906	valid_0's binary_logloss: 0.171525
    [4454]	valid_0's auc: 0.90391	valid_0's binary_logloss: 0.171536
    [4455]	valid_0's auc: 0.903911	valid_0's binary_logloss: 0.17152
    [4456]	valid_0's auc: 0.903914	valid_0's binary_logloss: 0.171537
    [4457]	valid_0's auc: 0.903909	valid_0's binary_logloss: 0.171518
    [4458]	valid_0's auc: 0.903908	valid_0's binary_logloss: 0.17151
    [4459]	valid_0's auc: 0.903908	valid_0's binary_logloss: 0.17152
    [4460]	valid_0's auc: 0.903905	valid_0's binary_logloss: 0.171508
    [4461]	valid_0's auc: 0.903906	valid_0's binary_logloss: 0.171523
    [4462]	valid_0's auc: 0.903933	valid_0's binary_logloss: 0.171499
    [4463]	valid_0's auc: 0.903951	valid_0's binary_logloss: 0.171479
    [4464]	valid_0's auc: 0.903944	valid_0's binary_logloss: 0.171461
    [4465]	valid_0's auc: 0.903949	valid_0's binary_logloss: 0.171441
    [4466]	valid_0's auc: 0.903947	valid_0's binary_logloss: 0.171426
    [4467]	valid_0's auc: 0.903949	valid_0's binary_logloss: 0.171407
    [4468]	valid_0's auc: 0.903928	valid_0's binary_logloss: 0.171392
    [4469]	valid_0's auc: 0.903927	valid_0's binary_logloss: 0.171311
    [4470]	valid_0's auc: 0.90393	valid_0's binary_logloss: 0.17132
    [4471]	valid_0's auc: 0.90393	valid_0's binary_logloss: 0.171331
    [4472]	valid_0's auc: 0.903923	valid_0's binary_logloss: 0.171288
    [4473]	valid_0's auc: 0.903925	valid_0's binary_logloss: 0.171306
    [4474]	valid_0's auc: 0.903948	valid_0's binary_logloss: 0.171254
    [4475]	valid_0's auc: 0.90395	valid_0's binary_logloss: 0.171266
    [4476]	valid_0's auc: 0.903948	valid_0's binary_logloss: 0.171282
    [4477]	valid_0's auc: 0.903965	valid_0's binary_logloss: 0.171255
    [4478]	valid_0's auc: 0.90396	valid_0's binary_logloss: 0.171231
    [4479]	valid_0's auc: 0.903961	valid_0's binary_logloss: 0.171248
    [4480]	valid_0's auc: 0.903971	valid_0's binary_logloss: 0.171222
    [4481]	valid_0's auc: 0.903958	valid_0's binary_logloss: 0.171211
    [4482]	valid_0's auc: 0.903959	valid_0's binary_logloss: 0.171227
    [4483]	valid_0's auc: 0.903966	valid_0's binary_logloss: 0.171243
    [4484]	valid_0's auc: 0.903967	valid_0's binary_logloss: 0.171255
    [4485]	valid_0's auc: 0.90397	valid_0's binary_logloss: 0.17127
    [4486]	valid_0's auc: 0.903982	valid_0's binary_logloss: 0.171248
    [4487]	valid_0's auc: 0.903988	valid_0's binary_logloss: 0.171227
    [4488]	valid_0's auc: 0.90399	valid_0's binary_logloss: 0.171236
    [4489]	valid_0's auc: 0.90399	valid_0's binary_logloss: 0.17125
    [4490]	valid_0's auc: 0.903993	valid_0's binary_logloss: 0.171264
    [4491]	valid_0's auc: 0.903997	valid_0's binary_logloss: 0.171248
    [4492]	valid_0's auc: 0.903997	valid_0's binary_logloss: 0.171256
    [4493]	valid_0's auc: 0.903998	valid_0's binary_logloss: 0.171267
    [4494]	valid_0's auc: 0.903999	valid_0's binary_logloss: 0.171278
    [4495]	valid_0's auc: 0.903984	valid_0's binary_logloss: 0.171253
    [4496]	valid_0's auc: 0.903984	valid_0's binary_logloss: 0.171266
    [4497]	valid_0's auc: 0.903971	valid_0's binary_logloss: 0.171258
    [4498]	valid_0's auc: 0.903961	valid_0's binary_logloss: 0.171244
    [4499]	valid_0's auc: 0.903964	valid_0's binary_logloss: 0.171258
    [4500]	valid_0's auc: 0.903968	valid_0's binary_logloss: 0.171272
    [4501]	valid_0's auc: 0.903968	valid_0's binary_logloss: 0.171282
    [4502]	valid_0's auc: 0.903971	valid_0's binary_logloss: 0.171295
    [4503]	valid_0's auc: 0.903976	valid_0's binary_logloss: 0.17128
    [4504]	valid_0's auc: 0.903977	valid_0's binary_logloss: 0.171293
    [4505]	valid_0's auc: 0.903979	valid_0's binary_logloss: 0.171304
    [4506]	valid_0's auc: 0.903995	valid_0's binary_logloss: 0.171267
    [4507]	valid_0's auc: 0.903996	valid_0's binary_logloss: 0.171287
    [4508]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.171248
    [4509]	valid_0's auc: 0.904013	valid_0's binary_logloss: 0.171259
    [4510]	valid_0's auc: 0.903999	valid_0's binary_logloss: 0.171254
    [4511]	valid_0's auc: 0.904024	valid_0's binary_logloss: 0.171217
    [4512]	valid_0's auc: 0.904033	valid_0's binary_logloss: 0.171163
    [4513]	valid_0's auc: 0.904035	valid_0's binary_logloss: 0.171177
    [4514]	valid_0's auc: 0.904038	valid_0's binary_logloss: 0.171193
    [4515]	valid_0's auc: 0.904031	valid_0's binary_logloss: 0.171178
    [4516]	valid_0's auc: 0.904037	valid_0's binary_logloss: 0.171195
    [4517]	valid_0's auc: 0.90404	valid_0's binary_logloss: 0.171208
    [4518]	valid_0's auc: 0.904045	valid_0's binary_logloss: 0.171226
    [4519]	valid_0's auc: 0.904024	valid_0's binary_logloss: 0.171214
    [4520]	valid_0's auc: 0.904023	valid_0's binary_logloss: 0.171147
    [4521]	valid_0's auc: 0.904021	valid_0's binary_logloss: 0.171097
    [4522]	valid_0's auc: 0.904022	valid_0's binary_logloss: 0.171077
    [4523]	valid_0's auc: 0.904042	valid_0's binary_logloss: 0.171052
    [4524]	valid_0's auc: 0.904043	valid_0's binary_logloss: 0.17104
    [4525]	valid_0's auc: 0.904026	valid_0's binary_logloss: 0.171024
    [4526]	valid_0's auc: 0.904027	valid_0's binary_logloss: 0.171034
    [4527]	valid_0's auc: 0.904029	valid_0's binary_logloss: 0.171045
    [4528]	valid_0's auc: 0.904029	valid_0's binary_logloss: 0.171031
    [4529]	valid_0's auc: 0.904032	valid_0's binary_logloss: 0.171042
    [4530]	valid_0's auc: 0.904011	valid_0's binary_logloss: 0.171031
    [4531]	valid_0's auc: 0.904013	valid_0's binary_logloss: 0.171044
    [4532]	valid_0's auc: 0.903991	valid_0's binary_logloss: 0.171032
    [4533]	valid_0's auc: 0.903996	valid_0's binary_logloss: 0.171043
    [4534]	valid_0's auc: 0.903996	valid_0's binary_logloss: 0.171052
    [4535]	valid_0's auc: 0.903993	valid_0's binary_logloss: 0.171034
    [4536]	valid_0's auc: 0.903996	valid_0's binary_logloss: 0.171049
    [4537]	valid_0's auc: 0.903998	valid_0's binary_logloss: 0.171058
    [4538]	valid_0's auc: 0.903995	valid_0's binary_logloss: 0.171039
    [4539]	valid_0's auc: 0.904012	valid_0's binary_logloss: 0.171006
    [4540]	valid_0's auc: 0.904015	valid_0's binary_logloss: 0.171019
    [4541]	valid_0's auc: 0.904001	valid_0's binary_logloss: 0.171008
    [4542]	valid_0's auc: 0.903993	valid_0's binary_logloss: 0.171001
    [4543]	valid_0's auc: 0.903999	valid_0's binary_logloss: 0.171015
    [4544]	valid_0's auc: 0.903998	valid_0's binary_logloss: 0.171028
    [4545]	valid_0's auc: 0.903983	valid_0's binary_logloss: 0.171024
    [4546]	valid_0's auc: 0.904004	valid_0's binary_logloss: 0.171007
    [4547]	valid_0's auc: 0.904	valid_0's binary_logloss: 0.170993
    [4548]	valid_0's auc: 0.903999	valid_0's binary_logloss: 0.171
    [4549]	valid_0's auc: 0.904006	valid_0's binary_logloss: 0.171011
    [4550]	valid_0's auc: 0.904006	valid_0's binary_logloss: 0.171021
    [4551]	valid_0's auc: 0.904019	valid_0's binary_logloss: 0.171006
    [4552]	valid_0's auc: 0.904007	valid_0's binary_logloss: 0.170991
    [4553]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.171009
    [4554]	valid_0's auc: 0.904009	valid_0's binary_logloss: 0.170967
    [4555]	valid_0's auc: 0.904012	valid_0's binary_logloss: 0.170978
    [4556]	valid_0's auc: 0.904016	valid_0's binary_logloss: 0.170986
    [4557]	valid_0's auc: 0.904021	valid_0's binary_logloss: 0.170994
    [4558]	valid_0's auc: 0.904008	valid_0's binary_logloss: 0.170929
    [4559]	valid_0's auc: 0.904011	valid_0's binary_logloss: 0.17094
    [4560]	valid_0's auc: 0.904018	valid_0's binary_logloss: 0.170921
    [4561]	valid_0's auc: 0.904018	valid_0's binary_logloss: 0.170939
    [4562]	valid_0's auc: 0.904021	valid_0's binary_logloss: 0.170954
    [4563]	valid_0's auc: 0.904022	valid_0's binary_logloss: 0.170968
    [4564]	valid_0's auc: 0.904017	valid_0's binary_logloss: 0.17095
    [4565]	valid_0's auc: 0.904002	valid_0's binary_logloss: 0.170946
    [4566]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.170927
    [4567]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.170944
    [4568]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.170958
    [4569]	valid_0's auc: 0.904008	valid_0's binary_logloss: 0.170965
    [4570]	valid_0's auc: 0.904012	valid_0's binary_logloss: 0.170973
    [4571]	valid_0's auc: 0.903995	valid_0's binary_logloss: 0.170961
    [4572]	valid_0's auc: 0.904	valid_0's binary_logloss: 0.170942
    [4573]	valid_0's auc: 0.904002	valid_0's binary_logloss: 0.17095
    [4574]	valid_0's auc: 0.904001	valid_0's binary_logloss: 0.170963
    [4575]	valid_0's auc: 0.904006	valid_0's binary_logloss: 0.170981
    [4576]	valid_0's auc: 0.904007	valid_0's binary_logloss: 0.170996
    [4577]	valid_0's auc: 0.904007	valid_0's binary_logloss: 0.17101
    [4578]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.171024
    [4579]	valid_0's auc: 0.90401	valid_0's binary_logloss: 0.17103
    [4580]	valid_0's auc: 0.904012	valid_0's binary_logloss: 0.171043
    [4581]	valid_0's auc: 0.903999	valid_0's binary_logloss: 0.171032
    [4582]	valid_0's auc: 0.904005	valid_0's binary_logloss: 0.170986
    [4583]	valid_0's auc: 0.904017	valid_0's binary_logloss: 0.170941
    [4584]	valid_0's auc: 0.90402	valid_0's binary_logloss: 0.17095
    [4585]	valid_0's auc: 0.904025	valid_0's binary_logloss: 0.170964
    [4586]	valid_0's auc: 0.904034	valid_0's binary_logloss: 0.17094
    [4587]	valid_0's auc: 0.904035	valid_0's binary_logloss: 0.170956
    [4588]	valid_0's auc: 0.904036	valid_0's binary_logloss: 0.170969
    [4589]	valid_0's auc: 0.904033	valid_0's binary_logloss: 0.170956
    [4590]	valid_0's auc: 0.904032	valid_0's binary_logloss: 0.170964
    [4591]	valid_0's auc: 0.904037	valid_0's binary_logloss: 0.170977
    [4592]	valid_0's auc: 0.904037	valid_0's binary_logloss: 0.170991
    [4593]	valid_0's auc: 0.904038	valid_0's binary_logloss: 0.171007
    [4594]	valid_0's auc: 0.904038	valid_0's binary_logloss: 0.171023
    [4595]	valid_0's auc: 0.90404	valid_0's binary_logloss: 0.171037
    [4596]	valid_0's auc: 0.904064	valid_0's binary_logloss: 0.170984
    [4597]	valid_0's auc: 0.904065	valid_0's binary_logloss: 0.170998
    [4598]	valid_0's auc: 0.904067	valid_0's binary_logloss: 0.171008
    [4599]	valid_0's auc: 0.904053	valid_0's binary_logloss: 0.170992
    [4600]	valid_0's auc: 0.90405	valid_0's binary_logloss: 0.170979
    [4601]	valid_0's auc: 0.904051	valid_0's binary_logloss: 0.170994
    [4602]	valid_0's auc: 0.904053	valid_0's binary_logloss: 0.171005
    [4603]	valid_0's auc: 0.90405	valid_0's binary_logloss: 0.170992
    [4604]	valid_0's auc: 0.904061	valid_0's binary_logloss: 0.170975
    [4605]	valid_0's auc: 0.904061	valid_0's binary_logloss: 0.170996
    [4606]	valid_0's auc: 0.904066	valid_0's binary_logloss: 0.170912
    [4607]	valid_0's auc: 0.904066	valid_0's binary_logloss: 0.170925
    [4608]	valid_0's auc: 0.904074	valid_0's binary_logloss: 0.170906
    [4609]	valid_0's auc: 0.904077	valid_0's binary_logloss: 0.170915
    [4610]	valid_0's auc: 0.904061	valid_0's binary_logloss: 0.170901
    [4611]	valid_0's auc: 0.90408	valid_0's binary_logloss: 0.170875
    [4612]	valid_0's auc: 0.904117	valid_0's binary_logloss: 0.170847
    [4613]	valid_0's auc: 0.904118	valid_0's binary_logloss: 0.170855
    [4614]	valid_0's auc: 0.904121	valid_0's binary_logloss: 0.170866
    [4615]	valid_0's auc: 0.904121	valid_0's binary_logloss: 0.170877
    [4616]	valid_0's auc: 0.904096	valid_0's binary_logloss: 0.170864
    [4617]	valid_0's auc: 0.904093	valid_0's binary_logloss: 0.170851
    [4618]	valid_0's auc: 0.904067	valid_0's binary_logloss: 0.170851
    [4619]	valid_0's auc: 0.904065	valid_0's binary_logloss: 0.170837
    [4620]	valid_0's auc: 0.904068	valid_0's binary_logloss: 0.170853
    [4621]	valid_0's auc: 0.90407	valid_0's binary_logloss: 0.170868
    [4622]	valid_0's auc: 0.904071	valid_0's binary_logloss: 0.170879
    [4623]	valid_0's auc: 0.904073	valid_0's binary_logloss: 0.170902
    [4624]	valid_0's auc: 0.904073	valid_0's binary_logloss: 0.17091
    [4625]	valid_0's auc: 0.904076	valid_0's binary_logloss: 0.170917
    [4626]	valid_0's auc: 0.904078	valid_0's binary_logloss: 0.170934
    [4627]	valid_0's auc: 0.904047	valid_0's binary_logloss: 0.170926
    [4628]	valid_0's auc: 0.904048	valid_0's binary_logloss: 0.170886
    [4629]	valid_0's auc: 0.904051	valid_0's binary_logloss: 0.170897
    [4630]	valid_0's auc: 0.904035	valid_0's binary_logloss: 0.170882
    [4631]	valid_0's auc: 0.904037	valid_0's binary_logloss: 0.170892
    [4632]	valid_0's auc: 0.904017	valid_0's binary_logloss: 0.170882
    [4633]	valid_0's auc: 0.904016	valid_0's binary_logloss: 0.17085
    [4634]	valid_0's auc: 0.903999	valid_0's binary_logloss: 0.17084
    [4635]	valid_0's auc: 0.903983	valid_0's binary_logloss: 0.170829
    [4636]	valid_0's auc: 0.903987	valid_0's binary_logloss: 0.170841
    [4637]	valid_0's auc: 0.903987	valid_0's binary_logloss: 0.170852
    [4638]	valid_0's auc: 0.903991	valid_0's binary_logloss: 0.170824
    [4639]	valid_0's auc: 0.90399	valid_0's binary_logloss: 0.170814
    [4640]	valid_0's auc: 0.903992	valid_0's binary_logloss: 0.170823
    [4641]	valid_0's auc: 0.903977	valid_0's binary_logloss: 0.17081
    [4642]	valid_0's auc: 0.903981	valid_0's binary_logloss: 0.17083
    [4643]	valid_0's auc: 0.903986	valid_0's binary_logloss: 0.170847
    [4644]	valid_0's auc: 0.904004	valid_0's binary_logloss: 0.170826
    [4645]	valid_0's auc: 0.904005	valid_0's binary_logloss: 0.170839
    [4646]	valid_0's auc: 0.904013	valid_0's binary_logloss: 0.170818
    [4647]	valid_0's auc: 0.904014	valid_0's binary_logloss: 0.170827
    [4648]	valid_0's auc: 0.904021	valid_0's binary_logloss: 0.170806
    [4649]	valid_0's auc: 0.904017	valid_0's binary_logloss: 0.170734
    [4650]	valid_0's auc: 0.904026	valid_0's binary_logloss: 0.170697
    [4651]	valid_0's auc: 0.904015	valid_0's binary_logloss: 0.170689
    [4652]	valid_0's auc: 0.904016	valid_0's binary_logloss: 0.170698
    [4653]	valid_0's auc: 0.904019	valid_0's binary_logloss: 0.170706
    [4654]	valid_0's auc: 0.904018	valid_0's binary_logloss: 0.170693
    [4655]	valid_0's auc: 0.904016	valid_0's binary_logloss: 0.170678
    [4656]	valid_0's auc: 0.904017	valid_0's binary_logloss: 0.170693
    [4657]	valid_0's auc: 0.904022	valid_0's binary_logloss: 0.170707
    [4658]	valid_0's auc: 0.904021	valid_0's binary_logloss: 0.17072
    [4659]	valid_0's auc: 0.904022	valid_0's binary_logloss: 0.170735
    [4660]	valid_0's auc: 0.904025	valid_0's binary_logloss: 0.170744
    [4661]	valid_0's auc: 0.904041	valid_0's binary_logloss: 0.170716
    [4662]	valid_0's auc: 0.904044	valid_0's binary_logloss: 0.170729
    [4663]	valid_0's auc: 0.904047	valid_0's binary_logloss: 0.170739
    [4664]	valid_0's auc: 0.90405	valid_0's binary_logloss: 0.170762
    [4665]	valid_0's auc: 0.904049	valid_0's binary_logloss: 0.170769
    [4666]	valid_0's auc: 0.904051	valid_0's binary_logloss: 0.170777
    [4667]	valid_0's auc: 0.904072	valid_0's binary_logloss: 0.170753
    [4668]	valid_0's auc: 0.904095	valid_0's binary_logloss: 0.170726
    [4669]	valid_0's auc: 0.904092	valid_0's binary_logloss: 0.170673
    [4670]	valid_0's auc: 0.904107	valid_0's binary_logloss: 0.170652
    [4671]	valid_0's auc: 0.904112	valid_0's binary_logloss: 0.170667
    [4672]	valid_0's auc: 0.904118	valid_0's binary_logloss: 0.170647
    [4673]	valid_0's auc: 0.904121	valid_0's binary_logloss: 0.170663
    [4674]	valid_0's auc: 0.904125	valid_0's binary_logloss: 0.17067
    [4675]	valid_0's auc: 0.904129	valid_0's binary_logloss: 0.17064
    [4676]	valid_0's auc: 0.90413	valid_0's binary_logloss: 0.170649
    [4677]	valid_0's auc: 0.904129	valid_0's binary_logloss: 0.170618
    [4678]	valid_0's auc: 0.904121	valid_0's binary_logloss: 0.170549
    [4679]	valid_0's auc: 0.904121	valid_0's binary_logloss: 0.170564
    [4680]	valid_0's auc: 0.904122	valid_0's binary_logloss: 0.17057
    [4681]	valid_0's auc: 0.90412	valid_0's binary_logloss: 0.170583
    [4682]	valid_0's auc: 0.904115	valid_0's binary_logloss: 0.170539
    [4683]	valid_0's auc: 0.904112	valid_0's binary_logloss: 0.170519
    [4684]	valid_0's auc: 0.904119	valid_0's binary_logloss: 0.170445
    [4685]	valid_0's auc: 0.904108	valid_0's binary_logloss: 0.17043
    [4686]	valid_0's auc: 0.904111	valid_0's binary_logloss: 0.17044
    [4687]	valid_0's auc: 0.904111	valid_0's binary_logloss: 0.170426
    [4688]	valid_0's auc: 0.904096	valid_0's binary_logloss: 0.170386
    [4689]	valid_0's auc: 0.904084	valid_0's binary_logloss: 0.170379
    [4690]	valid_0's auc: 0.904096	valid_0's binary_logloss: 0.170359
    [4691]	valid_0's auc: 0.904097	valid_0's binary_logloss: 0.170369
    [4692]	valid_0's auc: 0.904098	valid_0's binary_logloss: 0.170378
    [4693]	valid_0's auc: 0.904101	valid_0's binary_logloss: 0.170388
    [4694]	valid_0's auc: 0.904102	valid_0's binary_logloss: 0.170396
    [4695]	valid_0's auc: 0.904103	valid_0's binary_logloss: 0.170411
    [4696]	valid_0's auc: 0.904104	valid_0's binary_logloss: 0.170401
    [4697]	valid_0's auc: 0.904086	valid_0's binary_logloss: 0.170392
    [4698]	valid_0's auc: 0.904141	valid_0's binary_logloss: 0.170347
    [4699]	valid_0's auc: 0.904144	valid_0's binary_logloss: 0.170363
    [4700]	valid_0's auc: 0.904163	valid_0's binary_logloss: 0.170334
    [4701]	valid_0's auc: 0.904143	valid_0's binary_logloss: 0.170317
    [4702]	valid_0's auc: 0.904146	valid_0's binary_logloss: 0.170331
    [4703]	valid_0's auc: 0.90415	valid_0's binary_logloss: 0.17034
    [4704]	valid_0's auc: 0.904151	valid_0's binary_logloss: 0.170356
    [4705]	valid_0's auc: 0.904166	valid_0's binary_logloss: 0.170337
    [4706]	valid_0's auc: 0.904181	valid_0's binary_logloss: 0.170316
    [4707]	valid_0's auc: 0.904179	valid_0's binary_logloss: 0.170294
    [4708]	valid_0's auc: 0.90422	valid_0's binary_logloss: 0.170248
    [4709]	valid_0's auc: 0.904206	valid_0's binary_logloss: 0.170221
    [4710]	valid_0's auc: 0.904188	valid_0's binary_logloss: 0.170211
    [4711]	valid_0's auc: 0.904193	valid_0's binary_logloss: 0.170154
    [4712]	valid_0's auc: 0.904177	valid_0's binary_logloss: 0.170149
    [4713]	valid_0's auc: 0.904178	valid_0's binary_logloss: 0.170156
    [4714]	valid_0's auc: 0.904179	valid_0's binary_logloss: 0.170164
    [4715]	valid_0's auc: 0.904178	valid_0's binary_logloss: 0.170152
    [4716]	valid_0's auc: 0.904179	valid_0's binary_logloss: 0.17016
    [4717]	valid_0's auc: 0.90418	valid_0's binary_logloss: 0.170103
    [4718]	valid_0's auc: 0.904182	valid_0's binary_logloss: 0.170109
    [4719]	valid_0's auc: 0.904179	valid_0's binary_logloss: 0.170095
    [4720]	valid_0's auc: 0.90417	valid_0's binary_logloss: 0.170087
    [4721]	valid_0's auc: 0.904169	valid_0's binary_logloss: 0.170099
    [4722]	valid_0's auc: 0.904172	valid_0's binary_logloss: 0.170111
    [4723]	valid_0's auc: 0.904178	valid_0's binary_logloss: 0.170103
    [4724]	valid_0's auc: 0.904181	valid_0's binary_logloss: 0.170114
    [4725]	valid_0's auc: 0.904178	valid_0's binary_logloss: 0.170126
    [4726]	valid_0's auc: 0.904159	valid_0's binary_logloss: 0.170118
    [4727]	valid_0's auc: 0.904148	valid_0's binary_logloss: 0.170107
    [4728]	valid_0's auc: 0.904139	valid_0's binary_logloss: 0.170089
    [4729]	valid_0's auc: 0.904176	valid_0's binary_logloss: 0.170052
    [4730]	valid_0's auc: 0.904178	valid_0's binary_logloss: 0.170038
    [4731]	valid_0's auc: 0.904176	valid_0's binary_logloss: 0.170047
    [4732]	valid_0's auc: 0.904162	valid_0's binary_logloss: 0.170038
    [4733]	valid_0's auc: 0.904152	valid_0's binary_logloss: 0.170021
    [4734]	valid_0's auc: 0.904139	valid_0's binary_logloss: 0.170015
    [4735]	valid_0's auc: 0.904154	valid_0's binary_logloss: 0.170003
    [4736]	valid_0's auc: 0.904153	valid_0's binary_logloss: 0.169987
    [4737]	valid_0's auc: 0.904144	valid_0's binary_logloss: 0.169976
    [4738]	valid_0's auc: 0.904147	valid_0's binary_logloss: 0.169983
    [4739]	valid_0's auc: 0.904147	valid_0's binary_logloss: 0.16999
    [4740]	valid_0's auc: 0.904148	valid_0's binary_logloss: 0.170001
    [4741]	valid_0's auc: 0.90415	valid_0's binary_logloss: 0.169982
    [4742]	valid_0's auc: 0.90415	valid_0's binary_logloss: 0.16999
    [4743]	valid_0's auc: 0.904149	valid_0's binary_logloss: 0.170002
    [4744]	valid_0's auc: 0.904139	valid_0's binary_logloss: 0.169985
    [4745]	valid_0's auc: 0.904142	valid_0's binary_logloss: 0.169966
    [4746]	valid_0's auc: 0.904151	valid_0's binary_logloss: 0.169942
    [4747]	valid_0's auc: 0.904153	valid_0's binary_logloss: 0.169957
    [4748]	valid_0's auc: 0.904153	valid_0's binary_logloss: 0.169965
    [4749]	valid_0's auc: 0.904155	valid_0's binary_logloss: 0.169981
    [4750]	valid_0's auc: 0.904137	valid_0's binary_logloss: 0.169972
    [4751]	valid_0's auc: 0.904135	valid_0's binary_logloss: 0.16998
    [4752]	valid_0's auc: 0.904138	valid_0's binary_logloss: 0.169952
    [4753]	valid_0's auc: 0.904134	valid_0's binary_logloss: 0.169936
    [4754]	valid_0's auc: 0.904135	valid_0's binary_logloss: 0.169949
    [4755]	valid_0's auc: 0.904134	valid_0's binary_logloss: 0.169964
    [4756]	valid_0's auc: 0.904135	valid_0's binary_logloss: 0.169971
    [4757]	valid_0's auc: 0.904138	valid_0's binary_logloss: 0.169978
    [4758]	valid_0's auc: 0.904143	valid_0's binary_logloss: 0.169972
    [4759]	valid_0's auc: 0.904145	valid_0's binary_logloss: 0.169986
    [4760]	valid_0's auc: 0.904151	valid_0's binary_logloss: 0.169974
    [4761]	valid_0's auc: 0.904153	valid_0's binary_logloss: 0.169984
    [4762]	valid_0's auc: 0.904142	valid_0's binary_logloss: 0.169972
    [4763]	valid_0's auc: 0.904148	valid_0's binary_logloss: 0.169958
    [4764]	valid_0's auc: 0.904149	valid_0's binary_logloss: 0.169966
    [4765]	valid_0's auc: 0.904148	valid_0's binary_logloss: 0.169976
    [4766]	valid_0's auc: 0.904147	valid_0's binary_logloss: 0.169993
    [4767]	valid_0's auc: 0.904136	valid_0's binary_logloss: 0.169986
    [4768]	valid_0's auc: 0.904142	valid_0's binary_logloss: 0.169968
    [4769]	valid_0's auc: 0.90416	valid_0's binary_logloss: 0.169908
    [4770]	valid_0's auc: 0.904162	valid_0's binary_logloss: 0.169918
    [4771]	valid_0's auc: 0.90416	valid_0's binary_logloss: 0.169842
    [4772]	valid_0's auc: 0.904163	valid_0's binary_logloss: 0.169824
    [4773]	valid_0's auc: 0.904162	valid_0's binary_logloss: 0.169797
    [4774]	valid_0's auc: 0.904156	valid_0's binary_logloss: 0.169787
    [4775]	valid_0's auc: 0.904162	valid_0's binary_logloss: 0.169802
    [4776]	valid_0's auc: 0.904163	valid_0's binary_logloss: 0.169812
    [4777]	valid_0's auc: 0.904164	valid_0's binary_logloss: 0.169822
    [4778]	valid_0's auc: 0.904188	valid_0's binary_logloss: 0.169791
    [4779]	valid_0's auc: 0.904187	valid_0's binary_logloss: 0.169754
    [4780]	valid_0's auc: 0.904191	valid_0's binary_logloss: 0.169743
    [4781]	valid_0's auc: 0.904186	valid_0's binary_logloss: 0.169685
    [4782]	valid_0's auc: 0.904185	valid_0's binary_logloss: 0.169694
    [4783]	valid_0's auc: 0.904188	valid_0's binary_logloss: 0.169709
    [4784]	valid_0's auc: 0.904194	valid_0's binary_logloss: 0.169726
    [4785]	valid_0's auc: 0.904194	valid_0's binary_logloss: 0.169732
    [4786]	valid_0's auc: 0.904203	valid_0's binary_logloss: 0.169688
    [4787]	valid_0's auc: 0.904206	valid_0's binary_logloss: 0.169696
    [4788]	valid_0's auc: 0.904208	valid_0's binary_logloss: 0.169704
    [4789]	valid_0's auc: 0.904198	valid_0's binary_logloss: 0.1697
    [4790]	valid_0's auc: 0.904197	valid_0's binary_logloss: 0.169712
    [4791]	valid_0's auc: 0.904201	valid_0's binary_logloss: 0.169725
    [4792]	valid_0's auc: 0.904203	valid_0's binary_logloss: 0.16974
    [4793]	valid_0's auc: 0.904202	valid_0's binary_logloss: 0.169753
    [4794]	valid_0's auc: 0.904249	valid_0's binary_logloss: 0.169725
    [4795]	valid_0's auc: 0.904249	valid_0's binary_logloss: 0.169735
    [4796]	valid_0's auc: 0.904245	valid_0's binary_logloss: 0.169679
    [4797]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.169668
    [4798]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.169646
    [4799]	valid_0's auc: 0.904242	valid_0's binary_logloss: 0.169635
    [4800]	valid_0's auc: 0.904242	valid_0's binary_logloss: 0.169644
    [4801]	valid_0's auc: 0.904245	valid_0's binary_logloss: 0.169652
    [4802]	valid_0's auc: 0.904245	valid_0's binary_logloss: 0.169663
    [4803]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.169607
    [4804]	valid_0's auc: 0.904232	valid_0's binary_logloss: 0.16959
    [4805]	valid_0's auc: 0.904215	valid_0's binary_logloss: 0.169588
    [4806]	valid_0's auc: 0.904218	valid_0's binary_logloss: 0.169598
    [4807]	valid_0's auc: 0.904221	valid_0's binary_logloss: 0.169612
    [4808]	valid_0's auc: 0.90422	valid_0's binary_logloss: 0.169622
    [4809]	valid_0's auc: 0.904225	valid_0's binary_logloss: 0.169634
    [4810]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.169645
    [4811]	valid_0's auc: 0.904228	valid_0's binary_logloss: 0.169661
    [4812]	valid_0's auc: 0.904197	valid_0's binary_logloss: 0.169633
    [4813]	valid_0's auc: 0.904187	valid_0's binary_logloss: 0.169622
    [4814]	valid_0's auc: 0.904184	valid_0's binary_logloss: 0.169608
    [4815]	valid_0's auc: 0.904176	valid_0's binary_logloss: 0.169596
    [4816]	valid_0's auc: 0.904159	valid_0's binary_logloss: 0.169591
    [4817]	valid_0's auc: 0.904161	valid_0's binary_logloss: 0.169603
    [4818]	valid_0's auc: 0.904161	valid_0's binary_logloss: 0.169573
    [4819]	valid_0's auc: 0.904165	valid_0's binary_logloss: 0.169559
    [4820]	valid_0's auc: 0.904164	valid_0's binary_logloss: 0.169573
    [4821]	valid_0's auc: 0.904169	valid_0's binary_logloss: 0.169591
    [4822]	valid_0's auc: 0.90417	valid_0's binary_logloss: 0.169598
    [4823]	valid_0's auc: 0.904169	valid_0's binary_logloss: 0.16961
    [4824]	valid_0's auc: 0.904172	valid_0's binary_logloss: 0.169619
    [4825]	valid_0's auc: 0.904168	valid_0's binary_logloss: 0.16963
    [4826]	valid_0's auc: 0.904181	valid_0's binary_logloss: 0.169612
    [4827]	valid_0's auc: 0.904182	valid_0's binary_logloss: 0.169618
    [4828]	valid_0's auc: 0.904186	valid_0's binary_logloss: 0.16963
    [4829]	valid_0's auc: 0.904188	valid_0's binary_logloss: 0.169637
    [4830]	valid_0's auc: 0.904189	valid_0's binary_logloss: 0.169649
    [4831]	valid_0's auc: 0.904191	valid_0's binary_logloss: 0.169661
    [4832]	valid_0's auc: 0.904178	valid_0's binary_logloss: 0.169642
    [4833]	valid_0's auc: 0.904175	valid_0's binary_logloss: 0.169627
    [4834]	valid_0's auc: 0.904175	valid_0's binary_logloss: 0.16964
    [4835]	valid_0's auc: 0.904177	valid_0's binary_logloss: 0.169647
    [4836]	valid_0's auc: 0.904186	valid_0's binary_logloss: 0.169627
    [4837]	valid_0's auc: 0.904188	valid_0's binary_logloss: 0.169644
    [4838]	valid_0's auc: 0.90419	valid_0's binary_logloss: 0.169656
    [4839]	valid_0's auc: 0.904191	valid_0's binary_logloss: 0.169665
    [4840]	valid_0's auc: 0.904189	valid_0's binary_logloss: 0.169677
    [4841]	valid_0's auc: 0.904207	valid_0's binary_logloss: 0.169661
    [4842]	valid_0's auc: 0.904196	valid_0's binary_logloss: 0.169654
    [4843]	valid_0's auc: 0.904198	valid_0's binary_logloss: 0.169664
    [4844]	valid_0's auc: 0.904194	valid_0's binary_logloss: 0.169652
    [4845]	valid_0's auc: 0.904233	valid_0's binary_logloss: 0.169598
    [4846]	valid_0's auc: 0.904242	valid_0's binary_logloss: 0.169561
    [4847]	valid_0's auc: 0.90424	valid_0's binary_logloss: 0.16957
    [4848]	valid_0's auc: 0.90424	valid_0's binary_logloss: 0.169581
    [4849]	valid_0's auc: 0.904244	valid_0's binary_logloss: 0.169594
    [4850]	valid_0's auc: 0.904252	valid_0's binary_logloss: 0.16958
    [4851]	valid_0's auc: 0.904253	valid_0's binary_logloss: 0.169587
    [4852]	valid_0's auc: 0.904252	valid_0's binary_logloss: 0.169603
    [4853]	valid_0's auc: 0.904253	valid_0's binary_logloss: 0.16961
    [4854]	valid_0's auc: 0.904243	valid_0's binary_logloss: 0.169598
    [4855]	valid_0's auc: 0.904245	valid_0's binary_logloss: 0.169607
    [4856]	valid_0's auc: 0.904234	valid_0's binary_logloss: 0.169555
    [4857]	valid_0's auc: 0.904235	valid_0's binary_logloss: 0.169564
    [4858]	valid_0's auc: 0.904235	valid_0's binary_logloss: 0.169579
    [4859]	valid_0's auc: 0.904227	valid_0's binary_logloss: 0.169571
    [4860]	valid_0's auc: 0.904228	valid_0's binary_logloss: 0.169587
    [4861]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.169597
    [4862]	valid_0's auc: 0.904228	valid_0's binary_logloss: 0.169608
    [4863]	valid_0's auc: 0.90422	valid_0's binary_logloss: 0.169593
    [4864]	valid_0's auc: 0.904222	valid_0's binary_logloss: 0.16961
    [4865]	valid_0's auc: 0.904212	valid_0's binary_logloss: 0.169604
    [4866]	valid_0's auc: 0.904194	valid_0's binary_logloss: 0.169593
    [4867]	valid_0's auc: 0.904188	valid_0's binary_logloss: 0.169582
    [4868]	valid_0's auc: 0.904191	valid_0's binary_logloss: 0.169589
    [4869]	valid_0's auc: 0.904218	valid_0's binary_logloss: 0.169564
    [4870]	valid_0's auc: 0.904219	valid_0's binary_logloss: 0.169571
    [4871]	valid_0's auc: 0.904221	valid_0's binary_logloss: 0.169582
    [4872]	valid_0's auc: 0.904228	valid_0's binary_logloss: 0.169592
    [4873]	valid_0's auc: 0.904228	valid_0's binary_logloss: 0.169561
    [4874]	valid_0's auc: 0.904243	valid_0's binary_logloss: 0.169541
    [4875]	valid_0's auc: 0.90424	valid_0's binary_logloss: 0.169531
    [4876]	valid_0's auc: 0.904241	valid_0's binary_logloss: 0.169538
    [4877]	valid_0's auc: 0.904243	valid_0's binary_logloss: 0.169549
    [4878]	valid_0's auc: 0.904236	valid_0's binary_logloss: 0.169535
    [4879]	valid_0's auc: 0.904225	valid_0's binary_logloss: 0.169527
    [4880]	valid_0's auc: 0.904236	valid_0's binary_logloss: 0.169509
    [4881]	valid_0's auc: 0.904237	valid_0's binary_logloss: 0.169522
    [4882]	valid_0's auc: 0.904237	valid_0's binary_logloss: 0.16953
    [4883]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.169515
    [4884]	valid_0's auc: 0.904222	valid_0's binary_logloss: 0.169523
    [4885]	valid_0's auc: 0.904232	valid_0's binary_logloss: 0.16951
    [4886]	valid_0's auc: 0.904237	valid_0's binary_logloss: 0.169524
    [4887]	valid_0's auc: 0.904238	valid_0's binary_logloss: 0.169516
    [4888]	valid_0's auc: 0.90425	valid_0's binary_logloss: 0.169499
    [4889]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.169495
    [4890]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.169475
    [4891]	valid_0's auc: 0.904243	valid_0's binary_logloss: 0.169462
    [4892]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.169472
    [4893]	valid_0's auc: 0.90425	valid_0's binary_logloss: 0.169486
    [4894]	valid_0's auc: 0.904254	valid_0's binary_logloss: 0.169499
    [4895]	valid_0's auc: 0.904259	valid_0's binary_logloss: 0.16949
    [4896]	valid_0's auc: 0.904258	valid_0's binary_logloss: 0.169499
    [4897]	valid_0's auc: 0.904259	valid_0's binary_logloss: 0.169512
    [4898]	valid_0's auc: 0.904243	valid_0's binary_logloss: 0.169505
    [4899]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.1695
    [4900]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.169472
    [4901]	valid_0's auc: 0.904235	valid_0's binary_logloss: 0.169445
    [4902]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.169436
    [4903]	valid_0's auc: 0.904225	valid_0's binary_logloss: 0.169449
    [4904]	valid_0's auc: 0.904229	valid_0's binary_logloss: 0.169434
    [4905]	valid_0's auc: 0.904226	valid_0's binary_logloss: 0.169424
    [4906]	valid_0's auc: 0.904215	valid_0's binary_logloss: 0.169398
    [4907]	valid_0's auc: 0.904218	valid_0's binary_logloss: 0.169406
    [4908]	valid_0's auc: 0.904224	valid_0's binary_logloss: 0.16942
    [4909]	valid_0's auc: 0.904238	valid_0's binary_logloss: 0.169381
    [4910]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.169364
    [4911]	valid_0's auc: 0.904263	valid_0's binary_logloss: 0.169339
    [4912]	valid_0's auc: 0.904254	valid_0's binary_logloss: 0.169331
    [4913]	valid_0's auc: 0.904253	valid_0's binary_logloss: 0.169339
    [4914]	valid_0's auc: 0.904249	valid_0's binary_logloss: 0.169324
    [4915]	valid_0's auc: 0.904239	valid_0's binary_logloss: 0.169321
    [4916]	valid_0's auc: 0.904227	valid_0's binary_logloss: 0.169312
    [4917]	valid_0's auc: 0.904222	valid_0's binary_logloss: 0.169304
    [4918]	valid_0's auc: 0.904215	valid_0's binary_logloss: 0.16929
    [4919]	valid_0's auc: 0.904216	valid_0's binary_logloss: 0.169298
    [4920]	valid_0's auc: 0.904217	valid_0's binary_logloss: 0.169312
    [4921]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.169292
    [4922]	valid_0's auc: 0.90422	valid_0's binary_logloss: 0.169289
    [4923]	valid_0's auc: 0.904223	valid_0's binary_logloss: 0.169298
    [4924]	valid_0's auc: 0.904224	valid_0's binary_logloss: 0.169264
    [4925]	valid_0's auc: 0.904217	valid_0's binary_logloss: 0.16925
    [4926]	valid_0's auc: 0.904226	valid_0's binary_logloss: 0.169264
    [4927]	valid_0's auc: 0.904228	valid_0's binary_logloss: 0.169274
    [4928]	valid_0's auc: 0.904234	valid_0's binary_logloss: 0.169232
    [4929]	valid_0's auc: 0.904237	valid_0's binary_logloss: 0.169187
    [4930]	valid_0's auc: 0.904242	valid_0's binary_logloss: 0.169147
    [4931]	valid_0's auc: 0.904251	valid_0's binary_logloss: 0.169134
    [4932]	valid_0's auc: 0.904254	valid_0's binary_logloss: 0.16915
    [4933]	valid_0's auc: 0.904255	valid_0's binary_logloss: 0.169162
    [4934]	valid_0's auc: 0.904255	valid_0's binary_logloss: 0.169153
    [4935]	valid_0's auc: 0.904273	valid_0's binary_logloss: 0.169137
    [4936]	valid_0's auc: 0.904266	valid_0's binary_logloss: 0.169124
    [4937]	valid_0's auc: 0.904264	valid_0's binary_logloss: 0.169109
    [4938]	valid_0's auc: 0.904268	valid_0's binary_logloss: 0.169127
    [4939]	valid_0's auc: 0.904292	valid_0's binary_logloss: 0.169069
    [4940]	valid_0's auc: 0.904291	valid_0's binary_logloss: 0.169047
    [4941]	valid_0's auc: 0.90429	valid_0's binary_logloss: 0.169055
    [4942]	valid_0's auc: 0.904293	valid_0's binary_logloss: 0.169041
    [4943]	valid_0's auc: 0.904289	valid_0's binary_logloss: 0.169027
    [4944]	valid_0's auc: 0.904271	valid_0's binary_logloss: 0.168995
    [4945]	valid_0's auc: 0.904271	valid_0's binary_logloss: 0.16897
    [4946]	valid_0's auc: 0.904272	valid_0's binary_logloss: 0.168961
    [4947]	valid_0's auc: 0.904262	valid_0's binary_logloss: 0.168956
    [4948]	valid_0's auc: 0.904262	valid_0's binary_logloss: 0.168942
    [4949]	valid_0's auc: 0.904261	valid_0's binary_logloss: 0.168934
    [4950]	valid_0's auc: 0.904263	valid_0's binary_logloss: 0.168942
    [4951]	valid_0's auc: 0.904266	valid_0's binary_logloss: 0.168953
    [4952]	valid_0's auc: 0.904267	valid_0's binary_logloss: 0.168961
    [4953]	valid_0's auc: 0.904267	valid_0's binary_logloss: 0.16897
    [4954]	valid_0's auc: 0.90427	valid_0's binary_logloss: 0.168978
    [4955]	valid_0's auc: 0.904269	valid_0's binary_logloss: 0.168989
    [4956]	valid_0's auc: 0.904273	valid_0's binary_logloss: 0.169004
    [4957]	valid_0's auc: 0.904276	valid_0's binary_logloss: 0.168989
    [4958]	valid_0's auc: 0.904305	valid_0's binary_logloss: 0.168972
    [4959]	valid_0's auc: 0.904288	valid_0's binary_logloss: 0.168968
    [4960]	valid_0's auc: 0.904291	valid_0's binary_logloss: 0.168977
    [4961]	valid_0's auc: 0.90429	valid_0's binary_logloss: 0.168984
    [4962]	valid_0's auc: 0.904296	valid_0's binary_logloss: 0.16897
    [4963]	valid_0's auc: 0.904277	valid_0's binary_logloss: 0.168969
    [4964]	valid_0's auc: 0.904277	valid_0's binary_logloss: 0.168975
    [4965]	valid_0's auc: 0.904278	valid_0's binary_logloss: 0.168987
    [4966]	valid_0's auc: 0.904282	valid_0's binary_logloss: 0.168975
    [4967]	valid_0's auc: 0.904283	valid_0's binary_logloss: 0.168988
    [4968]	valid_0's auc: 0.904276	valid_0's binary_logloss: 0.168975
    [4969]	valid_0's auc: 0.904279	valid_0's binary_logloss: 0.16899
    [4970]	valid_0's auc: 0.904282	valid_0's binary_logloss: 0.169002
    [4971]	valid_0's auc: 0.904282	valid_0's binary_logloss: 0.169012
    [4972]	valid_0's auc: 0.904282	valid_0's binary_logloss: 0.169019
    [4973]	valid_0's auc: 0.904291	valid_0's binary_logloss: 0.169005
    [4974]	valid_0's auc: 0.904294	valid_0's binary_logloss: 0.169012
    [4975]	valid_0's auc: 0.904294	valid_0's binary_logloss: 0.169021
    [4976]	valid_0's auc: 0.904297	valid_0's binary_logloss: 0.169034
    [4977]	valid_0's auc: 0.904307	valid_0's binary_logloss: 0.169019
    [4978]	valid_0's auc: 0.904309	valid_0's binary_logloss: 0.168989
    [4979]	valid_0's auc: 0.904316	valid_0's binary_logloss: 0.168975
    [4980]	valid_0's auc: 0.904335	valid_0's binary_logloss: 0.168945
    [4981]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.168908
    [4982]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.168888
    [4983]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.1689
    [4984]	valid_0's auc: 0.904336	valid_0's binary_logloss: 0.168884
    [4985]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.168896
    [4986]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.168882
    [4987]	valid_0's auc: 0.904333	valid_0's binary_logloss: 0.168878
    [4988]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.168812
    [4989]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168799
    [4990]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.168785
    [4991]	valid_0's auc: 0.904339	valid_0's binary_logloss: 0.168774
    [4992]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.168786
    [4993]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.168795
    [4994]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.168805
    [4995]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.168811
    [4996]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168771
    [4997]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.168759
    [4998]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.168749
    [4999]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.168758
    [5000]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168767
    [5001]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.168781
    [5002]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.168791
    [5003]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.16878
    [5004]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.168775
    [5005]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.168767
    [5006]	valid_0's auc: 0.904335	valid_0's binary_logloss: 0.168778
    [5007]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.168788
    [5008]	valid_0's auc: 0.904345	valid_0's binary_logloss: 0.168772
    [5009]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.168757
    [5010]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.168765
    [5011]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.168774
    [5012]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.16878
    [5013]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.168785
    [5014]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.168793
    [5015]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.168783
    [5016]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.168745
    [5017]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.168754
    [5018]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.168744
    [5019]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.168734
    [5020]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.168741
    [5021]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.168752
    [5022]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.168765
    [5023]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.168776
    [5024]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.168767
    [5025]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.168777
    [5026]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.168791
    [5027]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.168769
    [5028]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.168778
    [5029]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.168789
    [5030]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.168796
    [5031]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.16878
    [5032]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.16879
    [5033]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.168781
    [5034]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.168746
    [5035]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.16873
    [5036]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.168719
    [5037]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.168727
    [5038]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.168742
    [5039]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.168678
    [5040]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.168687
    [5041]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.168695
    [5042]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.168689
    [5043]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.168679
    [5044]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.168655
    [5045]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.168642
    [5046]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.168631
    [5047]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.168639
    [5048]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.168603
    [5049]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.168568
    [5050]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.168577
    [5051]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.168585
    [5052]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.168575
    [5053]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.16856
    [5054]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.16855
    [5055]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.168561
    [5056]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.168573
    [5057]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.168582
    [5058]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.168593
    [5059]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.168601
    [5060]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.168589
    [5061]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.168597
    [5062]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.168582
    [5063]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.168594
    [5064]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.168603
    [5065]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.168545
    [5066]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.168536
    [5067]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.168552
    [5068]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.168562
    [5069]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.168525
    [5070]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.168535
    [5071]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.168543
    [5072]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.168508
    [5073]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.168499
    [5074]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.168507
    [5075]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.168504
    [5076]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.168448
    [5077]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.168439
    [5078]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.168424
    [5079]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.168439
    [5080]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.168445
    [5081]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.168431
    [5082]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.168438
    [5083]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.168388
    [5084]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.168395
    [5085]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.168406
    [5086]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.168414
    [5087]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.168424
    [5088]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.168411
    [5089]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.168404
    [5090]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.168413
    [5091]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.168408
    [5092]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.168398
    [5093]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.168389
    [5094]	valid_0's auc: 0.904356	valid_0's binary_logloss: 0.168398
    [5095]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.168407
    [5096]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.168418
    [5097]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.168427
    [5098]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.168434
    [5099]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.168442
    [5100]	valid_0's auc: 0.904345	valid_0's binary_logloss: 0.168435
    [5101]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.168425
    [5102]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168432
    [5103]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168442
    [5104]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.168428
    [5105]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.168418
    [5106]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.168428
    [5107]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.168426
    [5108]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.168433
    [5109]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.168444
    [5110]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.168433
    [5111]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.16844
    [5112]	valid_0's auc: 0.904333	valid_0's binary_logloss: 0.168433
    [5113]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.168446
    [5114]	valid_0's auc: 0.904335	valid_0's binary_logloss: 0.168453
    [5115]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.168439
    [5116]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.168448
    [5117]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168436
    [5118]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.16843
    [5119]	valid_0's auc: 0.904328	valid_0's binary_logloss: 0.168419
    [5120]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.168424
    [5121]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.168395
    [5122]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.16836
    [5123]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.168365
    [5124]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.168361
    [5125]	valid_0's auc: 0.904328	valid_0's binary_logloss: 0.168348
    [5126]	valid_0's auc: 0.904329	valid_0's binary_logloss: 0.168358
    [5127]	valid_0's auc: 0.904319	valid_0's binary_logloss: 0.168353
    [5128]	valid_0's auc: 0.904319	valid_0's binary_logloss: 0.168306
    [5129]	valid_0's auc: 0.904303	valid_0's binary_logloss: 0.168298
    [5130]	valid_0's auc: 0.904306	valid_0's binary_logloss: 0.168309
    [5131]	valid_0's auc: 0.904301	valid_0's binary_logloss: 0.168303
    [5132]	valid_0's auc: 0.904309	valid_0's binary_logloss: 0.16829
    [5133]	valid_0's auc: 0.904311	valid_0's binary_logloss: 0.168299
    [5134]	valid_0's auc: 0.904307	valid_0's binary_logloss: 0.168314
    [5135]	valid_0's auc: 0.904307	valid_0's binary_logloss: 0.168326
    [5136]	valid_0's auc: 0.904308	valid_0's binary_logloss: 0.168337
    [5137]	valid_0's auc: 0.904308	valid_0's binary_logloss: 0.16835
    [5138]	valid_0's auc: 0.904305	valid_0's binary_logloss: 0.168336
    [5139]	valid_0's auc: 0.904307	valid_0's binary_logloss: 0.168347
    [5140]	valid_0's auc: 0.904296	valid_0's binary_logloss: 0.168347
    [5141]	valid_0's auc: 0.904301	valid_0's binary_logloss: 0.168359
    [5142]	valid_0's auc: 0.904309	valid_0's binary_logloss: 0.168349
    [5143]	valid_0's auc: 0.90431	valid_0's binary_logloss: 0.16836
    [5144]	valid_0's auc: 0.904316	valid_0's binary_logloss: 0.168334
    [5145]	valid_0's auc: 0.90433	valid_0's binary_logloss: 0.168314
    [5146]	valid_0's auc: 0.904332	valid_0's binary_logloss: 0.168321
    [5147]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168307
    [5148]	valid_0's auc: 0.904348	valid_0's binary_logloss: 0.168315
    [5149]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.168321
    [5150]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.16833
    [5151]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.168311
    [5152]	valid_0's auc: 0.904329	valid_0's binary_logloss: 0.168306
    [5153]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.168281
    [5154]	valid_0's auc: 0.904325	valid_0's binary_logloss: 0.168273
    [5155]	valid_0's auc: 0.904316	valid_0's binary_logloss: 0.168264
    [5156]	valid_0's auc: 0.904319	valid_0's binary_logloss: 0.168271
    [5157]	valid_0's auc: 0.904323	valid_0's binary_logloss: 0.168283
    [5158]	valid_0's auc: 0.904324	valid_0's binary_logloss: 0.16827
    [5159]	valid_0's auc: 0.904327	valid_0's binary_logloss: 0.168278
    [5160]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.168263
    [5161]	valid_0's auc: 0.904336	valid_0's binary_logloss: 0.168251
    [5162]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.168234
    [5163]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.168244
    [5164]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.168253
    [5165]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.168261
    [5166]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.168275
    [5167]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.168287
    [5168]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.168295
    [5169]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.168305
    [5170]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.168291
    [5171]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.168299
    [5172]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.168311
    [5173]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.168321
    [5174]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.168304
    [5175]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.168294
    [5176]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.168284
    [5177]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.168291
    [5178]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.168273
    [5179]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.168264
    [5180]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.168256
    [5181]	valid_0's auc: 0.904365	valid_0's binary_logloss: 0.168264
    [5182]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.168273
    [5183]	valid_0's auc: 0.904367	valid_0's binary_logloss: 0.168282
    [5184]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.168269
    [5185]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.168261
    [5186]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.168268
    [5187]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.168275
    [5188]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.168272
    [5189]	valid_0's auc: 0.904338	valid_0's binary_logloss: 0.168283
    [5190]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.168292
    [5191]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.168302
    [5192]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.16829
    [5193]	valid_0's auc: 0.904339	valid_0's binary_logloss: 0.168282
    [5194]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.168265
    [5195]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.16826
    [5196]	valid_0's auc: 0.904348	valid_0's binary_logloss: 0.168236
    [5197]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.168229
    [5198]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.168214
    [5199]	valid_0's auc: 0.904356	valid_0's binary_logloss: 0.168167
    [5200]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.168157
    [5201]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.168162
    [5202]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.16817
    [5203]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.16812
    [5204]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.168105
    [5205]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.168114
    [5206]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.168104
    [5207]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.168094
    [5208]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.168105
    [5209]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.168113
    [5210]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.168122
    [5211]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.168133
    [5212]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.168142
    [5213]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.168151
    [5214]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.168165
    [5215]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.168153
    [5216]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.168149
    [5217]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.168146
    [5218]	valid_0's auc: 0.904338	valid_0's binary_logloss: 0.168114
    [5219]	valid_0's auc: 0.904335	valid_0's binary_logloss: 0.168123
    [5220]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.168129
    [5221]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.168136
    [5222]	valid_0's auc: 0.904322	valid_0's binary_logloss: 0.168132
    [5223]	valid_0's auc: 0.904322	valid_0's binary_logloss: 0.168138
    [5224]	valid_0's auc: 0.904311	valid_0's binary_logloss: 0.168132
    [5225]	valid_0's auc: 0.9043	valid_0's binary_logloss: 0.168123
    [5226]	valid_0's auc: 0.904302	valid_0's binary_logloss: 0.168138
    [5227]	valid_0's auc: 0.904301	valid_0's binary_logloss: 0.168147
    [5228]	valid_0's auc: 0.904305	valid_0's binary_logloss: 0.16816
    [5229]	valid_0's auc: 0.904313	valid_0's binary_logloss: 0.168142
    [5230]	valid_0's auc: 0.904311	valid_0's binary_logloss: 0.16813
    [5231]	valid_0's auc: 0.904321	valid_0's binary_logloss: 0.168096
    [5232]	valid_0's auc: 0.904302	valid_0's binary_logloss: 0.168097
    [5233]	valid_0's auc: 0.904303	valid_0's binary_logloss: 0.168102
    [5234]	valid_0's auc: 0.904308	valid_0's binary_logloss: 0.168046
    [5235]	valid_0's auc: 0.904311	valid_0's binary_logloss: 0.168054
    [5236]	valid_0's auc: 0.904295	valid_0's binary_logloss: 0.168047
    [5237]	valid_0's auc: 0.904298	valid_0's binary_logloss: 0.168057
    [5238]	valid_0's auc: 0.904299	valid_0's binary_logloss: 0.168062
    [5239]	valid_0's auc: 0.904304	valid_0's binary_logloss: 0.168069
    [5240]	valid_0's auc: 0.904306	valid_0's binary_logloss: 0.168081
    [5241]	valid_0's auc: 0.904288	valid_0's binary_logloss: 0.168033
    [5242]	valid_0's auc: 0.904282	valid_0's binary_logloss: 0.16801
    [5243]	valid_0's auc: 0.90428	valid_0's binary_logloss: 0.168018
    [5244]	valid_0's auc: 0.904261	valid_0's binary_logloss: 0.168016
    [5245]	valid_0's auc: 0.904239	valid_0's binary_logloss: 0.168002
    [5246]	valid_0's auc: 0.904234	valid_0's binary_logloss: 0.167992
    [5247]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.167981
    [5248]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.167988
    [5249]	valid_0's auc: 0.904255	valid_0's binary_logloss: 0.167996
    [5250]	valid_0's auc: 0.904252	valid_0's binary_logloss: 0.167989
    [5251]	valid_0's auc: 0.904251	valid_0's binary_logloss: 0.167996
    [5252]	valid_0's auc: 0.904246	valid_0's binary_logloss: 0.167982
    [5253]	valid_0's auc: 0.904258	valid_0's binary_logloss: 0.167975
    [5254]	valid_0's auc: 0.904258	valid_0's binary_logloss: 0.167984
    [5255]	valid_0's auc: 0.904263	valid_0's binary_logloss: 0.16799
    [5256]	valid_0's auc: 0.904262	valid_0's binary_logloss: 0.167995
    [5257]	valid_0's auc: 0.904264	valid_0's binary_logloss: 0.168004
    [5258]	valid_0's auc: 0.904267	valid_0's binary_logloss: 0.167982
    [5259]	valid_0's auc: 0.904269	valid_0's binary_logloss: 0.167991
    [5260]	valid_0's auc: 0.904269	valid_0's binary_logloss: 0.168002
    [5261]	valid_0's auc: 0.90426	valid_0's binary_logloss: 0.16797
    [5262]	valid_0's auc: 0.904252	valid_0's binary_logloss: 0.167948
    [5263]	valid_0's auc: 0.904255	valid_0's binary_logloss: 0.167935
    [5264]	valid_0's auc: 0.904256	valid_0's binary_logloss: 0.167944
    [5265]	valid_0's auc: 0.904256	valid_0's binary_logloss: 0.16795
    [5266]	valid_0's auc: 0.904258	valid_0's binary_logloss: 0.167958
    [5267]	valid_0's auc: 0.90425	valid_0's binary_logloss: 0.167951
    [5268]	valid_0's auc: 0.90426	valid_0's binary_logloss: 0.167937
    [5269]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.16793
    [5270]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.167935
    [5271]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.167944
    [5272]	valid_0's auc: 0.904257	valid_0's binary_logloss: 0.167932
    [5273]	valid_0's auc: 0.904257	valid_0's binary_logloss: 0.167938
    [5274]	valid_0's auc: 0.904245	valid_0's binary_logloss: 0.167933
    [5275]	valid_0's auc: 0.904246	valid_0's binary_logloss: 0.167923
    [5276]	valid_0's auc: 0.904231	valid_0's binary_logloss: 0.16792
    [5277]	valid_0's auc: 0.904239	valid_0's binary_logloss: 0.167908
    [5278]	valid_0's auc: 0.904241	valid_0's binary_logloss: 0.167918
    [5279]	valid_0's auc: 0.904255	valid_0's binary_logloss: 0.167868
    [5280]	valid_0's auc: 0.904257	valid_0's binary_logloss: 0.167875
    [5281]	valid_0's auc: 0.904244	valid_0's binary_logloss: 0.167858
    [5282]	valid_0's auc: 0.904247	valid_0's binary_logloss: 0.167871
    [5283]	valid_0's auc: 0.904227	valid_0's binary_logloss: 0.167852
    [5284]	valid_0's auc: 0.904236	valid_0's binary_logloss: 0.167819
    [5285]	valid_0's auc: 0.904242	valid_0's binary_logloss: 0.167807
    [5286]	valid_0's auc: 0.904245	valid_0's binary_logloss: 0.167814
    [5287]	valid_0's auc: 0.90425	valid_0's binary_logloss: 0.167824
    [5288]	valid_0's auc: 0.904283	valid_0's binary_logloss: 0.167767
    [5289]	valid_0's auc: 0.904285	valid_0's binary_logloss: 0.167756
    [5290]	valid_0's auc: 0.904289	valid_0's binary_logloss: 0.167763
    [5291]	valid_0's auc: 0.904297	valid_0's binary_logloss: 0.167755
    [5292]	valid_0's auc: 0.904285	valid_0's binary_logloss: 0.167749
    [5293]	valid_0's auc: 0.904287	valid_0's binary_logloss: 0.167755
    [5294]	valid_0's auc: 0.904288	valid_0's binary_logloss: 0.167763
    [5295]	valid_0's auc: 0.904292	valid_0's binary_logloss: 0.167753
    [5296]	valid_0's auc: 0.904295	valid_0's binary_logloss: 0.167742
    [5297]	valid_0's auc: 0.904289	valid_0's binary_logloss: 0.167716
    [5298]	valid_0's auc: 0.904293	valid_0's binary_logloss: 0.167722
    [5299]	valid_0's auc: 0.904286	valid_0's binary_logloss: 0.167719
    [5300]	valid_0's auc: 0.904318	valid_0's binary_logloss: 0.167695
    [5301]	valid_0's auc: 0.904319	valid_0's binary_logloss: 0.16771
    [5302]	valid_0's auc: 0.904316	valid_0's binary_logloss: 0.167687
    [5303]	valid_0's auc: 0.904323	valid_0's binary_logloss: 0.167693
    [5304]	valid_0's auc: 0.904328	valid_0's binary_logloss: 0.167676
    [5305]	valid_0's auc: 0.904327	valid_0's binary_logloss: 0.167687
    [5306]	valid_0's auc: 0.904327	valid_0's binary_logloss: 0.167694
    [5307]	valid_0's auc: 0.904319	valid_0's binary_logloss: 0.167686
    [5308]	valid_0's auc: 0.90432	valid_0's binary_logloss: 0.167694
    [5309]	valid_0's auc: 0.904311	valid_0's binary_logloss: 0.167687
    [5310]	valid_0's auc: 0.904311	valid_0's binary_logloss: 0.167692
    [5311]	valid_0's auc: 0.904313	valid_0's binary_logloss: 0.167701
    [5312]	valid_0's auc: 0.904333	valid_0's binary_logloss: 0.167687
    [5313]	valid_0's auc: 0.904338	valid_0's binary_logloss: 0.167698
    [5314]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.167708
    [5315]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.167692
    [5316]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.16768
    [5317]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.167645
    [5318]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.167651
    [5319]	valid_0's auc: 0.904338	valid_0's binary_logloss: 0.167641
    [5320]	valid_0's auc: 0.904338	valid_0's binary_logloss: 0.167654
    [5321]	valid_0's auc: 0.904338	valid_0's binary_logloss: 0.167632
    [5322]	valid_0's auc: 0.904336	valid_0's binary_logloss: 0.167638
    [5323]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.167628
    [5324]	valid_0's auc: 0.904335	valid_0's binary_logloss: 0.167606
    [5325]	valid_0's auc: 0.904328	valid_0's binary_logloss: 0.1676
    [5326]	valid_0's auc: 0.90433	valid_0's binary_logloss: 0.167606
    [5327]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.167597
    [5328]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.167604
    [5329]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.167598
    [5330]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.167584
    [5331]	valid_0's auc: 0.904339	valid_0's binary_logloss: 0.167591
    [5332]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.167598
    [5333]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.167607
    [5334]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.167591
    [5335]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.167582
    [5336]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.167569
    [5337]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167563
    [5338]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.16757
    [5339]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.167562
    [5340]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.167543
    [5341]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.16755
    [5342]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.167544
    [5343]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.167552
    [5344]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.167558
    [5345]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.167533
    [5346]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.167527
    [5347]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.16752
    [5348]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.167526
    [5349]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.167511
    [5350]	valid_0's auc: 0.904344	valid_0's binary_logloss: 0.16752
    [5351]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.167513
    [5352]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.167496
    [5353]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.167503
    [5354]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.167489
    [5355]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.167497
    [5356]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.167508
    [5357]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.167465
    [5358]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.167449
    [5359]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.167383
    [5360]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.167375
    [5361]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.167368
    [5362]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.167374
    [5363]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.167383
    [5364]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.167389
    [5365]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.167395
    [5366]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.16739
    [5367]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.167357
    [5368]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.167349
    [5369]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.167364
    [5370]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.167353
    [5371]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.167328
    [5372]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.167335
    [5373]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.16734
    [5374]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.167346
    [5375]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.167355
    [5376]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.167363
    [5377]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.16737
    [5378]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.167364
    [5379]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.16737
    [5380]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.16736
    [5381]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.167354
    [5382]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.16737
    [5383]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.167376
    [5384]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.167389
    [5385]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.167407
    [5386]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.167403
    [5387]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.167388
    [5388]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.167384
    [5389]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.167397
    [5390]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.167403
    [5391]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.167391
    [5392]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.167383
    [5393]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.167372
    [5394]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.16736
    [5395]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.167327
    [5396]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.167295
    [5397]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.167304
    [5398]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.167286
    [5399]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.167291
    [5400]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.167283
    [5401]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.167272
    [5402]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.16724
    [5403]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.167246
    [5404]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.167255
    [5405]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.167262
    [5406]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.167253
    [5407]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.167246
    [5408]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.167242
    [5409]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.167248
    [5410]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.167258
    [5411]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.16725
    [5412]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.167259
    [5413]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.167266
    [5414]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.167255
    [5415]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.167268
    [5416]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.167222
    [5417]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.167211
    [5418]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.167216
    [5419]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.167221
    [5420]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.167219
    [5421]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.167226
    [5422]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.167235
    [5423]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.167244
    [5424]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.167239
    [5425]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.167236
    [5426]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.167215
    [5427]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.167208
    [5428]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.167197
    [5429]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.167205
    [5430]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.16718
    [5431]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.167192
    [5432]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.167202
    [5433]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.167194
    [5434]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.167154
    [5435]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.167144
    [5436]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.167138
    [5437]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.167131
    [5438]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.167138
    [5439]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.167125
    [5440]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.16712
    [5441]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.167117
    [5442]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.16711
    [5443]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.167105
    [5444]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.167088
    [5445]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.167081
    [5446]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.167089
    [5447]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.167099
    [5448]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.167099
    [5449]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.167093
    [5450]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.167104
    [5451]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.167096
    [5452]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.167089
    [5453]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.167096
    [5454]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.167103
    [5455]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.167096
    [5456]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.167092
    [5457]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.167091
    [5458]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.167083
    [5459]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.167078
    [5460]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.167069
    [5461]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.167078
    [5462]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.167085
    [5463]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.167074
    [5464]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.167068
    [5465]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.167076
    [5466]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.167058
    [5467]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.167071
    [5468]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.167079
    [5469]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.167059
    [5470]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.167057
    [5471]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.167064
    [5472]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.167068
    [5473]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.167074
    [5474]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.16708
    [5475]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.167087
    [5476]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.16708
    [5477]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.167086
    [5478]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.16707
    [5479]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.167066
    [5480]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.167073
    [5481]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.167047
    [5482]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.167032
    [5483]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.167033
    [5484]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.16701
    [5485]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.167017
    [5486]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.167012
    [5487]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.167015
    [5488]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.167024
    [5489]	valid_0's auc: 0.904336	valid_0's binary_logloss: 0.167026
    [5490]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.167036
    [5491]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.167025
    [5492]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.167
    [5493]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.167006
    [5494]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.167013
    [5495]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.167024
    [5496]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.167031
    [5497]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.167019
    [5498]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.167027
    [5499]	valid_0's auc: 0.904367	valid_0's binary_logloss: 0.167018
    [5500]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.167023
    [5501]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167021
    [5502]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.167031
    [5503]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.167027
    [5504]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.167018
    [5505]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.167011
    [5506]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.167017
    [5507]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.167024
    [5508]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.167031
    [5509]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.16704
    [5510]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.167046
    [5511]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.167039
    [5512]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.167019
    [5513]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.167007
    [5514]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.167015
    [5515]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.167027
    [5516]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.167031
    [5517]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.167038
    [5518]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.167048
    [5519]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.167044
    [5520]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.167039
    [5521]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.167038
    [5522]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.167044
    [5523]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.16703
    [5524]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.167037
    [5525]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.167043
    [5526]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.167047
    [5527]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.167046
    [5528]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167041
    [5529]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.167036
    [5530]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.167045
    [5531]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167053
    [5532]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.167058
    [5533]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.167066
    [5534]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.167047
    [5535]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.167054
    [5536]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.167059
    [5537]	valid_0's auc: 0.904365	valid_0's binary_logloss: 0.167052
    [5538]	valid_0's auc: 0.904367	valid_0's binary_logloss: 0.167056
    [5539]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.167063
    [5540]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.167052
    [5541]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.16706
    [5542]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.167069
    [5543]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.16706
    [5544]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.167055
    [5545]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.167063
    [5546]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.167076
    [5547]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.167081
    [5548]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.16707
    [5549]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.167081
    [5550]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.167089
    [5551]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.167079
    [5552]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.167073
    [5553]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.167067
    [5554]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.167072
    [5555]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.167053
    [5556]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.167058
    [5557]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.167065
    [5558]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.167058
    [5559]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.167055
    [5560]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.167048
    [5561]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.167037
    [5562]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.167006
    [5563]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.167015
    [5564]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167023
    [5565]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.167029
    [5566]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.167039
    [5567]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.16705
    [5568]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.16706
    [5569]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.16705
    [5570]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167047
    [5571]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.167053
    [5572]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.167022
    [5573]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.167012
    [5574]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167006
    [5575]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.167012
    [5576]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.16698
    [5577]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.166975
    [5578]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.166925
    [5579]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.166932
    [5580]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166925
    [5581]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.166932
    [5582]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166939
    [5583]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.166931
    [5584]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.166918
    [5585]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.16693
    [5586]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.166906
    [5587]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.166912
    [5588]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.16689
    [5589]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.166883
    [5590]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.166859
    [5591]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.16685
    [5592]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.166857
    [5593]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.166855
    [5594]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.166852
    [5595]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166849
    [5596]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.166838
    [5597]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.166836
    [5598]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.166826
    [5599]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.166817
    [5600]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166772
    [5601]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.166769
    [5602]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.166759
    [5603]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.166773
    [5604]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.166772
    [5605]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.16678
    [5606]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.166786
    [5607]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.166796
    [5608]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.1668
    [5609]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166805
    [5610]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.166812
    [5611]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.166821
    [5612]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.166828
    [5613]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.166839
    [5614]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166846
    [5615]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.166842
    [5616]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.166847
    [5617]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.166842
    [5618]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.166839
    [5619]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.166851
    [5620]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.166858
    [5621]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.166853
    [5622]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.166862
    [5623]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.166845
    [5624]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.166844
    [5625]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.16685
    [5626]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.166857
    [5627]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.166842
    [5628]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.166849
    [5629]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.166856
    [5630]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.166849
    [5631]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.166856
    [5632]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.166861
    [5633]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.166813
    [5634]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.166819
    [5635]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.166806
    [5636]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.166811
    [5637]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.166819
    [5638]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.166813
    [5639]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.166801
    [5640]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.166792
    [5641]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.166799
    [5642]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.166796
    [5643]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.166793
    [5644]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.166788
    [5645]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166784
    [5646]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.166775
    [5647]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.166768
    [5648]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.166778
    [5649]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.166785
    [5650]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.166782
    [5651]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.166789
    [5652]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.166778
    [5653]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.166772
    [5654]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.166754
    [5655]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.166748
    [5656]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.166756
    [5657]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166746
    [5658]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166733
    [5659]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.166733
    [5660]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.166736
    [5661]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166719
    [5662]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166725
    [5663]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166732
    [5664]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.166727
    [5665]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.166734
    [5666]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.166717
    [5667]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.166708
    [5668]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.166702
    [5669]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.166693
    [5670]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.166701
    [5671]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.166709
    [5672]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.166698
    [5673]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.166693
    [5674]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.166699
    [5675]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.166703
    [5676]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.166702
    [5677]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.166709
    [5678]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.16668
    [5679]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.166635
    [5680]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.166609
    [5681]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.166604
    [5682]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.166612
    [5683]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.166617
    [5684]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.166602
    [5685]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.16661
    [5686]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.166617
    [5687]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.166631
    [5688]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.166638
    [5689]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.166633
    [5690]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.166628
    [5691]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.166601
    [5692]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.166605
    [5693]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.16661
    [5694]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.166618
    [5695]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.166612
    [5696]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.166621
    [5697]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.166612
    [5698]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.166609
    [5699]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.166618
    [5700]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.166602
    [5701]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.166607
    [5702]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.166614
    [5703]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.166623
    [5704]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.166629
    [5705]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.166637
    [5706]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.166644
    [5707]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.166655
    [5708]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.166661
    [5709]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.166669
    [5710]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.166669
    [5711]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166643
    [5712]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.166624
    [5713]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.166629
    [5714]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.166636
    [5715]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.16663
    [5716]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.166623
    [5717]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.16663
    [5718]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.166637
    [5719]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.166643
    [5720]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166651
    [5721]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.166647
    [5722]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.166641
    [5723]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.166625
    [5724]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166632
    [5725]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166638
    [5726]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.166648
    [5727]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.166658
    [5728]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.166665
    [5729]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.166656
    [5730]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.166646
    [5731]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.166652
    [5732]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.16664
    [5733]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.166646
    [5734]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.166651
    [5735]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.16665
    [5736]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.166659
    [5737]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166645
    [5738]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.166647
    [5739]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.166654
    [5740]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.166638
    [5741]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.166647
    [5742]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.166636
    [5743]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.16663
    [5744]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.166625
    [5745]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.166632
    [5746]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.166637
    [5747]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.166645
    [5748]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.166657
    [5749]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.166665
    [5750]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.166656
    [5751]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.166665
    [5752]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.166674
    [5753]	valid_0's auc: 0.904372	valid_0's binary_logloss: 0.166673
    [5754]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.166684
    [5755]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.166674
    [5756]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.166679
    [5757]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.166687
    [5758]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166695
    [5759]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.166703
    [5760]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.166709
    [5761]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.16667
    [5762]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166667
    [5763]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.166665
    [5764]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.166657
    [5765]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.166649
    [5766]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.16664
    [5767]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.166648
    [5768]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.166653
    [5769]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.166662
    [5770]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.166657
    [5771]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.166666
    [5772]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.166639
    [5773]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.166647
    [5774]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.166652
    [5775]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.16664
    [5776]	valid_0's auc: 0.904356	valid_0's binary_logloss: 0.166646
    [5777]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.166655
    [5778]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.166649
    [5779]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.166634
    [5780]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.166593
    [5781]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.166564
    [5782]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.166568
    [5783]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.166576
    [5784]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.166559
    [5785]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.166542
    [5786]	valid_0's auc: 0.904367	valid_0's binary_logloss: 0.166547
    [5787]	valid_0's auc: 0.904365	valid_0's binary_logloss: 0.166553
    [5788]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.166536
    [5789]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.166535
    [5790]	valid_0's auc: 0.904365	valid_0's binary_logloss: 0.166521
    [5791]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.166518
    [5792]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166506
    [5793]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.166513
    [5794]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166518
    [5795]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.166524
    [5796]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.166533
    [5797]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.16653
    [5798]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.166519
    [5799]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.166525
    [5800]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166532
    [5801]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.166539
    [5802]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.16653
    [5803]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166484
    [5804]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166495
    [5805]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166507
    [5806]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.166516
    [5807]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.166508
    [5808]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.166517
    [5809]	valid_0's auc: 0.904367	valid_0's binary_logloss: 0.166511
    [5810]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.166517
    [5811]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.166514
    [5812]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.166522
    [5813]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.166534
    [5814]	valid_0's auc: 0.904356	valid_0's binary_logloss: 0.166541
    [5815]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.16655
    [5816]	valid_0's auc: 0.904351	valid_0's binary_logloss: 0.166545
    [5817]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.166553
    [5818]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.166561
    [5819]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.16656
    [5820]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.16657
    [5821]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.166578
    [5822]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.16657
    [5823]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.16656
    [5824]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.166566
    [5825]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.166574
    [5826]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.166578
    [5827]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.166582
    [5828]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.166575
    [5829]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.166583
    [5830]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.166594
    [5831]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.166601
    [5832]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.16661
    [5833]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.166617
    [5834]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.166624
    [5835]	valid_0's auc: 0.904325	valid_0's binary_logloss: 0.16662
    [5836]	valid_0's auc: 0.904309	valid_0's binary_logloss: 0.166612
    [5837]	valid_0's auc: 0.904326	valid_0's binary_logloss: 0.166555
    [5838]	valid_0's auc: 0.904329	valid_0's binary_logloss: 0.166564
    [5839]	valid_0's auc: 0.904329	valid_0's binary_logloss: 0.166571
    [5840]	valid_0's auc: 0.904323	valid_0's binary_logloss: 0.166531
    [5841]	valid_0's auc: 0.904315	valid_0's binary_logloss: 0.16653
    [5842]	valid_0's auc: 0.904318	valid_0's binary_logloss: 0.166522
    [5843]	valid_0's auc: 0.904318	valid_0's binary_logloss: 0.166528
    [5844]	valid_0's auc: 0.904317	valid_0's binary_logloss: 0.16652
    [5845]	valid_0's auc: 0.904318	valid_0's binary_logloss: 0.166526
    [5846]	valid_0's auc: 0.904319	valid_0's binary_logloss: 0.166531
    [5847]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.166493
    [5848]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.166475
    [5849]	valid_0's auc: 0.904345	valid_0's binary_logloss: 0.166484
    [5850]	valid_0's auc: 0.904345	valid_0's binary_logloss: 0.166491
    [5851]	valid_0's auc: 0.904331	valid_0's binary_logloss: 0.16646
    [5852]	valid_0's auc: 0.90433	valid_0's binary_logloss: 0.166464
    [5853]	valid_0's auc: 0.904345	valid_0's binary_logloss: 0.166433
    [5854]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.166438
    [5855]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.166432
    [5856]	valid_0's auc: 0.904337	valid_0's binary_logloss: 0.166438
    [5857]	valid_0's auc: 0.904333	valid_0's binary_logloss: 0.166432
    [5858]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.166438
    [5859]	valid_0's auc: 0.904334	valid_0's binary_logloss: 0.166444
    [5860]	valid_0's auc: 0.904348	valid_0's binary_logloss: 0.166432
    [5861]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.166435
    [5862]	valid_0's auc: 0.904345	valid_0's binary_logloss: 0.166442
    [5863]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.166426
    [5864]	valid_0's auc: 0.904343	valid_0's binary_logloss: 0.166433
    [5865]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.16644
    [5866]	valid_0's auc: 0.904349	valid_0's binary_logloss: 0.166445
    [5867]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.166449
    [5868]	valid_0's auc: 0.904346	valid_0's binary_logloss: 0.166455
    [5869]	valid_0's auc: 0.904347	valid_0's binary_logloss: 0.166433
    [5870]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.166426
    [5871]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.166432
    [5872]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.166419
    [5873]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.166425
    [5874]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.166432
    [5875]	valid_0's auc: 0.904358	valid_0's binary_logloss: 0.166439
    [5876]	valid_0's auc: 0.904356	valid_0's binary_logloss: 0.166432
    [5877]	valid_0's auc: 0.904348	valid_0's binary_logloss: 0.166428
    [5878]	valid_0's auc: 0.904341	valid_0's binary_logloss: 0.166422
    [5879]	valid_0's auc: 0.90434	valid_0's binary_logloss: 0.166428
    [5880]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.166434
    [5881]	valid_0's auc: 0.904342	valid_0's binary_logloss: 0.16644
    [5882]	valid_0's auc: 0.904353	valid_0's binary_logloss: 0.16643
    [5883]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.166386
    [5884]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.16638
    [5885]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.166359
    [5886]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.16635
    [5887]	valid_0's auc: 0.904365	valid_0's binary_logloss: 0.166354
    [5888]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.166342
    [5889]	valid_0's auc: 0.90437	valid_0's binary_logloss: 0.166315
    [5890]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.166325
    [5891]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.166301
    [5892]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.166283
    [5893]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.166291
    [5894]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166285
    [5895]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.16629
    [5896]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.166296
    [5897]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166285
    [5898]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.166284
    [5899]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.16627
    [5900]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.166276
    [5901]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166282
    [5902]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166286
    [5903]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166293
    [5904]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166285
    [5905]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166291
    [5906]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.166297
    [5907]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.166287
    [5908]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.166293
    [5909]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.166298
    [5910]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.166297
    [5911]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.166304
    [5912]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.166298
    [5913]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.166305
    [5914]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.166313
    [5915]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.166308
    [5916]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166313
    [5917]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166318
    [5918]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.166277
    [5919]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.166261
    [5920]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.166268
    [5921]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.16626
    [5922]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.166251
    [5923]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.166248
    [5924]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.166254
    [5925]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.166258
    [5926]	valid_0's auc: 0.904365	valid_0's binary_logloss: 0.166265
    [5927]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.166254
    [5928]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.166258
    [5929]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.166266
    [5930]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.166262
    [5931]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.166268
    [5932]	valid_0's auc: 0.904357	valid_0's binary_logloss: 0.166251
    [5933]	valid_0's auc: 0.904348	valid_0's binary_logloss: 0.166248
    [5934]	valid_0's auc: 0.904352	valid_0's binary_logloss: 0.166256
    [5935]	valid_0's auc: 0.90435	valid_0's binary_logloss: 0.166264
    [5936]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.166246
    [5937]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.166252
    [5938]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166225
    [5939]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166231
    [5940]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166222
    [5941]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.166205
    [5942]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.166212
    [5943]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.166209
    [5944]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.166197
    [5945]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.166203
    [5946]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.166162
    [5947]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.16617
    [5948]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.166176
    [5949]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.166171
    [5950]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.166146
    [5951]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.166154
    [5952]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.166149
    [5953]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.16614
    [5954]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.166149
    [5955]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.166131
    [5956]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.166126
    [5957]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.166113
    [5958]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.166117
    [5959]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.166122
    [5960]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.166116
    [5961]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.166106
    [5962]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.166111
    [5963]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.166117
    [5964]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.166121
    [5965]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.166125
    [5966]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.166132
    [5967]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.166137
    [5968]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.166119
    [5969]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.16611
    [5970]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.1661
    [5971]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.166082
    [5972]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.166075
    [5973]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.166081
    [5974]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.166088
    [5975]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.166094
    [5976]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.166091
    [5977]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.166102
    [5978]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.166095
    [5979]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.166083
    [5980]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.166083
    [5981]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.166082
    [5982]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.166085
    [5983]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.166092
    [5984]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.166101
    [5985]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.166094
    [5986]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.166099
    [5987]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.166107
    [5988]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.16611
    [5989]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.166121
    [5990]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.166127
    [5991]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.166136
    [5992]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.166127
    [5993]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.166133
    [5994]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.166138
    [5995]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.166134
    [5996]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.166125
    [5997]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.166136
    [5998]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.166133
    [5999]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.166137
    [6000]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.166143
    [6001]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.166143
    [6002]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.166134
    [6003]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.166108
    [6004]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.166095
    [6005]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.166102
    [6006]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166087
    [6007]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.166077
    [6008]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.166071
    [6009]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.166069
    [6010]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.166065
    [6011]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.166075
    [6012]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166082
    [6013]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.16609
    [6014]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.166099
    [6015]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.166107
    [6016]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166118
    [6017]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.166124
    [6018]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166128
    [6019]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166135
    [6020]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166145
    [6021]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.166152
    [6022]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166158
    [6023]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.166163
    [6024]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.16617
    [6025]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.166175
    [6026]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.166166
    [6027]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.166158
    [6028]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.166147
    [6029]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.166139
    [6030]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.166148
    [6031]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.166116
    [6032]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.166113
    [6033]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.166105
    [6034]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.16611
    [6035]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.166103
    [6036]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.1661
    [6037]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.166092
    [6038]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.166086
    [6039]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.166096
    [6040]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.166104
    [6041]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.166111
    [6042]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.166103
    [6043]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.16611
    [6044]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.166116
    [6045]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.16612
    [6046]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.166127
    [6047]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.166132
    [6048]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.166127
    [6049]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.166116
    [6050]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.166122
    [6051]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.166122
    [6052]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.166114
    [6053]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.166102
    [6054]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.166092
    [6055]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.166072
    [6056]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.166066
    [6057]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.16606
    [6058]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.166065
    [6059]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.166045
    [6060]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.166051
    [6061]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.16606
    [6062]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.166049
    [6063]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.166053
    [6064]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.166061
    [6065]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.166052
    [6066]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.166058
    [6067]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.166055
    [6068]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.166026
    [6069]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.166022
    [6070]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165977
    [6071]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.165966
    [6072]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.165952
    [6073]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.165943
    [6074]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.165949
    [6075]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165955
    [6076]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.165961
    [6077]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.165954
    [6078]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.16595
    [6079]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.165936
    [6080]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.165945
    [6081]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.165953
    [6082]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.165959
    [6083]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.165946
    [6084]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.165942
    [6085]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.16595
    [6086]	valid_0's auc: 0.904494	valid_0's binary_logloss: 0.165962
    [6087]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.165955
    [6088]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.165949
    [6089]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165942
    [6090]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.165898
    [6091]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.16589
    [6092]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.165884
    [6093]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.165881
    [6094]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.165886
    [6095]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.165893
    [6096]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.165897
    [6097]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.165885
    [6098]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.16589
    [6099]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.165882
    [6100]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.165892
    [6101]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.165876
    [6102]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.165871
    [6103]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165879
    [6104]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.165873
    [6105]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.165878
    [6106]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.165884
    [6107]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.16589
    [6108]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.165898
    [6109]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.165873
    [6110]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.165877
    [6111]	valid_0's auc: 0.904505	valid_0's binary_logloss: 0.165882
    [6112]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.165868
    [6113]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.165874
    [6114]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.165879
    [6115]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.165889
    [6116]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.165894
    [6117]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.165899
    [6118]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.165889
    [6119]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.165882
    [6120]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.165868
    [6121]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165873
    [6122]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.165868
    [6123]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.165877
    [6124]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.165868
    [6125]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.165864
    [6126]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.165872
    [6127]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.165863
    [6128]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165866
    [6129]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.165871
    [6130]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.165877
    [6131]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.165881
    [6132]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.165861
    [6133]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.165854
    [6134]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.165826
    [6135]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.165838
    [6136]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.165843
    [6137]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.16585
    [6138]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.165849
    [6139]	valid_0's auc: 0.904515	valid_0's binary_logloss: 0.165821
    [6140]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165827
    [6141]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.16582
    [6142]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.16581
    [6143]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165798
    [6144]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.165777
    [6145]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.165785
    [6146]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165792
    [6147]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.1658
    [6148]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165805
    [6149]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.165795
    [6150]	valid_0's auc: 0.904495	valid_0's binary_logloss: 0.165803
    [6151]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.16581
    [6152]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.165806
    [6153]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.165813
    [6154]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.165807
    [6155]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.165801
    [6156]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.165805
    [6157]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.165808
    [6158]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.165813
    [6159]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.165818
    [6160]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.165822
    [6161]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.165818
    [6162]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.165823
    [6163]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.165827
    [6164]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.165833
    [6165]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.165839
    [6166]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.165843
    [6167]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.16585
    [6168]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.165856
    [6169]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.165859
    [6170]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.165859
    [6171]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.165858
    [6172]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.165867
    [6173]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.165875
    [6174]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165883
    [6175]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.165883
    [6176]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.165889
    [6177]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.165882
    [6178]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.16589
    [6179]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.165897
    [6180]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.165893
    [6181]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.165892
    [6182]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.165876
    [6183]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.16588
    [6184]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.165867
    [6185]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.165872
    [6186]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.165878
    [6187]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.165879
    [6188]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.165869
    [6189]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.165874
    [6190]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.165865
    [6191]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.165859
    [6192]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.165854
    [6193]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.165816
    [6194]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.165812
    [6195]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165818
    [6196]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165811
    [6197]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.16581
    [6198]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.165801
    [6199]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.165781
    [6200]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.165777
    [6201]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.16577
    [6202]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.165779
    [6203]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.165782
    [6204]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.165788
    [6205]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.165777
    [6206]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.165771
    [6207]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.165774
    [6208]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.165778
    [6209]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.165782
    [6210]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165776
    [6211]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.165782
    [6212]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.165741
    [6213]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.165746
    [6214]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.165719
    [6215]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.1657
    [6216]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.165694
    [6217]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.165688
    [6218]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.165683
    [6219]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.165688
    [6220]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.165699
    [6221]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.165689
    [6222]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165662
    [6223]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.165656
    [6224]	valid_0's auc: 0.904505	valid_0's binary_logloss: 0.165651
    [6225]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165624
    [6226]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.165623
    [6227]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.165627
    [6228]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.165625
    [6229]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.165609
    [6230]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.165615
    [6231]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.165621
    [6232]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.16563
    [6233]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.165635
    [6234]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165639
    [6235]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.165638
    [6236]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.165643
    [6237]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.165636
    [6238]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.165633
    [6239]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.165628
    [6240]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.165634
    [6241]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.16563
    [6242]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.165635
    [6243]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.165634
    [6244]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.16564
    [6245]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.165647
    [6246]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.165633
    [6247]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165637
    [6248]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.165629
    [6249]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.165624
    [6250]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.165624
    [6251]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.165608
    [6252]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.165599
    [6253]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.165604
    [6254]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.165613
    [6255]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.165619
    [6256]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.165605
    [6257]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.165598
    [6258]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.165594
    [6259]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.165593
    [6260]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.165588
    [6261]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.165584
    [6262]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.16559
    [6263]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.165574
    [6264]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165568
    [6265]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165575
    [6266]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165581
    [6267]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.165585
    [6268]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.165544
    [6269]	valid_0's auc: 0.904505	valid_0's binary_logloss: 0.165553
    [6270]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.165532
    [6271]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.165539
    [6272]	valid_0's auc: 0.904495	valid_0's binary_logloss: 0.165546
    [6273]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.165537
    [6274]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.165523
    [6275]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.165517
    [6276]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.165511
    [6277]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.165519
    [6278]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.165524
    [6279]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.165488
    [6280]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.165444
    [6281]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.165444
    [6282]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165437
    [6283]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.165446
    [6284]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.165443
    [6285]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.16545
    [6286]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.165453
    [6287]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.16546
    [6288]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.165464
    [6289]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.16547
    [6290]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.165476
    [6291]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.165484
    [6292]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.165489
    [6293]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.165478
    [6294]	valid_0's auc: 0.904571	valid_0's binary_logloss: 0.165471
    [6295]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.165464
    [6296]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.165469
    [6297]	valid_0's auc: 0.904586	valid_0's binary_logloss: 0.165474
    [6298]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.165477
    [6299]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.165475
    [6300]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.165468
    [6301]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.165475
    [6302]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.165482
    [6303]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.165489
    [6304]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.165493
    [6305]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.165458
    [6306]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.165462
    [6307]	valid_0's auc: 0.904593	valid_0's binary_logloss: 0.165468
    [6308]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.165465
    [6309]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.165473
    [6310]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.165481
    [6311]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.165475
    [6312]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.165482
    [6313]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.165479
    [6314]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.165483
    [6315]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.165479
    [6316]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.165485
    [6317]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.165482
    [6318]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.16549
    [6319]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.165484
    [6320]	valid_0's auc: 0.904565	valid_0's binary_logloss: 0.165492
    [6321]	valid_0's auc: 0.904565	valid_0's binary_logloss: 0.165495
    [6322]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.165503
    [6323]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.165498
    [6324]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.165481
    [6325]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.165471
    [6326]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.165464
    [6327]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.165463
    [6328]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.165455
    [6329]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.165463
    [6330]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.165468
    [6331]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.165464
    [6332]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.165467
    [6333]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.165472
    [6334]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.165477
    [6335]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.165475
    [6336]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.165473
    [6337]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.165482
    [6338]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.165476
    [6339]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.165484
    [6340]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.165476
    [6341]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.165469
    [6342]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.165466
    [6343]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165462
    [6344]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.165466
    [6345]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.16544
    [6346]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.165436
    [6347]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165432
    [6348]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.165435
    [6349]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.165441
    [6350]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.165448
    [6351]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.165452
    [6352]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.165458
    [6353]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165454
    [6354]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165462
    [6355]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.165451
    [6356]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165448
    [6357]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165454
    [6358]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.165462
    [6359]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.165465
    [6360]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.16545
    [6361]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.165456
    [6362]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.165449
    [6363]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.165454
    [6364]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.165451
    [6365]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.165446
    [6366]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.16545
    [6367]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.165456
    [6368]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.16546
    [6369]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165464
    [6370]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.16546
    [6371]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.165465
    [6372]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.165443
    [6373]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.165447
    [6374]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.165442
    [6375]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.165441
    [6376]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.165446
    [6377]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.165438
    [6378]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.165436
    [6379]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.16544
    [6380]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.165433
    [6381]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.16543
    [6382]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.165427
    [6383]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.165435
    [6384]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.165439
    [6385]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.165444
    [6386]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.165448
    [6387]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165446
    [6388]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.165449
    [6389]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.165443
    [6390]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.165448
    [6391]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.165439
    [6392]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.165429
    [6393]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.165432
    [6394]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.165439
    [6395]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.165447
    [6396]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.165446
    [6397]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.165455
    [6398]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.165452
    [6399]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.16546
    [6400]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.165433
    [6401]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.165426
    [6402]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.165429
    [6403]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.165419
    [6404]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.165428
    [6405]	valid_0's auc: 0.904559	valid_0's binary_logloss: 0.165433
    [6406]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.165412
    [6407]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.16541
    [6408]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.165397
    [6409]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.165403
    [6410]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.165408
    [6411]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.165404
    [6412]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.165392
    [6413]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.165399
    [6414]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.165408
    [6415]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.165397
    [6416]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.165389
    [6417]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.165383
    [6418]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.165387
    [6419]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.165393
    [6420]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.165393
    [6421]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165392
    [6422]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.165395
    [6423]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.165398
    [6424]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.165402
    [6425]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.165406
    [6426]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.165413
    [6427]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.16542
    [6428]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165429
    [6429]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.165434
    [6430]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.165431
    [6431]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.165436
    [6432]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.16544
    [6433]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.165435
    [6434]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.165439
    [6435]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.165408
    [6436]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165406
    [6437]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.165413
    [6438]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.16537
    [6439]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.165363
    [6440]	valid_0's auc: 0.904559	valid_0's binary_logloss: 0.165367
    [6441]	valid_0's auc: 0.904572	valid_0's binary_logloss: 0.165357
    [6442]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.165351
    [6443]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.165358
    [6444]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.165336
    [6445]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.165343
    [6446]	valid_0's auc: 0.904586	valid_0's binary_logloss: 0.165349
    [6447]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.165355
    [6448]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.165352
    [6449]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.165355
    [6450]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.165361
    [6451]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.165367
    [6452]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.16537
    [6453]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.165376
    [6454]	valid_0's auc: 0.904572	valid_0's binary_logloss: 0.165378
    [6455]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.165362
    [6456]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.165346
    [6457]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.165339
    [6458]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.165326
    [6459]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.16533
    [6460]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.165334
    [6461]	valid_0's auc: 0.904565	valid_0's binary_logloss: 0.165323
    [6462]	valid_0's auc: 0.904565	valid_0's binary_logloss: 0.165318
    [6463]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.165312
    [6464]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.165301
    [6465]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.165308
    [6466]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.16531
    [6467]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.165316
    [6468]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.16531
    [6469]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.165305
    [6470]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.165308
    [6471]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.165317
    [6472]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.165316
    [6473]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.165297
    [6474]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.16529
    [6475]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.16528
    [6476]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.165283
    [6477]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.165285
    [6478]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.165289
    [6479]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.165297
    [6480]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.165291
    [6481]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.165281
    [6482]	valid_0's auc: 0.904593	valid_0's binary_logloss: 0.165273
    [6483]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.165277
    [6484]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.165267
    [6485]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.165273
    [6486]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.16528
    [6487]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.165284
    [6488]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.165292
    [6489]	valid_0's auc: 0.904593	valid_0's binary_logloss: 0.165298
    [6490]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.165292
    [6491]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.165254
    [6492]	valid_0's auc: 0.904609	valid_0's binary_logloss: 0.165227
    [6493]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.165221
    [6494]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.165214
    [6495]	valid_0's auc: 0.904609	valid_0's binary_logloss: 0.165192
    [6496]	valid_0's auc: 0.904606	valid_0's binary_logloss: 0.165199
    [6497]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.165187
    [6498]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.165179
    [6499]	valid_0's auc: 0.90461	valid_0's binary_logloss: 0.165186
    [6500]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.165192
    [6501]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.165186
    [6502]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.165175
    [6503]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.165169
    [6504]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.165174
    [6505]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.165154
    [6506]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.165159
    [6507]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.165142
    [6508]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.165137
    [6509]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.165134
    [6510]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.165141
    [6511]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.165146
    [6512]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.165124
    [6513]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.165127
    [6514]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.165122
    [6515]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.165127
    [6516]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.165123
    [6517]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.165129
    [6518]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.16513
    [6519]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.16513
    [6520]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.165138
    [6521]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.165146
    [6522]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.165153
    [6523]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.165157
    [6524]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.165165
    [6525]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.165173
    [6526]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.165168
    [6527]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.165173
    [6528]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.165178
    [6529]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.165159
    [6530]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.165168
    [6531]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.165172
    [6532]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.165144
    [6533]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.165137
    [6534]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.165142
    [6535]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.165135
    [6536]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.165141
    [6537]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.165136
    [6538]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.165142
    [6539]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.165145
    [6540]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.165152
    [6541]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.165159
    [6542]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.165165
    [6543]	valid_0's auc: 0.904668	valid_0's binary_logloss: 0.16517
    [6544]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.165145
    [6545]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.165138
    [6546]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.165142
    [6547]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.165133
    [6548]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.165132
    [6549]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.165102
    [6550]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.165107
    [6551]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.165066
    [6552]	valid_0's auc: 0.904709	valid_0's binary_logloss: 0.165055
    [6553]	valid_0's auc: 0.904711	valid_0's binary_logloss: 0.165058
    [6554]	valid_0's auc: 0.904711	valid_0's binary_logloss: 0.165064
    [6555]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.165061
    [6556]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.16507
    [6557]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.165074
    [6558]	valid_0's auc: 0.904703	valid_0's binary_logloss: 0.165074
    [6559]	valid_0's auc: 0.904713	valid_0's binary_logloss: 0.165065
    [6560]	valid_0's auc: 0.904709	valid_0's binary_logloss: 0.165052
    [6561]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.165058
    [6562]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.165047
    [6563]	valid_0's auc: 0.90471	valid_0's binary_logloss: 0.165044
    [6564]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.165047
    [6565]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.165042
    [6566]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.165041
    [6567]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.165031
    [6568]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.165034
    [6569]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.165027
    [6570]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.165018
    [6571]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.165024
    [6572]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.165022
    [6573]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.165029
    [6574]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.165027
    [6575]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.165032
    [6576]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.165037
    [6577]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.165041
    [6578]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.165041
    [6579]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.16503
    [6580]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.165036
    [6581]	valid_0's auc: 0.90471	valid_0's binary_logloss: 0.165025
    [6582]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.16503
    [6583]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.165007
    [6584]	valid_0's auc: 0.904715	valid_0's binary_logloss: 0.165003
    [6585]	valid_0's auc: 0.904713	valid_0's binary_logloss: 0.165008
    [6586]	valid_0's auc: 0.904709	valid_0's binary_logloss: 0.165007
    [6587]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.165014
    [6588]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.16502
    [6589]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.165016
    [6590]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.165012
    [6591]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.165004
    [6592]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.164999
    [6593]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.164984
    [6594]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.164984
    [6595]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.164983
    [6596]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.164985
    [6597]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.164975
    [6598]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.164979
    [6599]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.164987
    [6600]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.164991
    [6601]	valid_0's auc: 0.904685	valid_0's binary_logloss: 0.164995
    [6602]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.165004
    [6603]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.165
    [6604]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.164992
    [6605]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.164997
    [6606]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164994
    [6607]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.164999
    [6608]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.165005
    [6609]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.165007
    [6610]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.165014
    [6611]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.165005
    [6612]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.165011
    [6613]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.164988
    [6614]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.164974
    [6615]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.164981
    [6616]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.16499
    [6617]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.164994
    [6618]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.164999
    [6619]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.165005
    [6620]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.165014
    [6621]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.165016
    [6622]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.165001
    [6623]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.164987
    [6624]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.164983
    [6625]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.164986
    [6626]	valid_0's auc: 0.904685	valid_0's binary_logloss: 0.164982
    [6627]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.164977
    [6628]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.164971
    [6629]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164965
    [6630]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.164971
    [6631]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.164966
    [6632]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.16497
    [6633]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164965
    [6634]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.16497
    [6635]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164974
    [6636]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.164963
    [6637]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.164961
    [6638]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.164945
    [6639]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.164952
    [6640]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164956
    [6641]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.164951
    [6642]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164954
    [6643]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.164961
    [6644]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.164957
    [6645]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.164962
    [6646]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.164956
    [6647]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.164951
    [6648]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.16493
    [6649]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.16492
    [6650]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.164915
    [6651]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164913
    [6652]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.164917
    [6653]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.164924
    [6654]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164929
    [6655]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.164932
    [6656]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.164914
    [6657]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164919
    [6658]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.164925
    [6659]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.16493
    [6660]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.164935
    [6661]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.16494
    [6662]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.164931
    [6663]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164932
    [6664]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.164922
    [6665]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.164914
    [6666]	valid_0's auc: 0.904661	valid_0's binary_logloss: 0.164909
    [6667]	valid_0's auc: 0.904661	valid_0's binary_logloss: 0.164917
    [6668]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.164924
    [6669]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.164928
    [6670]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.164934
    [6671]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.164939
    [6672]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.164944
    [6673]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.164951
    [6674]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164953
    [6675]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.16495
    [6676]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164956
    [6677]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.16496
    [6678]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.16494
    [6679]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164945
    [6680]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164948
    [6681]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164945
    [6682]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.16495
    [6683]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.164947
    [6684]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164942
    [6685]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164948
    [6686]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.164931
    [6687]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164936
    [6688]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164915
    [6689]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164921
    [6690]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164915
    [6691]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164912
    [6692]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164919
    [6693]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.164914
    [6694]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.164918
    [6695]	valid_0's auc: 0.904618	valid_0's binary_logloss: 0.164925
    [6696]	valid_0's auc: 0.904618	valid_0's binary_logloss: 0.164929
    [6697]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.164922
    [6698]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.164912
    [6699]	valid_0's auc: 0.90461	valid_0's binary_logloss: 0.164919
    [6700]	valid_0's auc: 0.904606	valid_0's binary_logloss: 0.164914
    [6701]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.164909
    [6702]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.164916
    [6703]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164913
    [6704]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164898
    [6705]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164902
    [6706]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164913
    [6707]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164888
    [6708]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164878
    [6709]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164883
    [6710]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164876
    [6711]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164879
    [6712]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.164886
    [6713]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.164882
    [6714]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164873
    [6715]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164878
    [6716]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164883
    [6717]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.16489
    [6718]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164873
    [6719]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164878
    [6720]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164877
    [6721]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164881
    [6722]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164876
    [6723]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164876
    [6724]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164868
    [6725]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164849
    [6726]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164854
    [6727]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.16482
    [6728]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164818
    [6729]	valid_0's auc: 0.904616	valid_0's binary_logloss: 0.164818
    [6730]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164811
    [6731]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.164811
    [6732]	valid_0's auc: 0.904586	valid_0's binary_logloss: 0.164806
    [6733]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.164807
    [6734]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.164809
    [6735]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.164814
    [6736]	valid_0's auc: 0.904565	valid_0's binary_logloss: 0.164811
    [6737]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.164807
    [6738]	valid_0's auc: 0.904572	valid_0's binary_logloss: 0.164791
    [6739]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.164795
    [6740]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.1648
    [6741]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.164806
    [6742]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.164788
    [6743]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.164782
    [6744]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.164789
    [6745]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.164776
    [6746]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.164781
    [6747]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.164787
    [6748]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.164792
    [6749]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164798
    [6750]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164792
    [6751]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.164766
    [6752]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.164771
    [6753]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.164776
    [6754]	valid_0's auc: 0.904615	valid_0's binary_logloss: 0.164781
    [6755]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.164774
    [6756]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.164768
    [6757]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164763
    [6758]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164768
    [6759]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164766
    [6760]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164754
    [6761]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.16476
    [6762]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164759
    [6763]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164764
    [6764]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164754
    [6765]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164758
    [6766]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164753
    [6767]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164756
    [6768]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164763
    [6769]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164768
    [6770]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164772
    [6771]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164777
    [6772]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164776
    [6773]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164784
    [6774]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164772
    [6775]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164777
    [6776]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164775
    [6777]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.16478
    [6778]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164773
    [6779]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164776
    [6780]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.164776
    [6781]	valid_0's auc: 0.904616	valid_0's binary_logloss: 0.164766
    [6782]	valid_0's auc: 0.904603	valid_0's binary_logloss: 0.164767
    [6783]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.164766
    [6784]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164766
    [6785]	valid_0's auc: 0.904605	valid_0's binary_logloss: 0.164757
    [6786]	valid_0's auc: 0.904602	valid_0's binary_logloss: 0.164763
    [6787]	valid_0's auc: 0.904601	valid_0's binary_logloss: 0.164768
    [6788]	valid_0's auc: 0.904603	valid_0's binary_logloss: 0.164773
    [6789]	valid_0's auc: 0.904605	valid_0's binary_logloss: 0.164778
    [6790]	valid_0's auc: 0.904603	valid_0's binary_logloss: 0.164784
    [6791]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.164781
    [6792]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.164786
    [6793]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.164789
    [6794]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164752
    [6795]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.164758
    [6796]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164763
    [6797]	valid_0's auc: 0.904603	valid_0's binary_logloss: 0.164756
    [6798]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.164751
    [6799]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.16475
    [6800]	valid_0's auc: 0.904602	valid_0's binary_logloss: 0.164745
    [6801]	valid_0's auc: 0.904601	valid_0's binary_logloss: 0.164724
    [6802]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.164729
    [6803]	valid_0's auc: 0.904604	valid_0's binary_logloss: 0.164714
    [6804]	valid_0's auc: 0.904609	valid_0's binary_logloss: 0.164709
    [6805]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.164711
    [6806]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.164712
    [6807]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.16472
    [6808]	valid_0's auc: 0.904587	valid_0's binary_logloss: 0.164726
    [6809]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.164729
    [6810]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.164724
    [6811]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.164707
    [6812]	valid_0's auc: 0.904601	valid_0's binary_logloss: 0.164694
    [6813]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.164697
    [6814]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.16469
    [6815]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164696
    [6816]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.1647
    [6817]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.164697
    [6818]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164701
    [6819]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.164702
    [6820]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.164709
    [6821]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.164715
    [6822]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.164721
    [6823]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.164722
    [6824]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.164712
    [6825]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.164718
    [6826]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.164723
    [6827]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.164729
    [6828]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.164729
    [6829]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.164717
    [6830]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.164726
    [6831]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.164729
    [6832]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.16472
    [6833]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.164712
    [6834]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.164706
    [6835]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.164711
    [6836]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.164696
    [6837]	valid_0's auc: 0.904586	valid_0's binary_logloss: 0.1647
    [6838]	valid_0's auc: 0.904585	valid_0's binary_logloss: 0.164704
    [6839]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.1647
    [6840]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164671
    [6841]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.164652
    [6842]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.164647
    [6843]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.164655
    [6844]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.164663
    [6845]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.164666
    [6846]	valid_0's auc: 0.904593	valid_0's binary_logloss: 0.164645
    [6847]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.16465
    [6848]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.164655
    [6849]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.16466
    [6850]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.164666
    [6851]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.164671
    [6852]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164674
    [6853]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.164677
    [6854]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.164676
    [6855]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.164681
    [6856]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164686
    [6857]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164691
    [6858]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.164685
    [6859]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.164682
    [6860]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.164666
    [6861]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164664
    [6862]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.164672
    [6863]	valid_0's auc: 0.904616	valid_0's binary_logloss: 0.164664
    [6864]	valid_0's auc: 0.904613	valid_0's binary_logloss: 0.16466
    [6865]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.164664
    [6866]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.164659
    [6867]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.164647
    [6868]	valid_0's auc: 0.904607	valid_0's binary_logloss: 0.164642
    [6869]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.164623
    [6870]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.164618
    [6871]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.164622
    [6872]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.164614
    [6873]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.164617
    [6874]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.164614
    [6875]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.164617
    [6876]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.16462
    [6877]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.164624
    [6878]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.164623
    [6879]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.164628
    [6880]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.164631
    [6881]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.164636
    [6882]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.164631
    [6883]	valid_0's auc: 0.90461	valid_0's binary_logloss: 0.16462
    [6884]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.164624
    [6885]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.164614
    [6886]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.164621
    [6887]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.164628
    [6888]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164621
    [6889]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164624
    [6890]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164628
    [6891]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164623
    [6892]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.16463
    [6893]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164636
    [6894]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.16463
    [6895]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164635
    [6896]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164625
    [6897]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.16462
    [6898]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164626
    [6899]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164613
    [6900]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164619
    [6901]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164604
    [6902]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164599
    [6903]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.164594
    [6904]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.164589
    [6905]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164585
    [6906]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.164562
    [6907]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.164566
    [6908]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.164537
    [6909]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.164542
    [6910]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.164547
    [6911]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.164545
    [6912]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.164549
    [6913]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.164554
    [6914]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.164545
    [6915]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164544
    [6916]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164548
    [6917]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164554
    [6918]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.16456
    [6919]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164552
    [6920]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164556
    [6921]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164556
    [6922]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164564
    [6923]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164569
    [6924]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164575
    [6925]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.164579
    [6926]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164583
    [6927]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.16459
    [6928]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164589
    [6929]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164592
    [6930]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164596
    [6931]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164589
    [6932]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164575
    [6933]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164582
    [6934]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164586
    [6935]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164589
    [6936]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.164569
    [6937]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164563
    [6938]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164567
    [6939]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164572
    [6940]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164566
    [6941]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164551
    [6942]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164531
    [6943]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.164524
    [6944]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164528
    [6945]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.164524
    [6946]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164492
    [6947]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164498
    [6948]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.164468
    [6949]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.164471
    [6950]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.164468
    [6951]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.164472
    [6952]	valid_0's auc: 0.904685	valid_0's binary_logloss: 0.164468
    [6953]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.164446
    [6954]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.164452
    [6955]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.164454
    [6956]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.164446
    [6957]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.164453
    [6958]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.16445
    [6959]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.164453
    [6960]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.16445
    [6961]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.164456
    [6962]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.164461
    [6963]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.164458
    [6964]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.164456
    [6965]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.164461
    [6966]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.16445
    [6967]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164446
    [6968]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164454
    [6969]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164456
    [6970]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164451
    [6971]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164453
    [6972]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.16445
    [6973]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164455
    [6974]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164461
    [6975]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164465
    [6976]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164467
    [6977]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.16445
    [6978]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164451
    [6979]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164455
    [6980]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164452
    [6981]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164454
    [6982]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164459
    [6983]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164463
    [6984]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164467
    [6985]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164473
    [6986]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164469
    [6987]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164464
    [6988]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164467
    [6989]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.164472
    [6990]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164479
    [6991]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164482
    [6992]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164488
    [6993]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164481
    [6994]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164479
    [6995]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164483
    [6996]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164482
    [6997]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164488
    [6998]	valid_0's auc: 0.904616	valid_0's binary_logloss: 0.164492
    [6999]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.164498
    [7000]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.164493
    [7001]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.164496
    [7002]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164474
    [7003]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164479
    [7004]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164471
    [7005]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.164448
    [7006]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164452
    [7007]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164457
    [7008]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164461
    [7009]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164454
    [7010]	valid_0's auc: 0.904664	valid_0's binary_logloss: 0.164425
    [7011]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.16442
    [7012]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.164425
    [7013]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164421
    [7014]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164418
    [7015]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164426
    [7016]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164429
    [7017]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164434
    [7018]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164438
    [7019]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.164435
    [7020]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164441
    [7021]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164437
    [7022]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164442
    [7023]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.164438
    [7024]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.164435
    [7025]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.16444
    [7026]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164434
    [7027]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164438
    [7028]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164441
    [7029]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164429
    [7030]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164422
    [7031]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164426
    [7032]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164431
    [7033]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164419
    [7034]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.164402
    [7035]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164397
    [7036]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.1644
    [7037]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164406
    [7038]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164413
    [7039]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164408
    [7040]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164396
    [7041]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164401
    [7042]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164403
    [7043]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164406
    [7044]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164376
    [7045]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.164369
    [7046]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164371
    [7047]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164378
    [7048]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.164384
    [7049]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164389
    [7050]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.164393
    [7051]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164392
    [7052]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164399
    [7053]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164396
    [7054]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164392
    [7055]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164397
    [7056]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164401
    [7057]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164406
    [7058]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164398
    [7059]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164402
    [7060]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.164394
    [7061]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164397
    [7062]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164395
    [7063]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164395
    [7064]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164388
    [7065]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164368
    [7066]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164373
    [7067]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164378
    [7068]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164376
    [7069]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.164372
    [7070]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.164376
    [7071]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.16438
    [7072]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.164384
    [7073]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.164388
    [7074]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.164384
    [7075]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.164374
    [7076]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.16438
    [7077]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.164386
    [7078]	valid_0's auc: 0.904607	valid_0's binary_logloss: 0.164391
    [7079]	valid_0's auc: 0.904607	valid_0's binary_logloss: 0.164395
    [7080]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164372
    [7081]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164368
    [7082]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164372
    [7083]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164377
    [7084]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.16438
    [7085]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164375
    [7086]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164379
    [7087]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.164385
    [7088]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.164389
    [7089]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164386
    [7090]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164394
    [7091]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164393
    [7092]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164398
    [7093]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164403
    [7094]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164406
    [7095]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.164399
    [7096]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.164363
    [7097]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164366
    [7098]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.164362
    [7099]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.164362
    [7100]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.164367
    [7101]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.164359
    [7102]	valid_0's auc: 0.904661	valid_0's binary_logloss: 0.164358
    [7103]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164356
    [7104]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164361
    [7105]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164366
    [7106]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164359
    [7107]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164364
    [7108]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.164367
    [7109]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164372
    [7110]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164375
    [7111]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164371
    [7112]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164365
    [7113]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164367
    [7114]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164364
    [7115]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164368
    [7116]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164366
    [7117]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.164369
    [7118]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164376
    [7119]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164371
    [7120]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164378
    [7121]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164369
    [7122]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164365
    [7123]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.16434
    [7124]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164345
    [7125]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164336
    [7126]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164332
    [7127]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164337
    [7128]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164333
    [7129]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.16434
    [7130]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164343
    [7131]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164348
    [7132]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164336
    [7133]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.16433
    [7134]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.164334
    [7135]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.164338
    [7136]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.164342
    [7137]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164335
    [7138]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164336
    [7139]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164339
    [7140]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164312
    [7141]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164316
    [7142]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164314
    [7143]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.164319
    [7144]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164325
    [7145]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164328
    [7146]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.164331
    [7147]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164335
    [7148]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164339
    [7149]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164343
    [7150]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164339
    [7151]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164335
    [7152]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164339
    [7153]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164334
    [7154]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164318
    [7155]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.164325
    [7156]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.164319
    [7157]	valid_0's auc: 0.904661	valid_0's binary_logloss: 0.164324
    [7158]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164324
    [7159]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164314
    [7160]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164309
    [7161]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164311
    [7162]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164317
    [7163]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164318
    [7164]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.164322
    [7165]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164327
    [7166]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.164309
    [7167]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.164289
    [7168]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.164293
    [7169]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.164298
    [7170]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.164302
    [7171]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.164292
    [7172]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.164298
    [7173]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.164289
    [7174]	valid_0's auc: 0.904714	valid_0's binary_logloss: 0.164294
    [7175]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.164282
    [7176]	valid_0's auc: 0.904716	valid_0's binary_logloss: 0.164287
    [7177]	valid_0's auc: 0.904715	valid_0's binary_logloss: 0.164294
    [7178]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.164289
    [7179]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.164276
    [7180]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.164274
    [7181]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.164276
    [7182]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.164281
    [7183]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.164285
    [7184]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.164289
    [7185]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.164292
    [7186]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.164295
    [7187]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.164291
    [7188]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.164292
    [7189]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.164297
    [7190]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.164301
    [7191]	valid_0's auc: 0.904664	valid_0's binary_logloss: 0.164303
    [7192]	valid_0's auc: 0.904663	valid_0's binary_logloss: 0.1643
    [7193]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.164275
    [7194]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.164252
    [7195]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.164256
    [7196]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.164259
    [7197]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.164263
    [7198]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.164268
    [7199]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.164261
    [7200]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.164265
    [7201]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.164266
    [7202]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.16426
    [7203]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.164265
    [7204]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.164269
    [7205]	valid_0's auc: 0.904668	valid_0's binary_logloss: 0.164261
    [7206]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.164259
    [7207]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.164258
    [7208]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.164262
    [7209]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.164266
    [7210]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.164255
    [7211]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.164251
    [7212]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.164257
    [7213]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.164262
    [7214]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.164242
    [7215]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.16424
    [7216]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.16423
    [7217]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.164232
    [7218]	valid_0's auc: 0.904685	valid_0's binary_logloss: 0.164235
    [7219]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.16424
    [7220]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.164221
    [7221]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.164214
    [7222]	valid_0's auc: 0.904703	valid_0's binary_logloss: 0.164212
    [7223]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.164187
    [7224]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.164192
    [7225]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.164196
    [7226]	valid_0's auc: 0.904721	valid_0's binary_logloss: 0.164168
    [7227]	valid_0's auc: 0.904723	valid_0's binary_logloss: 0.16417
    [7228]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.164175
    [7229]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.164172
    [7230]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.164179
    [7231]	valid_0's auc: 0.904714	valid_0's binary_logloss: 0.164185
    [7232]	valid_0's auc: 0.904715	valid_0's binary_logloss: 0.164192
    [7233]	valid_0's auc: 0.904716	valid_0's binary_logloss: 0.164195
    [7234]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.164189
    [7235]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.164181
    [7236]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.164186
    [7237]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.164193
    [7238]	valid_0's auc: 0.904718	valid_0's binary_logloss: 0.164191
    [7239]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.164188
    [7240]	valid_0's auc: 0.904713	valid_0's binary_logloss: 0.164184
    [7241]	valid_0's auc: 0.904714	valid_0's binary_logloss: 0.164188
    [7242]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.164179
    [7243]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.164165
    [7244]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.164167
    [7245]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.16417
    [7246]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.164156
    [7247]	valid_0's auc: 0.904723	valid_0's binary_logloss: 0.164147
    [7248]	valid_0's auc: 0.904725	valid_0's binary_logloss: 0.164151
    [7249]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.164155
    [7250]	valid_0's auc: 0.90473	valid_0's binary_logloss: 0.164151
    [7251]	valid_0's auc: 0.904723	valid_0's binary_logloss: 0.164142
    [7252]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.164125
    [7253]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.16413
    [7254]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.164135
    [7255]	valid_0's auc: 0.904727	valid_0's binary_logloss: 0.164139
    [7256]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.164142
    [7257]	valid_0's auc: 0.90473	valid_0's binary_logloss: 0.164145
    [7258]	valid_0's auc: 0.904731	valid_0's binary_logloss: 0.164148
    [7259]	valid_0's auc: 0.90473	valid_0's binary_logloss: 0.164154
    [7260]	valid_0's auc: 0.904731	valid_0's binary_logloss: 0.164157
    [7261]	valid_0's auc: 0.904734	valid_0's binary_logloss: 0.16415
    [7262]	valid_0's auc: 0.904734	valid_0's binary_logloss: 0.164153
    [7263]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.164154
    [7264]	valid_0's auc: 0.904711	valid_0's binary_logloss: 0.164159
    [7265]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164156
    [7266]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164158
    [7267]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164162
    [7268]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164156
    [7269]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164159
    [7270]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.164163
    [7271]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.164167
    [7272]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.164174
    [7273]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.164176
    [7274]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.164166
    [7275]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164164
    [7276]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.164168
    [7277]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.16417
    [7278]	valid_0's auc: 0.904711	valid_0's binary_logloss: 0.164176
    [7279]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.16415
    [7280]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.164153
    [7281]	valid_0's auc: 0.904722	valid_0's binary_logloss: 0.164157
    [7282]	valid_0's auc: 0.904716	valid_0's binary_logloss: 0.164151
    [7283]	valid_0's auc: 0.904734	valid_0's binary_logloss: 0.16414
    [7284]	valid_0's auc: 0.904731	valid_0's binary_logloss: 0.164136
    [7285]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.164134
    [7286]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.164138
    [7287]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.164133
    [7288]	valid_0's auc: 0.904714	valid_0's binary_logloss: 0.164128
    [7289]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.164121
    [7290]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.164126
    [7291]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.164131
    [7292]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.164134
    [7293]	valid_0's auc: 0.904721	valid_0's binary_logloss: 0.164137
    [7294]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.164139
    [7295]	valid_0's auc: 0.904709	valid_0's binary_logloss: 0.164135
    [7296]	valid_0's auc: 0.904716	valid_0's binary_logloss: 0.164129
    [7297]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.164133
    [7298]	valid_0's auc: 0.904732	valid_0's binary_logloss: 0.164104
    [7299]	valid_0's auc: 0.904729	valid_0's binary_logloss: 0.164101
    [7300]	valid_0's auc: 0.90473	valid_0's binary_logloss: 0.164104
    [7301]	valid_0's auc: 0.904729	valid_0's binary_logloss: 0.164108
    [7302]	valid_0's auc: 0.904718	valid_0's binary_logloss: 0.1641
    [7303]	valid_0's auc: 0.904717	valid_0's binary_logloss: 0.164105
    [7304]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.164109
    [7305]	valid_0's auc: 0.904717	valid_0's binary_logloss: 0.164107
    [7306]	valid_0's auc: 0.904715	valid_0's binary_logloss: 0.164111
    [7307]	valid_0's auc: 0.904715	valid_0's binary_logloss: 0.164109
    [7308]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.164108
    [7309]	valid_0's auc: 0.90471	valid_0's binary_logloss: 0.164102
    [7310]	valid_0's auc: 0.904703	valid_0's binary_logloss: 0.164086
    [7311]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.164083
    [7312]	valid_0's auc: 0.90471	valid_0's binary_logloss: 0.164078
    [7313]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.164079
    [7314]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.164071
    [7315]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.164075
    [7316]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.16408
    [7317]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.164083
    [7318]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.164079
    [7319]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.164078
    [7320]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.16407
    [7321]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.164077
    [7322]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.164081
    [7323]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.164063
    [7324]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.164057
    [7325]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.164062
    [7326]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.164067
    [7327]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.16407
    [7328]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164054
    [7329]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.164057
    [7330]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.164048
    [7331]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.164054
    [7332]	valid_0's auc: 0.904668	valid_0's binary_logloss: 0.164055
    [7333]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.164036
    [7334]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.16404
    [7335]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.164043
    [7336]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.164032
    [7337]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164032
    [7338]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164036
    [7339]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164031
    [7340]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164037
    [7341]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164042
    [7342]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164047
    [7343]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.164052
    [7344]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.164049
    [7345]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164049
    [7346]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164051
    [7347]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164055
    [7348]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164058
    [7349]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164056
    [7350]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164059
    [7351]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164057
    [7352]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164049
    [7353]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164047
    [7354]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164052
    [7355]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.164057
    [7356]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.164061
    [7357]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.164058
    [7358]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164063
    [7359]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.164067
    [7360]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164063
    [7361]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164066
    [7362]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.16405
    [7363]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.164054
    [7364]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.16406
    [7365]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164058
    [7366]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.164062
    [7367]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164066
    [7368]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.16406
    [7369]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.164066
    [7370]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.16407
    [7371]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164076
    [7372]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164077
    [7373]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.164066
    [7374]	valid_0's auc: 0.904623	valid_0's binary_logloss: 0.164062
    [7375]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.164069
    [7376]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.164072
    [7377]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.164062
    [7378]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.164061
    [7379]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.164059
    [7380]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164051
    [7381]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164056
    [7382]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164051
    [7383]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.164055
    [7384]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.164057
    [7385]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164062
    [7386]	valid_0's auc: 0.904636	valid_0's binary_logloss: 0.164066
    [7387]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.164071
    [7388]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.164068
    [7389]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.164074
    [7390]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.164054
    [7391]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.164046
    [7392]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164039
    [7393]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164033
    [7394]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164036
    [7395]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.164039
    [7396]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.164044
    [7397]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.164021
    [7398]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164016
    [7399]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.164021
    [7400]	valid_0's auc: 0.904664	valid_0's binary_logloss: 0.163996
    [7401]	valid_0's auc: 0.904661	valid_0's binary_logloss: 0.163993
    [7402]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.163993
    [7403]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.163997
    [7404]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.163995
    [7405]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163989
    [7406]	valid_0's auc: 0.904641	valid_0's binary_logloss: 0.163985
    [7407]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.163989
    [7408]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.163995
    [7409]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163992
    [7410]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.163996
    [7411]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163994
    [7412]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.163977
    [7413]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163972
    [7414]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.163969
    [7415]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.163964
    [7416]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.163968
    [7417]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.163945
    [7418]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.163949
    [7419]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.163952
    [7420]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163957
    [7421]	valid_0's auc: 0.904668	valid_0's binary_logloss: 0.16396
    [7422]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.163966
    [7423]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163959
    [7424]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.16396
    [7425]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.163967
    [7426]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163962
    [7427]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.163959
    [7428]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163951
    [7429]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163955
    [7430]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163952
    [7431]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163958
    [7432]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.16396
    [7433]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163963
    [7434]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163969
    [7435]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163972
    [7436]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163974
    [7437]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163976
    [7438]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163977
    [7439]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.16398
    [7440]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.16397
    [7441]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163966
    [7442]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163963
    [7443]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163967
    [7444]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163971
    [7445]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163976
    [7446]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163965
    [7447]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163967
    [7448]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163969
    [7449]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163974
    [7450]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.163977
    [7451]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.163971
    [7452]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163975
    [7453]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.163981
    [7454]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163986
    [7455]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163987
    [7456]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163981
    [7457]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163983
    [7458]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163988
    [7459]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163973
    [7460]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163978
    [7461]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.163969
    [7462]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.163973
    [7463]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163973
    [7464]	valid_0's auc: 0.904655	valid_0's binary_logloss: 0.163967
    [7465]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.163941
    [7466]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163942
    [7467]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.163937
    [7468]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.16394
    [7469]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.163946
    [7470]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.16394
    [7471]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.163942
    [7472]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163948
    [7473]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.163936
    [7474]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.163939
    [7475]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163946
    [7476]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.163951
    [7477]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163947
    [7478]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163943
    [7479]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.16394
    [7480]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.163943
    [7481]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.163918
    [7482]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163907
    [7483]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.163913
    [7484]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163917
    [7485]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163911
    [7486]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163914
    [7487]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.163886
    [7488]	valid_0's auc: 0.904707	valid_0's binary_logloss: 0.163854
    [7489]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163854
    [7490]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.163847
    [7491]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.163839
    [7492]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.163833
    [7493]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.163831
    [7494]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163834
    [7495]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.16383
    [7496]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.163821
    [7497]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163813
    [7498]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.163818
    [7499]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.163821
    [7500]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.163819
    [7501]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.163823
    [7502]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.163826
    [7503]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.163821
    [7504]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.163818
    [7505]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.16381
    [7506]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163804
    [7507]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.16381
    [7508]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.163814
    [7509]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.163808
    [7510]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163805
    [7511]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.1638
    [7512]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.163798
    [7513]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.163787
    [7514]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.163792
    [7515]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.163797
    [7516]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.163795
    [7517]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.163798
    [7518]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163802
    [7519]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.163805
    [7520]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163786
    [7521]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163788
    [7522]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.163786
    [7523]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.163781
    [7524]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.163785
    [7525]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.163784
    [7526]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163785
    [7527]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163786
    [7528]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163791
    [7529]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163796
    [7530]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163801
    [7531]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163803
    [7532]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163807
    [7533]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.163812
    [7534]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163814
    [7535]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163819
    [7536]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163823
    [7537]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163826
    [7538]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163828
    [7539]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163826
    [7540]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163828
    [7541]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163817
    [7542]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163803
    [7543]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163802
    [7544]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163806
    [7545]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163808
    [7546]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163813
    [7547]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163815
    [7548]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163818
    [7549]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163823
    [7550]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163823
    [7551]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163825
    [7552]	valid_0's auc: 0.904659	valid_0's binary_logloss: 0.163827
    [7553]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.163831
    [7554]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163834
    [7555]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.16384
    [7556]	valid_0's auc: 0.904656	valid_0's binary_logloss: 0.163836
    [7557]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163827
    [7558]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163832
    [7559]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163827
    [7560]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163832
    [7561]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163836
    [7562]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163842
    [7563]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163849
    [7564]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.16385
    [7565]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.16385
    [7566]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.163846
    [7567]	valid_0's auc: 0.904663	valid_0's binary_logloss: 0.163848
    [7568]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.163852
    [7569]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163846
    [7570]	valid_0's auc: 0.904664	valid_0's binary_logloss: 0.163849
    [7571]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163854
    [7572]	valid_0's auc: 0.904664	valid_0's binary_logloss: 0.163851
    [7573]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.16384
    [7574]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.163819
    [7575]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.163813
    [7576]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.163818
    [7577]	valid_0's auc: 0.904709	valid_0's binary_logloss: 0.163811
    [7578]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163815
    [7579]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163816
    [7580]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.163822
    [7581]	valid_0's auc: 0.904703	valid_0's binary_logloss: 0.163827
    [7582]	valid_0's auc: 0.904711	valid_0's binary_logloss: 0.163821
    [7583]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.163808
    [7584]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.163812
    [7585]	valid_0's auc: 0.90473	valid_0's binary_logloss: 0.163806
    [7586]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.163812
    [7587]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.163814
    [7588]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.163817
    [7589]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.163813
    [7590]	valid_0's auc: 0.904733	valid_0's binary_logloss: 0.163805
    [7591]	valid_0's auc: 0.904739	valid_0's binary_logloss: 0.163789
    [7592]	valid_0's auc: 0.904741	valid_0's binary_logloss: 0.163793
    [7593]	valid_0's auc: 0.904739	valid_0's binary_logloss: 0.163796
    [7594]	valid_0's auc: 0.904734	valid_0's binary_logloss: 0.163795
    [7595]	valid_0's auc: 0.904737	valid_0's binary_logloss: 0.163795
    [7596]	valid_0's auc: 0.904738	valid_0's binary_logloss: 0.163798
    [7597]	valid_0's auc: 0.904738	valid_0's binary_logloss: 0.163793
    [7598]	valid_0's auc: 0.904733	valid_0's binary_logloss: 0.16379
    [7599]	valid_0's auc: 0.904718	valid_0's binary_logloss: 0.163791
    [7600]	valid_0's auc: 0.904718	valid_0's binary_logloss: 0.163796
    [7601]	valid_0's auc: 0.904713	valid_0's binary_logloss: 0.163773
    [7602]	valid_0's auc: 0.904718	valid_0's binary_logloss: 0.163768
    [7603]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163754
    [7604]	valid_0's auc: 0.904714	valid_0's binary_logloss: 0.163748
    [7605]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163753
    [7606]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163758
    [7607]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163764
    [7608]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.163761
    [7609]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163765
    [7610]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.163754
    [7611]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163748
    [7612]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.163746
    [7613]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163745
    [7614]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163748
    [7615]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163744
    [7616]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163745
    [7617]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163727
    [7618]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163724
    [7619]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.16373
    [7620]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163733
    [7621]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163736
    [7622]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.16373
    [7623]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163735
    [7624]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.16374
    [7625]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163735
    [7626]	valid_0's auc: 0.904675	valid_0's binary_logloss: 0.163734
    [7627]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163737
    [7628]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163733
    [7629]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.163737
    [7630]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.16374
    [7631]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163738
    [7632]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.163737
    [7633]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163725
    [7634]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163726
    [7635]	valid_0's auc: 0.904663	valid_0's binary_logloss: 0.163723
    [7636]	valid_0's auc: 0.904663	valid_0's binary_logloss: 0.163728
    [7637]	valid_0's auc: 0.904668	valid_0's binary_logloss: 0.163721
    [7638]	valid_0's auc: 0.904668	valid_0's binary_logloss: 0.163724
    [7639]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163719
    [7640]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.163713
    [7641]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.163718
    [7642]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163722
    [7643]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163725
    [7644]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.163716
    [7645]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163704
    [7646]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163709
    [7647]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.163707
    [7648]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.163706
    [7649]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.163708
    [7650]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163713
    [7651]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.163709
    [7652]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163709
    [7653]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.163704
    [7654]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.163708
    [7655]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163711
    [7656]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163713
    [7657]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163717
    [7658]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163717
    [7659]	valid_0's auc: 0.904664	valid_0's binary_logloss: 0.163716
    [7660]	valid_0's auc: 0.904681	valid_0's binary_logloss: 0.163707
    [7661]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.163702
    [7662]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.163706
    [7663]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.163696
    [7664]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163699
    [7665]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.163698
    [7666]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.163692
    [7667]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.163687
    [7668]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163666
    [7669]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.163669
    [7670]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.163673
    [7671]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163674
    [7672]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.163678
    [7673]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.163662
    [7674]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.163667
    [7675]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.163671
    [7676]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.163675
    [7677]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.16368
    [7678]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.163684
    [7679]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163682
    [7680]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163684
    [7681]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.163688
    [7682]	valid_0's auc: 0.904683	valid_0's binary_logloss: 0.163678
    [7683]	valid_0's auc: 0.904684	valid_0's binary_logloss: 0.16368
    [7684]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163682
    [7685]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163685
    [7686]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163688
    [7687]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163691
    [7688]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163695
    [7689]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.1637
    [7690]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.163698
    [7691]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.163674
    [7692]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.163664
    [7693]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.16366
    [7694]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.163641
    [7695]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.163643
    [7696]	valid_0's auc: 0.904699	valid_0's binary_logloss: 0.163647
    [7697]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.163643
    [7698]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.163628
    [7699]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.16363
    [7700]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.163628
    [7701]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.163624
    [7702]	valid_0's auc: 0.904691	valid_0's binary_logloss: 0.163626
    [7703]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.163615
    [7704]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.16362
    [7705]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.163613
    [7706]	valid_0's auc: 0.904703	valid_0's binary_logloss: 0.163605
    [7707]	valid_0's auc: 0.904687	valid_0's binary_logloss: 0.163605
    [7708]	valid_0's auc: 0.904682	valid_0's binary_logloss: 0.163607
    [7709]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.163604
    [7710]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.163601
    [7711]	valid_0's auc: 0.904696	valid_0's binary_logloss: 0.163599
    [7712]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.163601
    [7713]	valid_0's auc: 0.9047	valid_0's binary_logloss: 0.163599
    [7714]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.163589
    [7715]	valid_0's auc: 0.904701	valid_0's binary_logloss: 0.163578
    [7716]	valid_0's auc: 0.904695	valid_0's binary_logloss: 0.163574
    [7717]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.163567
    [7718]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.163566
    [7719]	valid_0's auc: 0.904692	valid_0's binary_logloss: 0.163569
    [7720]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.163564
    [7721]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.16356
    [7722]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.163556
    [7723]	valid_0's auc: 0.904698	valid_0's binary_logloss: 0.163561
    [7724]	valid_0's auc: 0.904694	valid_0's binary_logloss: 0.163559
    [7725]	valid_0's auc: 0.904693	valid_0's binary_logloss: 0.163565
    [7726]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163559
    [7727]	valid_0's auc: 0.904688	valid_0's binary_logloss: 0.163563
    [7728]	valid_0's auc: 0.904697	valid_0's binary_logloss: 0.163546
    [7729]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.16355
    [7730]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163556
    [7731]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163556
    [7732]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.16356
    [7733]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163558
    [7734]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.163555
    [7735]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.163547
    [7736]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163542
    [7737]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163545
    [7738]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163548
    [7739]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.16355
    [7740]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.163554
    [7741]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163551
    [7742]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.163555
    [7743]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163557
    [7744]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.163555
    [7745]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.16356
    [7746]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.163563
    [7747]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.163571
    [7748]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.163575
    [7749]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163571
    [7750]	valid_0's auc: 0.90464	valid_0's binary_logloss: 0.163575
    [7751]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163576
    [7752]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.163573
    [7753]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.163566
    [7754]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163572
    [7755]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.163568
    [7756]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.163571
    [7757]	valid_0's auc: 0.904653	valid_0's binary_logloss: 0.163574
    [7758]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.163568
    [7759]	valid_0's auc: 0.904672	valid_0's binary_logloss: 0.163561
    [7760]	valid_0's auc: 0.904674	valid_0's binary_logloss: 0.163557
    [7761]	valid_0's auc: 0.904677	valid_0's binary_logloss: 0.163543
    [7762]	valid_0's auc: 0.904676	valid_0's binary_logloss: 0.163545
    [7763]	valid_0's auc: 0.90468	valid_0's binary_logloss: 0.163533
    [7764]	valid_0's auc: 0.904679	valid_0's binary_logloss: 0.163539
    [7765]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163543
    [7766]	valid_0's auc: 0.904673	valid_0's binary_logloss: 0.163547
    [7767]	valid_0's auc: 0.904678	valid_0's binary_logloss: 0.163544
    [7768]	valid_0's auc: 0.904686	valid_0's binary_logloss: 0.16354
    [7769]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.163537
    [7770]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.16353
    [7771]	valid_0's auc: 0.904704	valid_0's binary_logloss: 0.163535
    [7772]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.163538
    [7773]	valid_0's auc: 0.904702	valid_0's binary_logloss: 0.163544
    [7774]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.163538
    [7775]	valid_0's auc: 0.90472	valid_0's binary_logloss: 0.163525
    [7776]	valid_0's auc: 0.904729	valid_0's binary_logloss: 0.16352
    [7777]	valid_0's auc: 0.904731	valid_0's binary_logloss: 0.163517
    [7778]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.163511
    [7779]	valid_0's auc: 0.904733	valid_0's binary_logloss: 0.163496
    [7780]	valid_0's auc: 0.904746	valid_0's binary_logloss: 0.163486
    [7781]	valid_0's auc: 0.904733	valid_0's binary_logloss: 0.163485
    [7782]	valid_0's auc: 0.904734	valid_0's binary_logloss: 0.163482
    [7783]	valid_0's auc: 0.904734	valid_0's binary_logloss: 0.163487
    [7784]	valid_0's auc: 0.904733	valid_0's binary_logloss: 0.163488
    [7785]	valid_0's auc: 0.904745	valid_0's binary_logloss: 0.163483
    [7786]	valid_0's auc: 0.904741	valid_0's binary_logloss: 0.163477
    [7787]	valid_0's auc: 0.904752	valid_0's binary_logloss: 0.163475
    [7788]	valid_0's auc: 0.904751	valid_0's binary_logloss: 0.163478
    [7789]	valid_0's auc: 0.904758	valid_0's binary_logloss: 0.163471
    [7790]	valid_0's auc: 0.90476	valid_0's binary_logloss: 0.163468
    [7791]	valid_0's auc: 0.90476	valid_0's binary_logloss: 0.163472
    [7792]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163469
    [7793]	valid_0's auc: 0.904772	valid_0's binary_logloss: 0.163459
    [7794]	valid_0's auc: 0.904773	valid_0's binary_logloss: 0.163463
    [7795]	valid_0's auc: 0.904772	valid_0's binary_logloss: 0.163468
    [7796]	valid_0's auc: 0.904772	valid_0's binary_logloss: 0.163463
    [7797]	valid_0's auc: 0.904774	valid_0's binary_logloss: 0.163465
    [7798]	valid_0's auc: 0.904772	valid_0's binary_logloss: 0.163464
    [7799]	valid_0's auc: 0.904767	valid_0's binary_logloss: 0.163464
    [7800]	valid_0's auc: 0.904762	valid_0's binary_logloss: 0.163462
    [7801]	valid_0's auc: 0.904762	valid_0's binary_logloss: 0.163459
    [7802]	valid_0's auc: 0.904747	valid_0's binary_logloss: 0.163461
    [7803]	valid_0's auc: 0.904745	valid_0's binary_logloss: 0.163467
    [7804]	valid_0's auc: 0.904745	valid_0's binary_logloss: 0.163471
    [7805]	valid_0's auc: 0.904753	valid_0's binary_logloss: 0.163468
    [7806]	valid_0's auc: 0.90475	valid_0's binary_logloss: 0.163474
    [7807]	valid_0's auc: 0.904749	valid_0's binary_logloss: 0.163476
    [7808]	valid_0's auc: 0.904748	valid_0's binary_logloss: 0.163481
    [7809]	valid_0's auc: 0.904752	valid_0's binary_logloss: 0.16348
    [7810]	valid_0's auc: 0.904753	valid_0's binary_logloss: 0.163483
    [7811]	valid_0's auc: 0.90475	valid_0's binary_logloss: 0.163487
    [7812]	valid_0's auc: 0.904757	valid_0's binary_logloss: 0.163483
    [7813]	valid_0's auc: 0.904749	valid_0's binary_logloss: 0.163476
    [7814]	valid_0's auc: 0.904749	valid_0's binary_logloss: 0.163479
    [7815]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163472
    [7816]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163475
    [7817]	valid_0's auc: 0.904755	valid_0's binary_logloss: 0.16348
    [7818]	valid_0's auc: 0.904755	valid_0's binary_logloss: 0.163483
    [7819]	valid_0's auc: 0.904757	valid_0's binary_logloss: 0.163485
    [7820]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163488
    [7821]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163491
    [7822]	valid_0's auc: 0.904752	valid_0's binary_logloss: 0.163496
    [7823]	valid_0's auc: 0.904752	valid_0's binary_logloss: 0.163499
    [7824]	valid_0's auc: 0.90476	valid_0's binary_logloss: 0.163489
    [7825]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163473
    [7826]	valid_0's auc: 0.904757	valid_0's binary_logloss: 0.163477
    [7827]	valid_0's auc: 0.904757	valid_0's binary_logloss: 0.16348
    [7828]	valid_0's auc: 0.904757	valid_0's binary_logloss: 0.163485
    [7829]	valid_0's auc: 0.904756	valid_0's binary_logloss: 0.163483
    [7830]	valid_0's auc: 0.904753	valid_0's binary_logloss: 0.163481
    [7831]	valid_0's auc: 0.904758	valid_0's binary_logloss: 0.163473
    [7832]	valid_0's auc: 0.904758	valid_0's binary_logloss: 0.163478
    [7833]	valid_0's auc: 0.904755	valid_0's binary_logloss: 0.163474
    [7834]	valid_0's auc: 0.904751	valid_0's binary_logloss: 0.163461
    [7835]	valid_0's auc: 0.90475	valid_0's binary_logloss: 0.163464
    [7836]	valid_0's auc: 0.904755	valid_0's binary_logloss: 0.163463
    [7837]	valid_0's auc: 0.904746	valid_0's binary_logloss: 0.163461
    [7838]	valid_0's auc: 0.904737	valid_0's binary_logloss: 0.163461
    [7839]	valid_0's auc: 0.904735	valid_0's binary_logloss: 0.163467
    [7840]	valid_0's auc: 0.904736	valid_0's binary_logloss: 0.163466
    [7841]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.163464
    [7842]	valid_0's auc: 0.904723	valid_0's binary_logloss: 0.163464
    [7843]	valid_0's auc: 0.904725	valid_0's binary_logloss: 0.163463
    [7844]	valid_0's auc: 0.904721	valid_0's binary_logloss: 0.163461
    [7845]	valid_0's auc: 0.904723	valid_0's binary_logloss: 0.163467
    [7846]	valid_0's auc: 0.904721	valid_0's binary_logloss: 0.16347
    [7847]	valid_0's auc: 0.904721	valid_0's binary_logloss: 0.163474
    [7848]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.163468
    [7849]	valid_0's auc: 0.904724	valid_0's binary_logloss: 0.163472
    [7850]	valid_0's auc: 0.904728	valid_0's binary_logloss: 0.163469
    [7851]	valid_0's auc: 0.904726	valid_0's binary_logloss: 0.163473
    [7852]	valid_0's auc: 0.904731	valid_0's binary_logloss: 0.163469
    [7853]	valid_0's auc: 0.904717	valid_0's binary_logloss: 0.163472
    [7854]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163466
    [7855]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163469
    [7856]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163471
    [7857]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163473
    [7858]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163462
    [7859]	valid_0's auc: 0.904711	valid_0's binary_logloss: 0.163466
    [7860]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.16347
    [7861]	valid_0's auc: 0.904709	valid_0's binary_logloss: 0.163473
    [7862]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163478
    [7863]	valid_0's auc: 0.904715	valid_0's binary_logloss: 0.163471
    [7864]	valid_0's auc: 0.904721	valid_0's binary_logloss: 0.163466
    [7865]	valid_0's auc: 0.904712	valid_0's binary_logloss: 0.163467
    [7866]	valid_0's auc: 0.904714	valid_0's binary_logloss: 0.163469
    [7867]	valid_0's auc: 0.904713	valid_0's binary_logloss: 0.163474
    [7868]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.163474
    [7869]	valid_0's auc: 0.904705	valid_0's binary_logloss: 0.163479
    [7870]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163478
    [7871]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163481
    [7872]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.163487
    [7873]	valid_0's auc: 0.904719	valid_0's binary_logloss: 0.163467
    [7874]	valid_0's auc: 0.904718	valid_0's binary_logloss: 0.16347
    [7875]	valid_0's auc: 0.904717	valid_0's binary_logloss: 0.16347
    [7876]	valid_0's auc: 0.904706	valid_0's binary_logloss: 0.163467
    [7877]	valid_0's auc: 0.904703	valid_0's binary_logloss: 0.163472
    [7878]	valid_0's auc: 0.904708	valid_0's binary_logloss: 0.163463
    [7879]	valid_0's auc: 0.90469	valid_0's binary_logloss: 0.163466
    [7880]	valid_0's auc: 0.904689	valid_0's binary_logloss: 0.16347
    [7881]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163474
    [7882]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163479
    [7883]	valid_0's auc: 0.904669	valid_0's binary_logloss: 0.163482
    [7884]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163486
    [7885]	valid_0's auc: 0.904665	valid_0's binary_logloss: 0.163483
    [7886]	valid_0's auc: 0.904671	valid_0's binary_logloss: 0.163479
    [7887]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163478
    [7888]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163471
    [7889]	valid_0's auc: 0.904667	valid_0's binary_logloss: 0.163474
    [7890]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.163471
    [7891]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.163477
    [7892]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163476
    [7893]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163481
    [7894]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163486
    [7895]	valid_0's auc: 0.904651	valid_0's binary_logloss: 0.163485
    [7896]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.163487
    [7897]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163468
    [7898]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163471
    [7899]	valid_0's auc: 0.90466	valid_0's binary_logloss: 0.163465
    [7900]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.163469
    [7901]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163468
    [7902]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.163462
    [7903]	valid_0's auc: 0.90465	valid_0's binary_logloss: 0.163465
    [7904]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163467
    [7905]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.16347
    [7906]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163476
    [7907]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163478
    [7908]	valid_0's auc: 0.904649	valid_0's binary_logloss: 0.163481
    [7909]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.163486
    [7910]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163492
    [7911]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163489
    [7912]	valid_0's auc: 0.904666	valid_0's binary_logloss: 0.163474
    [7913]	valid_0's auc: 0.904662	valid_0's binary_logloss: 0.163473
    [7914]	valid_0's auc: 0.904658	valid_0's binary_logloss: 0.163472
    [7915]	valid_0's auc: 0.904657	valid_0's binary_logloss: 0.163474
    [7916]	valid_0's auc: 0.904654	valid_0's binary_logloss: 0.163476
    [7917]	valid_0's auc: 0.904652	valid_0's binary_logloss: 0.163472
    [7918]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.16347
    [7919]	valid_0's auc: 0.904647	valid_0's binary_logloss: 0.163474
    [7920]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163478
    [7921]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163475
    [7922]	valid_0's auc: 0.904648	valid_0's binary_logloss: 0.163474
    [7923]	valid_0's auc: 0.904643	valid_0's binary_logloss: 0.16347
    [7924]	valid_0's auc: 0.904645	valid_0's binary_logloss: 0.163453
    [7925]	valid_0's auc: 0.904646	valid_0's binary_logloss: 0.163458
    [7926]	valid_0's auc: 0.904642	valid_0's binary_logloss: 0.163463
    [7927]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.163462
    [7928]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.163465
    [7929]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.163467
    [7930]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.163463
    [7931]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.163468
    [7932]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.16347
    [7933]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.163468
    [7934]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163472
    [7935]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163475
    [7936]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.163474
    [7937]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.163477
    [7938]	valid_0's auc: 0.904618	valid_0's binary_logloss: 0.16348
    [7939]	valid_0's auc: 0.904616	valid_0's binary_logloss: 0.163478
    [7940]	valid_0's auc: 0.904613	valid_0's binary_logloss: 0.163484
    [7941]	valid_0's auc: 0.904613	valid_0's binary_logloss: 0.163481
    [7942]	valid_0's auc: 0.904616	valid_0's binary_logloss: 0.16348
    [7943]	valid_0's auc: 0.904615	valid_0's binary_logloss: 0.163485
    [7944]	valid_0's auc: 0.904615	valid_0's binary_logloss: 0.163487
    [7945]	valid_0's auc: 0.904613	valid_0's binary_logloss: 0.163491
    [7946]	valid_0's auc: 0.904629	valid_0's binary_logloss: 0.16348
    [7947]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.163481
    [7948]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.163488
    [7949]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.163481
    [7950]	valid_0's auc: 0.904627	valid_0's binary_logloss: 0.163485
    [7951]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.163479
    [7952]	valid_0's auc: 0.904644	valid_0's binary_logloss: 0.163475
    [7953]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.163475
    [7954]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.163478
    [7955]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.163481
    [7956]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.163483
    [7957]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.163486
    [7958]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.163485
    [7959]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.163489
    [7960]	valid_0's auc: 0.904631	valid_0's binary_logloss: 0.163492
    [7961]	valid_0's auc: 0.904632	valid_0's binary_logloss: 0.163496
    [7962]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.163499
    [7963]	valid_0's auc: 0.904628	valid_0's binary_logloss: 0.163501
    [7964]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.163494
    [7965]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.163497
    [7966]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.163498
    [7967]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.163495
    [7968]	valid_0's auc: 0.904624	valid_0's binary_logloss: 0.163491
    [7969]	valid_0's auc: 0.904625	valid_0's binary_logloss: 0.163496
    [7970]	valid_0's auc: 0.904622	valid_0's binary_logloss: 0.163499
    [7971]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163503
    [7972]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.1635
    [7973]	valid_0's auc: 0.904639	valid_0's binary_logloss: 0.163493
    [7974]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.163472
    [7975]	valid_0's auc: 0.904634	valid_0's binary_logloss: 0.163476
    [7976]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.16348
    [7977]	valid_0's auc: 0.904633	valid_0's binary_logloss: 0.163481
    [7978]	valid_0's auc: 0.90463	valid_0's binary_logloss: 0.163484
    [7979]	valid_0's auc: 0.904637	valid_0's binary_logloss: 0.163479
    [7980]	valid_0's auc: 0.904635	valid_0's binary_logloss: 0.163484
    [7981]	valid_0's auc: 0.904638	valid_0's binary_logloss: 0.163479
    [7982]	valid_0's auc: 0.904626	valid_0's binary_logloss: 0.163479
    [7983]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163486
    [7984]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163483
    [7985]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.16348
    [7986]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.163477
    [7987]	valid_0's auc: 0.904619	valid_0's binary_logloss: 0.163479
    [7988]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.163482
    [7989]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.163486
    [7990]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163473
    [7991]	valid_0's auc: 0.904621	valid_0's binary_logloss: 0.163476
    [7992]	valid_0's auc: 0.90462	valid_0's binary_logloss: 0.163479
    [7993]	valid_0's auc: 0.90461	valid_0's binary_logloss: 0.163476
    [7994]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.16348
    [7995]	valid_0's auc: 0.90461	valid_0's binary_logloss: 0.163483
    [7996]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.16348
    [7997]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.163484
    [7998]	valid_0's auc: 0.904607	valid_0's binary_logloss: 0.163487
    [7999]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.163484
    [8000]	valid_0's auc: 0.904604	valid_0's binary_logloss: 0.163483
    [8001]	valid_0's auc: 0.904617	valid_0's binary_logloss: 0.16348
    [8002]	valid_0's auc: 0.904615	valid_0's binary_logloss: 0.163482
    [8003]	valid_0's auc: 0.904615	valid_0's binary_logloss: 0.163485
    [8004]	valid_0's auc: 0.904614	valid_0's binary_logloss: 0.163487
    [8005]	valid_0's auc: 0.904615	valid_0's binary_logloss: 0.16349
    [8006]	valid_0's auc: 0.904613	valid_0's binary_logloss: 0.163492
    [8007]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.163494
    [8008]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.163498
    [8009]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.163502
    [8010]	valid_0's auc: 0.904611	valid_0's binary_logloss: 0.163507
    [8011]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.163507
    [8012]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.163498
    [8013]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.163491
    [8014]	valid_0's auc: 0.904605	valid_0's binary_logloss: 0.163494
    [8015]	valid_0's auc: 0.904604	valid_0's binary_logloss: 0.163499
    [8016]	valid_0's auc: 0.904605	valid_0's binary_logloss: 0.163495
    [8017]	valid_0's auc: 0.904605	valid_0's binary_logloss: 0.1635
    [8018]	valid_0's auc: 0.904609	valid_0's binary_logloss: 0.163496
    [8019]	valid_0's auc: 0.904609	valid_0's binary_logloss: 0.163501
    [8020]	valid_0's auc: 0.904607	valid_0's binary_logloss: 0.163504
    [8021]	valid_0's auc: 0.904606	valid_0's binary_logloss: 0.163506
    [8022]	valid_0's auc: 0.904606	valid_0's binary_logloss: 0.16351
    [8023]	valid_0's auc: 0.904604	valid_0's binary_logloss: 0.163513
    [8024]	valid_0's auc: 0.904601	valid_0's binary_logloss: 0.163511
    [8025]	valid_0's auc: 0.904605	valid_0's binary_logloss: 0.163495
    [8026]	valid_0's auc: 0.904601	valid_0's binary_logloss: 0.163495
    [8027]	valid_0's auc: 0.9046	valid_0's binary_logloss: 0.163485
    [8028]	valid_0's auc: 0.904604	valid_0's binary_logloss: 0.163479
    [8029]	valid_0's auc: 0.904608	valid_0's binary_logloss: 0.163477
    [8030]	valid_0's auc: 0.904607	valid_0's binary_logloss: 0.16348
    [8031]	valid_0's auc: 0.904612	valid_0's binary_logloss: 0.163478
    [8032]	valid_0's auc: 0.904609	valid_0's binary_logloss: 0.163484
    [8033]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.163483
    [8034]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.163487
    [8035]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.16348
    [8036]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.163477
    [8037]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.163479
    [8038]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.163474
    [8039]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.163477
    [8040]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.163481
    [8041]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.163485
    [8042]	valid_0's auc: 0.904589	valid_0's binary_logloss: 0.163476
    [8043]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.163478
    [8044]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.163481
    [8045]	valid_0's auc: 0.904585	valid_0's binary_logloss: 0.163486
    [8046]	valid_0's auc: 0.904587	valid_0's binary_logloss: 0.16349
    [8047]	valid_0's auc: 0.904584	valid_0's binary_logloss: 0.163489
    [8048]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.163475
    [8049]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.163471
    [8050]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.163474
    [8051]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.163477
    [8052]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.163465
    [8053]	valid_0's auc: 0.904598	valid_0's binary_logloss: 0.163459
    [8054]	valid_0's auc: 0.904599	valid_0's binary_logloss: 0.163461
    [8055]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.163462
    [8056]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.163464
    [8057]	valid_0's auc: 0.904594	valid_0's binary_logloss: 0.163468
    [8058]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.163473
    [8059]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.163471
    [8060]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.163473
    [8061]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.163477
    [8062]	valid_0's auc: 0.904591	valid_0's binary_logloss: 0.163481
    [8063]	valid_0's auc: 0.904593	valid_0's binary_logloss: 0.163481
    [8064]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.163476
    [8065]	valid_0's auc: 0.904596	valid_0's binary_logloss: 0.163471
    [8066]	valid_0's auc: 0.904595	valid_0's binary_logloss: 0.163476
    [8067]	valid_0's auc: 0.904597	valid_0's binary_logloss: 0.163472
    [8068]	valid_0's auc: 0.90459	valid_0's binary_logloss: 0.163466
    [8069]	valid_0's auc: 0.904589	valid_0's binary_logloss: 0.163469
    [8070]	valid_0's auc: 0.904592	valid_0's binary_logloss: 0.163465
    [8071]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.163464
    [8072]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.163467
    [8073]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.163457
    [8074]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.163461
    [8075]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.163466
    [8076]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.163466
    [8077]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.163471
    [8078]	valid_0's auc: 0.904565	valid_0's binary_logloss: 0.163475
    [8079]	valid_0's auc: 0.904571	valid_0's binary_logloss: 0.163469
    [8080]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.163472
    [8081]	valid_0's auc: 0.904571	valid_0's binary_logloss: 0.163469
    [8082]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.163466
    [8083]	valid_0's auc: 0.904571	valid_0's binary_logloss: 0.163469
    [8084]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.16346
    [8085]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.163457
    [8086]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.163453
    [8087]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.163452
    [8088]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.163455
    [8089]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.163457
    [8090]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163458
    [8091]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.163461
    [8092]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163465
    [8093]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163467
    [8094]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.163461
    [8095]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163468
    [8096]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163464
    [8097]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163467
    [8098]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.16347
    [8099]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163465
    [8100]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163467
    [8101]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163469
    [8102]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163474
    [8103]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163478
    [8104]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163483
    [8105]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.163486
    [8106]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.163491
    [8107]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.163497
    [8108]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163495
    [8109]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163494
    [8110]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163499
    [8111]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163503
    [8112]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163508
    [8113]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163511
    [8114]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163515
    [8115]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163516
    [8116]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163519
    [8117]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163519
    [8118]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163523
    [8119]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163514
    [8120]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163519
    [8121]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163514
    [8122]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163518
    [8123]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163522
    [8124]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.163524
    [8125]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.16352
    [8126]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.163522
    [8127]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.163514
    [8128]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163507
    [8129]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163509
    [8130]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163515
    [8131]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163519
    [8132]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163521
    [8133]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163525
    [8134]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.163528
    [8135]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.163523
    [8136]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.163521
    [8137]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.163525
    [8138]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.163528
    [8139]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.163533
    [8140]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.163536
    [8141]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.163537
    [8142]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.163529
    [8143]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.163532
    [8144]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.16353
    [8145]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.163532
    [8146]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163531
    [8147]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163532
    [8148]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163536
    [8149]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163537
    [8150]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.163538
    [8151]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163537
    [8152]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163527
    [8153]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163523
    [8154]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163515
    [8155]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.16351
    [8156]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163503
    [8157]	valid_0's auc: 0.904571	valid_0's binary_logloss: 0.163482
    [8158]	valid_0's auc: 0.904571	valid_0's binary_logloss: 0.163485
    [8159]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.163489
    [8160]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.163489
    [8161]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.163487
    [8162]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.16348
    [8163]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.163483
    [8164]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.163476
    [8165]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.163475
    [8166]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.163478
    [8167]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.163465
    [8168]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.163468
    [8169]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163468
    [8170]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.163461
    [8171]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.163457
    [8172]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.16346
    [8173]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163464
    [8174]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.16346
    [8175]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163446
    [8176]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163444
    [8177]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163444
    [8178]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163447
    [8179]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.16344
    [8180]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163443
    [8181]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163419
    [8182]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163423
    [8183]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163422
    [8184]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163425
    [8185]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163421
    [8186]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163418
    [8187]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163417
    [8188]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163393
    [8189]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163388
    [8190]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163389
    [8191]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.163388
    [8192]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163391
    [8193]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.16339
    [8194]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.163392
    [8195]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.163394
    [8196]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.163399
    [8197]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.163378
    [8198]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163372
    [8199]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163367
    [8200]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.163365
    [8201]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163368
    [8202]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163359
    [8203]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163362
    [8204]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163365
    [8205]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163371
    [8206]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163364
    [8207]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163364
    [8208]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163361
    [8209]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163356
    [8210]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.163355
    [8211]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163355
    [8212]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163358
    [8213]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163357
    [8214]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.16336
    [8215]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163352
    [8216]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.163356
    [8217]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.163359
    [8218]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163362
    [8219]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163366
    [8220]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163371
    [8221]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163374
    [8222]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163367
    [8223]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163352
    [8224]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163345
    [8225]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163343
    [8226]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.16333
    [8227]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163334
    [8228]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163337
    [8229]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163342
    [8230]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.16334
    [8231]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163343
    [8232]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.16334
    [8233]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163342
    [8234]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163344
    [8235]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163347
    [8236]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163352
    [8237]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.163356
    [8238]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.16335
    [8239]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.16334
    [8240]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.16334
    [8241]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163343
    [8242]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.16334
    [8243]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.163343
    [8244]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163346
    [8245]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163347
    [8246]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163352
    [8247]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163354
    [8248]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163337
    [8249]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163339
    [8250]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163342
    [8251]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.16333
    [8252]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163336
    [8253]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.16334
    [8254]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163331
    [8255]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.163328
    [8256]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.163303
    [8257]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.163304
    [8258]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.16331
    [8259]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163314
    [8260]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163312
    [8261]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163309
    [8262]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.163307
    [8263]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163306
    [8264]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163309
    [8265]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.1633
    [8266]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163304
    [8267]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.163295
    [8268]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163298
    [8269]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163301
    [8270]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.163291
    [8271]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.163294
    [8272]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.163296
    [8273]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.163299
    [8274]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.163301
    [8275]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163298
    [8276]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.1633
    [8277]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163306
    [8278]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163309
    [8279]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163306
    [8280]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.163309
    [8281]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.163309
    [8282]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163309
    [8283]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.16331
    [8284]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163314
    [8285]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163301
    [8286]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163296
    [8287]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.163298
    [8288]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.163302
    [8289]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.163306
    [8290]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163306
    [8291]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163308
    [8292]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163307
    [8293]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163311
    [8294]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163315
    [8295]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163317
    [8296]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.163313
    [8297]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.163307
    [8298]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.163304
    [8299]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.163307
    [8300]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.163311
    [8301]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.163313
    [8302]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.163314
    [8303]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.163311
    [8304]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.163298
    [8305]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.163302
    [8306]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.163307
    [8307]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.163309
    [8308]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163311
    [8309]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.163303
    [8310]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163307
    [8311]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163308
    [8312]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.163303
    [8313]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.163307
    [8314]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.163301
    [8315]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.163304
    [8316]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.163305
    [8317]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.163301
    [8318]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.163305
    [8319]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.163299
    [8320]	valid_0's auc: 0.904559	valid_0's binary_logloss: 0.1633
    [8321]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.163302
    [8322]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.163309
    [8323]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.163309
    [8324]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.163312
    [8325]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163303
    [8326]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163305
    [8327]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163309
    [8328]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163312
    [8329]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163317
    [8330]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.16332
    [8331]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163324
    [8332]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163323
    [8333]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163325
    [8334]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163328
    [8335]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.163331
    [8336]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163316
    [8337]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163319
    [8338]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163322
    [8339]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.16332
    [8340]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.163318
    [8341]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.163321
    [8342]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163309
    [8343]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163312
    [8344]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163315
    [8345]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.163311
    [8346]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163309
    [8347]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163305
    [8348]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163307
    [8349]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.16331
    [8350]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163312
    [8351]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163316
    [8352]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.16332
    [8353]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163324
    [8354]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163318
    [8355]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.16332
    [8356]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163324
    [8357]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163321
    [8358]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163324
    [8359]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163319
    [8360]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.16332
    [8361]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.163325
    [8362]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.163327
    [8363]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.16333
    [8364]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.163333
    [8365]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.163337
    [8366]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.16334
    [8367]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.163339
    [8368]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.163343
    [8369]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.163345
    [8370]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.163347
    [8371]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.163343
    [8372]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.163348
    [8373]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.163352
    [8374]	valid_0's auc: 0.904495	valid_0's binary_logloss: 0.163357
    [8375]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.163354
    [8376]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.163358
    [8377]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.16336
    [8378]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.163363
    [8379]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.163367
    [8380]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.163369
    [8381]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163367
    [8382]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.163365
    [8383]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163367
    [8384]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.163371
    [8385]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.163375
    [8386]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.16338
    [8387]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.163377
    [8388]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.163371
    [8389]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.163377
    [8390]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.16338
    [8391]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.163374
    [8392]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.163369
    [8393]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.163369
    [8394]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.163356
    [8395]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.163359
    [8396]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.163362
    [8397]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.163364
    [8398]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.163366
    [8399]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.163369
    [8400]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.16336
    [8401]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.163362
    [8402]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163363
    [8403]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.163358
    [8404]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.163361
    [8405]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.163363
    [8406]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.163363
    [8407]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.163365
    [8408]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.163368
    [8409]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.163353
    [8410]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.163357
    [8411]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.163357
    [8412]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.163354
    [8413]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163354
    [8414]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.163345
    [8415]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.163322
    [8416]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.163325
    [8417]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163321
    [8418]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163324
    [8419]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163328
    [8420]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.163322
    [8421]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.163324
    [8422]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.163327
    [8423]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.163333
    [8424]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.163336
    [8425]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.163338
    [8426]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.163342
    [8427]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.163347
    [8428]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.163351
    [8429]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163355
    [8430]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163353
    [8431]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.163355
    [8432]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163357
    [8433]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.163357
    [8434]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.163352
    [8435]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.163356
    [8436]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.163361
    [8437]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163357
    [8438]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163361
    [8439]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163363
    [8440]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.163364
    [8441]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.163368
    [8442]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.163372
    [8443]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.163368
    [8444]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163364
    [8445]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163368
    [8446]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.163364
    [8447]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.163367
    [8448]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.16336
    [8449]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163362
    [8450]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163363
    [8451]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.163368
    [8452]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.163354
    [8453]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.163356
    [8454]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.163353
    [8455]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.163344
    [8456]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.163346
    [8457]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.163348
    [8458]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.163352
    [8459]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.163353
    [8460]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163345
    [8461]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163327
    [8462]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163329
    [8463]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163333
    [8464]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.163331
    [8465]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.163334
    [8466]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.163329
    [8467]	valid_0's auc: 0.904425	valid_0's binary_logloss: 0.163324
    [8468]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.163321
    [8469]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.163326
    [8470]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.163322
    [8471]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163326
    [8472]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163329
    [8473]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163331
    [8474]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163335
    [8475]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163339
    [8476]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.163341
    [8477]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163336
    [8478]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163339
    [8479]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163342
    [8480]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.163345
    [8481]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.163347
    [8482]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.163349
    [8483]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.163352
    [8484]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.163352
    [8485]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.163345
    [8486]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163347
    [8487]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.16335
    [8488]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.163353
    [8489]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.163356
    [8490]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.163359
    [8491]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.163358
    [8492]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.163363
    [8493]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.16336
    [8494]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.16336
    [8495]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.163364
    [8496]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.163359
    [8497]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.163363
    [8498]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.163367
    [8499]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.163354
    [8500]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.163347
    [8501]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.163349
    [8502]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.163345
    [8503]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.163344
    [8504]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.163345
    [8505]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163347
    [8506]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.16335
    [8507]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.163353
    [8508]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.163351
    [8509]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.163349
    [8510]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163346
    [8511]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.163342
    [8512]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.163338
    [8513]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.163328
    [8514]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.163332
    [8515]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.163332
    [8516]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.163335
    [8517]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.163337
    [8518]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.163338
    [8519]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163341
    [8520]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.163345
    [8521]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.163344
    [8522]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.163346
    [8523]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.163348
    [8524]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.16335
    [8525]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.163356
    [8526]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.163357
    [8527]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163354
    [8528]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163357
    [8529]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.16336
    [8530]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.16335
    [8531]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163356
    [8532]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.163352
    [8533]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.163354
    [8534]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.163356
    [8535]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.16336
    [8536]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.163359
    [8537]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.163356
    [8538]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.163358
    [8539]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.163354
    [8540]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.163359
    [8541]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.163363
    [8542]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.163366
    [8543]	valid_0's auc: 0.904368	valid_0's binary_logloss: 0.163371
    [8544]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.163366
    [8545]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.163357
    [8546]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.16336
    [8547]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.163357
    [8548]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.163356
    [8549]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.163359
    [8550]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.163363
    [8551]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.163366
    [8552]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.163358
    [8553]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.16336
    [8554]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.163357
    [8555]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.163347
    [8556]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.163345
    [8557]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.163349
    [8558]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.16334
    [8559]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.16334
    [8560]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163339
    [8561]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.163334
    [8562]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.163336
    [8563]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.16332
    [8564]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163299
    [8565]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.163301
    [8566]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.163304
    [8567]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.1633
    [8568]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.163294
    [8569]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.163297
    [8570]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.163302
    [8571]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163305
    [8572]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.163308
    [8573]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.163294
    [8574]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.163297
    [8575]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.163299
    [8576]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.163287
    [8577]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.163291
    [8578]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163292
    [8579]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163296
    [8580]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163299
    [8581]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163302
    [8582]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163305
    [8583]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.16331
    [8584]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163313
    [8585]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163316
    [8586]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.163301
    [8587]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.163303
    [8588]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.163305
    [8589]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.163301
    [8590]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.163296
    [8591]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163297
    [8592]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163299
    [8593]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163301
    [8594]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163304
    [8595]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163306
    [8596]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163299
    [8597]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.163299
    [8598]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.163302
    [8599]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.163298
    [8600]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.163299
    [8601]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.163304
    [8602]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.163306
    [8603]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.163302
    [8604]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.163298
    [8605]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.163301
    [8606]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.163307
    [8607]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.163299
    [8608]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.163301
    [8609]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.163303
    [8610]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.163306
    [8611]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.163308
    [8612]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.163301
    [8613]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.163305
    [8614]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.163309
    [8615]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.163305
    [8616]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.163307
    [8617]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.163307
    [8618]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.16331
    [8619]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.163297
    [8620]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.163299
    [8621]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.163277
    [8622]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.16328
    [8623]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.163287
    [8624]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.163288
    [8625]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.163287
    [8626]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.163284
    [8627]	valid_0's auc: 0.904381	valid_0's binary_logloss: 0.163287
    [8628]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.163284
    [8629]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.163282
    [8630]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.163284
    [8631]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.163282
    [8632]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.163279
    [8633]	valid_0's auc: 0.904362	valid_0's binary_logloss: 0.163281
    [8634]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.163257
    [8635]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163237
    [8636]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.16324
    [8637]	valid_0's auc: 0.904374	valid_0's binary_logloss: 0.163235
    [8638]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.163234
    [8639]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.163237
    [8640]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.16324
    [8641]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.163238
    [8642]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.163234
    [8643]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.163225
    [8644]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.163224
    [8645]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.163221
    [8646]	valid_0's auc: 0.904364	valid_0's binary_logloss: 0.163213
    [8647]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.163209
    [8648]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.163212
    [8649]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.163206
    [8650]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.163203
    [8651]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.163201
    [8652]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.163198
    [8653]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163202
    [8654]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.163198
    [8655]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.163194
    [8656]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.163198
    [8657]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.163201
    [8658]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.163194
    [8659]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.163182
    [8660]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.163185
    [8661]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.163187
    [8662]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.163186
    [8663]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.163185
    [8664]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.163187
    [8665]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.163189
    [8666]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.163192
    [8667]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.163194
    [8668]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.163197
    [8669]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.163186
    [8670]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.16319
    [8671]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.163193
    [8672]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.163195
    [8673]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.163197
    [8674]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.163199
    [8675]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.1632
    [8676]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.163203
    [8677]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.163205
    [8678]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.163207
    [8679]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.163213
    [8680]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.163215
    [8681]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.163221
    [8682]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.163224
    [8683]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.163227
    [8684]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.163231
    [8685]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.163235
    [8686]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.163238
    [8687]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.163235
    [8688]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.16323
    [8689]	valid_0's auc: 0.904421	valid_0's binary_logloss: 0.163209
    [8690]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163204
    [8691]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163198
    [8692]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.16319
    [8693]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.163189
    [8694]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.163187
    [8695]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.163189
    [8696]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.163191
    [8697]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.16319
    [8698]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.163192
    [8699]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163196
    [8700]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.163189
    [8701]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163192
    [8702]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163197
    [8703]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.1632
    [8704]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.163192
    [8705]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.163188
    [8706]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163191
    [8707]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163193
    [8708]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.163194
    [8709]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.163189
    [8710]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.163189
    [8711]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.163185
    [8712]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.163182
    [8713]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.163178
    [8714]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.163172
    [8715]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163171
    [8716]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.163174
    [8717]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.163176
    [8718]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163181
    [8719]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.163183
    [8720]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.163183
    [8721]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.163184
    [8722]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.163181
    [8723]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.16318
    [8724]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163182
    [8725]	valid_0's auc: 0.904426	valid_0's binary_logloss: 0.163185
    [8726]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.163187
    [8727]	valid_0's auc: 0.904424	valid_0's binary_logloss: 0.163186
    [8728]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.16318
    [8729]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.16318
    [8730]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.163172
    [8731]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.163176
    [8732]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.163179
    [8733]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.163181
    [8734]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.163183
    [8735]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163173
    [8736]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.163151
    [8737]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.163155
    [8738]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.163159
    [8739]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.163163
    [8740]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.163165
    [8741]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.163162
    [8742]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163166
    [8743]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.163168
    [8744]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.163172
    [8745]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.163175
    [8746]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.163179
    [8747]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163174
    [8748]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.163179
    [8749]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.16317
    [8750]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.16317
    [8751]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.163167
    [8752]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.163165
    [8753]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.163169
    [8754]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.163168
    [8755]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.163171
    [8756]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163161
    [8757]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.163164
    [8758]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.163168
    [8759]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.163171
    [8760]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.163167
    [8761]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163166
    [8762]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.163162
    [8763]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.163161
    [8764]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.163163
    [8765]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.163157
    [8766]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.163153
    [8767]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.16315
    [8768]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.163149
    [8769]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.163147
    [8770]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.163146
    [8771]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163148
    [8772]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.163134
    [8773]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.163136
    [8774]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.163135
    [8775]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.163133
    [8776]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.163136
    [8777]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.163139
    [8778]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163141
    [8779]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163143
    [8780]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163148
    [8781]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.16315
    [8782]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.163154
    [8783]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.163158
    [8784]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.163156
    [8785]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.163156
    [8786]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.163158
    [8787]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.163155
    [8788]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.163158
    [8789]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.16316
    [8790]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.163163
    [8791]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.163161
    [8792]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.163165
    [8793]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.163167
    [8794]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.16317
    [8795]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.163165
    [8796]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.163168
    [8797]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.163161
    [8798]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.163163
    [8799]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.163166
    [8800]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.163167
    [8801]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.163163
    [8802]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.163166
    [8803]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.163161
    [8804]	valid_0's auc: 0.904495	valid_0's binary_logloss: 0.163151
    [8805]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.163152
    [8806]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.163157
    [8807]	valid_0's auc: 0.904494	valid_0's binary_logloss: 0.163161
    [8808]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.163158
    [8809]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.163153
    [8810]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.163148
    [8811]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.163149
    [8812]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.163151
    [8813]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.163152
    [8814]	valid_0's auc: 0.904494	valid_0's binary_logloss: 0.163138
    [8815]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.163139
    [8816]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.163143
    [8817]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.163148
    [8818]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.163144
    [8819]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.163146
    [8820]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.16315
    [8821]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.163148
    [8822]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.163144
    [8823]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.163148
    [8824]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163143
    [8825]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.163139
    [8826]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163143
    [8827]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.163147
    [8828]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.163146
    [8829]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163149
    [8830]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.163154
    [8831]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.163151
    [8832]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.163154
    [8833]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.163157
    [8834]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.163159
    [8835]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.163162
    [8836]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.163155
    [8837]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.16316
    [8838]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.163162
    [8839]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.163161
    [8840]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.163156
    [8841]	valid_0's auc: 0.904494	valid_0's binary_logloss: 0.163154
    [8842]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.163157
    [8843]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.163159
    [8844]	valid_0's auc: 0.904494	valid_0's binary_logloss: 0.16316
    [8845]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.163158
    [8846]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.163161
    [8847]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.163166
    [8848]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.163162
    [8849]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.16316
    [8850]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.163141
    [8851]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163145
    [8852]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163127
    [8853]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.16313
    [8854]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163133
    [8855]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.163136
    [8856]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163139
    [8857]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.163137
    [8858]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163135
    [8859]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163132
    [8860]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163134
    [8861]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163124
    [8862]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163127
    [8863]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.163131
    [8864]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163132
    [8865]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.163131
    [8866]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.163134
    [8867]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.16313
    [8868]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163133
    [8869]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163137
    [8870]	valid_0's auc: 0.904539	valid_0's binary_logloss: 0.163139
    [8871]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163141
    [8872]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163144
    [8873]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163142
    [8874]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163146
    [8875]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163149
    [8876]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163152
    [8877]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163154
    [8878]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163156
    [8879]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163159
    [8880]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163162
    [8881]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163164
    [8882]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.16316
    [8883]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.163158
    [8884]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.163144
    [8885]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163147
    [8886]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.16315
    [8887]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163152
    [8888]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.163151
    [8889]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163147
    [8890]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.163138
    [8891]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.16314
    [8892]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163143
    [8893]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.163145
    [8894]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.163145
    [8895]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.163136
    [8896]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.163135
    [8897]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163137
    [8898]	valid_0's auc: 0.90454	valid_0's binary_logloss: 0.163139
    [8899]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.16314
    [8900]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.163139
    [8901]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.163142
    [8902]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163146
    [8903]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.163145
    [8904]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.163143
    [8905]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.163142
    [8906]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.163139
    [8907]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.163135
    [8908]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163138
    [8909]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.163139
    [8910]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.163129
    [8911]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163125
    [8912]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163129
    [8913]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.163105
    [8914]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163109
    [8915]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163112
    [8916]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163115
    [8917]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163118
    [8918]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.163114
    [8919]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.163107
    [8920]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.163109
    [8921]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163115
    [8922]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.163118
    [8923]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163121
    [8924]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163124
    [8925]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163123
    [8926]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163126
    [8927]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163123
    [8928]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.163116
    [8929]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163114
    [8930]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163118
    [8931]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.16312
    [8932]	valid_0's auc: 0.904505	valid_0's binary_logloss: 0.163122
    [8933]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.16312
    [8934]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.163122
    [8935]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.163124
    [8936]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.163126
    [8937]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.16313
    [8938]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.163133
    [8939]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.163136
    [8940]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.163138
    [8941]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.163123
    [8942]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.163126
    [8943]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.163125
    [8944]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163113
    [8945]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.163116
    [8946]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.16312
    [8947]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.16312
    [8948]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.163114
    [8949]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.163114
    [8950]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.163115
    [8951]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.163114
    [8952]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.163118
    [8953]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.163098
    [8954]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163089
    [8955]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.163092
    [8956]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163078
    [8957]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163066
    [8958]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.16307
    [8959]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.16307
    [8960]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.16307
    [8961]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.163073
    [8962]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.163076
    [8963]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.163079
    [8964]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.163081
    [8965]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163084
    [8966]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.163083
    [8967]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163078
    [8968]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163073
    [8969]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.163078
    [8970]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163077
    [8971]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163077
    [8972]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163071
    [8973]	valid_0's auc: 0.904534	valid_0's binary_logloss: 0.163066
    [8974]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163056
    [8975]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163054
    [8976]	valid_0's auc: 0.904528	valid_0's binary_logloss: 0.163056
    [8977]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.163053
    [8978]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163049
    [8979]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.163043
    [8980]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.163034
    [8981]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163037
    [8982]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163041
    [8983]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.163039
    [8984]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163032
    [8985]	valid_0's auc: 0.904505	valid_0's binary_logloss: 0.16303
    [8986]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.163017
    [8987]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.16302
    [8988]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163022
    [8989]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163026
    [8990]	valid_0's auc: 0.904515	valid_0's binary_logloss: 0.163029
    [8991]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.163032
    [8992]	valid_0's auc: 0.904515	valid_0's binary_logloss: 0.163035
    [8993]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163031
    [8994]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.163032
    [8995]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.163033
    [8996]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.163023
    [8997]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.163023
    [8998]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163018
    [8999]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.163022
    [9000]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.163024
    [9001]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.163026
    [9002]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.163027
    [9003]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163018
    [9004]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.163017
    [9005]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163009
    [9006]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.16301
    [9007]	valid_0's auc: 0.904526	valid_0's binary_logloss: 0.163013
    [9008]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.163017
    [9009]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.163011
    [9010]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.163009
    [9011]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.163006
    [9012]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.163005
    [9013]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.163
    [9014]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.162978
    [9015]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.162974
    [9016]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.162976
    [9017]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162975
    [9018]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.162977
    [9019]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.162959
    [9020]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.162963
    [9021]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.162962
    [9022]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162958
    [9023]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162962
    [9024]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.162965
    [9025]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.162967
    [9026]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.162971
    [9027]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162973
    [9028]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.162972
    [9029]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.162974
    [9030]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162976
    [9031]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.162967
    [9032]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162967
    [9033]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.16297
    [9034]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.162961
    [9035]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162959
    [9036]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162961
    [9037]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.162958
    [9038]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.162956
    [9039]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.162955
    [9040]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.162952
    [9041]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162949
    [9042]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.162951
    [9043]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162954
    [9044]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.162947
    [9045]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.162946
    [9046]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.162932
    [9047]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.162934
    [9048]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.162938
    [9049]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162942
    [9050]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162947
    [9051]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.162949
    [9052]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.162943
    [9053]	valid_0's auc: 0.90456	valid_0's binary_logloss: 0.162941
    [9054]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.162938
    [9055]	valid_0's auc: 0.904557	valid_0's binary_logloss: 0.16294
    [9056]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162944
    [9057]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.16293
    [9058]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.162933
    [9059]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.162935
    [9060]	valid_0's auc: 0.904541	valid_0's binary_logloss: 0.162938
    [9061]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.16294
    [9062]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.162924
    [9063]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162926
    [9064]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162929
    [9065]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162933
    [9066]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162934
    [9067]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162934
    [9068]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.162936
    [9069]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.16294
    [9070]	valid_0's auc: 0.904547	valid_0's binary_logloss: 0.162941
    [9071]	valid_0's auc: 0.904548	valid_0's binary_logloss: 0.162943
    [9072]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.162941
    [9073]	valid_0's auc: 0.904546	valid_0's binary_logloss: 0.162938
    [9074]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.162941
    [9075]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.162943
    [9076]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.162937
    [9077]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.162938
    [9078]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.162942
    [9079]	valid_0's auc: 0.904542	valid_0's binary_logloss: 0.162944
    [9080]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.162937
    [9081]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162939
    [9082]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162934
    [9083]	valid_0's auc: 0.904555	valid_0's binary_logloss: 0.162936
    [9084]	valid_0's auc: 0.904556	valid_0's binary_logloss: 0.162931
    [9085]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.162916
    [9086]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.162916
    [9087]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.162913
    [9088]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.162916
    [9089]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.162913
    [9090]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.162915
    [9091]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.162917
    [9092]	valid_0's auc: 0.904575	valid_0's binary_logloss: 0.162919
    [9093]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.162921
    [9094]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.162924
    [9095]	valid_0's auc: 0.904579	valid_0's binary_logloss: 0.162926
    [9096]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.162929
    [9097]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.162931
    [9098]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.162931
    [9099]	valid_0's auc: 0.904574	valid_0's binary_logloss: 0.162933
    [9100]	valid_0's auc: 0.904573	valid_0's binary_logloss: 0.162936
    [9101]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.162931
    [9102]	valid_0's auc: 0.904583	valid_0's binary_logloss: 0.162933
    [9103]	valid_0's auc: 0.904593	valid_0's binary_logloss: 0.162926
    [9104]	valid_0's auc: 0.904588	valid_0's binary_logloss: 0.162924
    [9105]	valid_0's auc: 0.904582	valid_0's binary_logloss: 0.162925
    [9106]	valid_0's auc: 0.904587	valid_0's binary_logloss: 0.162923
    [9107]	valid_0's auc: 0.904586	valid_0's binary_logloss: 0.162927
    [9108]	valid_0's auc: 0.904586	valid_0's binary_logloss: 0.162928
    [9109]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.162923
    [9110]	valid_0's auc: 0.904581	valid_0's binary_logloss: 0.162925
    [9111]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.162927
    [9112]	valid_0's auc: 0.904578	valid_0's binary_logloss: 0.162929
    [9113]	valid_0's auc: 0.90458	valid_0's binary_logloss: 0.162931
    [9114]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.16293
    [9115]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.162931
    [9116]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.162935
    [9117]	valid_0's auc: 0.904577	valid_0's binary_logloss: 0.162938
    [9118]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.162939
    [9119]	valid_0's auc: 0.904576	valid_0's binary_logloss: 0.162942
    [9120]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.162941
    [9121]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.162943
    [9122]	valid_0's auc: 0.90457	valid_0's binary_logloss: 0.162946
    [9123]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.162949
    [9124]	valid_0's auc: 0.904569	valid_0's binary_logloss: 0.162951
    [9125]	valid_0's auc: 0.904568	valid_0's binary_logloss: 0.162953
    [9126]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.162957
    [9127]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.162952
    [9128]	valid_0's auc: 0.904563	valid_0's binary_logloss: 0.162954
    [9129]	valid_0's auc: 0.904564	valid_0's binary_logloss: 0.162956
    [9130]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.16296
    [9131]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.162963
    [9132]	valid_0's auc: 0.904561	valid_0's binary_logloss: 0.162965
    [9133]	valid_0's auc: 0.904558	valid_0's binary_logloss: 0.162968
    [9134]	valid_0's auc: 0.904567	valid_0's binary_logloss: 0.162961
    [9135]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.162964
    [9136]	valid_0's auc: 0.904559	valid_0's binary_logloss: 0.162963
    [9137]	valid_0's auc: 0.904562	valid_0's binary_logloss: 0.162951
    [9138]	valid_0's auc: 0.904566	valid_0's binary_logloss: 0.162941
    [9139]	valid_0's auc: 0.904554	valid_0's binary_logloss: 0.162941
    [9140]	valid_0's auc: 0.904553	valid_0's binary_logloss: 0.162943
    [9141]	valid_0's auc: 0.904552	valid_0's binary_logloss: 0.162945
    [9142]	valid_0's auc: 0.90455	valid_0's binary_logloss: 0.162944
    [9143]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162946
    [9144]	valid_0's auc: 0.904549	valid_0's binary_logloss: 0.162946
    [9145]	valid_0's auc: 0.904545	valid_0's binary_logloss: 0.162946
    [9146]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.162949
    [9147]	valid_0's auc: 0.904543	valid_0's binary_logloss: 0.162949
    [9148]	valid_0's auc: 0.904544	valid_0's binary_logloss: 0.162952
    [9149]	valid_0's auc: 0.904551	valid_0's binary_logloss: 0.162944
    [9150]	valid_0's auc: 0.904533	valid_0's binary_logloss: 0.162937
    [9151]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.16294
    [9152]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.162941
    [9153]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.162943
    [9154]	valid_0's auc: 0.904529	valid_0's binary_logloss: 0.162942
    [9155]	valid_0's auc: 0.904521	valid_0's binary_logloss: 0.162942
    [9156]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.162942
    [9157]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.162942
    [9158]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.162944
    [9159]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.162946
    [9160]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.16295
    [9161]	valid_0's auc: 0.904519	valid_0's binary_logloss: 0.162945
    [9162]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.162941
    [9163]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.162943
    [9164]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.162944
    [9165]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.162948
    [9166]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.162952
    [9167]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162954
    [9168]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162956
    [9169]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162957
    [9170]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.162955
    [9171]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162952
    [9172]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162947
    [9173]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162949
    [9174]	valid_0's auc: 0.904494	valid_0's binary_logloss: 0.162952
    [9175]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.162951
    [9176]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.162954
    [9177]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162947
    [9178]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.16295
    [9179]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162951
    [9180]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162948
    [9181]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162945
    [9182]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162947
    [9183]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162944
    [9184]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.162946
    [9185]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162948
    [9186]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.16295
    [9187]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162948
    [9188]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.16295
    [9189]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162949
    [9190]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162951
    [9191]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.16294
    [9192]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.162942
    [9193]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.162937
    [9194]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.16294
    [9195]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.16294
    [9196]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162942
    [9197]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.16294
    [9198]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.162943
    [9199]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.162944
    [9200]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.162941
    [9201]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162941
    [9202]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.162938
    [9203]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.162929
    [9204]	valid_0's auc: 0.904535	valid_0's binary_logloss: 0.162914
    [9205]	valid_0's auc: 0.904536	valid_0's binary_logloss: 0.162916
    [9206]	valid_0's auc: 0.904532	valid_0's binary_logloss: 0.162919
    [9207]	valid_0's auc: 0.90453	valid_0's binary_logloss: 0.162921
    [9208]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.162919
    [9209]	valid_0's auc: 0.904538	valid_0's binary_logloss: 0.162923
    [9210]	valid_0's auc: 0.904537	valid_0's binary_logloss: 0.162925
    [9211]	valid_0's auc: 0.904531	valid_0's binary_logloss: 0.162922
    [9212]	valid_0's auc: 0.904522	valid_0's binary_logloss: 0.162923
    [9213]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162927
    [9214]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162927
    [9215]	valid_0's auc: 0.904505	valid_0's binary_logloss: 0.16293
    [9216]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.162921
    [9217]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.162925
    [9218]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.162923
    [9219]	valid_0's auc: 0.904514	valid_0's binary_logloss: 0.16292
    [9220]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.162918
    [9221]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.162915
    [9222]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.162919
    [9223]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.162916
    [9224]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162915
    [9225]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162913
    [9226]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162917
    [9227]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.16292
    [9228]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162921
    [9229]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162923
    [9230]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162919
    [9231]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.162913
    [9232]	valid_0's auc: 0.904515	valid_0's binary_logloss: 0.162914
    [9233]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.162917
    [9234]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.16292
    [9235]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162922
    [9236]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162922
    [9237]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162925
    [9238]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162922
    [9239]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162925
    [9240]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.162922
    [9241]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162912
    [9242]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162916
    [9243]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.162915
    [9244]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.162904
    [9245]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.162906
    [9246]	valid_0's auc: 0.904506	valid_0's binary_logloss: 0.162906
    [9247]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162909
    [9248]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162912
    [9249]	valid_0's auc: 0.904495	valid_0's binary_logloss: 0.162909
    [9250]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162911
    [9251]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162898
    [9252]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.1629
    [9253]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162903
    [9254]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.1629
    [9255]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.162901
    [9256]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.162904
    [9257]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.162887
    [9258]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.162889
    [9259]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162876
    [9260]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162879
    [9261]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162882
    [9262]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162884
    [9263]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.162868
    [9264]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.162871
    [9265]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.162874
    [9266]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.16287
    [9267]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.162872
    [9268]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162877
    [9269]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162869
    [9270]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162872
    [9271]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162875
    [9272]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.162879
    [9273]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.162882
    [9274]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162884
    [9275]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162886
    [9276]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162889
    [9277]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162891
    [9278]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162893
    [9279]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162896
    [9280]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162895
    [9281]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162897
    [9282]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.162899
    [9283]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.162903
    [9284]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162902
    [9285]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162906
    [9286]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162909
    [9287]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162905
    [9288]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162909
    [9289]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162912
    [9290]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162903
    [9291]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162903
    [9292]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162898
    [9293]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.1629
    [9294]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162901
    [9295]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162902
    [9296]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.1629
    [9297]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162904
    [9298]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162903
    [9299]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162906
    [9300]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162908
    [9301]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162902
    [9302]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162905
    [9303]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162906
    [9304]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.162907
    [9305]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.162909
    [9306]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162898
    [9307]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162901
    [9308]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162904
    [9309]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162907
    [9310]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.162901
    [9311]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162904
    [9312]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.162895
    [9313]	valid_0's auc: 0.904513	valid_0's binary_logloss: 0.162897
    [9314]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.162896
    [9315]	valid_0's auc: 0.904523	valid_0's binary_logloss: 0.16289
    [9316]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.162887
    [9317]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.162891
    [9318]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.162892
    [9319]	valid_0's auc: 0.904524	valid_0's binary_logloss: 0.162896
    [9320]	valid_0's auc: 0.904527	valid_0's binary_logloss: 0.162898
    [9321]	valid_0's auc: 0.904525	valid_0's binary_logloss: 0.162897
    [9322]	valid_0's auc: 0.90452	valid_0's binary_logloss: 0.162898
    [9323]	valid_0's auc: 0.904509	valid_0's binary_logloss: 0.162897
    [9324]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.1629
    [9325]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162903
    [9326]	valid_0's auc: 0.904508	valid_0's binary_logloss: 0.162903
    [9327]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162906
    [9328]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.162902
    [9329]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162904
    [9330]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.162895
    [9331]	valid_0's auc: 0.90451	valid_0's binary_logloss: 0.162898
    [9332]	valid_0's auc: 0.904511	valid_0's binary_logloss: 0.1629
    [9333]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.1629
    [9334]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.162901
    [9335]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162903
    [9336]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162904
    [9337]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162907
    [9338]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.16291
    [9339]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.162909
    [9340]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162912
    [9341]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.16291
    [9342]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162914
    [9343]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162915
    [9344]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162915
    [9345]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.162916
    [9346]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.162919
    [9347]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.162914
    [9348]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.162913
    [9349]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162912
    [9350]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162907
    [9351]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.16291
    [9352]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162912
    [9353]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162914
    [9354]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162916
    [9355]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162908
    [9356]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.16291
    [9357]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.16291
    [9358]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162892
    [9359]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.16289
    [9360]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162893
    [9361]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162896
    [9362]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162893
    [9363]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162894
    [9364]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162895
    [9365]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162888
    [9366]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162891
    [9367]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.16289
    [9368]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.162892
    [9369]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162894
    [9370]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162896
    [9371]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.162896
    [9372]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162893
    [9373]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162895
    [9374]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162893
    [9375]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162895
    [9376]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162897
    [9377]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162899
    [9378]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162902
    [9379]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.162898
    [9380]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.162901
    [9381]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162901
    [9382]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.162904
    [9383]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.162907
    [9384]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162905
    [9385]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162898
    [9386]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.1629
    [9387]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162898
    [9388]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162897
    [9389]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.1629
    [9390]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162905
    [9391]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.162909
    [9392]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162911
    [9393]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162909
    [9394]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162911
    [9395]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162914
    [9396]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162917
    [9397]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.16292
    [9398]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162923
    [9399]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162925
    [9400]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162926
    [9401]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.16293
    [9402]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.162932
    [9403]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162934
    [9404]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.162938
    [9405]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.16294
    [9406]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.162937
    [9407]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162939
    [9408]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162934
    [9409]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162935
    [9410]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162939
    [9411]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.162936
    [9412]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162932
    [9413]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.162935
    [9414]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.162937
    [9415]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162936
    [9416]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162938
    [9417]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.16294
    [9418]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162941
    [9419]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.162941
    [9420]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.162944
    [9421]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.162948
    [9422]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.16295
    [9423]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162948
    [9424]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162952
    [9425]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.162956
    [9426]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.162957
    [9427]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.162954
    [9428]	valid_0's auc: 0.904416	valid_0's binary_logloss: 0.162958
    [9429]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.162963
    [9430]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.16296
    [9431]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.162962
    [9432]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.162964
    [9433]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.162963
    [9434]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.162966
    [9435]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162968
    [9436]	valid_0's auc: 0.904407	valid_0's binary_logloss: 0.16297
    [9437]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162969
    [9438]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.162968
    [9439]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.16297
    [9440]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.162972
    [9441]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162967
    [9442]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.162963
    [9443]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.162962
    [9444]	valid_0's auc: 0.904412	valid_0's binary_logloss: 0.162961
    [9445]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.162959
    [9446]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.16296
    [9447]	valid_0's auc: 0.904405	valid_0's binary_logloss: 0.162956
    [9448]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162959
    [9449]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162961
    [9450]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162964
    [9451]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162965
    [9452]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162968
    [9453]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.162969
    [9454]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.16297
    [9455]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.162966
    [9456]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162968
    [9457]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.162966
    [9458]	valid_0's auc: 0.90441	valid_0's binary_logloss: 0.16297
    [9459]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.162971
    [9460]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.162975
    [9461]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.162974
    [9462]	valid_0's auc: 0.90438	valid_0's binary_logloss: 0.16297
    [9463]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.162973
    [9464]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.162977
    [9465]	valid_0's auc: 0.904371	valid_0's binary_logloss: 0.162977
    [9466]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.162979
    [9467]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.162975
    [9468]	valid_0's auc: 0.904366	valid_0's binary_logloss: 0.162976
    [9469]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.16298
    [9470]	valid_0's auc: 0.904363	valid_0's binary_logloss: 0.162978
    [9471]	valid_0's auc: 0.904369	valid_0's binary_logloss: 0.162974
    [9472]	valid_0's auc: 0.904355	valid_0's binary_logloss: 0.162976
    [9473]	valid_0's auc: 0.904354	valid_0's binary_logloss: 0.162978
    [9474]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.162969
    [9475]	valid_0's auc: 0.90436	valid_0's binary_logloss: 0.16297
    [9476]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.162973
    [9477]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.162971
    [9478]	valid_0's auc: 0.904361	valid_0's binary_logloss: 0.162973
    [9479]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.162965
    [9480]	valid_0's auc: 0.904359	valid_0's binary_logloss: 0.162967
    [9481]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.162955
    [9482]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.162952
    [9483]	valid_0's auc: 0.904382	valid_0's binary_logloss: 0.162954
    [9484]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.162936
    [9485]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.162938
    [9486]	valid_0's auc: 0.904378	valid_0's binary_logloss: 0.162939
    [9487]	valid_0's auc: 0.904373	valid_0's binary_logloss: 0.162939
    [9488]	valid_0's auc: 0.904367	valid_0's binary_logloss: 0.162931
    [9489]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.162915
    [9490]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.162913
    [9491]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162917
    [9492]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.162916
    [9493]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.162918
    [9494]	valid_0's auc: 0.904389	valid_0's binary_logloss: 0.16292
    [9495]	valid_0's auc: 0.904388	valid_0's binary_logloss: 0.162922
    [9496]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.162923
    [9497]	valid_0's auc: 0.904387	valid_0's binary_logloss: 0.162926
    [9498]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.162924
    [9499]	valid_0's auc: 0.904386	valid_0's binary_logloss: 0.162919
    [9500]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.162903
    [9501]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.162907
    [9502]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162908
    [9503]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162911
    [9504]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162913
    [9505]	valid_0's auc: 0.904393	valid_0's binary_logloss: 0.162915
    [9506]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.162913
    [9507]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.162912
    [9508]	valid_0's auc: 0.904385	valid_0's binary_logloss: 0.162913
    [9509]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.162914
    [9510]	valid_0's auc: 0.904384	valid_0's binary_logloss: 0.162917
    [9511]	valid_0's auc: 0.904383	valid_0's binary_logloss: 0.16292
    [9512]	valid_0's auc: 0.904376	valid_0's binary_logloss: 0.162919
    [9513]	valid_0's auc: 0.904375	valid_0's binary_logloss: 0.162921
    [9514]	valid_0's auc: 0.904377	valid_0's binary_logloss: 0.162921
    [9515]	valid_0's auc: 0.904379	valid_0's binary_logloss: 0.162919
    [9516]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.162909
    [9517]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.162901
    [9518]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.162901
    [9519]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.162903
    [9520]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.1629
    [9521]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.162902
    [9522]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.162905
    [9523]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.162899
    [9524]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.162902
    [9525]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162904
    [9526]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.1629
    [9527]	valid_0's auc: 0.904409	valid_0's binary_logloss: 0.162903
    [9528]	valid_0's auc: 0.904408	valid_0's binary_logloss: 0.162906
    [9529]	valid_0's auc: 0.904411	valid_0's binary_logloss: 0.162899
    [9530]	valid_0's auc: 0.904414	valid_0's binary_logloss: 0.162898
    [9531]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.162901
    [9532]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.162905
    [9533]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.162907
    [9534]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.162907
    [9535]	valid_0's auc: 0.904397	valid_0's binary_logloss: 0.162905
    [9536]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162899
    [9537]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162901
    [9538]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.162893
    [9539]	valid_0's auc: 0.904391	valid_0's binary_logloss: 0.162895
    [9540]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162888
    [9541]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.16289
    [9542]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.162893
    [9543]	valid_0's auc: 0.904406	valid_0's binary_logloss: 0.162893
    [9544]	valid_0's auc: 0.904404	valid_0's binary_logloss: 0.162895
    [9545]	valid_0's auc: 0.904403	valid_0's binary_logloss: 0.162899
    [9546]	valid_0's auc: 0.90439	valid_0's binary_logloss: 0.162902
    [9547]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162899
    [9548]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.162889
    [9549]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.162891
    [9550]	valid_0's auc: 0.904396	valid_0's binary_logloss: 0.162891
    [9551]	valid_0's auc: 0.904398	valid_0's binary_logloss: 0.162893
    [9552]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.162895
    [9553]	valid_0's auc: 0.904395	valid_0's binary_logloss: 0.162889
    [9554]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162892
    [9555]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.162896
    [9556]	valid_0's auc: 0.904392	valid_0's binary_logloss: 0.162899
    [9557]	valid_0's auc: 0.904394	valid_0's binary_logloss: 0.162896
    [9558]	valid_0's auc: 0.904402	valid_0's binary_logloss: 0.162887
    [9559]	valid_0's auc: 0.904401	valid_0's binary_logloss: 0.162889
    [9560]	valid_0's auc: 0.9044	valid_0's binary_logloss: 0.162892
    [9561]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.162895
    [9562]	valid_0's auc: 0.904399	valid_0's binary_logloss: 0.162898
    [9563]	valid_0's auc: 0.904419	valid_0's binary_logloss: 0.162894
    [9564]	valid_0's auc: 0.904413	valid_0's binary_logloss: 0.162895
    [9565]	valid_0's auc: 0.904417	valid_0's binary_logloss: 0.162891
    [9566]	valid_0's auc: 0.904415	valid_0's binary_logloss: 0.16289
    [9567]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.162871
    [9568]	valid_0's auc: 0.904418	valid_0's binary_logloss: 0.162873
    [9569]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.162872
    [9570]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162858
    [9571]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.16285
    [9572]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162844
    [9573]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162841
    [9574]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162843
    [9575]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162844
    [9576]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162846
    [9577]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162849
    [9578]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162853
    [9579]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162854
    [9580]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162856
    [9581]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162855
    [9582]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.16285
    [9583]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.162848
    [9584]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162849
    [9585]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162851
    [9586]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.162849
    [9587]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162842
    [9588]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162841
    [9589]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162844
    [9590]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162841
    [9591]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162839
    [9592]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.162841
    [9593]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.162833
    [9594]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.162835
    [9595]	valid_0's auc: 0.904439	valid_0's binary_logloss: 0.162837
    [9596]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.162832
    [9597]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.162831
    [9598]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.162833
    [9599]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162834
    [9600]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.162835
    [9601]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162831
    [9602]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.162831
    [9603]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162832
    [9604]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.162835
    [9605]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162836
    [9606]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162834
    [9607]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.162813
    [9608]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162805
    [9609]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162806
    [9610]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162808
    [9611]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162805
    [9612]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.162808
    [9613]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.162794
    [9614]	valid_0's auc: 0.904467	valid_0's binary_logloss: 0.162793
    [9615]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162795
    [9616]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162797
    [9617]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162799
    [9618]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.1628
    [9619]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162802
    [9620]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.162804
    [9621]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162802
    [9622]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162801
    [9623]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162804
    [9624]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162806
    [9625]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.162808
    [9626]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.162809
    [9627]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162811
    [9628]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.162812
    [9629]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162816
    [9630]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162818
    [9631]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.162805
    [9632]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.162807
    [9633]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.16281
    [9634]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.162798
    [9635]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.1628
    [9636]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162799
    [9637]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.162802
    [9638]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.162801
    [9639]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.162803
    [9640]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162806
    [9641]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162799
    [9642]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.1628
    [9643]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162799
    [9644]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.1628
    [9645]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.162803
    [9646]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162797
    [9647]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162798
    [9648]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.162799
    [9649]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.16279
    [9650]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162792
    [9651]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.162796
    [9652]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162798
    [9653]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162795
    [9654]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.162784
    [9655]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.162787
    [9656]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162785
    [9657]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162787
    [9658]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162783
    [9659]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.162786
    [9660]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.162786
    [9661]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.16278
    [9662]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162774
    [9663]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162775
    [9664]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162774
    [9665]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.162771
    [9666]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.162773
    [9667]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162775
    [9668]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.162775
    [9669]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.162777
    [9670]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.162779
    [9671]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.162781
    [9672]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162784
    [9673]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162786
    [9674]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162789
    [9675]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162784
    [9676]	valid_0's auc: 0.904434	valid_0's binary_logloss: 0.162786
    [9677]	valid_0's auc: 0.904433	valid_0's binary_logloss: 0.162788
    [9678]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162784
    [9679]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162782
    [9680]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162782
    [9681]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.162781
    [9682]	valid_0's auc: 0.90442	valid_0's binary_logloss: 0.162785
    [9683]	valid_0's auc: 0.904423	valid_0's binary_logloss: 0.162782
    [9684]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162774
    [9685]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162767
    [9686]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.162767
    [9687]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.162768
    [9688]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.162766
    [9689]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162768
    [9690]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162771
    [9691]	valid_0's auc: 0.90444	valid_0's binary_logloss: 0.162767
    [9692]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.162762
    [9693]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.162763
    [9694]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.162765
    [9695]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162767
    [9696]	valid_0's auc: 0.904437	valid_0's binary_logloss: 0.162762
    [9697]	valid_0's auc: 0.904435	valid_0's binary_logloss: 0.162764
    [9698]	valid_0's auc: 0.904428	valid_0's binary_logloss: 0.162764
    [9699]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162766
    [9700]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.162769
    [9701]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.162765
    [9702]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162757
    [9703]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162755
    [9704]	valid_0's auc: 0.904436	valid_0's binary_logloss: 0.162751
    [9705]	valid_0's auc: 0.904438	valid_0's binary_logloss: 0.162753
    [9706]	valid_0's auc: 0.904422	valid_0's binary_logloss: 0.162755
    [9707]	valid_0's auc: 0.904429	valid_0's binary_logloss: 0.162746
    [9708]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162744
    [9709]	valid_0's auc: 0.904432	valid_0's binary_logloss: 0.162743
    [9710]	valid_0's auc: 0.904431	valid_0's binary_logloss: 0.16274
    [9711]	valid_0's auc: 0.90443	valid_0's binary_logloss: 0.162741
    [9712]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.16273
    [9713]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.162726
    [9714]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.162725
    [9715]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162722
    [9716]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162723
    [9717]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162725
    [9718]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162726
    [9719]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.162718
    [9720]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.16272
    [9721]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162722
    [9722]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162725
    [9723]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162719
    [9724]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.16272
    [9725]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162723
    [9726]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162724
    [9727]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.16272
    [9728]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.162713
    [9729]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162712
    [9730]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162713
    [9731]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162716
    [9732]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162719
    [9733]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162714
    [9734]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162718
    [9735]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.16272
    [9736]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.16272
    [9737]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.16272
    [9738]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162723
    [9739]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162727
    [9740]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162729
    [9741]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.16273
    [9742]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162729
    [9743]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162731
    [9744]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162733
    [9745]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162732
    [9746]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162734
    [9747]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.162736
    [9748]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162738
    [9749]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.162739
    [9750]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.16274
    [9751]	valid_0's auc: 0.904459	valid_0's binary_logloss: 0.162742
    [9752]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.162744
    [9753]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162741
    [9754]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162741
    [9755]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162734
    [9756]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.162728
    [9757]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162723
    [9758]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162724
    [9759]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.162727
    [9760]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.162729
    [9761]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.162732
    [9762]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.162735
    [9763]	valid_0's auc: 0.904448	valid_0's binary_logloss: 0.162729
    [9764]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162729
    [9765]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162731
    [9766]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162727
    [9767]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162729
    [9768]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162731
    [9769]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162734
    [9770]	valid_0's auc: 0.904447	valid_0's binary_logloss: 0.162733
    [9771]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162738
    [9772]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.16274
    [9773]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.162743
    [9774]	valid_0's auc: 0.904442	valid_0's binary_logloss: 0.162744
    [9775]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.162747
    [9776]	valid_0's auc: 0.904441	valid_0's binary_logloss: 0.162749
    [9777]	valid_0's auc: 0.904449	valid_0's binary_logloss: 0.162738
    [9778]	valid_0's auc: 0.904456	valid_0's binary_logloss: 0.162733
    [9779]	valid_0's auc: 0.904455	valid_0's binary_logloss: 0.162731
    [9780]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.162729
    [9781]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.16272
    [9782]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.16272
    [9783]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162722
    [9784]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162725
    [9785]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162727
    [9786]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162726
    [9787]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162723
    [9788]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.162722
    [9789]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162716
    [9790]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162712
    [9791]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162714
    [9792]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162715
    [9793]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162716
    [9794]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162711
    [9795]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162714
    [9796]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162718
    [9797]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162719
    [9798]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162723
    [9799]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162724
    [9800]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162726
    [9801]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162729
    [9802]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162732
    [9803]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162723
    [9804]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162726
    [9805]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162712
    [9806]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.162707
    [9807]	valid_0's auc: 0.904517	valid_0's binary_logloss: 0.162709
    [9808]	valid_0's auc: 0.904516	valid_0's binary_logloss: 0.162711
    [9809]	valid_0's auc: 0.904518	valid_0's binary_logloss: 0.16271
    [9810]	valid_0's auc: 0.904512	valid_0's binary_logloss: 0.162711
    [9811]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.16271
    [9812]	valid_0's auc: 0.904507	valid_0's binary_logloss: 0.162712
    [9813]	valid_0's auc: 0.904504	valid_0's binary_logloss: 0.162713
    [9814]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.162721
    [9815]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162718
    [9816]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.16272
    [9817]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162707
    [9818]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162708
    [9819]	valid_0's auc: 0.9045	valid_0's binary_logloss: 0.162703
    [9820]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162705
    [9821]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162701
    [9822]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.162703
    [9823]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.162706
    [9824]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.162709
    [9825]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.16271
    [9826]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162712
    [9827]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162708
    [9828]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.16271
    [9829]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162712
    [9830]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162714
    [9831]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162716
    [9832]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162717
    [9833]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162719
    [9834]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.162714
    [9835]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162715
    [9836]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162717
    [9837]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162716
    [9838]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162719
    [9839]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162717
    [9840]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.162701
    [9841]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162696
    [9842]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162698
    [9843]	valid_0's auc: 0.904499	valid_0's binary_logloss: 0.162699
    [9844]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162701
    [9845]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.162695
    [9846]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162696
    [9847]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.162698
    [9848]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162699
    [9849]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.162697
    [9850]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.162697
    [9851]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.162696
    [9852]	valid_0's auc: 0.904493	valid_0's binary_logloss: 0.162698
    [9853]	valid_0's auc: 0.904503	valid_0's binary_logloss: 0.162694
    [9854]	valid_0's auc: 0.904502	valid_0's binary_logloss: 0.162696
    [9855]	valid_0's auc: 0.904501	valid_0's binary_logloss: 0.162698
    [9856]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162702
    [9857]	valid_0's auc: 0.904489	valid_0's binary_logloss: 0.162699
    [9858]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.162696
    [9859]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.162698
    [9860]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.162699
    [9861]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.162697
    [9862]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.162699
    [9863]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.162691
    [9864]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162688
    [9865]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.16269
    [9866]	valid_0's auc: 0.904491	valid_0's binary_logloss: 0.162687
    [9867]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162681
    [9868]	valid_0's auc: 0.904498	valid_0's binary_logloss: 0.16268
    [9869]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162682
    [9870]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162684
    [9871]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162685
    [9872]	valid_0's auc: 0.904483	valid_0's binary_logloss: 0.162687
    [9873]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162683
    [9874]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162688
    [9875]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162687
    [9876]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162682
    [9877]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162684
    [9878]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162687
    [9879]	valid_0's auc: 0.904451	valid_0's binary_logloss: 0.162684
    [9880]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162686
    [9881]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162688
    [9882]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162691
    [9883]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162694
    [9884]	valid_0's auc: 0.90445	valid_0's binary_logloss: 0.162681
    [9885]	valid_0's auc: 0.904444	valid_0's binary_logloss: 0.162681
    [9886]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162674
    [9887]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162676
    [9888]	valid_0's auc: 0.904465	valid_0's binary_logloss: 0.16267
    [9889]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162672
    [9890]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162674
    [9891]	valid_0's auc: 0.904463	valid_0's binary_logloss: 0.162676
    [9892]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162677
    [9893]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162681
    [9894]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.162683
    [9895]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.162679
    [9896]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162681
    [9897]	valid_0's auc: 0.904457	valid_0's binary_logloss: 0.16268
    [9898]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162664
    [9899]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.162666
    [9900]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162663
    [9901]	valid_0's auc: 0.904497	valid_0's binary_logloss: 0.162647
    [9902]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162646
    [9903]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.162645
    [9904]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162645
    [9905]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162647
    [9906]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162649
    [9907]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162652
    [9908]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162649
    [9909]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162651
    [9910]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162653
    [9911]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162657
    [9912]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162653
    [9913]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162655
    [9914]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162657
    [9915]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162657
    [9916]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.16265
    [9917]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162651
    [9918]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162654
    [9919]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162655
    [9920]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.162646
    [9921]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162647
    [9922]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162644
    [9923]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162644
    [9924]	valid_0's auc: 0.904484	valid_0's binary_logloss: 0.162643
    [9925]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162645
    [9926]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162647
    [9927]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162649
    [9928]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.16265
    [9929]	valid_0's auc: 0.904485	valid_0's binary_logloss: 0.162652
    [9930]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162648
    [9931]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162651
    [9932]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162653
    [9933]	valid_0's auc: 0.904487	valid_0's binary_logloss: 0.162654
    [9934]	valid_0's auc: 0.904488	valid_0's binary_logloss: 0.162656
    [9935]	valid_0's auc: 0.904486	valid_0's binary_logloss: 0.162658
    [9936]	valid_0's auc: 0.904469	valid_0's binary_logloss: 0.162659
    [9937]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162658
    [9938]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162654
    [9939]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162647
    [9940]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162646
    [9941]	valid_0's auc: 0.90447	valid_0's binary_logloss: 0.162648
    [9942]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162648
    [9943]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162645
    [9944]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162647
    [9945]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162636
    [9946]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162639
    [9947]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162638
    [9948]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.16264
    [9949]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162642
    [9950]	valid_0's auc: 0.904472	valid_0's binary_logloss: 0.162644
    [9951]	valid_0's auc: 0.904471	valid_0's binary_logloss: 0.162645
    [9952]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162641
    [9953]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162637
    [9954]	valid_0's auc: 0.90448	valid_0's binary_logloss: 0.162637
    [9955]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162629
    [9956]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162631
    [9957]	valid_0's auc: 0.904477	valid_0's binary_logloss: 0.162628
    [9958]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.16263
    [9959]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162634
    [9960]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162636
    [9961]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162637
    [9962]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.16264
    [9963]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162641
    [9964]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162643
    [9965]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.162644
    [9966]	valid_0's auc: 0.904481	valid_0's binary_logloss: 0.162643
    [9967]	valid_0's auc: 0.904482	valid_0's binary_logloss: 0.162645
    [9968]	valid_0's auc: 0.904495	valid_0's binary_logloss: 0.162641
    [9969]	valid_0's auc: 0.904496	valid_0's binary_logloss: 0.162643
    [9970]	valid_0's auc: 0.904492	valid_0's binary_logloss: 0.162644
    [9971]	valid_0's auc: 0.90449	valid_0's binary_logloss: 0.162646
    [9972]	valid_0's auc: 0.904475	valid_0's binary_logloss: 0.162648
    [9973]	valid_0's auc: 0.904474	valid_0's binary_logloss: 0.16265
    [9974]	valid_0's auc: 0.904479	valid_0's binary_logloss: 0.162637
    [9975]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162642
    [9976]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162643
    [9977]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162645
    [9978]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162648
    [9979]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.16265
    [9980]	valid_0's auc: 0.904443	valid_0's binary_logloss: 0.162648
    [9981]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.162635
    [9982]	valid_0's auc: 0.904458	valid_0's binary_logloss: 0.162638
    [9983]	valid_0's auc: 0.904462	valid_0's binary_logloss: 0.162636
    [9984]	valid_0's auc: 0.904466	valid_0's binary_logloss: 0.162632
    [9985]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162634
    [9986]	valid_0's auc: 0.904464	valid_0's binary_logloss: 0.162636
    [9987]	valid_0's auc: 0.904468	valid_0's binary_logloss: 0.162633
    [9988]	valid_0's auc: 0.904473	valid_0's binary_logloss: 0.162628
    [9989]	valid_0's auc: 0.904476	valid_0's binary_logloss: 0.162622
    [9990]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162624
    [9991]	valid_0's auc: 0.904478	valid_0's binary_logloss: 0.162617
    [9992]	valid_0's auc: 0.904461	valid_0's binary_logloss: 0.162617
    [9993]	valid_0's auc: 0.90446	valid_0's binary_logloss: 0.162618
    [9994]	valid_0's auc: 0.904454	valid_0's binary_logloss: 0.162617
    [9995]	valid_0's auc: 0.904453	valid_0's binary_logloss: 0.162619
    [9996]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.16262
    [9997]	valid_0's auc: 0.904452	valid_0's binary_logloss: 0.162617
    [9998]	valid_0's auc: 0.904445	valid_0's binary_logloss: 0.162615
    [9999]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162617
    [10000]	valid_0's auc: 0.904446	valid_0's binary_logloss: 0.162619
    




    LGBMClassifier(boosting_type='dart', class_weight=None, colsample_bytree=0.3,
                   early_stopping_rounds=50, importance_type='split',
                   learning_rate=0.01, max_depth=6, min_child_samples=20,
                   min_child_weight=0.001, min_split_gain=0.0, n_estimators=10000,
                   n_jobs=-1, num_leaves=15, objective=None, random_state=None,
                   reg_alpha=0.0, reg_lambda=0.9, silent=True, subsample=0.9,
                   subsample_for_bin=200000, subsample_freq=0)



#### testing the model on the train set


```python
from sklearn import metrics
y_train_pred = gbm.predict(transformed_x_train)
# how did our model perform on the train set?
count_misclassified = (y_train != y_train_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_train_pred, y_train)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 1342
    Accuracy: 0.95
    

#### testing on the test set


```python
y_valid_pred = gbm.predict(transformed_x_valid)
# how did our model perform on the test set?
count_misclassified = (y_valid != y_valid_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_valid_pred, y_valid)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 249
    Accuracy: 0.94
    


```python
lbgo=y_valid_pred
```


```python

```




    0.9965200099428287



## checking out catboost


```python
from catboost import CatBoostClassifier, Pool
```


```python
test_pool = Pool(transformed_x_test)
```


```python

```


```python
eval_dataset = Pool(transformed_x_test, y_test)

catb_model = CatBoostClassifier(learning_rate=0.01,eval_metric='AUC',depth=6,bagging_temperature=0.15)
#904623 d=6
# catb_model.fit(transformed_x_train,
#           y_train,
#           eval_set=eval_dataset,
#           verbose=True)
```

#### test cat boost on the training set


```python
y_train_pred = catb_model.predict(transformed_x_train)
# how did our model perform on the train set?
count_misclassified = (y_train != y_train_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_train_pred, y_train)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 1382
    Accuracy: 0.94
    

#### test cat boost on the validation set


```python
y_valid_pred = catb_model.predict(transformed_x_valid)
# how did our model perform on the test set?
count_misclassified = (y_valid != y_valid_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_valid_pred, y_valid)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 253
    Accuracy: 0.94
    


```python
catgbo=y_valid_pred
```

#### On the verge of giving up due to fraustration and mental fatigue, in sheer desperation i have resorted to a randomForest model


```python
(catgbo==xgbo).mean()
```




    0.99602286850609




```python
(catgbo==lbgo).mean()
```




    0.9970171513795675




```python
(lbgo==xgbo).mean()
```




    0.9965200099428287




```python
from sklearn.ensemble import RandomForestClassifier
```


```python
forest=RandomForestClassifier(n_estimators=500,max_depth=4)
```


```python
forest.fit(balanced_x_train,balanced_y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=4, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=500,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



##### test forest model onn the training set


```python
y_train_pred = vc.predict(transformed_x_train)
# how did our model perform on the train set?
count_misclassified = (y_train != y_train_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_train_pred, y_train)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 1270
    Accuracy: 0.95
    

#### test random forest on the test set


```python
y_valid_pred = vc.predict(transformed_x_valid)
# how did our model perform on the test set?
count_misclassified = (y_valid != y_valid_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_valid_pred, y_valid)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 249
    Accuracy: 0.94
    


```python

```

### sheer desperation causes me to resort to underhanded tactics such as using a voting classifier....... pathetic


```python
from sklearn.ensemble import VotingClassifier
```


```python
vc = VotingClassifier(estimators =[('XGBOOST',clf), 
                                   ('LGBOOST', gbm), 
                                   ('CATBOOST',catb_model)], 
                      voting ='hard')
```


```python
vc.fit(balanced_x_train, balanced_y_train)
```

    C:\Users\Admin\Anaconda3\lib\site-packages\lightgbm\engine.py:123: UserWarning: Found `early_stopping_rounds` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    C:\Users\Admin\Anaconda3\lib\site-packages\lightgbm\callback.py:189: UserWarning: Early stopping is not available in dart mode
      warnings.warn('Early stopping is not available in dart mode')
    

    0:	total: 112ms	remaining: 1m 51s
    1:	total: 212ms	remaining: 1m 45s
    2:	total: 318ms	remaining: 1m 45s
    3:	total: 436ms	remaining: 1m 48s
    4:	total: 548ms	remaining: 1m 49s
    5:	total: 662ms	remaining: 1m 49s
    6:	total: 777ms	remaining: 1m 50s
    7:	total: 887ms	remaining: 1m 50s
    8:	total: 1s	remaining: 1m 50s
    9:	total: 1.12s	remaining: 1m 50s
    10:	total: 1.24s	remaining: 1m 51s
    11:	total: 1.35s	remaining: 1m 51s
    12:	total: 1.46s	remaining: 1m 51s
    13:	total: 1.57s	remaining: 1m 50s
    14:	total: 1.69s	remaining: 1m 50s
    15:	total: 1.8s	remaining: 1m 50s
    16:	total: 1.92s	remaining: 1m 51s
    17:	total: 2.03s	remaining: 1m 50s
    18:	total: 2.15s	remaining: 1m 50s
    19:	total: 2.26s	remaining: 1m 50s
    20:	total: 2.38s	remaining: 1m 50s
    21:	total: 2.49s	remaining: 1m 50s
    22:	total: 2.6s	remaining: 1m 50s
    23:	total: 2.71s	remaining: 1m 50s
    24:	total: 2.83s	remaining: 1m 50s
    25:	total: 2.95s	remaining: 1m 50s
    26:	total: 3.06s	remaining: 1m 50s
    27:	total: 3.17s	remaining: 1m 50s
    28:	total: 3.29s	remaining: 1m 50s
    29:	total: 3.4s	remaining: 1m 49s
    30:	total: 3.51s	remaining: 1m 49s
    31:	total: 3.62s	remaining: 1m 49s
    32:	total: 3.73s	remaining: 1m 49s
    33:	total: 3.85s	remaining: 1m 49s
    34:	total: 3.96s	remaining: 1m 49s
    35:	total: 4.07s	remaining: 1m 49s
    36:	total: 4.19s	remaining: 1m 49s
    37:	total: 4.3s	remaining: 1m 48s
    38:	total: 4.42s	remaining: 1m 48s
    39:	total: 4.53s	remaining: 1m 48s
    40:	total: 4.65s	remaining: 1m 48s
    41:	total: 4.76s	remaining: 1m 48s
    42:	total: 4.87s	remaining: 1m 48s
    43:	total: 4.98s	remaining: 1m 48s
    44:	total: 5.1s	remaining: 1m 48s
    45:	total: 5.21s	remaining: 1m 48s
    46:	total: 5.33s	remaining: 1m 47s
    47:	total: 5.44s	remaining: 1m 47s
    48:	total: 5.55s	remaining: 1m 47s
    49:	total: 5.66s	remaining: 1m 47s
    50:	total: 5.78s	remaining: 1m 47s
    51:	total: 5.89s	remaining: 1m 47s
    52:	total: 6s	remaining: 1m 47s
    53:	total: 6.12s	remaining: 1m 47s
    54:	total: 6.23s	remaining: 1m 47s
    55:	total: 6.35s	remaining: 1m 47s
    56:	total: 6.46s	remaining: 1m 46s
    57:	total: 6.57s	remaining: 1m 46s
    58:	total: 6.69s	remaining: 1m 46s
    59:	total: 6.8s	remaining: 1m 46s
    60:	total: 6.92s	remaining: 1m 46s
    61:	total: 7.03s	remaining: 1m 46s
    62:	total: 7.14s	remaining: 1m 46s
    63:	total: 7.27s	remaining: 1m 46s
    64:	total: 7.4s	remaining: 1m 46s
    65:	total: 7.53s	remaining: 1m 46s
    66:	total: 7.65s	remaining: 1m 46s
    67:	total: 7.77s	remaining: 1m 46s
    68:	total: 7.89s	remaining: 1m 46s
    69:	total: 8.01s	remaining: 1m 46s
    70:	total: 8.12s	remaining: 1m 46s
    71:	total: 8.23s	remaining: 1m 46s
    72:	total: 8.35s	remaining: 1m 46s
    73:	total: 8.46s	remaining: 1m 45s
    74:	total: 8.57s	remaining: 1m 45s
    75:	total: 8.69s	remaining: 1m 45s
    76:	total: 8.8s	remaining: 1m 45s
    77:	total: 8.91s	remaining: 1m 45s
    78:	total: 9.03s	remaining: 1m 45s
    79:	total: 9.14s	remaining: 1m 45s
    80:	total: 9.26s	remaining: 1m 45s
    81:	total: 9.37s	remaining: 1m 44s
    82:	total: 9.48s	remaining: 1m 44s
    83:	total: 9.6s	remaining: 1m 44s
    84:	total: 9.72s	remaining: 1m 44s
    85:	total: 9.83s	remaining: 1m 44s
    86:	total: 9.94s	remaining: 1m 44s
    87:	total: 10s	remaining: 1m 44s
    88:	total: 10.2s	remaining: 1m 44s
    89:	total: 10.3s	remaining: 1m 43s
    90:	total: 10.4s	remaining: 1m 43s
    91:	total: 10.5s	remaining: 1m 43s
    92:	total: 10.6s	remaining: 1m 43s
    93:	total: 10.7s	remaining: 1m 43s
    94:	total: 10.8s	remaining: 1m 43s
    95:	total: 10.9s	remaining: 1m 43s
    96:	total: 11.1s	remaining: 1m 42s
    97:	total: 11.2s	remaining: 1m 42s
    98:	total: 11.3s	remaining: 1m 42s
    99:	total: 11.4s	remaining: 1m 42s
    100:	total: 11.5s	remaining: 1m 42s
    101:	total: 11.6s	remaining: 1m 42s
    102:	total: 11.7s	remaining: 1m 42s
    103:	total: 11.9s	remaining: 1m 42s
    104:	total: 12s	remaining: 1m 42s
    105:	total: 12.1s	remaining: 1m 41s
    106:	total: 12.2s	remaining: 1m 41s
    107:	total: 12.3s	remaining: 1m 41s
    108:	total: 12.4s	remaining: 1m 41s
    109:	total: 12.5s	remaining: 1m 41s
    110:	total: 12.7s	remaining: 1m 41s
    111:	total: 12.8s	remaining: 1m 41s
    112:	total: 12.9s	remaining: 1m 41s
    113:	total: 13s	remaining: 1m 41s
    114:	total: 13.1s	remaining: 1m 40s
    115:	total: 13.2s	remaining: 1m 40s
    116:	total: 13.3s	remaining: 1m 40s
    117:	total: 13.5s	remaining: 1m 40s
    118:	total: 13.6s	remaining: 1m 40s
    119:	total: 13.7s	remaining: 1m 40s
    120:	total: 13.8s	remaining: 1m 40s
    121:	total: 13.9s	remaining: 1m 40s
    122:	total: 14s	remaining: 1m 39s
    123:	total: 14.1s	remaining: 1m 39s
    124:	total: 14.3s	remaining: 1m 39s
    125:	total: 14.4s	remaining: 1m 39s
    126:	total: 14.5s	remaining: 1m 39s
    127:	total: 14.6s	remaining: 1m 39s
    128:	total: 14.7s	remaining: 1m 39s
    129:	total: 14.8s	remaining: 1m 39s
    130:	total: 14.9s	remaining: 1m 39s
    131:	total: 15s	remaining: 1m 38s
    132:	total: 15.2s	remaining: 1m 38s
    133:	total: 15.3s	remaining: 1m 38s
    134:	total: 15.4s	remaining: 1m 38s
    135:	total: 15.5s	remaining: 1m 38s
    136:	total: 15.6s	remaining: 1m 38s
    137:	total: 15.7s	remaining: 1m 38s
    138:	total: 15.8s	remaining: 1m 38s
    139:	total: 15.9s	remaining: 1m 37s
    140:	total: 16.1s	remaining: 1m 37s
    141:	total: 16.2s	remaining: 1m 37s
    142:	total: 16.3s	remaining: 1m 37s
    143:	total: 16.4s	remaining: 1m 37s
    144:	total: 16.5s	remaining: 1m 37s
    145:	total: 16.6s	remaining: 1m 37s
    146:	total: 16.7s	remaining: 1m 37s
    147:	total: 16.8s	remaining: 1m 36s
    148:	total: 17s	remaining: 1m 36s
    149:	total: 17.1s	remaining: 1m 36s
    150:	total: 17.2s	remaining: 1m 36s
    151:	total: 17.3s	remaining: 1m 36s
    152:	total: 17.4s	remaining: 1m 36s
    153:	total: 17.5s	remaining: 1m 36s
    154:	total: 17.6s	remaining: 1m 36s
    155:	total: 17.8s	remaining: 1m 36s
    156:	total: 17.9s	remaining: 1m 35s
    157:	total: 18s	remaining: 1m 35s
    158:	total: 18.1s	remaining: 1m 35s
    159:	total: 18.2s	remaining: 1m 35s
    160:	total: 18.3s	remaining: 1m 35s
    161:	total: 18.4s	remaining: 1m 35s
    162:	total: 18.6s	remaining: 1m 35s
    163:	total: 18.7s	remaining: 1m 35s
    164:	total: 18.8s	remaining: 1m 35s
    165:	total: 18.9s	remaining: 1m 35s
    166:	total: 19s	remaining: 1m 34s
    167:	total: 19.1s	remaining: 1m 34s
    168:	total: 19.3s	remaining: 1m 34s
    169:	total: 19.4s	remaining: 1m 34s
    170:	total: 19.5s	remaining: 1m 34s
    171:	total: 19.6s	remaining: 1m 34s
    172:	total: 19.7s	remaining: 1m 34s
    173:	total: 19.9s	remaining: 1m 34s
    174:	total: 20s	remaining: 1m 34s
    175:	total: 20.1s	remaining: 1m 34s
    176:	total: 20.2s	remaining: 1m 33s
    177:	total: 20.3s	remaining: 1m 33s
    178:	total: 20.4s	remaining: 1m 33s
    179:	total: 20.6s	remaining: 1m 33s
    180:	total: 20.7s	remaining: 1m 33s
    181:	total: 20.8s	remaining: 1m 33s
    182:	total: 20.9s	remaining: 1m 33s
    183:	total: 21s	remaining: 1m 33s
    184:	total: 21.1s	remaining: 1m 33s
    185:	total: 21.2s	remaining: 1m 32s
    186:	total: 21.3s	remaining: 1m 32s
    187:	total: 21.5s	remaining: 1m 32s
    188:	total: 21.6s	remaining: 1m 32s
    189:	total: 21.7s	remaining: 1m 32s
    190:	total: 21.8s	remaining: 1m 32s
    191:	total: 21.9s	remaining: 1m 32s
    192:	total: 22s	remaining: 1m 32s
    193:	total: 22.1s	remaining: 1m 31s
    194:	total: 22.3s	remaining: 1m 31s
    195:	total: 22.4s	remaining: 1m 31s
    196:	total: 22.5s	remaining: 1m 31s
    197:	total: 22.6s	remaining: 1m 31s
    198:	total: 22.7s	remaining: 1m 31s
    199:	total: 22.8s	remaining: 1m 31s
    200:	total: 22.9s	remaining: 1m 31s
    201:	total: 23.1s	remaining: 1m 31s
    202:	total: 23.2s	remaining: 1m 31s
    203:	total: 23.3s	remaining: 1m 30s
    204:	total: 23.4s	remaining: 1m 30s
    205:	total: 23.5s	remaining: 1m 30s
    206:	total: 23.6s	remaining: 1m 30s
    207:	total: 23.7s	remaining: 1m 30s
    208:	total: 23.8s	remaining: 1m 30s
    209:	total: 24s	remaining: 1m 30s
    210:	total: 24.1s	remaining: 1m 30s
    211:	total: 24.2s	remaining: 1m 29s
    212:	total: 24.3s	remaining: 1m 29s
    213:	total: 24.4s	remaining: 1m 29s
    214:	total: 24.5s	remaining: 1m 29s
    215:	total: 24.6s	remaining: 1m 29s
    216:	total: 24.7s	remaining: 1m 29s
    217:	total: 24.9s	remaining: 1m 29s
    218:	total: 25s	remaining: 1m 29s
    219:	total: 25.1s	remaining: 1m 28s
    220:	total: 25.2s	remaining: 1m 28s
    221:	total: 25.3s	remaining: 1m 28s
    222:	total: 25.4s	remaining: 1m 28s
    223:	total: 25.5s	remaining: 1m 28s
    224:	total: 25.7s	remaining: 1m 28s
    225:	total: 25.8s	remaining: 1m 28s
    226:	total: 25.9s	remaining: 1m 28s
    227:	total: 26s	remaining: 1m 28s
    228:	total: 26.1s	remaining: 1m 27s
    229:	total: 26.2s	remaining: 1m 27s
    230:	total: 26.3s	remaining: 1m 27s
    231:	total: 26.4s	remaining: 1m 27s
    232:	total: 26.6s	remaining: 1m 27s
    233:	total: 26.7s	remaining: 1m 27s
    234:	total: 26.8s	remaining: 1m 27s
    235:	total: 26.9s	remaining: 1m 27s
    236:	total: 27s	remaining: 1m 26s
    237:	total: 27.1s	remaining: 1m 26s
    238:	total: 27.2s	remaining: 1m 26s
    239:	total: 27.4s	remaining: 1m 26s
    240:	total: 27.5s	remaining: 1m 26s
    241:	total: 27.6s	remaining: 1m 26s
    242:	total: 27.7s	remaining: 1m 26s
    243:	total: 27.8s	remaining: 1m 26s
    244:	total: 27.9s	remaining: 1m 25s
    245:	total: 28s	remaining: 1m 25s
    246:	total: 28.1s	remaining: 1m 25s
    247:	total: 28.2s	remaining: 1m 25s
    248:	total: 28.4s	remaining: 1m 25s
    249:	total: 28.5s	remaining: 1m 25s
    250:	total: 28.6s	remaining: 1m 25s
    251:	total: 28.7s	remaining: 1m 25s
    252:	total: 28.8s	remaining: 1m 25s
    253:	total: 28.9s	remaining: 1m 24s
    254:	total: 29s	remaining: 1m 24s
    255:	total: 29.1s	remaining: 1m 24s
    256:	total: 29.3s	remaining: 1m 24s
    257:	total: 29.4s	remaining: 1m 24s
    258:	total: 29.5s	remaining: 1m 24s
    259:	total: 29.6s	remaining: 1m 24s
    260:	total: 29.7s	remaining: 1m 24s
    261:	total: 29.8s	remaining: 1m 24s
    262:	total: 29.9s	remaining: 1m 23s
    263:	total: 30.1s	remaining: 1m 23s
    264:	total: 30.2s	remaining: 1m 23s
    265:	total: 30.3s	remaining: 1m 23s
    266:	total: 30.4s	remaining: 1m 23s
    267:	total: 30.5s	remaining: 1m 23s
    268:	total: 30.6s	remaining: 1m 23s
    269:	total: 30.7s	remaining: 1m 23s
    270:	total: 30.8s	remaining: 1m 22s
    271:	total: 31s	remaining: 1m 22s
    272:	total: 31.1s	remaining: 1m 22s
    273:	total: 31.2s	remaining: 1m 22s
    274:	total: 31.3s	remaining: 1m 22s
    275:	total: 31.5s	remaining: 1m 22s
    276:	total: 31.6s	remaining: 1m 22s
    277:	total: 31.7s	remaining: 1m 22s
    278:	total: 31.8s	remaining: 1m 22s
    279:	total: 32s	remaining: 1m 22s
    280:	total: 32.1s	remaining: 1m 22s
    281:	total: 32.2s	remaining: 1m 21s
    282:	total: 32.3s	remaining: 1m 21s
    283:	total: 32.4s	remaining: 1m 21s
    284:	total: 32.5s	remaining: 1m 21s
    285:	total: 32.7s	remaining: 1m 21s
    286:	total: 32.8s	remaining: 1m 21s
    287:	total: 32.9s	remaining: 1m 21s
    288:	total: 33s	remaining: 1m 21s
    289:	total: 33.1s	remaining: 1m 21s
    290:	total: 33.2s	remaining: 1m 20s
    291:	total: 33.3s	remaining: 1m 20s
    292:	total: 33.4s	remaining: 1m 20s
    293:	total: 33.6s	remaining: 1m 20s
    294:	total: 33.7s	remaining: 1m 20s
    295:	total: 33.8s	remaining: 1m 20s
    296:	total: 33.9s	remaining: 1m 20s
    297:	total: 34s	remaining: 1m 20s
    298:	total: 34.1s	remaining: 1m 19s
    299:	total: 34.2s	remaining: 1m 19s
    300:	total: 34.3s	remaining: 1m 19s
    301:	total: 34.4s	remaining: 1m 19s
    302:	total: 34.6s	remaining: 1m 19s
    303:	total: 34.7s	remaining: 1m 19s
    304:	total: 34.8s	remaining: 1m 19s
    305:	total: 34.9s	remaining: 1m 19s
    306:	total: 35s	remaining: 1m 19s
    307:	total: 35.1s	remaining: 1m 18s
    308:	total: 35.2s	remaining: 1m 18s
    309:	total: 35.3s	remaining: 1m 18s
    310:	total: 35.5s	remaining: 1m 18s
    311:	total: 35.6s	remaining: 1m 18s
    312:	total: 35.7s	remaining: 1m 18s
    313:	total: 35.8s	remaining: 1m 18s
    314:	total: 35.9s	remaining: 1m 18s
    315:	total: 36s	remaining: 1m 17s
    316:	total: 36.1s	remaining: 1m 17s
    317:	total: 36.3s	remaining: 1m 17s
    318:	total: 36.4s	remaining: 1m 17s
    319:	total: 36.5s	remaining: 1m 17s
    320:	total: 36.6s	remaining: 1m 17s
    321:	total: 36.7s	remaining: 1m 17s
    322:	total: 36.8s	remaining: 1m 17s
    323:	total: 36.9s	remaining: 1m 17s
    324:	total: 37s	remaining: 1m 16s
    325:	total: 37.2s	remaining: 1m 16s
    326:	total: 37.3s	remaining: 1m 16s
    327:	total: 37.4s	remaining: 1m 16s
    328:	total: 37.5s	remaining: 1m 16s
    329:	total: 37.6s	remaining: 1m 16s
    330:	total: 37.7s	remaining: 1m 16s
    331:	total: 37.8s	remaining: 1m 16s
    332:	total: 37.9s	remaining: 1m 15s
    333:	total: 38s	remaining: 1m 15s
    334:	total: 38.1s	remaining: 1m 15s
    335:	total: 38.3s	remaining: 1m 15s
    336:	total: 38.4s	remaining: 1m 15s
    337:	total: 38.5s	remaining: 1m 15s
    338:	total: 38.6s	remaining: 1m 15s
    339:	total: 38.7s	remaining: 1m 15s
    340:	total: 38.8s	remaining: 1m 15s
    341:	total: 38.9s	remaining: 1m 14s
    342:	total: 39.1s	remaining: 1m 14s
    343:	total: 39.2s	remaining: 1m 14s
    344:	total: 39.3s	remaining: 1m 14s
    345:	total: 39.4s	remaining: 1m 14s
    346:	total: 39.5s	remaining: 1m 14s
    347:	total: 39.6s	remaining: 1m 14s
    348:	total: 39.7s	remaining: 1m 14s
    349:	total: 39.8s	remaining: 1m 13s
    350:	total: 39.9s	remaining: 1m 13s
    351:	total: 40.1s	remaining: 1m 13s
    352:	total: 40.2s	remaining: 1m 13s
    353:	total: 40.3s	remaining: 1m 13s
    354:	total: 40.4s	remaining: 1m 13s
    355:	total: 40.5s	remaining: 1m 13s
    356:	total: 40.6s	remaining: 1m 13s
    357:	total: 40.7s	remaining: 1m 13s
    358:	total: 40.8s	remaining: 1m 12s
    359:	total: 41s	remaining: 1m 12s
    360:	total: 41.1s	remaining: 1m 12s
    361:	total: 41.2s	remaining: 1m 12s
    362:	total: 41.3s	remaining: 1m 12s
    363:	total: 41.4s	remaining: 1m 12s
    364:	total: 41.5s	remaining: 1m 12s
    365:	total: 41.6s	remaining: 1m 12s
    366:	total: 41.7s	remaining: 1m 11s
    367:	total: 41.9s	remaining: 1m 11s
    368:	total: 42s	remaining: 1m 11s
    369:	total: 42.1s	remaining: 1m 11s
    370:	total: 42.2s	remaining: 1m 11s
    371:	total: 42.3s	remaining: 1m 11s
    372:	total: 42.4s	remaining: 1m 11s
    373:	total: 42.5s	remaining: 1m 11s
    374:	total: 42.6s	remaining: 1m 11s
    375:	total: 42.7s	remaining: 1m 10s
    376:	total: 42.9s	remaining: 1m 10s
    377:	total: 43s	remaining: 1m 10s
    378:	total: 43.1s	remaining: 1m 10s
    379:	total: 43.2s	remaining: 1m 10s
    380:	total: 43.3s	remaining: 1m 10s
    381:	total: 43.4s	remaining: 1m 10s
    382:	total: 43.5s	remaining: 1m 10s
    383:	total: 43.6s	remaining: 1m 9s
    384:	total: 43.7s	remaining: 1m 9s
    385:	total: 43.9s	remaining: 1m 9s
    386:	total: 44s	remaining: 1m 9s
    387:	total: 44.1s	remaining: 1m 9s
    388:	total: 44.2s	remaining: 1m 9s
    389:	total: 44.3s	remaining: 1m 9s
    390:	total: 44.4s	remaining: 1m 9s
    391:	total: 44.5s	remaining: 1m 9s
    392:	total: 44.6s	remaining: 1m 8s
    393:	total: 44.8s	remaining: 1m 8s
    394:	total: 44.9s	remaining: 1m 8s
    395:	total: 45s	remaining: 1m 8s
    396:	total: 45.1s	remaining: 1m 8s
    397:	total: 45.2s	remaining: 1m 8s
    398:	total: 45.3s	remaining: 1m 8s
    399:	total: 45.4s	remaining: 1m 8s
    400:	total: 45.5s	remaining: 1m 8s
    401:	total: 45.7s	remaining: 1m 7s
    402:	total: 45.8s	remaining: 1m 7s
    403:	total: 45.9s	remaining: 1m 7s
    404:	total: 46s	remaining: 1m 7s
    405:	total: 46.1s	remaining: 1m 7s
    406:	total: 46.2s	remaining: 1m 7s
    407:	total: 46.3s	remaining: 1m 7s
    408:	total: 46.4s	remaining: 1m 7s
    409:	total: 46.6s	remaining: 1m 6s
    410:	total: 46.7s	remaining: 1m 6s
    411:	total: 46.8s	remaining: 1m 6s
    412:	total: 46.9s	remaining: 1m 6s
    413:	total: 47s	remaining: 1m 6s
    414:	total: 47.1s	remaining: 1m 6s
    415:	total: 47.2s	remaining: 1m 6s
    416:	total: 47.4s	remaining: 1m 6s
    417:	total: 47.5s	remaining: 1m 6s
    418:	total: 47.6s	remaining: 1m 5s
    419:	total: 47.7s	remaining: 1m 5s
    420:	total: 47.8s	remaining: 1m 5s
    421:	total: 47.9s	remaining: 1m 5s
    422:	total: 48s	remaining: 1m 5s
    423:	total: 48.2s	remaining: 1m 5s
    424:	total: 48.3s	remaining: 1m 5s
    425:	total: 48.4s	remaining: 1m 5s
    426:	total: 48.5s	remaining: 1m 5s
    427:	total: 48.6s	remaining: 1m 4s
    428:	total: 48.7s	remaining: 1m 4s
    429:	total: 48.8s	remaining: 1m 4s
    430:	total: 48.9s	remaining: 1m 4s
    431:	total: 49s	remaining: 1m 4s
    432:	total: 49.2s	remaining: 1m 4s
    433:	total: 49.3s	remaining: 1m 4s
    434:	total: 49.4s	remaining: 1m 4s
    435:	total: 49.5s	remaining: 1m 4s
    436:	total: 49.6s	remaining: 1m 3s
    437:	total: 49.7s	remaining: 1m 3s
    438:	total: 49.8s	remaining: 1m 3s
    439:	total: 49.9s	remaining: 1m 3s
    440:	total: 50s	remaining: 1m 3s
    441:	total: 50.2s	remaining: 1m 3s
    442:	total: 50.3s	remaining: 1m 3s
    443:	total: 50.4s	remaining: 1m 3s
    444:	total: 50.5s	remaining: 1m 2s
    445:	total: 50.6s	remaining: 1m 2s
    446:	total: 50.7s	remaining: 1m 2s
    447:	total: 50.8s	remaining: 1m 2s
    448:	total: 50.9s	remaining: 1m 2s
    449:	total: 51.1s	remaining: 1m 2s
    450:	total: 51.2s	remaining: 1m 2s
    451:	total: 51.3s	remaining: 1m 2s
    452:	total: 51.4s	remaining: 1m 2s
    453:	total: 51.5s	remaining: 1m 1s
    454:	total: 51.6s	remaining: 1m 1s
    455:	total: 51.7s	remaining: 1m 1s
    456:	total: 51.8s	remaining: 1m 1s
    457:	total: 51.9s	remaining: 1m 1s
    458:	total: 52.1s	remaining: 1m 1s
    459:	total: 52.2s	remaining: 1m 1s
    460:	total: 52.3s	remaining: 1m 1s
    461:	total: 52.4s	remaining: 1m 1s
    462:	total: 52.5s	remaining: 1m
    463:	total: 52.6s	remaining: 1m
    464:	total: 52.7s	remaining: 1m
    465:	total: 52.8s	remaining: 1m
    466:	total: 52.9s	remaining: 1m
    467:	total: 53.1s	remaining: 1m
    468:	total: 53.2s	remaining: 1m
    469:	total: 53.3s	remaining: 1m
    470:	total: 53.4s	remaining: 60s
    471:	total: 53.5s	remaining: 59.9s
    472:	total: 53.6s	remaining: 59.7s
    473:	total: 53.7s	remaining: 59.6s
    474:	total: 53.8s	remaining: 59.5s
    475:	total: 54s	remaining: 59.4s
    476:	total: 54.1s	remaining: 59.3s
    477:	total: 54.2s	remaining: 59.2s
    478:	total: 54.3s	remaining: 59.1s
    479:	total: 54.4s	remaining: 58.9s
    480:	total: 54.5s	remaining: 58.8s
    481:	total: 54.6s	remaining: 58.7s
    482:	total: 54.7s	remaining: 58.6s
    483:	total: 54.9s	remaining: 58.5s
    484:	total: 55s	remaining: 58.4s
    485:	total: 55.1s	remaining: 58.2s
    486:	total: 55.2s	remaining: 58.1s
    487:	total: 55.3s	remaining: 58s
    488:	total: 55.4s	remaining: 57.9s
    489:	total: 55.5s	remaining: 57.8s
    490:	total: 55.6s	remaining: 57.7s
    491:	total: 55.7s	remaining: 57.6s
    492:	total: 55.9s	remaining: 57.4s
    493:	total: 56s	remaining: 57.3s
    494:	total: 56.1s	remaining: 57.2s
    495:	total: 56.2s	remaining: 57.1s
    496:	total: 56.3s	remaining: 57s
    497:	total: 56.4s	remaining: 56.9s
    498:	total: 56.5s	remaining: 56.8s
    499:	total: 56.6s	remaining: 56.6s
    500:	total: 56.8s	remaining: 56.5s
    501:	total: 56.9s	remaining: 56.4s
    502:	total: 57s	remaining: 56.3s
    503:	total: 57.1s	remaining: 56.2s
    504:	total: 57.2s	remaining: 56.1s
    505:	total: 57.3s	remaining: 56s
    506:	total: 57.4s	remaining: 55.8s
    507:	total: 57.5s	remaining: 55.7s
    508:	total: 57.7s	remaining: 55.6s
    509:	total: 57.8s	remaining: 55.5s
    510:	total: 57.9s	remaining: 55.4s
    511:	total: 58s	remaining: 55.3s
    512:	total: 58.1s	remaining: 55.2s
    513:	total: 58.2s	remaining: 55s
    514:	total: 58.3s	remaining: 54.9s
    515:	total: 58.4s	remaining: 54.8s
    516:	total: 58.6s	remaining: 54.7s
    517:	total: 58.7s	remaining: 54.6s
    518:	total: 58.8s	remaining: 54.5s
    519:	total: 58.9s	remaining: 54.4s
    520:	total: 59s	remaining: 54.3s
    521:	total: 59.1s	remaining: 54.1s
    522:	total: 59.2s	remaining: 54s
    523:	total: 59.4s	remaining: 53.9s
    524:	total: 59.5s	remaining: 53.8s
    525:	total: 59.6s	remaining: 53.7s
    526:	total: 59.7s	remaining: 53.6s
    527:	total: 59.8s	remaining: 53.5s
    528:	total: 59.9s	remaining: 53.4s
    529:	total: 1m	remaining: 53.3s
    530:	total: 1m	remaining: 53.1s
    531:	total: 1m	remaining: 53s
    532:	total: 1m	remaining: 52.9s
    533:	total: 1m	remaining: 52.8s
    534:	total: 1m	remaining: 52.7s
    535:	total: 1m	remaining: 52.6s
    536:	total: 1m	remaining: 52.4s
    537:	total: 1m	remaining: 52.3s
    538:	total: 1m 1s	remaining: 52.2s
    539:	total: 1m 1s	remaining: 52.1s
    540:	total: 1m 1s	remaining: 52s
    541:	total: 1m 1s	remaining: 51.9s
    542:	total: 1m 1s	remaining: 51.8s
    543:	total: 1m 1s	remaining: 51.7s
    544:	total: 1m 1s	remaining: 51.5s
    545:	total: 1m 1s	remaining: 51.4s
    546:	total: 1m 1s	remaining: 51.3s
    547:	total: 1m 2s	remaining: 51.2s
    548:	total: 1m 2s	remaining: 51.1s
    549:	total: 1m 2s	remaining: 51s
    550:	total: 1m 2s	remaining: 50.9s
    551:	total: 1m 2s	remaining: 50.8s
    552:	total: 1m 2s	remaining: 50.7s
    553:	total: 1m 2s	remaining: 50.5s
    554:	total: 1m 2s	remaining: 50.4s
    555:	total: 1m 3s	remaining: 50.3s
    556:	total: 1m 3s	remaining: 50.2s
    557:	total: 1m 3s	remaining: 50.1s
    558:	total: 1m 3s	remaining: 50s
    559:	total: 1m 3s	remaining: 49.9s
    560:	total: 1m 3s	remaining: 49.8s
    561:	total: 1m 3s	remaining: 49.7s
    562:	total: 1m 3s	remaining: 49.5s
    563:	total: 1m 3s	remaining: 49.4s
    564:	total: 1m 4s	remaining: 49.3s
    565:	total: 1m 4s	remaining: 49.2s
    566:	total: 1m 4s	remaining: 49.1s
    567:	total: 1m 4s	remaining: 49s
    568:	total: 1m 4s	remaining: 48.9s
    569:	total: 1m 4s	remaining: 48.7s
    570:	total: 1m 4s	remaining: 48.6s
    571:	total: 1m 4s	remaining: 48.5s
    572:	total: 1m 4s	remaining: 48.4s
    573:	total: 1m 5s	remaining: 48.3s
    574:	total: 1m 5s	remaining: 48.2s
    575:	total: 1m 5s	remaining: 48.1s
    576:	total: 1m 5s	remaining: 48s
    577:	total: 1m 5s	remaining: 47.8s
    578:	total: 1m 5s	remaining: 47.7s
    579:	total: 1m 5s	remaining: 47.6s
    580:	total: 1m 5s	remaining: 47.5s
    581:	total: 1m 5s	remaining: 47.4s
    582:	total: 1m 6s	remaining: 47.3s
    583:	total: 1m 6s	remaining: 47.2s
    584:	total: 1m 6s	remaining: 47.1s
    585:	total: 1m 6s	remaining: 47s
    586:	total: 1m 6s	remaining: 46.9s
    587:	total: 1m 6s	remaining: 46.8s
    588:	total: 1m 6s	remaining: 46.7s
    589:	total: 1m 6s	remaining: 46.6s
    590:	total: 1m 7s	remaining: 46.4s
    591:	total: 1m 7s	remaining: 46.3s
    592:	total: 1m 7s	remaining: 46.2s
    593:	total: 1m 7s	remaining: 46.1s
    594:	total: 1m 7s	remaining: 46s
    595:	total: 1m 7s	remaining: 45.9s
    596:	total: 1m 7s	remaining: 45.8s
    597:	total: 1m 7s	remaining: 45.7s
    598:	total: 1m 8s	remaining: 45.6s
    599:	total: 1m 8s	remaining: 45.4s
    600:	total: 1m 8s	remaining: 45.3s
    601:	total: 1m 8s	remaining: 45.2s
    602:	total: 1m 8s	remaining: 45.1s
    603:	total: 1m 8s	remaining: 45s
    604:	total: 1m 8s	remaining: 44.9s
    605:	total: 1m 8s	remaining: 44.8s
    606:	total: 1m 8s	remaining: 44.6s
    607:	total: 1m 9s	remaining: 44.5s
    608:	total: 1m 9s	remaining: 44.4s
    609:	total: 1m 9s	remaining: 44.3s
    610:	total: 1m 9s	remaining: 44.2s
    611:	total: 1m 9s	remaining: 44.1s
    612:	total: 1m 9s	remaining: 44s
    613:	total: 1m 9s	remaining: 43.8s
    614:	total: 1m 9s	remaining: 43.7s
    615:	total: 1m 9s	remaining: 43.6s
    616:	total: 1m 10s	remaining: 43.5s
    617:	total: 1m 10s	remaining: 43.4s
    618:	total: 1m 10s	remaining: 43.3s
    619:	total: 1m 10s	remaining: 43.2s
    620:	total: 1m 10s	remaining: 43s
    621:	total: 1m 10s	remaining: 42.9s
    622:	total: 1m 10s	remaining: 42.8s
    623:	total: 1m 10s	remaining: 42.7s
    624:	total: 1m 11s	remaining: 42.6s
    625:	total: 1m 11s	remaining: 42.5s
    626:	total: 1m 11s	remaining: 42.4s
    627:	total: 1m 11s	remaining: 42.3s
    628:	total: 1m 11s	remaining: 42.1s
    629:	total: 1m 11s	remaining: 42s
    630:	total: 1m 11s	remaining: 41.9s
    631:	total: 1m 11s	remaining: 41.8s
    632:	total: 1m 11s	remaining: 41.7s
    633:	total: 1m 12s	remaining: 41.6s
    634:	total: 1m 12s	remaining: 41.4s
    635:	total: 1m 12s	remaining: 41.3s
    636:	total: 1m 12s	remaining: 41.2s
    637:	total: 1m 12s	remaining: 41.1s
    638:	total: 1m 12s	remaining: 41s
    639:	total: 1m 12s	remaining: 40.9s
    640:	total: 1m 12s	remaining: 40.8s
    641:	total: 1m 13s	remaining: 40.7s
    642:	total: 1m 13s	remaining: 40.6s
    643:	total: 1m 13s	remaining: 40.5s
    644:	total: 1m 13s	remaining: 40.4s
    645:	total: 1m 13s	remaining: 40.3s
    646:	total: 1m 13s	remaining: 40.2s
    647:	total: 1m 13s	remaining: 40.1s
    648:	total: 1m 13s	remaining: 40s
    649:	total: 1m 14s	remaining: 39.9s
    650:	total: 1m 14s	remaining: 39.8s
    651:	total: 1m 14s	remaining: 39.6s
    652:	total: 1m 14s	remaining: 39.5s
    653:	total: 1m 14s	remaining: 39.4s
    654:	total: 1m 14s	remaining: 39.3s
    655:	total: 1m 14s	remaining: 39.2s
    656:	total: 1m 14s	remaining: 39.1s
    657:	total: 1m 15s	remaining: 39s
    658:	total: 1m 15s	remaining: 38.9s
    659:	total: 1m 15s	remaining: 38.8s
    660:	total: 1m 15s	remaining: 38.7s
    661:	total: 1m 15s	remaining: 38.6s
    662:	total: 1m 15s	remaining: 38.5s
    663:	total: 1m 15s	remaining: 38.4s
    664:	total: 1m 15s	remaining: 38.3s
    665:	total: 1m 16s	remaining: 38.2s
    666:	total: 1m 16s	remaining: 38s
    667:	total: 1m 16s	remaining: 37.9s
    668:	total: 1m 16s	remaining: 37.8s
    669:	total: 1m 16s	remaining: 37.7s
    670:	total: 1m 16s	remaining: 37.6s
    671:	total: 1m 16s	remaining: 37.5s
    672:	total: 1m 16s	remaining: 37.4s
    673:	total: 1m 17s	remaining: 37.3s
    674:	total: 1m 17s	remaining: 37.1s
    675:	total: 1m 17s	remaining: 37s
    676:	total: 1m 17s	remaining: 36.9s
    677:	total: 1m 17s	remaining: 36.8s
    678:	total: 1m 17s	remaining: 36.7s
    679:	total: 1m 17s	remaining: 36.6s
    680:	total: 1m 17s	remaining: 36.5s
    681:	total: 1m 17s	remaining: 36.3s
    682:	total: 1m 18s	remaining: 36.2s
    683:	total: 1m 18s	remaining: 36.1s
    684:	total: 1m 18s	remaining: 36s
    685:	total: 1m 18s	remaining: 35.9s
    686:	total: 1m 18s	remaining: 35.8s
    687:	total: 1m 18s	remaining: 35.7s
    688:	total: 1m 18s	remaining: 35.6s
    689:	total: 1m 19s	remaining: 35.5s
    690:	total: 1m 19s	remaining: 35.4s
    691:	total: 1m 19s	remaining: 35.3s
    692:	total: 1m 19s	remaining: 35.2s
    693:	total: 1m 19s	remaining: 35.1s
    694:	total: 1m 19s	remaining: 35s
    695:	total: 1m 19s	remaining: 34.9s
    696:	total: 1m 19s	remaining: 34.8s
    697:	total: 1m 20s	remaining: 34.6s
    698:	total: 1m 20s	remaining: 34.5s
    699:	total: 1m 20s	remaining: 34.4s
    700:	total: 1m 20s	remaining: 34.3s
    701:	total: 1m 20s	remaining: 34.2s
    702:	total: 1m 20s	remaining: 34.1s
    703:	total: 1m 20s	remaining: 34s
    704:	total: 1m 20s	remaining: 33.9s
    705:	total: 1m 21s	remaining: 33.8s
    706:	total: 1m 21s	remaining: 33.6s
    707:	total: 1m 21s	remaining: 33.5s
    708:	total: 1m 21s	remaining: 33.4s
    709:	total: 1m 21s	remaining: 33.3s
    710:	total: 1m 21s	remaining: 33.2s
    711:	total: 1m 21s	remaining: 33.1s
    712:	total: 1m 21s	remaining: 33s
    713:	total: 1m 22s	remaining: 32.9s
    714:	total: 1m 22s	remaining: 32.7s
    715:	total: 1m 22s	remaining: 32.6s
    716:	total: 1m 22s	remaining: 32.5s
    717:	total: 1m 22s	remaining: 32.4s
    718:	total: 1m 22s	remaining: 32.3s
    719:	total: 1m 22s	remaining: 32.2s
    720:	total: 1m 22s	remaining: 32.1s
    721:	total: 1m 22s	remaining: 31.9s
    722:	total: 1m 23s	remaining: 31.8s
    723:	total: 1m 23s	remaining: 31.7s
    724:	total: 1m 23s	remaining: 31.6s
    725:	total: 1m 23s	remaining: 31.5s
    726:	total: 1m 23s	remaining: 31.4s
    727:	total: 1m 23s	remaining: 31.3s
    728:	total: 1m 23s	remaining: 31.2s
    729:	total: 1m 23s	remaining: 31.1s
    730:	total: 1m 24s	remaining: 31s
    731:	total: 1m 24s	remaining: 30.8s
    732:	total: 1m 24s	remaining: 30.7s
    733:	total: 1m 24s	remaining: 30.6s
    734:	total: 1m 24s	remaining: 30.5s
    735:	total: 1m 24s	remaining: 30.4s
    736:	total: 1m 24s	remaining: 30.3s
    737:	total: 1m 24s	remaining: 30.2s
    738:	total: 1m 25s	remaining: 30.1s
    739:	total: 1m 25s	remaining: 29.9s
    740:	total: 1m 25s	remaining: 29.8s
    741:	total: 1m 25s	remaining: 29.7s
    742:	total: 1m 25s	remaining: 29.6s
    743:	total: 1m 25s	remaining: 29.5s
    744:	total: 1m 25s	remaining: 29.4s
    745:	total: 1m 26s	remaining: 29.3s
    746:	total: 1m 26s	remaining: 29.2s
    747:	total: 1m 26s	remaining: 29.1s
    748:	total: 1m 26s	remaining: 28.9s
    749:	total: 1m 26s	remaining: 28.8s
    750:	total: 1m 26s	remaining: 28.7s
    751:	total: 1m 26s	remaining: 28.6s
    752:	total: 1m 26s	remaining: 28.5s
    753:	total: 1m 26s	remaining: 28.4s
    754:	total: 1m 27s	remaining: 28.2s
    755:	total: 1m 27s	remaining: 28.1s
    756:	total: 1m 27s	remaining: 28s
    757:	total: 1m 27s	remaining: 27.9s
    758:	total: 1m 27s	remaining: 27.8s
    759:	total: 1m 27s	remaining: 27.7s
    760:	total: 1m 27s	remaining: 27.6s
    761:	total: 1m 27s	remaining: 27.4s
    762:	total: 1m 28s	remaining: 27.3s
    763:	total: 1m 28s	remaining: 27.2s
    764:	total: 1m 28s	remaining: 27.1s
    765:	total: 1m 28s	remaining: 27s
    766:	total: 1m 28s	remaining: 26.9s
    767:	total: 1m 28s	remaining: 26.8s
    768:	total: 1m 28s	remaining: 26.6s
    769:	total: 1m 28s	remaining: 26.5s
    770:	total: 1m 28s	remaining: 26.4s
    771:	total: 1m 29s	remaining: 26.3s
    772:	total: 1m 29s	remaining: 26.2s
    773:	total: 1m 29s	remaining: 26.1s
    774:	total: 1m 29s	remaining: 26s
    775:	total: 1m 29s	remaining: 25.8s
    776:	total: 1m 29s	remaining: 25.7s
    777:	total: 1m 29s	remaining: 25.6s
    778:	total: 1m 29s	remaining: 25.5s
    779:	total: 1m 29s	remaining: 25.4s
    780:	total: 1m 30s	remaining: 25.3s
    781:	total: 1m 30s	remaining: 25.1s
    782:	total: 1m 30s	remaining: 25s
    783:	total: 1m 30s	remaining: 24.9s
    784:	total: 1m 30s	remaining: 24.8s
    785:	total: 1m 30s	remaining: 24.7s
    786:	total: 1m 30s	remaining: 24.6s
    787:	total: 1m 30s	remaining: 24.5s
    788:	total: 1m 31s	remaining: 24.4s
    789:	total: 1m 31s	remaining: 24.3s
    790:	total: 1m 31s	remaining: 24.2s
    791:	total: 1m 31s	remaining: 24.1s
    792:	total: 1m 31s	remaining: 24s
    793:	total: 1m 31s	remaining: 23.8s
    794:	total: 1m 32s	remaining: 23.7s
    795:	total: 1m 32s	remaining: 23.6s
    796:	total: 1m 32s	remaining: 23.5s
    797:	total: 1m 32s	remaining: 23.4s
    798:	total: 1m 32s	remaining: 23.3s
    799:	total: 1m 32s	remaining: 23.2s
    800:	total: 1m 33s	remaining: 23.1s
    801:	total: 1m 33s	remaining: 23s
    802:	total: 1m 33s	remaining: 22.9s
    803:	total: 1m 33s	remaining: 22.8s
    804:	total: 1m 33s	remaining: 22.7s
    805:	total: 1m 33s	remaining: 22.6s
    806:	total: 1m 33s	remaining: 22.5s
    807:	total: 1m 34s	remaining: 22.4s
    808:	total: 1m 34s	remaining: 22.2s
    809:	total: 1m 34s	remaining: 22.1s
    810:	total: 1m 34s	remaining: 22s
    811:	total: 1m 34s	remaining: 21.9s
    812:	total: 1m 34s	remaining: 21.8s
    813:	total: 1m 34s	remaining: 21.7s
    814:	total: 1m 35s	remaining: 21.6s
    815:	total: 1m 35s	remaining: 21.5s
    816:	total: 1m 35s	remaining: 21.4s
    817:	total: 1m 35s	remaining: 21.2s
    818:	total: 1m 35s	remaining: 21.1s
    819:	total: 1m 35s	remaining: 21s
    820:	total: 1m 35s	remaining: 20.9s
    821:	total: 1m 36s	remaining: 20.8s
    822:	total: 1m 36s	remaining: 20.7s
    823:	total: 1m 36s	remaining: 20.6s
    824:	total: 1m 36s	remaining: 20.5s
    825:	total: 1m 36s	remaining: 20.4s
    826:	total: 1m 36s	remaining: 20.3s
    827:	total: 1m 37s	remaining: 20.2s
    828:	total: 1m 37s	remaining: 20.1s
    829:	total: 1m 37s	remaining: 20s
    830:	total: 1m 37s	remaining: 19.9s
    831:	total: 1m 37s	remaining: 19.8s
    832:	total: 1m 37s	remaining: 19.6s
    833:	total: 1m 38s	remaining: 19.5s
    834:	total: 1m 38s	remaining: 19.4s
    835:	total: 1m 38s	remaining: 19.3s
    836:	total: 1m 38s	remaining: 19.2s
    837:	total: 1m 38s	remaining: 19.1s
    838:	total: 1m 38s	remaining: 19s
    839:	total: 1m 38s	remaining: 18.9s
    840:	total: 1m 39s	remaining: 18.7s
    841:	total: 1m 39s	remaining: 18.6s
    842:	total: 1m 39s	remaining: 18.5s
    843:	total: 1m 39s	remaining: 18.4s
    844:	total: 1m 39s	remaining: 18.3s
    845:	total: 1m 40s	remaining: 18.2s
    846:	total: 1m 40s	remaining: 18.1s
    847:	total: 1m 40s	remaining: 18s
    848:	total: 1m 40s	remaining: 17.9s
    849:	total: 1m 40s	remaining: 17.8s
    850:	total: 1m 41s	remaining: 17.7s
    851:	total: 1m 41s	remaining: 17.6s
    852:	total: 1m 41s	remaining: 17.5s
    853:	total: 1m 41s	remaining: 17.4s
    854:	total: 1m 42s	remaining: 17.3s
    855:	total: 1m 42s	remaining: 17.2s
    856:	total: 1m 42s	remaining: 17.1s
    857:	total: 1m 42s	remaining: 17s
    858:	total: 1m 42s	remaining: 16.9s
    859:	total: 1m 43s	remaining: 16.8s
    860:	total: 1m 43s	remaining: 16.7s
    861:	total: 1m 43s	remaining: 16.6s
    862:	total: 1m 43s	remaining: 16.4s
    863:	total: 1m 43s	remaining: 16.3s
    864:	total: 1m 43s	remaining: 16.2s
    865:	total: 1m 44s	remaining: 16.1s
    866:	total: 1m 44s	remaining: 16s
    867:	total: 1m 44s	remaining: 15.9s
    868:	total: 1m 44s	remaining: 15.8s
    869:	total: 1m 44s	remaining: 15.6s
    870:	total: 1m 44s	remaining: 15.5s
    871:	total: 1m 44s	remaining: 15.4s
    872:	total: 1m 45s	remaining: 15.3s
    873:	total: 1m 45s	remaining: 15.2s
    874:	total: 1m 45s	remaining: 15.1s
    875:	total: 1m 45s	remaining: 14.9s
    876:	total: 1m 45s	remaining: 14.8s
    877:	total: 1m 45s	remaining: 14.7s
    878:	total: 1m 45s	remaining: 14.6s
    879:	total: 1m 46s	remaining: 14.5s
    880:	total: 1m 46s	remaining: 14.4s
    881:	total: 1m 46s	remaining: 14.2s
    882:	total: 1m 46s	remaining: 14.1s
    883:	total: 1m 46s	remaining: 14s
    884:	total: 1m 46s	remaining: 13.9s
    885:	total: 1m 47s	remaining: 13.8s
    886:	total: 1m 47s	remaining: 13.7s
    887:	total: 1m 47s	remaining: 13.5s
    888:	total: 1m 47s	remaining: 13.4s
    889:	total: 1m 47s	remaining: 13.3s
    890:	total: 1m 47s	remaining: 13.2s
    891:	total: 1m 47s	remaining: 13.1s
    892:	total: 1m 48s	remaining: 13s
    893:	total: 1m 48s	remaining: 12.8s
    894:	total: 1m 48s	remaining: 12.7s
    895:	total: 1m 48s	remaining: 12.6s
    896:	total: 1m 48s	remaining: 12.5s
    897:	total: 1m 48s	remaining: 12.4s
    898:	total: 1m 48s	remaining: 12.2s
    899:	total: 1m 49s	remaining: 12.1s
    900:	total: 1m 49s	remaining: 12s
    901:	total: 1m 49s	remaining: 11.9s
    902:	total: 1m 49s	remaining: 11.8s
    903:	total: 1m 49s	remaining: 11.6s
    904:	total: 1m 49s	remaining: 11.5s
    905:	total: 1m 49s	remaining: 11.4s
    906:	total: 1m 50s	remaining: 11.3s
    907:	total: 1m 50s	remaining: 11.2s
    908:	total: 1m 50s	remaining: 11s
    909:	total: 1m 50s	remaining: 10.9s
    910:	total: 1m 50s	remaining: 10.8s
    911:	total: 1m 50s	remaining: 10.7s
    912:	total: 1m 50s	remaining: 10.6s
    913:	total: 1m 51s	remaining: 10.4s
    914:	total: 1m 51s	remaining: 10.3s
    915:	total: 1m 51s	remaining: 10.2s
    916:	total: 1m 51s	remaining: 10.1s
    917:	total: 1m 51s	remaining: 9.97s
    918:	total: 1m 51s	remaining: 9.85s
    919:	total: 1m 51s	remaining: 9.73s
    920:	total: 1m 52s	remaining: 9.61s
    921:	total: 1m 52s	remaining: 9.49s
    922:	total: 1m 52s	remaining: 9.37s
    923:	total: 1m 52s	remaining: 9.25s
    924:	total: 1m 52s	remaining: 9.13s
    925:	total: 1m 52s	remaining: 9.01s
    926:	total: 1m 52s	remaining: 8.89s
    927:	total: 1m 52s	remaining: 8.77s
    928:	total: 1m 53s	remaining: 8.64s
    929:	total: 1m 53s	remaining: 8.52s
    930:	total: 1m 53s	remaining: 8.4s
    931:	total: 1m 53s	remaining: 8.28s
    932:	total: 1m 53s	remaining: 8.16s
    933:	total: 1m 53s	remaining: 8.04s
    934:	total: 1m 53s	remaining: 7.92s
    935:	total: 1m 53s	remaining: 7.79s
    936:	total: 1m 54s	remaining: 7.68s
    937:	total: 1m 54s	remaining: 7.56s
    938:	total: 1m 54s	remaining: 7.44s
    939:	total: 1m 54s	remaining: 7.32s
    940:	total: 1m 54s	remaining: 7.2s
    941:	total: 1m 54s	remaining: 7.08s
    942:	total: 1m 55s	remaining: 6.96s
    943:	total: 1m 55s	remaining: 6.84s
    944:	total: 1m 55s	remaining: 6.71s
    945:	total: 1m 55s	remaining: 6.6s
    946:	total: 1m 55s	remaining: 6.47s
    947:	total: 1m 55s	remaining: 6.35s
    948:	total: 1m 56s	remaining: 6.24s
    949:	total: 1m 56s	remaining: 6.11s
    950:	total: 1m 56s	remaining: 5.99s
    951:	total: 1m 56s	remaining: 5.87s
    952:	total: 1m 56s	remaining: 5.75s
    953:	total: 1m 56s	remaining: 5.63s
    954:	total: 1m 56s	remaining: 5.51s
    955:	total: 1m 56s	remaining: 5.38s
    956:	total: 1m 57s	remaining: 5.26s
    957:	total: 1m 57s	remaining: 5.14s
    958:	total: 1m 57s	remaining: 5.02s
    959:	total: 1m 57s	remaining: 4.9s
    960:	total: 1m 57s	remaining: 4.78s
    961:	total: 1m 57s	remaining: 4.66s
    962:	total: 1m 58s	remaining: 4.54s
    963:	total: 1m 58s	remaining: 4.42s
    964:	total: 1m 58s	remaining: 4.29s
    965:	total: 1m 58s	remaining: 4.17s
    966:	total: 1m 58s	remaining: 4.05s
    967:	total: 1m 58s	remaining: 3.93s
    968:	total: 1m 58s	remaining: 3.81s
    969:	total: 1m 59s	remaining: 3.68s
    970:	total: 1m 59s	remaining: 3.56s
    971:	total: 1m 59s	remaining: 3.44s
    972:	total: 1m 59s	remaining: 3.32s
    973:	total: 1m 59s	remaining: 3.19s
    974:	total: 1m 59s	remaining: 3.07s
    975:	total: 2m	remaining: 2.95s
    976:	total: 2m	remaining: 2.83s
    977:	total: 2m	remaining: 2.71s
    978:	total: 2m	remaining: 2.58s
    979:	total: 2m	remaining: 2.46s
    980:	total: 2m	remaining: 2.34s
    981:	total: 2m	remaining: 2.21s
    982:	total: 2m 1s	remaining: 2.09s
    983:	total: 2m 1s	remaining: 1.97s
    984:	total: 2m 1s	remaining: 1.85s
    985:	total: 2m 1s	remaining: 1.73s
    986:	total: 2m 1s	remaining: 1.6s
    987:	total: 2m 1s	remaining: 1.48s
    988:	total: 2m 2s	remaining: 1.36s
    989:	total: 2m 2s	remaining: 1.23s
    990:	total: 2m 2s	remaining: 1.11s
    991:	total: 2m 2s	remaining: 988ms
    992:	total: 2m 2s	remaining: 865ms
    993:	total: 2m 2s	remaining: 741ms
    994:	total: 2m 2s	remaining: 618ms
    995:	total: 2m 3s	remaining: 494ms
    996:	total: 2m 3s	remaining: 371ms
    997:	total: 2m 3s	remaining: 247ms
    998:	total: 2m 3s	remaining: 124ms
    999:	total: 2m 3s	remaining: 0us
    




    VotingClassifier(estimators=[('XGBOOST',
                                  XGBClassifier(base_score=0.7, booster='dart',
                                                colsample_bylevel=1,
                                                colsample_bynode=1,
                                                colsample_bytree=1, gamma=0,
                                                learning_rate=0.01,
                                                max_delta_step=0, max_depth=8,
                                                min_child_weight=1, missing=None,
                                                n_estimators=3000, n_jobs=1,
                                                nthread=None,
                                                objective='binary:logistic',
                                                random_state=0, reg_alpha=0,
                                                reg_lambda=0.03, scale_pos...
                                                 min_child_weight=0.001,
                                                 min_split_gain=0.0,
                                                 n_estimators=10000, n_jobs=-1,
                                                 num_leaves=15, objective=None,
                                                 random_state=None, reg_alpha=0.0,
                                                 reg_lambda=0.9, silent=True,
                                                 subsample=0.9,
                                                 subsample_for_bin=200000,
                                                 subsample_freq=0)),
                                 ('CATBOOST',
                                  <catboost.core.CatBoostClassifier object at 0x000002776B61CCC0>)],
                     flatten_transform=True, n_jobs=None, voting='hard',
                     weights=None)




```python
vc.score(transformed_x_valid, y_valid)
```




    0.9381058911260254



### saving my output to a file


```python
def save_predictions(model,filename):
    test=pd.read_csv("test.csv")
    transformed_test=transformer.transform(test)
    model_predictions=model.predict(transformed_test)
    df=pd.DataFrame()
    df['EmployeeNo']=test['EmployeeNo']
    df['Promoted_or_Not']=model_predictions
    df['Promoted_or_Not']=df['Promoted_or_Not'].map(lambda x :int(x))
    df.to_csv(filename,index=False)
    return "saved"

save_predictions(vc,"voting_classifier.csv")
```




    'saved'




```python
test=pd.read_csv("test.csv")
transformed_test=transformer.transform(test)
clf_prediction=vc.predict(transformed_test)
```


```python
merged=pd.read_csv("hydra.csv")
merged_values=merged['Promoted_or_Not'].values
```


```python
sums=clf_prediction+merged_values
merged['Promoted_or_Not']=sums
```


```python
merged['Promoted_or_Not']=merged['Promoted_or_Not'].map(lambda x: x if x==0 else 1)
```


```python
pd.value_counts(sums)
```




    0    15882
    2      518
    1       96
    dtype: int64




```python
merged.to_csv("hydra_test.csv",index=False)
```


```python
merged["Promoted_or_Not"].value_counts()
```




    0    15882
    1      614
    Name: Promoted_or_Not, dtype: int64




```python
merged['Promoted_or_Not']=merged['Promoted_or_Not'].map(lambda x: x if x==0 else 1)
```


```python

```
