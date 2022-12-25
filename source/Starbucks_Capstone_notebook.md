
# Starbucks Capstone Challenge

### Introduction

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

### Example

To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.

However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

### Cleaning

This makes data cleaning especially important and tricky.

You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.

### Final Advice

Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

**Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  

You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:

<img src="pic1.png"/>

Then you will want to run the above command:

<img src="pic2.png"/>

Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# Install and Upgrade packages


```python
!pip install --upgrade pandas  # conda did not work. pip upgrade needs to be executed each time after restarting the workspace. A kernel restart is necessary after the pip upgrade.
#!pip install -U matplotlib
!pip install --upgrade seaborn
```

    Requirement already up-to-date: pandas in /opt/conda/lib/python3.6/site-packages (1.1.5)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/lib/python3.6/site-packages (from pandas) (2.8.2)
    Requirement already satisfied, skipping upgrade: numpy>=1.15.4 in /opt/conda/lib/python3.6/site-packages (from pandas) (1.19.5)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas) (2017.3)
    Requirement already satisfied, skipping upgrade: six>=1.5 in /opt/conda/lib/python3.6/site-packages (from python-dateutil>=2.7.3->pandas) (1.11.0)
    Requirement already up-to-date: seaborn in /opt/conda/lib/python3.6/site-packages (0.11.2)
    Requirement already satisfied, skipping upgrade: scipy>=1.0 in /opt/conda/lib/python3.6/site-packages (from seaborn) (1.2.1)
    Requirement already satisfied, skipping upgrade: pandas>=0.23 in /opt/conda/lib/python3.6/site-packages (from seaborn) (1.1.5)
    Requirement already satisfied, skipping upgrade: numpy>=1.15 in /opt/conda/lib/python3.6/site-packages (from seaborn) (1.19.5)
    Requirement already satisfied, skipping upgrade: matplotlib>=2.2 in /opt/conda/lib/python3.6/site-packages (from seaborn) (3.3.4)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas>=0.23->seaborn) (2017.3)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/lib/python3.6/site-packages (from pandas>=0.23->seaborn) (2.8.2)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /opt/conda/lib/python3.6/site-packages (from matplotlib>=2.2->seaborn) (2.2.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg (from matplotlib>=2.2->seaborn) (0.10.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/lib/python3.6/site-packages (from matplotlib>=2.2->seaborn) (1.3.1)
    Requirement already satisfied, skipping upgrade: pillow>=6.2.0 in /opt/conda/lib/python3.6/site-packages (from matplotlib>=2.2->seaborn) (8.4.0)
    Requirement already satisfied, skipping upgrade: six>=1.5 in /opt/conda/lib/python3.6/site-packages (from python-dateutil>=2.7.3->pandas>=0.23->seaborn) (1.11.0)



```python
!pip uninstall statsmodels -y
!pip install numpy scipy patsy pandas
!pip install statsmodels
!pip install --upgrade sklearn
!pip install scikit-learn --upgrade
```

    Uninstalling statsmodels-0.12.2:
      Successfully uninstalled statsmodels-0.12.2
    Requirement already satisfied: numpy in /opt/conda/lib/python3.6/site-packages (1.19.5)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.6/site-packages (1.2.1)
    Requirement already satisfied: patsy in /opt/conda/lib/python3.6/site-packages (0.5.3)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.6/site-packages (1.1.5)
    Requirement already satisfied: six in /opt/conda/lib/python3.6/site-packages (from patsy) (1.11.0)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas) (2017.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/lib/python3.6/site-packages (from pandas) (2.8.2)
    Collecting statsmodels
      Using cached https://files.pythonhosted.org/packages/0d/7b/c17815648dc31396af865b9c6627cc3f95705954e30f61106795361c39ee/statsmodels-0.12.2-cp36-cp36m-manylinux1_x86_64.whl
    Requirement already satisfied: numpy>=1.15 in /opt/conda/lib/python3.6/site-packages (from statsmodels) (1.19.5)
    Requirement already satisfied: pandas>=0.21 in /opt/conda/lib/python3.6/site-packages (from statsmodels) (1.1.5)
    Requirement already satisfied: scipy>=1.1 in /opt/conda/lib/python3.6/site-packages (from statsmodels) (1.2.1)
    Requirement already satisfied: patsy>=0.5 in /opt/conda/lib/python3.6/site-packages (from statsmodels) (0.5.3)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas>=0.21->statsmodels) (2017.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/lib/python3.6/site-packages (from pandas>=0.21->statsmodels) (2.8.2)
    Requirement already satisfied: six in /opt/conda/lib/python3.6/site-packages (from patsy>=0.5->statsmodels) (1.11.0)
    Installing collected packages: statsmodels
    Successfully installed statsmodels-0.12.2
    Requirement already up-to-date: sklearn in /opt/conda/lib/python3.6/site-packages (0.0.post1)
    Requirement already up-to-date: scikit-learn in /opt/conda/lib/python3.6/site-packages (0.24.2)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/lib/python3.6/site-packages (from scikit-learn) (1.2.1)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/lib/python3.6/site-packages (from scikit-learn) (0.11)
    Requirement already satisfied, skipping upgrade: numpy>=1.13.3 in /opt/conda/lib/python3.6/site-packages (from scikit-learn) (1.19.5)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.6/site-packages (from scikit-learn) (3.1.0)



```python
#!pip install pipreqsnb
#!pipreqsnb .
```

# Imports


```python
import pandas as pd
import numpy as np
import math
import json
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
#% matplotlib notebook

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
```


```python
# models
import statsmodels.api as sm

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```


```python
pd.__version__, sns.__version__
```




    ('1.1.5', '0.11.2')




```python
# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
```

# Expore Data And First Cleaning

## Portfolio


```python
portfolio.head()
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
      <th>reward</th>
      <th>channels</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>[email, mobile, social]</td>
      <td>10</td>
      <td>7</td>
      <td>bogo</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>[web, email, mobile, social]</td>
      <td>10</td>
      <td>5</td>
      <td>bogo</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>[web, email, mobile]</td>
      <td>0</td>
      <td>4</td>
      <td>informational</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>[web, email, mobile]</td>
      <td>5</td>
      <td>7</td>
      <td>bogo</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>[web, email]</td>
      <td>20</td>
      <td>10</td>
      <td>discount</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
    </tr>
  </tbody>
</table>
</div>




```python
portfolio.shape
```




    (10, 6)




```python
portfolio.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 6 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   reward      10 non-null     int64 
     1   channels    10 non-null     object
     2   difficulty  10 non-null     int64 
     3   duration    10 non-null     int64 
     4   offer_type  10 non-null     object
     5   id          10 non-null     object
    dtypes: int64(3), object(3)
    memory usage: 608.0+ bytes



```python
portfolio.describe()
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
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.200000</td>
      <td>7.700000</td>
      <td>6.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.583915</td>
      <td>5.831905</td>
      <td>2.321398</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>8.500000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>20.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#portfolio.duplicated()
```

the portfolio data contains 10 rows with all non-null values. For further exploration the lists contained in column 'channels' need to be separated in to individual columns.


```python
pd.get_dummies(portfolio['channels'].explode()).sum(level=0)  # as suggested on https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies

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
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# as each row contains "1" for emails, email will be treated as the default and the column email is removed to avoid collinearity effects
portfolio = portfolio.join(pd.get_dummies(portfolio['channels'].explode(), drop_first=True).sum(level=0), rsuffix='_r')
portfolio = portfolio.drop(['channels'], axis=1)
```


```python
portfolio.duplicated().any()
```




    False




```python
# as profile also contains a column named 'id', column is renamed to avoid confusing features. Use 'offer_id' as in transcript
portfolio = portfolio.rename(columns={'id': 'offer_id'})
portfolio
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
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>offer_id</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>bogo</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>bogo</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>informational</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>bogo</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>20</td>
      <td>10</td>
      <td>discount</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>discount</td>
      <td>2298d6c36e964ae4a3e7e9706d1fb8c2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>10</td>
      <td>10</td>
      <td>discount</td>
      <td>fafdcd668e3743c1bb461111dcafc2a4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>informational</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>bogo</td>
      <td>f19421c1d4aa40978ebb69ca19b0e20d</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>10</td>
      <td>7</td>
      <td>discount</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
portfolio['reward'].sort_values().unique(), portfolio['difficulty'].sort_values().unique()
```




    (array([ 0,  2,  3,  5, 10]), array([ 0,  5,  7, 10, 20]))




```python
round((portfolio['reward'] / portfolio['difficulty']).sort_values(), 2).unique()
```




    array([0.2 , 0.25, 0.43, 1.  ,  nan])



Except for the bogo offer (100% discount), there are discounts of 20%, 25% and 43% (if maxed out by just fulfilling the difficulty). Informational offer has 0% discount. 

## Profile


```python
profile.head()
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
      <th>gender</th>
      <th>age</th>
      <th>id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>118</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>20170212</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>55</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>20170715</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>118</td>
      <td>38fe809add3b4fcf9315a9694bb96ff5</td>
      <td>20180712</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>75</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>20170509</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>118</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>20170804</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
profile.shape
```




    (17000, 5)




```python
profile.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17000 entries, 0 to 16999
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   gender            14825 non-null  object 
     1   age               17000 non-null  int64  
     2   id                17000 non-null  object 
     3   became_member_on  17000 non-null  int64  
     4   income            14825 non-null  float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 664.2+ KB



```python
profile.describe()
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
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.000000</td>
      <td>1.700000e+04</td>
      <td>14825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>62.531412</td>
      <td>2.016703e+07</td>
      <td>65404.991568</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.738580</td>
      <td>1.167750e+04</td>
      <td>21598.299410</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>2.013073e+07</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45.000000</td>
      <td>2.016053e+07</td>
      <td>49000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>58.000000</td>
      <td>2.017080e+07</td>
      <td>64000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.000000</td>
      <td>2.017123e+07</td>
      <td>80000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>118.000000</td>
      <td>2.018073e+07</td>
      <td>120000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
profile.duplicated().any()
```




    False




```python
profile[profile['age']==118].shape[0], profile[profile['gender'].isnull()].shape[0], profile[profile['income'].isnull()].shape[0]
```




    (2175, 2175, 2175)



columns gender and income contain same number of null values, other columns are non-null at first glimpse. However, the age column contains, as stated in the project description, max values of 118 instead of NaNs. 'became_member_on' has been converted to datetime and will probably be replaced with the number of membership years. There are no duplicate rows.
Look into the 'age = 118' rows:


```python
profile.loc[(profile['age']==118) & ((profile['gender'].isnull() == False) | (profile['income'].isnull() == False))].shape[0]
```




    0



All of the 2175 rows that lack the age information, do not contain information on gender nor income. As in rows without demographic information and only the joining date will not be too useful in our analysis and losing 12,8% of the customer rows seems reasonable, we drop these rows (knowing that joining date may be useful for some extended evaluation):


```python
profile = profile[profile['age']!=118]
```


```python
profile.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 14825 entries, 1 to 16999
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   gender            14825 non-null  object 
     1   age               14825 non-null  int64  
     2   id                14825 non-null  object 
     3   became_member_on  14825 non-null  int64  
     4   income            14825 non-null  float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 694.9+ KB



```python
profile.describe()
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
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14825.000000</td>
      <td>1.482500e+04</td>
      <td>14825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.393524</td>
      <td>2.016689e+07</td>
      <td>65404.991568</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.383705</td>
      <td>1.188565e+04</td>
      <td>21598.299410</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>2.013073e+07</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>42.000000</td>
      <td>2.016052e+07</td>
      <td>49000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.000000</td>
      <td>2.017080e+07</td>
      <td>64000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>66.000000</td>
      <td>2.017123e+07</td>
      <td>80000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>101.000000</td>
      <td>2.018073e+07</td>
      <td>120000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
profile['id'].nunique()
```




    14825




```python
profile['gender'].hist()
plt.xlabel('gender');
plt.ylabel('count');
plt.title('Histogram: Genders');

profile['gender'].value_counts()
```




    M    8484
    F    6129
    O     212
    Name: gender, dtype: int64




![png](output_40_1.png)



```python
profile['gender'].value_counts()[0]/profile.shape[0], profile['gender'].value_counts()[1]/profile.shape[0], profile['gender'].value_counts()[2]/profile.shape[0], 
```




    (0.5722765598650927, 0.4134232715008432, 0.014300168634064081)



genders represented in dataset are imbalanced. We will dummy the gender,  dropping 'O' because of its low representation and to avoid high correlation between gender columns. Ethics of doing so not discussed here.
('O' could have been represented as 0 in both 'female' and 'male' columns, if used)


```python
#profile[['female', 'male', 'other']] = pd.get_dummies(profile['gender'])
#profile = profile.drop(['gender', 'other'], axis=1)
profile = profile[profile['gender'] != 'O']
profile[['female', 'male']] = pd.get_dummies(profile['gender'])
profile = profile.drop(['gender', 'male'], axis=1)
profile
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
      <th>age</th>
      <th>id</th>
      <th>became_member_on</th>
      <th>income</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>20170715</td>
      <td>112000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>20170509</td>
      <td>100000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68</td>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>20180426</td>
      <td>70000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>65</td>
      <td>389bc3fa690240e798340f5a15918d5c</td>
      <td>20180209</td>
      <td>53000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>58</td>
      <td>2eeac8d8feae4a8cad5a6af0499a211d</td>
      <td>20171111</td>
      <td>51000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16995</th>
      <td>45</td>
      <td>6d5f3a774f3d4714ab0c092238f3a1d7</td>
      <td>20180604</td>
      <td>54000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16996</th>
      <td>61</td>
      <td>2cb4f97358b841b9a9773a7aa05a9d77</td>
      <td>20180713</td>
      <td>72000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16997</th>
      <td>49</td>
      <td>01d26f638c274aa0b965d24cefe3183f</td>
      <td>20170126</td>
      <td>73000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16998</th>
      <td>83</td>
      <td>9dc1421481194dcd9400aec7c9ae6366</td>
      <td>20160307</td>
      <td>50000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16999</th>
      <td>62</td>
      <td>e4052622e5ba45a8b96b59aba68cf068</td>
      <td>20170722</td>
      <td>82000.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>14613 rows × 5 columns</p>
</div>




```python
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
```


```python
profile['became_member_on'].min(), profile['became_member_on'].max()
```




    (Timestamp('2013-07-29 00:00:00'), Timestamp('2018-07-26 00:00:00'))




```python
reference_date = profile['became_member_on'].max() + timedelta(days=7)
reference_date
```




    Timestamp('2018-08-02 00:00:00')




```python
profile['membership months'] = (reference_date - profile['became_member_on']).astype('timedelta64[M]').astype('int64')
profile = profile.drop(['became_member_on'], axis=1)
```


```python
profile['membership months'].hist()
plt.xlabel('months');
plt.ylabel('count');
plt.title('Histogram: Membership');
```


![png](output_48_0.png)


the membership time distribution is skewed to short memberships within the dataset


```python
profile.income.hist()
plt.xlabel('USD');
plt.ylabel('count');
plt.title('Histogram: Income');
```


![png](output_50_0.png)


the income distribution is skewed to the lower incomes


```python
profile.head()
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
      <th>age</th>
      <th>id</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>112000.0</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68</td>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>70000.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>65</td>
      <td>389bc3fa690240e798340f5a15918d5c</td>
      <td>53000.0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>58</td>
      <td>2eeac8d8feae4a8cad5a6af0499a211d</td>
      <td>51000.0</td>
      <td>0</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



No nulls left in the profile dataset.
Rename column 'id' to 'customer_id' to be specific:


```python
profile = profile.rename(columns={'id': 'customer_id'})
```


```python
plt.scatter(profile['age'], profile['income'])
plt.xlabel('age (years)');
plt.ylabel('income (USD)');
plt.title('Income vs. Age');
```


![png](output_55_0.png)


weird pattern but probably due to random value creation for the dataset. nothing to be cleaned.

## Transcript


```python
transcript.head()
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
      <th>person</th>
      <th>event</th>
      <th>value</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>offer received</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>offer received</td>
      <td>{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
value_df = pd.json_normalize(transcript['value'])  # as proposed on https://stackoverflow.com/questions/46391291/how-to-convert-json-data-inside-a-pandas-column-into-new-columns
```


```python
value_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 4 columns):
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   offer id  134002 non-null  object 
     1   amount    138953 non-null  float64
     2   offer_id  33579 non-null   object 
     3   reward    33579 non-null   float64
    dtypes: float64(2), object(2)
    memory usage: 9.4+ MB



```python
value_df[~value_df['offer_id'].isnull() & ~value_df['offer id'].isnull()].shape[0]
```




    0



After the extraction, there are 2 'offer_id' columns: 'offer_id' and 'offer id'. As there is no overlap, we will simply copy all values from 'offer id' to 'offer_id' and then drop 'offer id'


```python
value_df.loc[value_df['offer_id'].isnull(), 'offer_id'] = value_df['offer id']
value_df = value_df.drop(['offer id'], axis=1)
value_df.head()
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
      <th>amount</th>
      <th>offer_id</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>fafdcd668e3743c1bb461111dcafc2a4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
value_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 3 columns):
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   amount    138953 non-null  float64
     1   offer_id  167581 non-null  object 
     2   reward    33579 non-null   float64
    dtypes: float64(2), object(1)
    memory usage: 7.0+ MB


check: the number of non-null values in 'offer_id' is now equivalent to the sum of the two columns before the operation.
Join value_df with transcript and drop original 'value' column:


```python
transcript = transcript.drop(['value'], axis=1)
transcript = transcript.join(value_df, rsuffix='_r')
transcript.head()
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
      <th>person</th>
      <th>event</th>
      <th>time</th>
      <th>amount</th>
      <th>offer_id</th>
      <th>reward</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>fafdcd668e3743c1bb461111dcafc2a4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
transcript['event'].value_counts()
```




    transaction        138953
    offer received      76277
    offer viewed        57725
    offer completed     33579
    Name: event, dtype: int64




```python
transcript.duplicated().sum()
```




    397




```python
transcript[transcript.duplicated()]['event'].value_counts()
```




    offer completed    397
    Name: event, dtype: int64



There are 397 duplicates which are all events of type 'offer completed'. This could be due to multiple purchases of the same item/offer by the same customer within one hour or if the duplicates should be removed. However, for our evaluation we will remove that small subset (about 400 out of about 139000).


```python
transcript = transcript.drop_duplicates()
transcript.shape
```




    (306137, 6)



Rename 'person' to 'customer_id' for consistency:


```python
transcript = transcript.rename(columns={'person': 'customer_id'})
```

### Merge into 1 dataset


```python
df = pd.merge(transcript, profile, on='customer_id')
df.head()
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
      <th>customer_id</th>
      <th>event</th>
      <th>time</th>
      <th>amount</th>
      <th>offer_id</th>
      <th>reward</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>6</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>132</td>
      <td>19.89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>132</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>5.0</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>144</td>
      <td>17.78</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.merge(df, portfolio, how='left', on='offer_id')
df.head()
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
      <th>customer_id</th>
      <th>event</th>
      <th>time</th>
      <th>amount</th>
      <th>offer_id</th>
      <th>reward_x</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward_y</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>6</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>132</td>
      <td>19.89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>132</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>5.0</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>144</td>
      <td>17.78</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (268423, 17)



rename rewards to make specific


```python
df[~df['reward_x'].isnull()]['event'].value_counts()
```




    offer completed    31575
    Name: event, dtype: int64




```python
df[~df['reward_y'].isnull()]['event'].value_counts()
```




    offer received     65585
    offer viewed       49087
    offer completed    31575
    Name: event, dtype: int64




```python
df.query('event=="offer completed" & reward_x==reward_y').shape[0]
```




    31575



'reward_x' is identical to 'reward_y' for all rows of 'event' "offer received" and is NaN for all other events. Thus 'reward_x' can be dropped and 'reward_y' is renamed to 'reward':


```python
df = df.drop(['reward_x'], axis=1)
df = df.rename(columns={'reward_y': 'reward'})
```


```python
df.head(10)
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
      <th>customer_id</th>
      <th>event</th>
      <th>time</th>
      <th>amount</th>
      <th>offer_id</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>6</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>132</td>
      <td>19.89</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>132</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>144</td>
      <td>17.78</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>168</td>
      <td>NaN</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>216</td>
      <td>NaN</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>222</td>
      <td>19.67</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>240</td>
      <td>29.72</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>378</td>
      <td>23.93</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>268423.000000</td>
      <td>122176.000000</td>
      <td>268423.000000</td>
      <td>268423.000000</td>
      <td>268423.000000</td>
      <td>268423.000000</td>
      <td>146247.000000</td>
      <td>146247.000000</td>
      <td>146247.000000</td>
      <td>146247.000000</td>
      <td>146247.000000</td>
      <td>146247.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>366.614903</td>
      <td>13.984628</td>
      <td>53.834396</td>
      <td>64349.187663</td>
      <td>0.420754</td>
      <td>18.860515</td>
      <td>4.442854</td>
      <td>7.882958</td>
      <td>6.621927</td>
      <td>0.917735</td>
      <td>0.659316</td>
      <td>0.806629</td>
    </tr>
    <tr>
      <th>std</th>
      <td>200.368088</td>
      <td>31.828472</td>
      <td>17.571553</td>
      <td>21271.958211</td>
      <td>0.493681</td>
      <td>14.112490</td>
      <td>3.374746</td>
      <td>5.034415</td>
      <td>2.132900</td>
      <td>0.274769</td>
      <td>0.473941</td>
      <td>0.394943</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>18.000000</td>
      <td>30000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>186.000000</td>
      <td>3.630000</td>
      <td>41.000000</td>
      <td>48000.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>408.000000</td>
      <td>10.760000</td>
      <td>55.000000</td>
      <td>62000.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>528.000000</td>
      <td>19.130000</td>
      <td>66.000000</td>
      <td>78000.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>714.000000</td>
      <td>1062.280000</td>
      <td>101.000000</td>
      <td>120000.000000</td>
      <td>1.000000</td>
      <td>60.000000</td>
      <td>10.000000</td>
      <td>20.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['time'].max()/24
```




    29.75



The test was run for about 30 days

# Data Preparation


```python
df2 = df.copy()
```


```python
df2['valid until'] = df2[df2['event'] == 'offer received']['time'] + df2[df2['event'] == 'offer received']['duration']*24
```

def event_within_validity(df, row, event):
    customer = row['customer_id']
    offer = row['offer_id']
    df = df.query('customer_id == @customer  & offer_id == @offer & event == @event')
    #df = df[(df['customer_id'] == customer_id)  & (df['offer_id'] == offer_id) & (df['event'] == event)]
    start = row['time']
    end = row['valid until']
    view_time = df['time']
    for view in view_time:
        if (start <= view) and (view <= end):
            return True
    
    return False

df2['offer viewed in time'] = df2[df2['event'] == 'offer received'][['customer_id', 'time', 'valid until', 'offer_id']].apply(lambda row: event_within_validity(df2, row, 'offer viewed'), axis=1)
df2

above function to check whether offer has been "viewed" within validity timeframe runs for too long on full dataset. Thus we just assume that if an offer has been recorded as "viewed", it was within the range of validity. Same for "offer completed"


```python
viewed_df = df2[df2['event'] == 'offer viewed'][['customer_id', 'offer_id']]
df2['offer viewed'] = np.nan
df2['offer viewed'] = (df2['customer_id'].isin(viewed_df['customer_id']) & df2['offer_id'].isin(viewed_df['offer_id'])).astype('int')

completed_df = df2[df2['event'] == 'offer completed'][['customer_id', 'offer_id']]
df2['offer completed'] = np.nan
df2['offer completed'] = (df2['customer_id'].isin(completed_df['customer_id']) & df2['offer_id'].isin(completed_df['offer_id'])).astype('int')
df2
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
      <th>customer_id</th>
      <th>event</th>
      <th>time</th>
      <th>amount</th>
      <th>offer_id</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>6</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>132</td>
      <td>19.89</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>132</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>144</td>
      <td>17.78</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>268418</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>offer viewed</td>
      <td>504</td>
      <td>NaN</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268419</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>offer received</td>
      <td>576</td>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>696.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268420</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>offer viewed</td>
      <td>576</td>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268421</th>
      <td>3045af4e98794a04a5542d3eac939b1f</td>
      <td>offer received</td>
      <td>576</td>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>58</td>
      <td>78000.0</td>
      <td>1</td>
      <td>21</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>696.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268422</th>
      <td>3045af4e98794a04a5542d3eac939b1f</td>
      <td>offer viewed</td>
      <td>576</td>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>58</td>
      <td>78000.0</td>
      <td>1</td>
      <td>21</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>268423 rows × 19 columns</p>
</div>




```python
df3 = df2[df2['event'].isin(['offer received', 'transaction'])]
df3.loc[df3['event'] == 'transaction', 'offer_type'] = 'transaction'
df3 = df3.drop(['offer_id', 'event', 'duration'], axis=1)
df3 = df3.rename(columns={'offer_type': 'type'})
df3.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/indexing.py:1763: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      isetter(loc, value)





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
      <th>customer_id</th>
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>type</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>132</td>
      <td>19.89</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>transaction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>144</td>
      <td>17.78</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>transaction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>168</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>240.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>222</td>
      <td>19.67</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>transaction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have built an understanding of the data, done some cleaning and have merged everything to a single dataframe, we can start working with the data. We can go into some deeper analysis and visualize findings.


```python
df3.describe()
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
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>187761.000000</td>
      <td>122176.000000</td>
      <td>187761.000000</td>
      <td>187761.000000</td>
      <td>187761.000000</td>
      <td>187761.000000</td>
      <td>65585.000000</td>
      <td>65585.000000</td>
      <td>65585.000000</td>
      <td>65585.000000</td>
      <td>65585.000000</td>
      <td>65585.000000</td>
      <td>187761.000000</td>
      <td>187761.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>364.368958</td>
      <td>13.984628</td>
      <td>53.288580</td>
      <td>63094.279430</td>
      <td>0.409233</td>
      <td>19.286439</td>
      <td>4.200778</td>
      <td>7.715499</td>
      <td>0.898956</td>
      <td>0.598765</td>
      <td>0.799741</td>
      <td>488.864466</td>
      <td>0.346787</td>
      <td>0.231193</td>
    </tr>
    <tr>
      <th>std</th>
      <td>201.323402</td>
      <td>31.828472</td>
      <td>17.766384</td>
      <td>21096.859661</td>
      <td>0.491694</td>
      <td>14.326498</td>
      <td>3.399038</td>
      <td>5.545371</td>
      <td>0.301390</td>
      <td>0.490152</td>
      <td>0.400197</td>
      <td>203.315589</td>
      <td>0.475948</td>
      <td>0.421597</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>18.000000</td>
      <td>30000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>180.000000</td>
      <td>3.630000</td>
      <td>40.000000</td>
      <td>47000.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>336.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>402.000000</td>
      <td>10.760000</td>
      <td>54.000000</td>
      <td>61000.000000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>528.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>528.000000</td>
      <td>19.130000</td>
      <td>66.000000</td>
      <td>75000.000000</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>672.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>714.000000</td>
      <td>1062.280000</td>
      <td>101.000000</td>
      <td>120000.000000</td>
      <td>1.000000</td>
      <td>60.000000</td>
      <td>10.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>816.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3['reward'].max(), df3['difficulty'].max()
```




    (10.0, 20.0)




```python
df3[df3['type'] == 'transaction']['amount'].describe()
```




    count    122176.000000
    mean         13.984628
    std          31.828472
    min           0.050000
    25%           3.630000
    50%          10.760000
    75%          19.130000
    max        1062.280000
    Name: amount, dtype: float64




```python
# amount is only non-null for transactions
df3[df3['type'] == 'transaction']['amount'].hist()
np.percentile(df3[df3['type'] == 'transaction']['amount'], 99.5)
```




    51.25874999999999




![png](output_99_1.png)


in this evaluation we will focus on the about 99.5% of customers whose amount is 50 or less, assuming that the few customers buying bigger amounts would most likely do so without having received offers (negligible savings with bogos, discounts between 20% and 1%). This needs to be investigated in another evaluation.


```python
df4 = df3[(df3['amount'] < 200) | df3['amount'].isnull()]
```


```python
df4.head()
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
      <th>customer_id</th>
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>type</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>132</td>
      <td>19.89</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>transaction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>144</td>
      <td>17.78</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>transaction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>168</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>240.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>222</td>
      <td>19.67</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>transaction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# amount is only non-null for transactions
df4[df4['type'] == 'transaction']['amount'].hist()
```




    <AxesSubplot:>




![png](output_103_1.png)



```python
fig, ax = plt.subplots(figsize=(15,5))
#df4.drop(['income', 'time', 'valid until', 'amount'], axis=1).boxplot()
df4[['age', 'membership months']].boxplot()
#df4[['reward', 'difficulty']].boxplot()
plt.title('Boxplots: Age, Membership');
```


![png](output_104_0.png)



```python
df4[df4['type'] != 'transaction'].isnull().any()
```




    customer_id          False
    time                 False
    amount                True
    age                  False
    income               False
    female               False
    membership months    False
    reward               False
    difficulty           False
    type                 False
    mobile               False
    social               False
    web                  False
    valid until          False
    offer viewed         False
    offer completed      False
    dtype: bool




```python
df4[df4['type'] == 'transaction'].isnull().any()
```




    customer_id          False
    time                 False
    amount               False
    age                  False
    income               False
    female               False
    membership months    False
    reward                True
    difficulty            True
    type                 False
    mobile                True
    social                True
    web                   True
    valid until           True
    offer viewed         False
    offer completed      False
    dtype: bool



# Feature Engineering


```python
df = df4.copy()
```


```python
df[['bogo', 'discount', 'informational', 'transaction']] = pd.get_dummies(df['type'])
```


```python
df['rel discount'] = round(df['reward'] / df['difficulty'], 2)
#df[df['rel discount'].isnull()]['rel discount'] = 0
df.loc[df['rel discount'].isnull(), 'rel discount'] = 0
df[(df['type']=='discount') & (df['rel discount']==0)].shape[0], df[(df['type']=='discount') & (df['rel discount']==1)].shape[0]
```




    (0, 0)




```python
# dummy rel discount, 0% = information, 100% = 
df[['0.00', 'discount 0.20', 'discount 0.25', 'discount 0.43', '1.00']] = pd.get_dummies(df['rel discount'])
df = df.drop(['type', '0.00', '1.00', 'discount', 'rel discount'], axis=1)
df
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
      <th>customer_id</th>
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>mobile</th>
      <th>...</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>bogo</th>
      <th>informational</th>
      <th>transaction</th>
      <th>discount 0.20</th>
      <th>discount 0.25</th>
      <th>discount 0.43</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>132</td>
      <td>19.89</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>144</td>
      <td>17.78</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>168</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>240.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>222</td>
      <td>19.67</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>268415</th>
      <td>8578196a074a4f328976e334fa9383a3</td>
      <td>576</td>
      <td>NaN</td>
      <td>48</td>
      <td>58000.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>744.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268416</th>
      <td>8578196a074a4f328976e334fa9383a3</td>
      <td>702</td>
      <td>4.62</td>
      <td>48</td>
      <td>58000.0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268417</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>504</td>
      <td>NaN</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>600.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268419</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>576</td>
      <td>NaN</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>696.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268421</th>
      <td>3045af4e98794a04a5542d3eac939b1f</td>
      <td>576</td>
      <td>NaN</td>
      <td>58</td>
      <td>78000.0</td>
      <td>1</td>
      <td>21</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>696.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>187394 rows × 21 columns</p>
</div>




```python
df5 = df.copy()
```


```python
df[df['transaction'] == 0]
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
      <th>customer_id</th>
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>mobile</th>
      <th>...</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>bogo</th>
      <th>informational</th>
      <th>transaction</th>
      <th>discount 0.20</th>
      <th>discount 0.25</th>
      <th>discount 0.43</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>168</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>240.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>408</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>576.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>504</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>624.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>0</td>
      <td>NaN</td>
      <td>68</td>
      <td>70000.0</td>
      <td>0</td>
      <td>3</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>268413</th>
      <td>8578196a074a4f328976e334fa9383a3</td>
      <td>504</td>
      <td>NaN</td>
      <td>48</td>
      <td>58000.0</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>624.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268415</th>
      <td>8578196a074a4f328976e334fa9383a3</td>
      <td>576</td>
      <td>NaN</td>
      <td>48</td>
      <td>58000.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>744.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268417</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>504</td>
      <td>NaN</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>600.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268419</th>
      <td>9fcbff4f8d7241faa4ab8a9d19c8a812</td>
      <td>576</td>
      <td>NaN</td>
      <td>47</td>
      <td>94000.0</td>
      <td>0</td>
      <td>9</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>696.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>268421</th>
      <td>3045af4e98794a04a5542d3eac939b1f</td>
      <td>576</td>
      <td>NaN</td>
      <td>58</td>
      <td>78000.0</td>
      <td>1</td>
      <td>21</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>696.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>65585 rows × 21 columns</p>
</div>



#### Funnel


```python
offer_cols = ['bogo', 'informational', 'discount 0.20', 'discount 0.25', 'discount 0.43']
offers_count = (df[offer_cols].sum(axis=1)).sum()
offers_viewed_count = (df['offer viewed'] == 1).sum()
offers_completed_count = (df['offer completed'] == 1).sum()
informational_count = (df['informational'] == 1).sum()
```


```python
offers_count, offers_viewed_count, offers_completed_count, informational_count
```




    (65585, 65113, 43409, 13105)




```python
offers_viewed_count / offers_count, offers_completed_count / offers_viewed_count, offers_completed_count / offers_count
```




    (0.9928032324464435, 0.6666717859720793, 0.6618739040939239)




```python
offers_completed_count / (offers_count - informational_count)
```




    0.8271532012195122




```python
#analyse sent vs completed for different offer types except for informaitonal:
bogo = df[df['bogo'] == 1]
discount43 = df[df['discount 0.43'] == 1]
discount25 = df[df['discount 0.25'] == 1]
discount20 = df[df['discount 0.20'] == 1]

print(f"bogo: {(bogo['offer completed'] ==1).sum()} - {(bogo['offer completed'] ==1).sum() / bogo.shape[0]}")
print(f"discount 43: {(discount43['offer completed'] ==1).sum()} - {(discount43['offer completed'] ==1).sum() / discount43.shape[0]}")
print(f"discount 25: {(discount25['offer completed'] ==1).sum()} - {(discount25['offer completed'] ==1).sum() / discount25.shape[0]}")
print(f"discount 20: {(discount20['offer completed'] ==1).sum()} - {(discount20['offer completed'] ==1).sum() / discount20.shape[0]}")
```

    bogo: 21523 - 0.8220219226215484
    discount 43: 5658 - 0.8621057443242419
    discount 25: 5290 - 0.798249585030934
    discount 20: 10938 - 0.8345159075303273


- 99% of all offers are viewes (including informational)
- 83% of all offers are completed (if excluding informational, which has no record for completion)
    All offers have almost the same acceptance rate. 25% discount has lowest completion rate. Lower than 20% discount, which is 'cheaper' for the company -> suggest A/B test dropping the 25% discount, if further analysis shows that demographic subgroups show similar results (Simpson's Paradox!). Not investigated here


```python
number_of_offers_per_customer_no_info = df[df['informational'] == 0].groupby('customer_id', as_index=False).size()
number_of_offers_per_customer_no_info.hist()
plt.xlabel('offers per customer');
plt.ylabel('count');
plt.title('Histogram: Offers per Customer');
```


![png](output_121_0.png)



```python
rows_completed = df[df['offer completed'] == 1]
number_of_offers_completed_per_customer = rows_completed.groupby('customer_id', as_index=False).size()
number_of_offers_completed_per_customer.hist()
```




    array([[<AxesSubplot:title={'center':'size'}>]], dtype=object)




![png](output_122_1.png)



```python
number_of_offered_and_completed = number_of_offers_per_customer_no_info.merge(number_of_offers_completed_per_customer, how='left', on='customer_id')
number_of_offered_and_completed.rename(columns={'size_x': 'offers', 'size_y': 'completed'}, inplace=True)
#number_of_offered_and_completed[number_of_offered_and_completed['completed']=='NaN']#['offers'].value_counts()
number_of_offered_and_completed['completed rel'] = number_of_offered_and_completed['completed'] / number_of_offered_and_completed['offers']
number_of_offered_and_completed['completed rel'].hist()
number_of_offered_and_completed['completed rel'].describe()
```




    count    11807.000000
    mean         0.323903
    std          0.142414
    min          0.055556
    25%          0.222222
    50%          0.300000
    75%          0.400000
    max          0.857143
    Name: completed rel, dtype: float64




![png](output_123_1.png)



```python
plt.scatter(number_of_offered_and_completed['offers'], number_of_offered_and_completed['completed'])
```




    <matplotlib.collections.PathCollection at 0x7f0870b7a588>




![png](output_124_1.png)



```python
number_of_offered_and_completed[number_of_offered_and_completed['completed']==6]['offers'].hist()
```




    <AxesSubplot:>




![png](output_125_1.png)


50% of the customers complete between 22% and 40% of the offers that they receive but 6 at max, even when offered up to 40 discounts.

### Correlation analysis


```python
df.columns
```




    Index(['customer_id', 'time', 'amount', 'age', 'income', 'female',
           'membership months', 'reward', 'difficulty', 'mobile', 'social', 'web',
           'valid until', 'offer viewed', 'offer completed', 'bogo',
           'informational', 'transaction', 'discount 0.20', 'discount 0.25',
           'discount 0.43'],
          dtype='object')




```python
cols = ['offer completed', 'offer viewed', 'bogo', 'informational', 'discount 0.20', 'discount 0.25', 'discount 0.43', 'reward', 'difficulty', 'age', 'income', 'female', 'membership months', 'mobile', 'social', 'web']
cols.reverse()
df = df[cols]
```


```python
def nice_corr_matrix(df, save=False):
    """creates a nice correlation matrix plot (triangle)
    
    """
    ## nice correlation matrix plot inspired by: https://lost-stats.github.io/Presentation/Figures/heatmap_colored_correlation_matrix.html
    # Create the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # More details at https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        corr,          # The data to plot
        mask=mask,     # Mask some cells
        cmap=cmap,     # What colors to plot the heatmap as
        annot=True,    # Should the values be plotted in the cells?
        vmax=1.0,       # The maximum value of the legend. All higher vals will be same color
        vmin=-1.0,      # The minimum value of the legend. All lower vals will be same color
        center=0,      # The center value of the legend. With divergent cmap, where white is
        square=True,   # Force cells to be square
        linewidths=.5, # Width of lines that divide cells
        cbar_kws={"shrink": .5},  # Extra kwargs for the legend; in this case, shrink by 50%
        fmt=".2f"      # decimals display format
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    
    if save:
        f.savefig('heatmap_colored_correlation_matrix.png') # save as png
```


```python
nice_corr_matrix(df)
```


![png](output_131_0.png)



```python
df_no_inf = df[df['informational'] == 0].drop(['informational'], axis=1)
nice_corr_matrix(df_no_inf)
```


![png](output_132_0.png)


- No strong correlations between "offers viewed" and demographic data or channels. If an offer is viewed or not, seems to depend on the offer tpye, with bogo being the most likely one to be viewed.
- Strong correlation between "offer viewed" and "offer completed", as expected.


```python
#sns.pairplot(df)
```

# Modeling


```python
df = df5.copy()
df.head()
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
      <th>customer_id</th>
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>mobile</th>
      <th>...</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>bogo</th>
      <th>informational</th>
      <th>transaction</th>
      <th>discount 0.20</th>
      <th>discount 0.25</th>
      <th>discount 0.43</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>132</td>
      <td>19.89</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>144</td>
      <td>17.78</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>168</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>240.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>222</td>
      <td>19.67</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df = df[df['informational'] == 0].drop(['informational'], axis=1)
df = df[df['transaction'] == 0].drop(['transaction'], axis=1)
df.head()
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
      <th>customer_id</th>
      <th>time</th>
      <th>amount</th>
      <th>age</th>
      <th>income</th>
      <th>female</th>
      <th>membership months</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>mobile</th>
      <th>social</th>
      <th>web</th>
      <th>valid until</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>bogo</th>
      <th>discount 0.20</th>
      <th>discount 0.25</th>
      <th>discount 0.43</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>0</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>408</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>576.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>504</td>
      <td>NaN</td>
      <td>75</td>
      <td>100000.0</td>
      <td>1</td>
      <td>14</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>624.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>0</td>
      <td>NaN</td>
      <td>68</td>
      <td>70000.0</td>
      <td>0</td>
      <td>3</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>168.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>408</td>
      <td>NaN</td>
      <td>68</td>
      <td>70000.0</td>
      <td>0</td>
      <td>3</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>576.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Selecting Features

### Multicollinearity: VIF


```python
def vif(X):
    """print variance inflation factors for given column selection of pandas DataFrame.
    
    """
    vif_data = pd.DataFrame()  # VIF dataframe  # inspired by: https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]

    print(vif_data)
```


```python
X_all = df[['age', 'income', 'female', 'membership months', 'reward', 'difficulty', 'bogo', 'discount 0.20', 'discount 0.25', 'discount 0.43']]
X_rel = df[['age', 'income', 'female', 'membership months', 'bogo', 'discount 0.20', 'discount 0.25', 'discount 0.43']]
X_abs = df[['age', 'income', 'female', 'membership months', 'reward', 'difficulty']]
X_demo =  df[['age', 'income', 'female', 'membership months']]
xlist = [X_all, X_rel, X_abs, X_demo]
```


```python
for X in xlist:
    vif(X)
    print()
```

    /opt/conda/lib/python3.6/site-packages/statsmodels/stats/outliers_influence.py:193: RuntimeWarning: divide by zero encountered in double_scalars
      vif = 1. / (1. - r_squared_i)


                 feature        VIF
    0                age   1.113177
    1             income   1.147413
    2             female   1.064623
    3  membership months   1.000954
    4             reward        inf
    5         difficulty        inf
    6               bogo  17.886203
    7      discount 0.20        inf
    8      discount 0.25        inf
    9      discount 0.43        inf
    
                 feature       VIF
    0                age  1.113147
    1             income  1.147385
    2             female  1.064622
    3  membership months  1.000945
    4               bogo  8.881590
    5      discount 0.20  4.942397
    6      discount 0.25  2.992615
    7      discount 0.43  2.961348
    
                 feature       VIF
    0                age  9.326740
    1             income  9.461316
    2             female  1.827985
    3  membership months  2.367720
    4             reward  3.657212
    5         difficulty  4.703589
    
                 feature       VIF
    0                age  8.195107
    1             income  8.531994
    2             female  1.827984
    3  membership months  2.303266
    


X_all has, as expected, too strong collinearity to be used for further modelling. We will test the other feature sets.


```python
xlist.remove(X_all)
```

## Model for Predicting Whether a Given Customer Will Complete an Offer


```python
y = df['offer completed']
#X = df[['age', 'income', 'female', 'membership months', 'reward', 'difficulty', 'bogo', 'discount 0.20', 'discount 0.25', 'discount 0.43']]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)
```


```python
def score_model(y_test, y_preds):
    """Print a set of scores and the confusion matrix for classifications.
    
    """
    prediction = list(map(round, y_preds))
    print(str(precision_score(y_test, prediction)) + " (Precision)")
    print(str(recall_score(y_test, prediction)) + " (Recall)")
    print(str(accuracy_score(y_test, prediction)) + " (Accuracy)")
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction, target_names=['not completed', 'completed']))
```

Reminder confusion matrix (from: https://levelup.gitconnected.com/an-introduction-to-logistic-regression-in-python-with-statsmodels-and-scikit-learn-1a1fb5ce1c13)
![image.png](attachment:image.png)


### statsmodels


```python
def sm_log_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

    #X2_train = StandardScaler().fit_transform(X_train) non-robust when scaling
    #X2_test = StandardScaler().fit_transform(X_test)

    model = sm.Logit(y_train, X_train)
    results = model.fit()
    print(results.summary())
    print()

    y_preds = results.predict(X_test)
    score_model(y_test, y_preds)
    
    return results
```


```python
for X in xlist:
    sm_log_reg(X, y)
```

    Optimization terminated successfully.
             Current function value: 0.384300
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:        offer completed   No. Observations:                36736
    Model:                          Logit   Df Residuals:                    36728
    Method:                           MLE   Df Model:                            7
    Date:                Sun, 25 Dec 2022   Pseudo R-squ.:                  0.1622
    Time:                        17:03:30   Log-Likelihood:                -14118.
    converged:                       True   LL-Null:                       -16851.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =====================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    age                   0.0036      0.001      4.136      0.000       0.002       0.005
    income             2.979e-05   8.45e-07     35.260      0.000    2.81e-05    3.14e-05
    female                0.6427      0.033     19.367      0.000       0.578       0.708
    membership months     0.0702      0.002     43.612      0.000       0.067       0.073
    bogo                 -1.6205      0.064    -25.496      0.000      -1.745      -1.496
    discount 0.20        -1.5280      0.067    -22.779      0.000      -1.659      -1.397
    discount 0.25        -1.7829      0.072    -24.626      0.000      -1.925      -1.641
    discount 0.43        -1.2647      0.075    -16.965      0.000      -1.411      -1.119
    =====================================================================================
    
    0.8367346938775511 (Precision)
    0.9820548367221196 (Recall)
    0.827172256097561 (Accuracy)
    [[  272  2488]
     [  233 12751]]
                   precision    recall  f1-score   support
    
    not completed       0.54      0.10      0.17      2760
        completed       0.84      0.98      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.69      0.54      0.54     15744
     weighted avg       0.78      0.83      0.77     15744
    
    Optimization terminated successfully.
             Current function value: 0.388333
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:        offer completed   No. Observations:                36736
    Model:                          Logit   Df Residuals:                    36730
    Method:                           MLE   Df Model:                            5
    Date:                Sun, 25 Dec 2022   Pseudo R-squ.:                  0.1534
    Time:                        17:03:30   Log-Likelihood:                -14266.
    converged:                       True   LL-Null:                       -16851.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =====================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    age                  -0.0034      0.001     -4.368      0.000      -0.005      -0.002
    income             2.311e-05   7.19e-07     32.167      0.000    2.17e-05    2.45e-05
    female                0.6465      0.033     19.512      0.000       0.582       0.711
    membership months     0.0647      0.002     41.962      0.000       0.062       0.068
    reward               -0.0559      0.005    -12.084      0.000      -0.065      -0.047
    difficulty           -0.0427      0.003    -14.657      0.000      -0.048      -0.037
    =====================================================================================
    
    0.8301521795202477 (Precision)
    0.991528034504005 (Recall)
    0.8257113821138211 (Accuracy)
    [[  126  2634]
     [  110 12874]]
                   precision    recall  f1-score   support
    
    not completed       0.53      0.05      0.08      2760
        completed       0.83      0.99      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.68      0.52      0.49     15744
     weighted avg       0.78      0.83      0.76     15744
    
    Optimization terminated successfully.
             Current function value: 0.394536
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:        offer completed   No. Observations:                36736
    Model:                          Logit   Df Residuals:                    36732
    Method:                           MLE   Df Model:                            3
    Date:                Sun, 25 Dec 2022   Pseudo R-squ.:                  0.1399
    Time:                        17:03:30   Log-Likelihood:                -14494.
    converged:                       True   LL-Null:                       -16851.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =====================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    age                  -0.0089      0.001    -12.349      0.000      -0.010      -0.007
    income             1.755e-05   6.51e-07     26.962      0.000    1.63e-05    1.88e-05
    female                0.6434      0.033     19.518      0.000       0.579       0.708
    membership months     0.0597      0.001     40.031      0.000       0.057       0.063
    =====================================================================================
    
    0.8251797188116292 (Precision)
    0.9989987677141097 (Recall)
    0.8246316056910569 (Accuracy)
    [[   12  2748]
     [   13 12971]]
                   precision    recall  f1-score   support
    
    not completed       0.48      0.00      0.01      2760
        completed       0.83      1.00      0.90     12984
    
         accuracy                           0.82     15744
        macro avg       0.65      0.50      0.46     15744
     weighted avg       0.76      0.82      0.75     15744
    



```python
results = sm_log_reg(X_rel, y)
summary_df = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]
coeffs = summary_df['coef'].values
```

    Optimization terminated successfully.
             Current function value: 0.384300
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:        offer completed   No. Observations:                36736
    Model:                          Logit   Df Residuals:                    36728
    Method:                           MLE   Df Model:                            7
    Date:                Sun, 25 Dec 2022   Pseudo R-squ.:                  0.1622
    Time:                        17:03:30   Log-Likelihood:                -14118.
    converged:                       True   LL-Null:                       -16851.
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =====================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    age                   0.0036      0.001      4.136      0.000       0.002       0.005
    income             2.979e-05   8.45e-07     35.260      0.000    2.81e-05    3.14e-05
    female                0.6427      0.033     19.367      0.000       0.578       0.708
    membership months     0.0702      0.002     43.612      0.000       0.067       0.073
    bogo                 -1.6205      0.064    -25.496      0.000      -1.745      -1.496
    discount 0.20        -1.5280      0.067    -22.779      0.000      -1.659      -1.397
    discount 0.25        -1.7829      0.072    -24.626      0.000      -1.925      -1.641
    discount 0.43        -1.2647      0.075    -16.965      0.000      -1.411      -1.119
    =====================================================================================
    
    0.8367346938775511 (Precision)
    0.9820548367221196 (Recall)
    0.827172256097561 (Accuracy)
    [[  272  2488]
     [  233 12751]]
                   precision    recall  f1-score   support
    
    not completed       0.54      0.10      0.17      2760
        completed       0.84      0.98      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.69      0.54      0.54     15744
     weighted avg       0.78      0.83      0.77     15744
    



```python
summary_df['real coeff'] = np.exp(summary_df['coef'])
summary_df
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
      <th>coef</th>
      <th>std err</th>
      <th>z</th>
      <th>P&gt;|z|</th>
      <th>[0.025</th>
      <th>0.975]</th>
      <th>real coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>0.00360</td>
      <td>1.000000e-03</td>
      <td>4.136</td>
      <td>0.0</td>
      <td>0.002000</td>
      <td>0.005000</td>
      <td>1.003606</td>
    </tr>
    <tr>
      <th>income</th>
      <td>0.00003</td>
      <td>8.450000e-07</td>
      <td>35.260</td>
      <td>0.0</td>
      <td>0.000028</td>
      <td>0.000031</td>
      <td>1.000030</td>
    </tr>
    <tr>
      <th>female</th>
      <td>0.64270</td>
      <td>3.300000e-02</td>
      <td>19.367</td>
      <td>0.0</td>
      <td>0.578000</td>
      <td>0.708000</td>
      <td>1.901608</td>
    </tr>
    <tr>
      <th>membership months</th>
      <td>0.07020</td>
      <td>2.000000e-03</td>
      <td>43.612</td>
      <td>0.0</td>
      <td>0.067000</td>
      <td>0.073000</td>
      <td>1.072723</td>
    </tr>
    <tr>
      <th>bogo</th>
      <td>-1.62050</td>
      <td>6.400000e-02</td>
      <td>-25.496</td>
      <td>0.0</td>
      <td>-1.745000</td>
      <td>-1.496000</td>
      <td>0.197800</td>
    </tr>
    <tr>
      <th>discount 0.20</th>
      <td>-1.52800</td>
      <td>6.700000e-02</td>
      <td>-22.779</td>
      <td>0.0</td>
      <td>-1.659000</td>
      <td>-1.397000</td>
      <td>0.216969</td>
    </tr>
    <tr>
      <th>discount 0.25</th>
      <td>-1.78290</td>
      <td>7.200000e-02</td>
      <td>-24.626</td>
      <td>0.0</td>
      <td>-1.925000</td>
      <td>-1.641000</td>
      <td>0.168150</td>
    </tr>
    <tr>
      <th>discount 0.43</th>
      <td>-1.26470</td>
      <td>7.500000e-02</td>
      <td>-16.965</td>
      <td>0.0</td>
      <td>-1.411000</td>
      <td>-1.119000</td>
      <td>0.282324</td>
    </tr>
  </tbody>
</table>
</div>



the p values for all features are 0 -> all selected coefficients of X_rel are statistically siginificant.

### sklearn

#### Quick comparison sklearn LogisticRegression


```python
pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression())])
pipeline.steps
```




    [('scale', StandardScaler()), ('clf', LogisticRegression())]




```python
pipeline.set_params(clf = LogisticRegression())

def sklearn_log_reg(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)
    score_model(y_test, y_preds)
```


```python
for X in xlist:
    sklearn_log_reg(X,y)
```

    0.8367346938775511 (Precision)
    0.9820548367221196 (Recall)
    0.827172256097561 (Accuracy)
    [[  272  2488]
     [  233 12751]]
                   precision    recall  f1-score   support
    
    not completed       0.54      0.10      0.17      2760
        completed       0.84      0.98      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.69      0.54      0.54     15744
     weighted avg       0.78      0.83      0.77     15744
    
    0.8368137641187287 (Precision)
    0.9814386937769563 (Recall)
    0.826854674796748 (Accuracy)
    [[  275  2485]
     [  241 12743]]
                   precision    recall  f1-score   support
    
    not completed       0.53      0.10      0.17      2760
        completed       0.84      0.98      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.68      0.54      0.54     15744
     weighted avg       0.78      0.83      0.77     15744
    
    0.8352001048012052 (Precision)
    0.9820548367221196 (Recall)
    0.8253938008130082 (Accuracy)
    [[  244  2516]
     [  233 12751]]
                   precision    recall  f1-score   support
    
    not completed       0.51      0.09      0.15      2760
        completed       0.84      0.98      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.67      0.54      0.53     15744
     weighted avg       0.78      0.83      0.77     15744
    


#### Model Comparisons

we will choose X_res for all further steps


```python
X_train, X_test, y_train, y_test = train_test_split(X_rel, y, test_size = .30, random_state=42)
```


```python
# selection of models inspired by: https://www.kaggle.com/code/gautham11/building-a-scikit-learn-classification-pipeline/notebook
clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())

for classifier in clfs:
    print(classifier)
    pipeline.set_params(clf = classifier)
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)
    score_model(y_test, y_preds)
```

    LogisticRegression()
    0.8367346938775511 (Precision)
    0.9820548367221196 (Recall)
    0.827172256097561 (Accuracy)
    [[  272  2488]
     [  233 12751]]
                   precision    recall  f1-score   support
    
    not completed       0.54      0.10      0.17      2760
        completed       0.84      0.98      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.69      0.54      0.54     15744
     weighted avg       0.78      0.83      0.77     15744
    
    SVC()
    0.8247887681849946 (Precision)
    0.9999229821318546 (Recall)
    0.8247586382113821 (Accuracy)
    [[    2  2758]
     [    1 12983]]
                   precision    recall  f1-score   support
    
    not completed       0.67      0.00      0.00      2760
        completed       0.82      1.00      0.90     12984
    
         accuracy                           0.82     15744
        macro avg       0.75      0.50      0.45     15744
     weighted avg       0.80      0.82      0.75     15744
    
    KNeighborsClassifier(n_neighbors=3)
    0.8917131178000743 (Precision)
    0.924060382008626 (Recall)
    0.8448297764227642 (Accuracy)
    [[ 1303  1457]
     [  986 11998]]
                   precision    recall  f1-score   support
    
    not completed       0.57      0.47      0.52      2760
        completed       0.89      0.92      0.91     12984
    
         accuracy                           0.84     15744
        macro avg       0.73      0.70      0.71     15744
     weighted avg       0.84      0.84      0.84     15744
    
    DecisionTreeClassifier()
    0.9647013487475915 (Precision)
    0.9640326555760936 (Recall)
    0.9412474593495935 (Accuracy)
    [[ 2302   458]
     [  467 12517]]
                   precision    recall  f1-score   support
    
    not completed       0.83      0.83      0.83      2760
        completed       0.96      0.96      0.96     12984
    
         accuracy                           0.94     15744
        macro avg       0.90      0.90      0.90     15744
     weighted avg       0.94      0.94      0.94     15744
    
    RandomForestClassifier()
    0.9216858576911722 (Precision)
    0.9617221195317314 (Recall)
    0.9010416666666666 (Accuracy)
    [[ 1699  1061]
     [  497 12487]]
                   precision    recall  f1-score   support
    
    not completed       0.77      0.62      0.69      2760
        completed       0.92      0.96      0.94     12984
    
         accuracy                           0.90     15744
        macro avg       0.85      0.79      0.81     15744
     weighted avg       0.90      0.90      0.90     15744
    
    GradientBoostingClassifier()
    0.8465509150633506 (Precision)
    0.9725816389402341 (Recall)
    0.8319994918699187 (Accuracy)
    [[  471  2289]
     [  356 12628]]
                   precision    recall  f1-score   support
    
    not completed       0.57      0.17      0.26      2760
        completed       0.85      0.97      0.91     12984
    
         accuracy                           0.83     15744
        macro avg       0.71      0.57      0.58     15744
     weighted avg       0.80      0.83      0.79     15744
    


DecisionTree looks promising as a classifier, thus we select this one to run GridSearchCV for tuning hyperparameters:


```python
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}  # parameter taken from: https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
```

    Fitting 3 folds for each of 294 candidates, totalling 882 fits





    GridSearchCV(cv=3, estimator=DecisionTreeClassifier(random_state=42),
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 verbose=1)




```python
grid_search_cv.best_estimator_
```




    DecisionTreeClassifier(max_leaf_nodes=96, random_state=42)




```python
pipeline.set_params(clf = grid_search_cv.best_estimator_)
pipeline.fit(X_train, y_train)
y_preds = pipeline.predict(X_test)
score_model(y_test, y_preds)
```

    0.8527121191419593 (Precision)
    0.9613370301910044 (Recall)
    0.8311737804878049 (Accuracy)
    [[  604  2156]
     [  502 12482]]
                   precision    recall  f1-score   support
    
    not completed       0.55      0.22      0.31      2760
        completed       0.85      0.96      0.90     12984
    
         accuracy                           0.83     15744
        macro avg       0.70      0.59      0.61     15744
     weighted avg       0.80      0.83      0.80     15744
    



```python
pipeline.set_params(clf = DecisionTreeClassifier())
model = pipeline.fit(X_train, y_train)
y_preds = pipeline.predict(X_test)
score_model(y_test, y_preds)
```

    0.9645963210959747 (Precision)
    0.9652649414664202 (Recall)
    0.9421366869918699 (Accuracy)
    [[ 2300   460]
     [  451 12533]]
                   precision    recall  f1-score   support
    
    not completed       0.84      0.83      0.83      2760
        completed       0.96      0.97      0.96     12984
    
         accuracy                           0.94     15744
        macro avg       0.90      0.90      0.90     15744
     weighted avg       0.94      0.94      0.94     15744
    


since the GridSearchCV optimized version of the model scores worse than with the default hyperparameters, we will keep the default model, which scores quite high.

# Addendum


```python
# inspired by: https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
tree_rules = export_text(pipeline['clf'], feature_names=list(X_rel.columns))
print(tree_rules)
```

    |--- membership months <= -0.39
    |   |--- income <= 0.28
    |   |   |--- female <= 0.16
    |   |   |   |--- income <= -0.82
    |   |   |   |   |--- membership months <= -0.90
    |   |   |   |   |   |--- income <= -1.01
    |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |--- income <= -1.15
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- income >  -1.15
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |--- age <= -1.43
    |   |   |   |   |   |   |   |   |   |--- membership months <= -1.05
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- membership months >  -1.05
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |--- age >  -1.43
    |   |   |   |   |   |   |   |   |   |--- income <= -1.06
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 19
    |   |   |   |   |   |   |   |   |   |--- income >  -1.06
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.20
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.20
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |   |   |--- age <= -2.03
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -2.03
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.22
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.22
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |   |   |--- income <= -1.49
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  -1.49
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |--- income <= -1.47
    |   |   |   |   |   |   |   |   |   |--- age <= 0.21
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.61
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- age >  0.21
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -1.47
    |   |   |   |   |   |   |   |   |   |--- age <= 1.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.29
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.29
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |--- age >  1.22
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.30
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.30
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |--- income >  -1.01
    |   |   |   |   |   |   |--- membership months <= -1.05
    |   |   |   |   |   |   |   |--- age <= -1.00
    |   |   |   |   |   |   |   |   |--- income <= -0.96
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -0.96
    |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.17
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.17
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.92
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.92
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |--- age >  -1.00
    |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |--- age <= 0.44
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- age >  0.44
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |--- age <= -0.51
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.51
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.45
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.45
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |--- membership months >  -1.05
    |   |   |   |   |   |   |   |--- age <= 0.82
    |   |   |   |   |   |   |   |   |--- age <= -0.34
    |   |   |   |   |   |   |   |   |   |--- age <= -2.01
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -2.01
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.03
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.03
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -0.34
    |   |   |   |   |   |   |   |   |   |--- age <= 0.30
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- age >  0.30
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.73
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.73
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  0.82
    |   |   |   |   |   |   |   |   |--- age <= 0.99
    |   |   |   |   |   |   |   |   |   |--- income <= -0.92
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  -0.92
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.87
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.87
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.99
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- membership months >  -0.90
    |   |   |   |   |   |--- age <= 1.45
    |   |   |   |   |   |   |--- age <= 0.99
    |   |   |   |   |   |   |   |--- age <= 0.70
    |   |   |   |   |   |   |   |   |--- age <= 0.35
    |   |   |   |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.26
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.26
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.29
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 17
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.29
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 20
    |   |   |   |   |   |   |   |   |--- age >  0.35
    |   |   |   |   |   |   |   |   |   |--- age <= 0.47
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.96
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.96
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- age >  0.47
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.38
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.38
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |--- age >  0.70
    |   |   |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.61
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.61
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.06
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.06
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |--- age >  0.99
    |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |--- age <= 1.39
    |   |   |   |   |   |   |   |   |   |--- age <= 1.10
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.68
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.68
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- age >  1.10
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |--- age >  1.39
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  1.45
    |   |   |   |   |   |   |--- income <= -1.29
    |   |   |   |   |   |   |   |--- income <= -1.42
    |   |   |   |   |   |   |   |   |--- income <= -1.61
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -1.61
    |   |   |   |   |   |   |   |   |   |--- age <= 2.17
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  2.17
    |   |   |   |   |   |   |   |   |   |   |--- age <= 2.37
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  2.37
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- income >  -1.42
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  -1.29
    |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |--- membership months <= -0.61
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  -0.61
    |   |   |   |   |   |   |   |   |   |--- income <= -1.15
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -1.15
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |--- income >  -0.82
    |   |   |   |   |--- membership months <= -0.76
    |   |   |   |   |   |--- income <= -0.09
    |   |   |   |   |   |   |--- age <= -1.66
    |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |--- age <= -1.72
    |   |   |   |   |   |   |   |   |   |--- age <= -1.77
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.77
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.34
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.34
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -1.72
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |--- income <= -0.69
    |   |   |   |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.77
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.77
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -0.69
    |   |   |   |   |   |   |   |   |   |--- income <= -0.59
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -0.59
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.46
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.46
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |--- age >  -1.66
    |   |   |   |   |   |   |   |--- age <= 0.01
    |   |   |   |   |   |   |   |   |--- income <= -0.36
    |   |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.68
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.68
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |--- income >  -0.36
    |   |   |   |   |   |   |   |   |   |--- income <= -0.27
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.31
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.31
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- income >  -0.27
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |--- age >  0.01
    |   |   |   |   |   |   |   |   |--- income <= -0.46
    |   |   |   |   |   |   |   |   |   |--- age <= 2.43
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.50
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 13
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.50
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- age >  2.43
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -0.46
    |   |   |   |   |   |   |   |   |   |--- income <= -0.32
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 11
    |   |   |   |   |   |   |   |   |   |--- income >  -0.32
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 14
    |   |   |   |   |   |--- income >  -0.09
    |   |   |   |   |   |   |--- age <= -0.28
    |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |--- age <= -0.57
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |--- age >  -0.57
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.05
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.05
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |--- income <= 0.14
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  0.14
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.90
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.90
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |--- age <= -1.40
    |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |--- age >  -1.40
    |   |   |   |   |   |   |   |   |   |--- income <= 0.19
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.62
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.62
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |--- income >  0.19
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- age >  -0.28
    |   |   |   |   |   |   |   |--- income <= 0.05
    |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |--- age <= 0.35
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.11
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.11
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- age >  0.35
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |--- age <= 0.41
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.41
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.93
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.93
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |--- income >  0.05
    |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |--- age <= 1.68
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- age >  1.68
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.21
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.21
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- membership months >  -0.76
    |   |   |   |   |   |--- age <= -1.20
    |   |   |   |   |   |   |--- income <= 0.24
    |   |   |   |   |   |   |   |--- age <= -2.06
    |   |   |   |   |   |   |   |   |--- membership months <= -0.68
    |   |   |   |   |   |   |   |   |   |--- income <= -0.55
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -0.55
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  -0.68
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -2.06
    |   |   |   |   |   |   |   |   |--- age <= -2.01
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.55
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.55
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -2.01
    |   |   |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.06
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.06
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |--- income >  0.24
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- age >  -1.20
    |   |   |   |   |   |   |--- age <= 1.91
    |   |   |   |   |   |   |   |--- age <= -1.03
    |   |   |   |   |   |   |   |   |--- income <= -0.59
    |   |   |   |   |   |   |   |   |   |--- income <= -0.64
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.61
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -0.64
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -0.59
    |   |   |   |   |   |   |   |   |   |--- income <= 0.14
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  0.14
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  -1.03
    |   |   |   |   |   |   |   |   |--- age <= 1.45
    |   |   |   |   |   |   |   |   |   |--- age <= 0.53
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.32
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.32
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 14
    |   |   |   |   |   |   |   |   |   |--- age >  0.53
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |--- age >  1.45
    |   |   |   |   |   |   |   |   |   |--- age <= 1.79
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  1.79
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- age >  1.91
    |   |   |   |   |   |   |   |--- income <= -0.20
    |   |   |   |   |   |   |   |   |--- age <= 2.20
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  2.20
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.65
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.65
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- income >  -0.20
    |   |   |   |   |   |   |   |   |--- income <= -0.09
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -0.09
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |--- female >  0.16
    |   |   |   |--- membership months <= -0.83
    |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |--- income <= -0.59
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- income >  -0.59
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |--- income <= -1.56
    |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |--- age <= -1.52
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -1.52
    |   |   |   |   |   |   |   |   |--- age <= 0.35
    |   |   |   |   |   |   |   |   |   |--- age <= -0.25
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- age >  -0.25
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.35
    |   |   |   |   |   |   |   |   |   |--- age <= 0.87
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.87
    |   |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- income >  -1.56
    |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |--- income <= -0.09
    |   |   |   |   |   |   |   |   |--- income <= -0.32
    |   |   |   |   |   |   |   |   |   |--- income <= -1.29
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.62
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.62
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 11
    |   |   |   |   |   |   |   |   |   |--- income >  -1.29
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.15
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.15
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 23
    |   |   |   |   |   |   |   |   |--- income >  -0.32
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.90
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.90
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.23
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.23
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- income >  -0.09
    |   |   |   |   |   |   |   |   |--- age <= 0.58
    |   |   |   |   |   |   |   |   |   |--- age <= -0.05
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.37
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.37
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |   |   |--- age >  -0.05
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.27
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.27
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |--- age >  0.58
    |   |   |   |   |   |   |   |   |   |--- age <= 1.39
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.02
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.02
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- age >  1.39
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |--- membership months >  -0.83
    |   |   |   |   |--- income <= -0.64
    |   |   |   |   |   |--- age <= 1.62
    |   |   |   |   |   |   |--- income <= -1.61
    |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |--- age <= -0.88
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.88
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |--- age <= -0.16
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.16
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.27
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.27
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- income >  -1.61
    |   |   |   |   |   |   |   |--- income <= -1.33
    |   |   |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |   |   |--- age <= 0.70
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |--- age >  0.70
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.42
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.42
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |   |   |--- age <= 1.42
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  1.42
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.51
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.51
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  -1.33
    |   |   |   |   |   |   |   |   |--- age <= 1.33
    |   |   |   |   |   |   |   |   |   |--- age <= -0.85
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |--- age >  -0.85
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |--- age >  1.33
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- age >  1.62
    |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |--- income <= -1.08
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -1.08
    |   |   |   |   |   |   |   |   |   |--- income <= -0.92
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.91
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.91
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  -0.92
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |--- age <= 1.76
    |   |   |   |   |   |   |   |   |   |--- age <= 1.68
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  1.68
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  1.76
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.68
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.99
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.99
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.68
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- income >  -0.64
    |   |   |   |   |   |--- age <= 2.08
    |   |   |   |   |   |   |--- age <= -1.72
    |   |   |   |   |   |   |   |--- income <= -0.41
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  -0.41
    |   |   |   |   |   |   |   |   |--- income <= -0.18
    |   |   |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.77
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.77
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -0.18
    |   |   |   |   |   |   |   |   |   |--- age <= -2.06
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -2.06
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.65
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.65
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |--- age >  -1.72
    |   |   |   |   |   |   |   |--- income <= 0.10
    |   |   |   |   |   |   |   |   |--- membership months <= -0.61
    |   |   |   |   |   |   |   |   |   |--- age <= 0.93
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.64
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.64
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- age >  0.93
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.10
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.10
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |--- membership months >  -0.61
    |   |   |   |   |   |   |   |   |   |--- age <= 1.71
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |   |   |--- age >  1.71
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.05
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.05
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- income >  0.10
    |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |--- age <= -0.05
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |--- age >  -0.05
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.44
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.44
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |--- age <= -0.77
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.77
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |--- age >  2.08
    |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |--- age <= 2.40
    |   |   |   |   |   |   |   |   |--- age <= 2.25
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.46
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.46
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  2.25
    |   |   |   |   |   |   |   |   |   |--- income <= -0.36
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -0.36
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  2.40
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |--- age <= 2.17
    |   |   |   |   |   |   |   |   |--- income <= -0.23
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -0.23
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  2.17
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |--- income >  0.28
    |   |   |--- age <= -0.39
    |   |   |   |--- female <= 0.16
    |   |   |   |   |--- income <= 0.47
    |   |   |   |   |   |--- membership months <= -0.68
    |   |   |   |   |   |   |--- age <= -0.74
    |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |--- age <= -1.77
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- age >  -1.77
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.72
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.72
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |--- age <= -1.66
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.66
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |--- age <= -1.14
    |   |   |   |   |   |   |   |   |   |--- income <= 0.33
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- income >  0.33
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -1.14
    |   |   |   |   |   |   |   |   |   |--- age <= -0.85
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- age >  -0.85
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -0.74
    |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.35
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.35
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.90
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.90
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |--- age <= -0.54
    |   |   |   |   |   |   |   |   |   |--- age <= -0.68
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.68
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -0.54
    |   |   |   |   |   |   |   |   |   |--- membership months <= -1.01
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  -1.01
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.33
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.33
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- membership months >  -0.68
    |   |   |   |   |   |   |--- age <= -0.91
    |   |   |   |   |   |   |   |--- income <= 0.33
    |   |   |   |   |   |   |   |   |--- age <= -1.86
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -1.86
    |   |   |   |   |   |   |   |   |   |--- age <= -1.40
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.40
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  0.33
    |   |   |   |   |   |   |   |   |--- age <= -1.83
    |   |   |   |   |   |   |   |   |   |--- age <= -2.06
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -2.06
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.98
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.98
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -1.83
    |   |   |   |   |   |   |   |   |   |--- age <= -1.75
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.75
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.66
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.66
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |--- age >  -0.91
    |   |   |   |   |   |   |   |--- age <= -0.57
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -0.57
    |   |   |   |   |   |   |   |   |--- membership months <= -0.54
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  -0.54
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- income >  0.47
    |   |   |   |   |   |--- age <= -0.45
    |   |   |   |   |   |   |--- membership months <= -1.05
    |   |   |   |   |   |   |   |--- age <= -0.57
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -0.57
    |   |   |   |   |   |   |   |   |--- income <= 0.61
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  0.61
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- membership months >  -1.05
    |   |   |   |   |   |   |   |--- age <= -1.03
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  -1.03
    |   |   |   |   |   |   |   |   |--- income <= 0.56
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  0.56
    |   |   |   |   |   |   |   |   |   |--- income <= 1.25
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.11
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.11
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- income >  1.25
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.44
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.44
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  -0.45
    |   |   |   |   |   |   |--- income <= 0.95
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  0.95
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |--- female >  0.16
    |   |   |   |   |--- membership months <= -0.76
    |   |   |   |   |   |--- income <= 1.44
    |   |   |   |   |   |   |--- income <= 1.07
    |   |   |   |   |   |   |   |--- age <= -1.40
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -1.40
    |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |--- age <= -0.71
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.71
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |--- income <= 0.79
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.05
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.05
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |--- income >  0.79
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- income >  1.07
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- income >  1.44
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- membership months >  -0.76
    |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |--- income <= 0.33
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- income >  0.33
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |--- membership months <= -0.61
    |   |   |   |   |   |   |   |--- membership months <= -0.68
    |   |   |   |   |   |   |   |   |--- income <= 0.35
    |   |   |   |   |   |   |   |   |   |--- age <= -0.83
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.31
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.31
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.83
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  0.35
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  -0.68
    |   |   |   |   |   |   |   |   |--- income <= 1.11
    |   |   |   |   |   |   |   |   |   |--- age <= -1.60
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.75
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.75
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -1.60
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  1.11
    |   |   |   |   |   |   |   |   |   |--- income <= 1.32
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  1.32
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- membership months >  -0.61
    |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |--- income <= 0.37
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  0.37
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |--- age >  -0.39
    |   |   |   |--- membership months <= -1.05
    |   |   |   |   |--- income <= 0.51
    |   |   |   |   |   |--- income <= 0.47
    |   |   |   |   |   |   |--- age <= -0.11
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -0.11
    |   |   |   |   |   |   |   |--- age <= 1.45
    |   |   |   |   |   |   |   |   |--- age <= 0.87
    |   |   |   |   |   |   |   |   |   |--- age <= 0.07
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.07
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.44
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.44
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |--- age >  0.87
    |   |   |   |   |   |   |   |   |   |--- income <= 0.37
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  0.37
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  1.45
    |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |--- age <= 1.76
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.35
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.35
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  1.76
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- income >  0.47
    |   |   |   |   |   |   |--- age <= 0.58
    |   |   |   |   |   |   |   |--- age <= -0.08
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  -0.08
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  0.58
    |   |   |   |   |   |   |   |--- membership months <= -1.19
    |   |   |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |   |   |--- age <= 0.67
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.67
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  -1.19
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |--- income >  0.51
    |   |   |   |   |   |--- age <= 1.85
    |   |   |   |   |   |   |--- age <= 1.79
    |   |   |   |   |   |   |   |--- income <= 0.79
    |   |   |   |   |   |   |   |   |--- age <= 0.24
    |   |   |   |   |   |   |   |   |   |--- income <= 0.65
    |   |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- income >  0.65
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.24
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  0.79
    |   |   |   |   |   |   |   |   |--- membership months <= -1.12
    |   |   |   |   |   |   |   |   |   |--- income <= 1.94
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.81
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.81
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |--- income >  1.94
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -1.12
    |   |   |   |   |   |   |   |   |   |--- age <= -0.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= 2.17
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  2.17
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.84
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.84
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |--- age >  1.79
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- age >  1.85
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |--- membership months >  -1.05
    |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |--- membership months <= -0.90
    |   |   |   |   |   |   |--- income <= 1.25
    |   |   |   |   |   |   |   |--- income <= 1.02
    |   |   |   |   |   |   |   |   |--- income <= 0.84
    |   |   |   |   |   |   |   |   |   |--- income <= 0.74
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.18
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.18
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |--- income >  0.74
    |   |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |--- income >  0.84
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  1.02
    |   |   |   |   |   |   |   |   |--- age <= -0.11
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -0.11
    |   |   |   |   |   |   |   |   |   |--- age <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  1.25
    |   |   |   |   |   |   |   |--- income <= 1.46
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  1.46
    |   |   |   |   |   |   |   |   |--- age <= 0.07
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.07
    |   |   |   |   |   |   |   |   |   |--- income <= 1.53
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  1.53
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.18
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.18
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |--- membership months >  -0.90
    |   |   |   |   |   |   |--- income <= 0.70
    |   |   |   |   |   |   |   |--- age <= 2.37
    |   |   |   |   |   |   |   |   |--- age <= 1.22
    |   |   |   |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 15
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.61
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.61
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |--- age >  1.22
    |   |   |   |   |   |   |   |   |   |--- age <= 1.85
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.51
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.51
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  1.85
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.99
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.99
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  2.37
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- income >  0.70
    |   |   |   |   |   |   |   |--- income <= 1.16
    |   |   |   |   |   |   |   |   |--- age <= -0.16
    |   |   |   |   |   |   |   |   |   |--- age <= -0.28
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.28
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |--- age >  -0.16
    |   |   |   |   |   |   |   |   |   |--- age <= 0.35
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- age >  0.35
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.41
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.41
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |--- income >  1.16
    |   |   |   |   |   |   |   |   |--- income <= 1.25
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.07
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.07
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  1.25
    |   |   |   |   |   |   |   |   |   |--- income <= 2.13
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.12
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |--- income >  2.13
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |--- age <= -0.34
    |   |   |   |   |   |   |--- income <= 1.37
    |   |   |   |   |   |   |   |--- membership months <= -0.47
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  -0.47
    |   |   |   |   |   |   |   |   |--- income <= 0.86
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  0.86
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  1.37
    |   |   |   |   |   |   |   |--- income <= 1.53
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- income >  1.53
    |   |   |   |   |   |   |   |   |--- membership months <= -0.83
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  -0.83
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  -0.34
    |   |   |   |   |   |   |--- income <= 0.51
    |   |   |   |   |   |   |   |--- age <= 1.22
    |   |   |   |   |   |   |   |   |--- age <= 0.99
    |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 11
    |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.99
    |   |   |   |   |   |   |   |   |   |--- income <= 0.47
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.16
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.16
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  0.47
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  1.22
    |   |   |   |   |   |   |   |   |--- membership months <= -0.76
    |   |   |   |   |   |   |   |   |   |--- income <= 0.37
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.53
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.53
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- income >  0.37
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -0.76
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  0.51
    |   |   |   |   |   |   |   |--- age <= 1.79
    |   |   |   |   |   |   |   |   |--- age <= 1.28
    |   |   |   |   |   |   |   |   |   |--- age <= 1.22
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 17
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 11
    |   |   |   |   |   |   |   |   |   |--- age >  1.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.71
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.71
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  1.28
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.88
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.88
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.69
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.69
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |--- age >  1.79
    |   |   |   |   |   |   |   |   |--- age <= 1.97
    |   |   |   |   |   |   |   |   |   |--- income <= 2.11
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.79
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.79
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- income >  2.11
    |   |   |   |   |   |   |   |   |   |   |--- income <= 2.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  2.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  1.97
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.97
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.97
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.90
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.90
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |--- membership months >  -0.39
    |   |--- income <= -0.23
    |   |   |--- membership months <= 1.35
    |   |   |   |--- membership months <= -0.32
    |   |   |   |   |--- income <= -1.56
    |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |--- age <= -0.45
    |   |   |   |   |   |   |   |--- age <= -1.08
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  -1.08
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -0.45
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- income >  -1.56
    |   |   |   |   |   |--- age <= -1.72
    |   |   |   |   |   |   |--- income <= -1.06
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  -1.06
    |   |   |   |   |   |   |   |--- age <= -2.01
    |   |   |   |   |   |   |   |   |--- income <= -0.46
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -0.46
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  -2.01
    |   |   |   |   |   |   |   |   |--- income <= -0.64
    |   |   |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -0.64
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- age >  -1.72
    |   |   |   |   |   |   |--- age <= -1.08
    |   |   |   |   |   |   |   |--- age <= -1.54
    |   |   |   |   |   |   |   |   |--- income <= -1.42
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -1.42
    |   |   |   |   |   |   |   |   |   |--- income <= -1.15
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.33
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.33
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  -1.15
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -1.54
    |   |   |   |   |   |   |   |   |--- income <= -0.62
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  -0.62
    |   |   |   |   |   |   |   |   |   |--- income <= -0.39
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  -0.39
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -1.08
    |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |--- age <= -0.74
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.42
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.42
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- age >  -0.74
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.82
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 13
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.82
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |--- age <= -0.97
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.97
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.87
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.87
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |--- age <= 1.13
    |   |   |   |   |   |   |   |   |   |--- age <= -0.54
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.62
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.62
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.54
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  1.13
    |   |   |   |   |   |   |   |   |   |--- age <= 1.36
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  1.36
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |--- membership months >  -0.32
    |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |--- age <= 1.28
    |   |   |   |   |   |   |   |   |--- income <= -1.47
    |   |   |   |   |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 0.55
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  0.55
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -1.47
    |   |   |   |   |   |   |   |   |   |--- age <= -0.62
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.36
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 14
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.36
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |--- age >  -0.62
    |   |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 11
    |   |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 12
    |   |   |   |   |   |   |   |--- age >  1.28
    |   |   |   |   |   |   |   |   |--- age <= 1.33
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.95
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.45
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.45
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  1.33
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.12
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.87
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.87
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.12
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.99
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.99
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |--- membership months <= 0.48
    |   |   |   |   |   |   |   |   |--- age <= -0.51
    |   |   |   |   |   |   |   |   |   |--- age <= -1.49
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.49
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.10
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.10
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |--- age >  -0.51
    |   |   |   |   |   |   |   |   |   |--- age <= 0.93
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.47
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.47
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.93
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.36
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.36
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |--- membership months >  0.48
    |   |   |   |   |   |   |   |   |--- age <= -1.03
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.28
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.28
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.89
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.89
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -1.03
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.55
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.15
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.15
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.55
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 14
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |--- age <= -2.01
    |   |   |   |   |   |   |   |--- membership months <= 0.55
    |   |   |   |   |   |   |   |   |--- membership months <= 0.44
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  0.44
    |   |   |   |   |   |   |   |   |   |--- age <= -2.06
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -2.06
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  0.55
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -2.01
    |   |   |   |   |   |   |   |--- age <= 0.18
    |   |   |   |   |   |   |   |   |--- age <= -1.49
    |   |   |   |   |   |   |   |   |   |--- age <= -1.54
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.54
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.01
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -1.49
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  0.18
    |   |   |   |   |   |   |   |   |--- age <= 0.58
    |   |   |   |   |   |   |   |   |   |--- age <= 0.53
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.56
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.56
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- age >  0.53
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.14
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |--- age >  0.58
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.13
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.13
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.55
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.55
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |--- income <= -1.52
    |   |   |   |   |   |   |--- membership months <= -0.18
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- membership months >  -0.18
    |   |   |   |   |   |   |   |--- membership months <= 0.33
    |   |   |   |   |   |   |   |   |--- age <= -0.11
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.26
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.39
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.26
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -0.11
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  0.33
    |   |   |   |   |   |   |   |   |--- age <= 1.07
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  1.07
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.84
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.84
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- income >  -1.52
    |   |   |   |   |   |   |--- membership months <= 0.77
    |   |   |   |   |   |   |   |--- income <= -0.96
    |   |   |   |   |   |   |   |   |--- age <= -2.01
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.41
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.41
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -2.01
    |   |   |   |   |   |   |   |   |   |--- membership months <= -0.25
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.60
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.60
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  -0.25
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |--- income >  -0.96
    |   |   |   |   |   |   |   |   |--- membership months <= -0.18
    |   |   |   |   |   |   |   |   |   |--- age <= 0.61
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.61
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.73
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.73
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  -0.18
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- membership months >  0.77
    |   |   |   |   |   |   |   |--- income <= -1.42
    |   |   |   |   |   |   |   |   |--- membership months <= 0.84
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  0.84
    |   |   |   |   |   |   |   |   |   |--- age <= 0.56
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.56
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.07
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.07
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  -1.42
    |   |   |   |   |   |   |   |   |--- age <= -0.68
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.84
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.78
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.78
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.84
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.80
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.80
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |--- age >  -0.68
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.06
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.06
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.51
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.51
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |--- membership months >  1.35
    |   |   |   |--- income <= -1.24
    |   |   |   |   |--- income <= -1.47
    |   |   |   |   |   |--- income <= -1.56
    |   |   |   |   |   |   |--- age <= -1.29
    |   |   |   |   |   |   |   |--- membership months <= 2.36
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  2.36
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -1.29
    |   |   |   |   |   |   |   |--- membership months <= 1.46
    |   |   |   |   |   |   |   |   |--- age <= -0.05
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -0.05
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  1.46
    |   |   |   |   |   |   |   |   |--- age <= -0.62
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -0.62
    |   |   |   |   |   |   |   |   |   |--- age <= -0.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.61
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.61
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- income >  -1.56
    |   |   |   |   |   |   |--- membership months <= 2.87
    |   |   |   |   |   |   |   |--- age <= -0.68
    |   |   |   |   |   |   |   |   |--- age <= -0.77
    |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.20
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.20
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -0.77
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.97
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.97
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  -0.68
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- membership months >  2.87
    |   |   |   |   |   |   |   |--- membership months <= 2.95
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  2.95
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- income >  -1.47
    |   |   |   |   |   |--- age <= -0.91
    |   |   |   |   |   |   |--- age <= -1.54
    |   |   |   |   |   |   |   |--- membership months <= 2.51
    |   |   |   |   |   |   |   |   |--- membership months <= 2.00
    |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.83
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.83
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  2.00
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.75
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.75
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  2.51
    |   |   |   |   |   |   |   |   |--- membership months <= 2.87
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  2.87
    |   |   |   |   |   |   |   |   |   |--- age <= -1.89
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -1.89
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  -1.54
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  -0.91
    |   |   |   |   |   |   |--- membership months <= 2.80
    |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |--- age <= 0.01
    |   |   |   |   |   |   |   |   |   |--- age <= -0.11
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.68
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.68
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- age >  -0.11
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.01
    |   |   |   |   |   |   |   |   |   |--- age <= 1.33
    |   |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- age >  1.33
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |--- age <= 0.24
    |   |   |   |   |   |   |   |   |   |--- age <= 0.01
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.82
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.82
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |--- age >  0.01
    |   |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.24
    |   |   |   |   |   |   |   |   |   |--- income <= -1.38
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 2.33
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  2.33
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -1.38
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.29
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.29
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |--- membership months >  2.80
    |   |   |   |   |   |   |   |--- age <= 1.10
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  1.10
    |   |   |   |   |   |   |   |   |--- age <= 1.42
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  1.42
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |--- income >  -1.24
    |   |   |   |   |--- income <= -0.27
    |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |--- income <= -0.73
    |   |   |   |   |   |   |   |--- income <= -1.01
    |   |   |   |   |   |   |   |   |--- age <= 1.42
    |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |--- income <= -1.06
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- income >  -1.06
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.53
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.53
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |--- age >  1.42
    |   |   |   |   |   |   |   |   |   |--- income <= -1.10
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.71
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.71
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  -1.10
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  -1.01
    |   |   |   |   |   |   |   |   |--- age <= -1.66
    |   |   |   |   |   |   |   |   |   |--- income <= -0.87
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.80
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.80
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- income >  -0.87
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.42
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.42
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -1.66
    |   |   |   |   |   |   |   |   |   |--- income <= -0.92
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 2.22
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  2.22
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |--- income >  -0.92
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.31
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.31
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |--- income >  -0.73
    |   |   |   |   |   |   |   |--- income <= -0.46
    |   |   |   |   |   |   |   |   |--- membership months <= 2.07
    |   |   |   |   |   |   |   |   |   |--- age <= -0.31
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.31
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.22
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.22
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |--- membership months >  2.07
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.29
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.56
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.56
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.29
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 2.73
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  2.73
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |--- income >  -0.46
    |   |   |   |   |   |   |   |   |--- membership months <= 2.44
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.07
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.85
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.85
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.07
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.36
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.36
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |--- membership months >  2.44
    |   |   |   |   |   |   |   |   |   |--- age <= -1.60
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.41
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.41
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.60
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.67
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.67
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |--- membership months <= 2.29
    |   |   |   |   |   |   |   |--- membership months <= 2.15
    |   |   |   |   |   |   |   |   |--- income <= -1.19
    |   |   |   |   |   |   |   |   |   |--- age <= -0.97
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.97
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -1.19
    |   |   |   |   |   |   |   |   |   |--- age <= -0.97
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.03
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.03
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.97
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  2.15
    |   |   |   |   |   |   |   |   |--- age <= 0.84
    |   |   |   |   |   |   |   |   |   |--- age <= -0.65
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.65
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.84
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- membership months >  2.29
    |   |   |   |   |   |   |   |--- age <= -1.63
    |   |   |   |   |   |   |   |   |--- income <= -0.87
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  -0.87
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -1.63
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- income >  -0.27
    |   |   |   |   |   |--- age <= 0.47
    |   |   |   |   |   |   |--- membership months <= 1.42
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- membership months >  1.42
    |   |   |   |   |   |   |   |--- membership months <= 2.51
    |   |   |   |   |   |   |   |   |--- membership months <= 2.07
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.93
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.93
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  2.07
    |   |   |   |   |   |   |   |   |   |--- age <= -1.92
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.92
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  2.51
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  0.47
    |   |   |   |   |   |   |--- class: 1
    |   |--- income >  -0.23
    |   |   |--- membership months <= -0.32
    |   |   |   |--- income <= 0.37
    |   |   |   |   |--- income <= -0.18
    |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- income >  -0.18
    |   |   |   |   |   |--- income <= 0.14
    |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |--- age <= -0.45
    |   |   |   |   |   |   |   |   |--- age <= -1.46
    |   |   |   |   |   |   |   |   |   |--- age <= -1.54
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.54
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -1.46
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  -0.45
    |   |   |   |   |   |   |   |   |--- age <= 0.27
    |   |   |   |   |   |   |   |   |   |--- age <= 0.07
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.19
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.19
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.07
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.27
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.50
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.50
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |--- age <= 0.15
    |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.08
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.08
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |--- income <= 0.03
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- income >  0.03
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.36
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.36
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- age >  0.15
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- income >  0.14
    |   |   |   |   |   |   |--- age <= -1.86
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- age >  -1.86
    |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |--- age <= 0.47
    |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.19
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.47
    |   |   |   |   |   |   |   |   |   |--- age <= 0.53
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  0.53
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.45
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.45
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |--- income <= 0.24
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  0.24
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |--- income >  0.37
    |   |   |   |   |--- age <= -0.65
    |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |--- income <= 0.70
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  0.70
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- age >  -0.65
    |   |   |   |   |   |--- age <= 1.74
    |   |   |   |   |   |   |--- income <= 0.54
    |   |   |   |   |   |   |   |--- age <= 0.73
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  0.73
    |   |   |   |   |   |   |   |   |--- age <= 0.96
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.96
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  0.54
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  1.74
    |   |   |   |   |   |   |--- age <= 1.82
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |--- age >  1.82
    |   |   |   |   |   |   |   |--- class: 1
    |   |   |--- membership months >  -0.32
    |   |   |   |--- membership months <= 1.35
    |   |   |   |   |--- age <= 2.43
    |   |   |   |   |   |--- age <= -0.97
    |   |   |   |   |   |   |--- age <= -1.20
    |   |   |   |   |   |   |   |--- membership months <= -0.10
    |   |   |   |   |   |   |   |   |--- income <= 0.10
    |   |   |   |   |   |   |   |   |   |--- income <= 0.03
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  0.03
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= -0.18
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  -0.18
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  0.10
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  -0.10
    |   |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.20
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.20
    |   |   |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |   |--- age <= -1.26
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.26
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  0.59
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |--- age >  -1.20
    |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |--- income <= 0.47
    |   |   |   |   |   |   |   |   |   |--- income <= 0.42
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |--- income >  0.42
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 0.70
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  0.70
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- income >  0.47
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |--- age <= -1.08
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.66
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.66
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -1.08
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.22
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.22
    |   |   |   |   |   |   |   |   |   |   |--- age <= -1.03
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- age >  -1.03
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- age >  -0.97
    |   |   |   |   |   |   |--- membership months <= 0.91
    |   |   |   |   |   |   |   |--- membership months <= 0.70
    |   |   |   |   |   |   |   |   |--- membership months <= 0.55
    |   |   |   |   |   |   |   |   |   |--- discount 0.25 <= 1.13
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.22
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.22
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 13
    |   |   |   |   |   |   |   |   |   |--- discount 0.25 >  1.13
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 10
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.24
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 6
    |   |   |   |   |   |   |   |   |--- membership months >  0.55
    |   |   |   |   |   |   |   |   |   |--- membership months <= 0.62
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.85
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 9
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.85
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- membership months >  0.62
    |   |   |   |   |   |   |   |   |   |   |--- age <= 1.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- age >  1.39
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |--- membership months >  0.70
    |   |   |   |   |   |   |   |   |--- age <= -0.22
    |   |   |   |   |   |   |   |   |   |--- age <= -0.28
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -0.28
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  -0.22
    |   |   |   |   |   |   |   |   |   |--- income <= 1.71
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- income >  1.71
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.76
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.76
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- membership months >  0.91
    |   |   |   |   |   |   |   |--- income <= -0.18
    |   |   |   |   |   |   |   |   |--- membership months <= 0.99
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- membership months >  0.99
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 <= 0.58
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- discount 0.20 >  0.58
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.31
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.31
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  -0.18
    |   |   |   |   |   |   |   |   |--- income <= 1.25
    |   |   |   |   |   |   |   |   |   |--- membership months <= 1.20
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.10
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.10
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 13
    |   |   |   |   |   |   |   |   |   |--- membership months >  1.20
    |   |   |   |   |   |   |   |   |   |   |--- income <= 1.21
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 7
    |   |   |   |   |   |   |   |   |   |   |--- income >  1.21
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |--- income >  1.25
    |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |--- income <= 2.17
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- income >  2.17
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.35
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.35
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- age >  2.43
    |   |   |   |   |   |--- membership months <= 0.19
    |   |   |   |   |   |   |--- membership months <= 0.08
    |   |   |   |   |   |   |   |--- income <= 0.10
    |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- income >  0.10
    |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- membership months >  0.08
    |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- membership months >  0.19
    |   |   |   |   |   |   |--- class: 1
    |   |   |   |--- membership months >  1.35
    |   |   |   |   |--- age <= 2.23
    |   |   |   |   |   |--- income <= 0.24
    |   |   |   |   |   |   |--- age <= 0.76
    |   |   |   |   |   |   |   |--- membership months <= 1.42
    |   |   |   |   |   |   |   |   |--- age <= 0.18
    |   |   |   |   |   |   |   |   |   |--- age <= -1.95
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 <= 1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- discount 0.43 >  1.14
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  -1.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.18
    |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |--- membership months >  1.42
    |   |   |   |   |   |   |   |   |--- income <= 0.19
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.95
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.49
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.49
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 8
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.95
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.39
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.39
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- income >  0.19
    |   |   |   |   |   |   |   |   |   |--- bogo <= 0.00
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- bogo >  0.00
    |   |   |   |   |   |   |   |   |   |   |--- age <= 0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 5
    |   |   |   |   |   |   |   |   |   |   |--- age >  0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- age >  0.76
    |   |   |   |   |   |   |   |--- age <= 1.05
    |   |   |   |   |   |   |   |   |--- age <= 0.93
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.87
    |   |   |   |   |   |   |   |   |   |   |--- income <= -0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
    |   |   |   |   |   |   |   |   |   |   |--- income >  -0.04
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 3
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.87
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |--- age >  0.93
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.04
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.12
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.12
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.04
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- age >  1.05
    |   |   |   |   |   |   |   |   |--- membership months <= 2.07
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- membership months >  2.07
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.15
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.15
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |--- income >  0.24
    |   |   |   |   |   |   |--- income <= 0.65
    |   |   |   |   |   |   |   |--- income <= 0.61
    |   |   |   |   |   |   |   |   |--- age <= 1.16
    |   |   |   |   |   |   |   |   |   |--- age <= -0.39
    |   |   |   |   |   |   |   |   |   |   |--- age <= -0.45
    |   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 4
    |   |   |   |   |   |   |   |   |   |   |--- age >  -0.45
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- age >  -0.39
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  1.16
    |   |   |   |   |   |   |   |   |   |--- age <= 1.33
    |   |   |   |   |   |   |   |   |   |   |--- income <= 0.44
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- income >  0.44
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  1.33
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- income >  0.61
    |   |   |   |   |   |   |   |   |--- age <= -0.16
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.62
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 2.26
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  2.26
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.62
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  -0.16
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |--- income >  0.65
    |   |   |   |   |   |   |   |--- membership months <= 2.87
    |   |   |   |   |   |   |   |   |--- age <= 0.93
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- age >  0.93
    |   |   |   |   |   |   |   |   |   |--- age <= 0.99
    |   |   |   |   |   |   |   |   |   |   |--- membership months <= 1.57
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |   |--- membership months >  1.57
    |   |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |   |--- age >  0.99
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |--- membership months >  2.87
    |   |   |   |   |   |   |   |   |--- female <= 0.16
    |   |   |   |   |   |   |   |   |   |--- membership months <= 2.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |   |   |   |   |--- membership months >  2.95
    |   |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |   |   |   |   |--- female >  0.16
    |   |   |   |   |   |   |   |   |   |--- class: 1
    |   |   |   |   |--- age >  2.23
    |   |   |   |   |   |--- income <= 1.00
    |   |   |   |   |   |   |--- class: 0
    |   |   |   |   |   |--- income >  1.00
    |   |   |   |   |   |   |--- class: 1
    



```python

```
