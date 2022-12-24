# Udacity_DataScience_Capstone_Starbucks
Udacity DataScience Capstone Project on Starbucks Dataset


## Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Files](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)


# Summary <a name="summary"></a>
Blogpost-Link here


## Motivation and Project Description


## Key Results


## Dataset Description (by Udacity/Starbucks)

The data is contained in three files:

- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**

- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)

**profile.json**

- age (int) - age of the customer
- became\_member\_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

**transcript.json**

- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record



# Installation <a name="installation"></a>
- install packages from provided requirements.txt
- if using udacity workspace, you might have to uninstall and reinstall some packages first, in order to being able to update those outdated packages. Otherwise might get errors when executing the notebook. See first cells of notebook and uncomment, if necessary.
- you may have to restart the kernels once or twice after installations
- installations are necessary in every new session.


# Files <a name=files></a>
All input data is made available in the ´data´ folder, the notebook code is to be found in the ´source´ folder.

A ´requirements.txt´ is also provided to facilitate the installation of necessary packages.

# Licensing, Authors, Acknowledgements<a name="licensing"></a>
The and original project statement are IP of Udacity and Starbucks. 

However, feel free to use the code of the Jupyter Notebook in this repo as you like.
