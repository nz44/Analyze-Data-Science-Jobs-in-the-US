import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn
import collections


df = pd.read_csv('alldata.csv')

#print(df)

##########################################################
# Some descriptive statistics in order to impute NaN in the amount invested
##########################################################

df.company.unique().size
df.location.unique().size
# df.City.unique().size
# df.State.unique().size
# df.Zipcode.unique().size

#######################################################
#                                                                                                                                                               #
#                                                                                                                                                               #
#        Part I Preprocessing                                                                                                                     #
#                                                                                                                                                               #
#       1.1 Descriptive Statistics                                                                                                               #
#                                                                                                                                                               #
#                                                                                                                                                               #
#######################################################



############################################################
# Split the location in to City, State and Zipcode
############################################################

df2 = df

df2[['city', 'state']] = df['location'].str.split(", ", n=1, expand=True)

df3 = df2

df3[['state', 'zipcode']] = df['state'].str.split(" ", n=1, expand=True)



############################################
### Extract information from Locations #####################
############################################

df4 = df3

city = list(df4['city'])


# Print the companies that post the most number of data scientist jobs
# https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0

city_count = collections.Counter(city)


n_print = 30

print()
print('The top 30 cities where most data scientists jobs are located:')
for keyword, count in city_count.most_common(n_print):
    print(keyword, ": ", count)

city_count_sorted = sorted(city_count.items(),  reverse=True, key=lambda city_count: city_count[1])
city_count_sorted =  collections.OrderedDict(city_count_sorted)
# print(city_count_sorted)

#https://stackoverflow.com/questions/21925007/using-dictionary-to-make-a-matplotlib-graph

city_jobs = list(city_count_sorted.values())
city_names = list(city_count_sorted.keys())
sObject = slice(20)

plt.subplot()
plt.xticks(rotation=70)
plt.bar(city_names[sObject], city_jobs[sObject])
# plt.xticks(range(len(city_names[sObject])),)
plt.title('Figure 3. The Cities that Offer the Most Data Science Jobs')
plt.subplots_adjust( bottom=0.35)
plt.savefig('cities_count.png')
plt.show()
############################################################
# Extract State information into a bar chart
##############################################
state = list(df4['state'])

print('The states that appeared in the dataset include:')
print(np.unique(np.array(state)))


print()
print('Take a closer look at those states that are coded "nan":')
df5 = df.loc[pd.isna(df4.state) == True]
print(df5)

print('Since all state == nan is blank page, which I double checked in CSV\
      so I am going to drop rows where state == nan')

print('The shape before dropping state == nan is: ', df4.shape)
df6 = df4.dropna(subset=['state'])
print('The shape after dropping state == nan is: ', df6.shape)


state = list(df6['state'])
state_count = collections.Counter(state)

n_print = 20

print()
print('The number of data science jobs located in the states:')
for keyword, count in state_count.most_common(n_print):
    print(keyword, ": ", count)

state_count_sorted = sorted(state_count.items(),  reverse=True, key=lambda state_count: state_count[1])
state_count_sorted =  collections.OrderedDict(state_count_sorted)


state_jobs = list(state_count_sorted.values())
state_names = list(state_count_sorted.keys())


plt.subplot()
plt.xticks(rotation=30)
plt.bar(state_names,  state_jobs)
plt.title('Figure 4. The States that Offer the Most Data Science Jobs')
plt.subplots_adjust( bottom=0.15)
plt.savefig('states_count.png')
plt.show()


#######################################################
#                                                                                                                                                               #
#                                                                                                                                                               #
#        Part I Preprocessing                                                                                                                     #
#                                                                                                                                                               #
#       1.1 Generate Target for Classification                                                                                          #
#                                                                                                                                                               #
#                                                                                                                                                               #
#######################################################


################################################
###### assiign categorical variables using city and company variables ########
################################################
# In order to make the geographical location a variable that can be classified,
# we need to assign categories to them.

# First I thought of assigning by the number of data science jobs each city could offer
# but I realized that some small cities are located right next to a big city, so geographically they
# should belong to the same metropolitan area.
# Thus, I decide to group cities in term of metropolitan areas.

# According to the wiki-pedia list below
# https://en.wikipedia.org/wiki/List_of_metropolitan_statistical_areas

##################################################
# New York metropolitan area
# Generate a new pandas column called "nymetro" that is 1 if state is NY or NJ, 0 otherwise
# https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition

df8 = df6.copy()
def f(row):
    if row['state'] == 'NY':
        val = 1
    elif row['state']  == 'NJ':
        val = 1
    else:
        val = 0
    return val


df8['NY_metro'] = df8.apply(f, axis=1)


##################################################
# Since we have two metropolitan areas in California, we need to list the cities in CA
# and assign them according to the following lists.
# Los Angeles metropolitan area
# https://en.wikipedia.org/wiki/Los_Angeles_metropolitan_area#Metropolitan_Statistical_Area
# San Francisco bay area
# https://en.wikipedia.org/wiki/San_Francisco%E2%80%93Oakland%E2%80%93Hayward,_CA_Metropolitan_Statistical_Area

CA = ['CA']
df9 = df6[df6.state.isin(CA) ]
city = list(df9['city'])
print()
print('The cities in California:')
CA_cities = list(np.unique(np.array(city)))
print(CA_cities)

# By checking the above cities on Google Map one by one, the following two lists are sorted:

SF_bay_area = ['Alameda', 'Belmont', 'Berkeley', 'Burlingame', 'Daly City', 'Emeryville', \
               'Foster City', 'Fremont', 'Hayward', 'Kentfield', 'Lafayette', 'Martinez', \
               'Menlo Park', 'Mill Valley', 'Mountain View', 'Novato', 'Oakland', 'Pleasant Hill', \
               'Redwood City', 'Richmond', 'San Bruno', 'San Carlos', 'San Francisco', \
               'San Francisco Bay Area', 'San Leandro', 'San Mateo', 'San Rafael', \
               'San Ramon', 'South San Francisco', 'Sunnyvale', 'Union City', 'Walnut Creek']
LA_metro_cities =  ['Los Angeles', 'San Diego']


def f(row):
    if row['city']  in SF_bay_area and row['state'] == 'CA':
        val = 1
    else:
        val = 0
    return val

df8['SF_bay_area'] = df8.apply(f, axis=1)

print()
print('The number of positions located in San Francisco Bay area:')
print(df8[df8['SF_bay_area'] == 1].count())


def f(row):
    if row['city']  in LA_metro_cities and row['state'] == 'CA':
        val = 1
    else:
        val = 0
    return val

df8['LA_metro'] = df8.apply(f, axis=1)


##################################################
# Boston metropolitan area
# https://en.wikipedia.org/wiki/Greater_Boston#Metropolitan_Statistical_Area
# The only state in the dataset that belong to Boston metropolitan area is MA

def f(row):
    if row['state'] == 'MA':
        val = 1
    else:
        val = 0
    return val

df8['Boston_metro'] = df8.apply(f, axis=1)

##################################################
# Seattle metropolitan area
# https://en.wikipedia.org/wiki/Seattle_metropolitan_area#Metropolitan_Statistical_Area
# The only state in the dataset that belong to Seattle metropolitan area is WA
WA = ['WA']
df9 = df6[df6.state.isin(WA) ]
city = list(df9['city'])
print()
print('The cities in Washington State:')
WA_cities = list(np.unique(np.array(city)))
print(WA_cities)


def f(row):
    if row['state'] == 'WA':
        val = 1
    else:
        val = 0
    return val

df8['Seattle_metro'] = df8.apply(f, axis=1)


##################################################
# Chicago metropolitan area
# https://en.wikipedia.org/wiki/Chicago_metropolitan_area#Metropolitan_Statistical_Area
# The only state in the dataset that belong to Seattle metropolitan area is IL

IL = ['IL']
df9 = df6[df6.state.isin(IL) ]
city = list(df9['city'])
print()
print('The cities in Illinois State:')
IL_cities = list(np.unique(np.array(city)))
print(IL_cities)


def f(row):
    if row['state'] == 'IL':
        val = 1
    else:
        val = 0
    return val

df8['Chicago_metro'] = df8.apply(f, axis=1)



##################################################
# Washington metropolitan area
# https://en.wikipedia.org/wiki/Washington_metropolitan_area#Metropolitan_Statistical_Area
# The only state in the dataset that belong to Seattle metropolitan area is DC

DC = ['DC']
df9 = df6[df6.state.isin(DC) ]
city = list(df9['city'])
print()
print('The cities in DC Metro:')
DC_cities = list(np.unique(np.array(city)))
print(DC_cities)

def f(row):
    if row['state'] == 'DC':
        val = 1
    else:
        val = 0
    return val

df8['DC_metro'] = df8.apply(f, axis=1)


##################################################
# Atlanta metropolitan area
# https://en.wikipedia.org/wiki/Atlanta_metropolitan_area#Metropolitan_Statistical_Area
# The only state in the dataset that belong to Seattle metropolitan area is GA
GA = ['GA']
df9 = df6[df6.state.isin(GA) ]
city = list(df9['city'])
print()
print('The cities in Georgia State:')
GA_cities = list(np.unique(np.array(city)))
print(GA_cities)

def f(row):
    if row['state'] == 'GA':
        val = 1
    else:
        val = 0
    return val

df8['Atlanta_metro'] = df8.apply(f, axis=1)



##################################################
# Denver metropolitan area
# https://en.wikipedia.org/wiki/Denver_metropolitan_area
# The only state in the dataset that belong to Seattle metropolitan area is CO
CO = ['CO']
df9 = df6[df6.state.isin(CO) ]
city = list(df9['city'])
print()
print('The cities in Colorado State:')
CO_cities = list(np.unique(np.array(city)))
print(CO_cities)


def f(row):
    if row['state'] == 'CO':
        val = 1
    else:
        val = 0
    return val

df8['Denver_metro'] = df8.apply(f, axis=1)

##################################################
# Texas (combine Dallas and Huston metropolitan areas)
# The only state in the dataset that belong to Seattle metropolitan area is TX

TX = ['TX']
df9 = df6[df6.state.isin(TX)]
city = list(df9['city'])
print()
print('The cities in Texas State:')
TX_cities = list(np.unique(np.array(city)))
print(TX_cities)



def f(row):
    if row['state'] == 'TX':
        val = 1
    else:
        val = 0
    return val

df8['Austin_metro'] = df8.apply(f, axis=1)





#######################################################
#                                                                                                                                                               #
#                                                                                                                                                               #
#        Part I Preprocessing                                                                                                                     #
#                                                                                                                                                               #
#       1.1 Extract features from Text                                                                                                     #
#       1.1.1 Extract from Position Title                                                                                                    #                                                 #
#                                                                                                                                                               #
#######################################################

# Turn the position column into a string and lower case and concatenate it

position = (df8.position
           .str.lower()
           .str.cat(sep=' ')
)
#print(position)

# remove all the unwanted charaters inside the position title string

position = position.replace('&', '').replace('– ', '').replace('--', '').replace(' -', '').replace('-', '')\
    .replace('/', '').replace("\\","").replace('(', '').replace(')', '')\
    .replace(':', '').replace('.', '').replace(',', '').replace(' !', '')\
    .replace('|', '').replace(' and ', '').replace(' of ', '')\
    .replace('i ', '').replace('ii ', '').replace('iii ', '')\
    .replace(' sr ', '').replace(' or ', '').replace(' in ', '')\
    .replace('for ', '')

# remove all numeric from the string
position = ''.join(i for i in position if not i.isdigit())
position_list = position.split(' ')


# # remove extra spaces in the string
# https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
position_trim = list(filter(lambda a: a != '', position_list))

##########################################################

# Print the most frequently occurred element in the list
# https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0

position_keyword = collections.Counter(position_trim)

n_print = 150

print()
print('The top 150 most mentioned words in data science job titles:')
for keyword, count in position_keyword.most_common(n_print):
    print(keyword, ": ", count)


# Visualize the most frequently occured word with word cloud
# https://www.datacamp.com/community/tutorials/wordcloud-python
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wordcloud = WordCloud(max_font_size=60, max_words=100, background_color="white", width=1000, height=500).generate(position)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Figure 1. Word Cloud of Position Title')
wordcloud.to_file(r"C:\Users\naixi\OneDrive\_____GWU_ECON_PHD_____\__2019_Spring__\Intro to Data Mining\___Final_Project___\push_to_git\Graphs\position_title_wordcloud.png")
plt.show()

JT_senior_keywords = ['senior',  'principal', 'lead', 'director', 'leader', 'executive', 'vp','manager']

JT_junior_keywords = ['junior', 'intern']

###############################################
# Assign binary variables to senior level jobs
def f(row):
    if any(x in row['position'].lower() for x in JT_senior_keywords):
        val = 1
    else:
        val = 0
    return val

df8['Senior_Position'] = df8.apply(f, axis=1)

# Check to see how many senior jobs are there in database
print()
print('Number of senior positions in the dataset：')
print(len(df8[(df8['Senior_Position'] == 1) ]))

#######################################################
#                                                                                                                                                               #
#                                                                                                                                                               #
#        Part I Preprocessing                                                                                                                     #
#                                                                                                                                                               #
#       1.1 Extract features from Text                                                                                                     #
#       1.1.2 Extract from Company Names                                                                                                #                                                 #
#                                                                                                                                                               #
#######################################################

company = list(df8['company'])


# Print the companies that post the most number of data scientist jobs
# https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0

company_count = collections.Counter(company)

n_print = 100

print()
print('The top 100 companies that hire most data scientists:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
##########################################################
# https://stackoverflow.com/questions/16054870/how-to-convert-list-into-string-with-quotes-in-python
company_str = ','.join("'{0}'".format(x) for x in company)

# Visualize the most frequently occured word with word cloud
# https://www.datacamp.com/community/tutorials/wordcloud-python
wordcloud = WordCloud(max_font_size=60, max_words=100, background_color="white", width=1000, height=500).generate(company_str)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Figure 2. Word Cloud of Top 100 Companies')
wordcloud.to_file(r"C:\Users\naixi\OneDrive\_____GWU_ECON_PHD_____\__2019_Spring__\Intro to Data Mining\___Final_Project___\push_to_git\Graphs\company_name_wordcloud.png")

###############################################
# Assign binary variables to tech companies within the top 100 companies
# Tech include software, hardware, telecommunication, internet companies
# Does not include consulting companies that solely provide data solutions

TECH_COMP = ['Amazon', 'Microsoft', 'Google', 'Lab126', 'Facebook', 'Harnham', 'Cymer',\
             'JD.com', 'Zillow Group', 'The Climate Corporation', 'Uber', 'NOKIA', 'Civis Analytics',\
             'Tempus', 'Autodesk', 'Twitter', 'Quora', 'Apple', 'IBM', 'BuzzFeed', 'Spotify', 'Samsung Research America',\
             'ASML', 'Pandora Media, Inc.', 'Dremio', 'DataRobot']

def f(row):
    if any(x in row['company'] for x in TECH_COMP):
        val = 1
    else:
        val = 0
    return val

df8['Tech_Company'] = df8.apply(f, axis=1)

# Check to see how many senior jobs are there in database
print()
print('Number of positions hired by tech companies (among top 100 employers)：')
print(len(df8[(df8['Tech_Company'] == 1) ]))








#######################################################
#                                                                                                                                                               #
#                                                                                                                                                               #
#        Part I Preprocessing                                                                                                                     #
#                                                                                                                                                               #
#       1.1 Extract features from Text                                                                                                     #
#       1.1.3 Extract from Job Descriptions                                                                                                                                                        #
#                                                                                                                                                               #
#######################################################

description = (df8.description
           .str.lower()
           .str.cat(sep=' ')
)

# remove all the unwanted charaters inside the position title string
# https://gist.github.com/sebleier/554280
blacklist = [' me',
' my ',
' myself',
' we ',
' our ',
' ours ',
' ourselves',
' you ',
' your ',
' yours ',
' yourself ',
' yourselves ',
' he ',
' him',
' his',
' himself',
' she',
' her',
' hers',
' herself',
' it ',
' its ',
'itself',
' they ',
' them',
' their',
' theirs',
'themselves',
' what',
' which',
' who',
'whom',
' this',
' that',
'these',
'those',
' am ',
' is ',
' are ',
' was ',
' were ',
' be ',
' been ',
' being ',
' have ',
' has ',
' had ',
' having ',
' do ',
' does ',
' did ',
' doing ',
' a ',
' an ',
' the ',
' and ',
' but ',
' if ',
' or ',
' because ',
' as ',
' until ',
' while ',
' of ',
' at ',
' by ',
' for ',
' with ',
' about ',
'against',
'between',
' into ',
' through ',
' during ',
' before ',
' after ',
' above ',
' below ',
' to ',
' from ',
' up ',
' down ',
' in ',
' out ',
' on ',
' off ',
' over ',
' under ',
' again ',
' further',
' then ',
' once ',
' here ',
' there ',
' when ',
' where',
' why',
' how',
' all',
' any',
' both',
' each',
' few',
' more',
' most',
' other',
' some',
' such',
' no ',
' nor ',
' not ',
' only ',
' own ',
' same',
' so ',
' than ',
' too ',
' very ',
' can ',
' will ',
' just ',
' don',
' should',
' now ',
'& ',
'– ',
'/',
"\\",
'(',
')',
':',
',',
'.',
'!',
'|',
'&amp;',
' andor ',
' also ',
" we're ",
" you'll ",
' ie ',
' etc ',
' eg ',
'-',
 ' within ',
' like ',
' using ',
' across ',
' please ',
' without ',
' may ',
' every ',
' also ',
' + ']

for word in blacklist:
    description = description.replace(word, "")


    # remove all numeric from the string
description = ''.join(i for i in description if not i.isdigit())
description_list = description.split(' ')


# # remove extra spaces in the string
# https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
description_trim = list(filter(lambda a: a != '', description_list))

# Print the most frequently occurred element in the list
# https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0

import collections
description_keyword = collections.Counter(description_trim)

n_print = 600

print()
print('The top 600 most mentioned words in data science job description:')
for keyword, count in description_keyword.most_common(n_print):
    print(keyword, ": ", count)

# I will pick some words that best characterize the nature of the job, in addition to general requirements.
# Such words include languages: r, python, java or biology and so on.

JD_coding_keywords = ['sql', 'software', 'program', 'python', 'cloud', 'web', 'r', \
                   'database', 'natural', 'java', 'algorithms', 'mining', 'c', 'code', 'ml', 'neural', \
                   'hardware', 'artificial', 'scripting', 'engineers', 'aws', 'spark', 'nlp', 'cloud',\
                      'backend', 'computational', 'ai', 'architecture', 'sas', 'c++', 'automation', 'coding']

# According to the most frequently appear words in job titles printed previously
# I pick lists of words that describe the seniority and functionality of the job

JI_general_keywords = ['research', 'business', 'work', 'management', 'technical', 'analysis',\
                       'communication', 'analytics', 'technology', 'analytical', 'modeling', 'statistics']

JT_bio_keywords = ['molecular', 'immunology', 'chemistry', 'drug', 'pharmacology', 'health',\
                   'biology', 'medical', 'bioinformatics', 'biologist', 'clinical',\
                   'cancer', 'healthcare', 'disease']


###############################################
# Assign binary variables to types of jobs
def f(row):
    if any(x in row['description'].lower() for x in JT_bio_keywords):
        val = 1
    else:
        val = 0
    return val

df8['Healthcare_Industry'] = df8.apply(f, axis=1)

# Check to see how many jobs contain healthcare industry keyword
print()
print('Number of job descriptions that contain healthcare industry keywords：')
print(len(df8[(df8['Healthcare_Industry'] == 1) ]))







#######################################################
#                                                                                                                                                               #
#                                                                                                                                                               #
#        Part I Preprocessing                                                                                                                     #
#                                                                                                                                                               #
#                                                                                                                                                               #
#       1.1.4 Impute Missing Value and Normalize                                                                                     #
#                                                                                                                                                               #
#######################################################

# Check the columns of df8 before proceeding
print()
print('The columns of the dataframe I am going to use from here:')
print(list(df8.columns.values))

# Impute the missing values in the number of reviews column
print()
print('The total number of instances in the dataset:')
print(len(df8.index))
print('The number of missing values in number of reviews:')
print(df8['reviews'].isnull().sum())

# Since the missing values account for about 23% for the total values, I cannot
# simply delete them.
# https://www.theanalysisfactor.com/mean-imputation/
# http://jmlr.org/papers/volume18/17-073/17-073.pdf
print()
print('First step: impute missing using group (company) means:')
#company_mean_review = df8.groupby('company')['reviews'].mean()
df8['reviews'] = df8['reviews'].fillna(df8.groupby('company')['reviews'].transform('mean'))

print()
print('The number of missing values after first step imputation:')
print(df8['reviews'].isnull().sum())

print()
print('Second step: impute missing using city means:')
df8['reviews'] = df8['reviews'].fillna(df8.groupby('city')['reviews'].transform('mean'))
print()
print('The number of missing values after second step imputation:')
print(df8['reviews'].isnull().sum())

print()
print('Third step: impute missing using state means:')
df8['reviews'] = df8['reviews'].fillna(df8.groupby('state')['reviews'].transform('mean'))
print()
print('The number of missing values after third step imputation:')
print(df8['reviews'].isnull().sum())



#######################################################
#                                                                                                                                                                #
#                                                                                                                                                                #
#        Part II   Algorithm                                                                                                                         #
#                                                                                                                                                                #
#       1.1   KNN                                                                                                                                          #
#                   1.1.1 The type of company as target                                                                                   #
#                                                                                                                                                                #
#######################################################

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Preprocess
# Standardize numerical variables
from sklearn.preprocessing import StandardScaler

# number of reviews before standard scaler
print()
print('reviews before standard scaler')
print(df8[['reviews']].head(5))
scaler = StandardScaler()
df8[['reviews']] = scaler.fit_transform(df8[['reviews']])
print()
print('reviews after standard scaler')
print(df8[['reviews']].head(5))


# https://www.ritchieng.com/machine-learning-k-nearest-neighbors-knn/
###############################################

y = df8['Tech_Company']
X = df8[['reviews', 'NY_metro', 'SF_bay_area', 'LA_metro', 'Boston_metro', 'Seattle_metro', 'Chicago_metro',\
         'DC_metro', 'Atlanta_metro', 'Denver_metro', 'Austin_metro', 'Senior_Position']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print()
print('The shapes of X_train, y_train, X_test, y_test, respectively:')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# https://www.ritchieng.com/machine-learning-k-nearest-neighbors-knn/
k_range = range(1, 26)

scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print()
print('The dependent variable is Tech_Company:')
print('The range of Accuracy Scores corresponding to different values of k')
print(scores)


plt.figure()
plt.plot(k_range, scores)
plt.title('Figure 5. KNN The Accuracy Score over Range of K')
plt.xlabel('Value of K in KNN')
plt.ylabel('Testing Accuracy for predicting Tech_Company')
plt.savefig('KNN_1.png')
plt.show()




#######################################################
#                                                                                                                                                                #
#                                                                                                                                                                #
#        Part II   Algorithm                                                                                                                         #
#                                                                                                                                                                #
#       1.1   KNN                                                                                                                                          #
#                   1.1.2 The location of the position as target                                                                        #
#                                                                                                                                                                #
#######################################################

# Create a new categorical variable metro_area

def f(row):
    if row['SF_bay_area'] == 1:
        val = 'SF_bay_area'
    if row['Boston_metro'] == 1:
        val = 'Boston_metro'
    if row['NY_metro'] == 1:
        val = 'NY_metro'
    if row['Seattle_metro'] == 1:
        val = 'Seattle_metro'
    if row['LA_metro'] == 1:
        val = 'LA_metro'
    if row['Chicago_metro'] == 1:
        val = 'Chicago_metro'
    if row['DC_metro'] == 1:
        val = 'DC_metro'
    if row['Atlanta_metro'] == 1:
        val = 'Atlanta_metro'
    if row['Denver_metro'] == 1:
        val = 'Denver_metro'
    if row['Austin_metro'] == 1:
        val = 'Austin_metro'
    return val

df8['metro_area'] = df8.apply(f, axis=1)


# Apply machine learning algorithm
y = df8['metro_area']
X = df8[['reviews', 'Senior_Position', 'Tech_Company', 'Healthcare_Industry']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

k_range = range(1, 26)

scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print()
print('The dependent variable is metro_area:')
print('The range of Accuracy Scores corresponding to different values of k')
print(scores)


plt.figure()
plt.plot(k_range, scores)
plt.title('Figure 6. KNN The Accuracy Score over Range of K')
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy for predicting metro_area')
plt.savefig('KNN_2.png')
plt.show()




#######################################################
#                                                                                                                                                                #
#                                                                                                                                                                #
#        Part III  Metro_area analytics                                                                                                     #
#                                                                                                                                                                #
#       1.1   Top Hiring Companies differ a lot by region                                                                            #
#                                                                                                                                                                #
#                                                                                                                                                                #
#######################################################

# After looking at the most frequently occurring words in position title and job descriptions, they do not
# much by region, only the top hiring companies differ, so I list them here.
df_SF = df8[df8['metro_area'] == 'SF_bay_area']
print(df_SF.shape)
company = list(df_SF['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in San Francisco Bay Area:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()


#########################################################
# First subset a dataframe just including Seattle metro position
df_Seattle = df8[df8['metro_area'] == 'Seattle_metro']
print(df_Seattle.shape)

company = list(df_Seattle['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in Seattle metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()

#######################################################
# First subset a dataframe just including NY metro position
df_NY = df8[df8['metro_area'] == 'NY_metro']
print(df_NY.shape)

company = list(df_NY['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in NY metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)



#######################################################
# First subset a dataframe just including Boston metro position
df_Boston = df8[df8['metro_area'] == 'Boston_metro']
print(df_Boston.shape)

company = list(df_Boston['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in Boston metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()

#######################################################
# First subset a dataframe just including LA metro position
df_LA = df8[df8['metro_area'] == 'LA_metro']
print(df_LA.shape)

company = list(df_LA['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in LA metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()

#######################################################
# First subset a dataframe just including Chicago metro position
df_Chicago = df8[df8['metro_area'] == 'Chicago_metro']
print(df_Chicago.shape)

company = list(df_Chicago['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in Chicago metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)



#######################################################
# First subset a dataframe just including DC metro position
df_DC = df8[df8['metro_area'] == 'DC_metro']
print(df_DC.shape)

company = list(df_DC['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in DC metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()

#######################################################
# First subset a dataframe just including Atlanta metro position
df_Atlanta = df8[df8['metro_area'] == 'Atlanta_metro']
print(df_Atlanta.shape)

company = list(df_Atlanta['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in Atlanta metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()


#######################################################
# First subset a dataframe just including Denver metro position
df_Denver = df8[df8['metro_area'] == 'Denver_metro']
print(df_Denver.shape)

company = list(df_Denver['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in Denver metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)
print()
print()


#######################################################
# First subset a dataframe just including Austin metro position
df_Austin = df8[df8['metro_area'] == 'Austin_metro']
print(df_Austin.shape)
company = list(df_Austin['company'])
company_count = collections.Counter(company)
n_print = 10
print()
print('The top 10 companies that hire most data scientists in Austin metro:')
for keyword, count in company_count.most_common(n_print):
    print(keyword, ": ", count)





#######################################################
#                                                                                                                                                                #
#                                                                                                                                                                #
#        Part IV   Decision Tree and Random Forest Analysis                                                                    #
#                                                                                                                                                                #
#       ---- Decision Tree                                                                                                                           #
#                                                                                                                                                                #
#                                                                                                                                                                #
#######################################################

# Preprocess, get the dummy variable for company column
# https://stackoverflow.com/questions/43588679/issue-with-onehotencoder-for-categorical-features
print()
df_companies = pd.get_dummies(df8, columns=['company'])
print('Shape and of df_comapny dummies:')
# print(df_companies.head(10))
print(df_companies.shape)
#print(list(df_companies.columns.values))





y = df_companies['metro_area']
X = df_companies.drop(['position', 'description', 'location', 'city', 'state', 'zipcode', \
                       'NY_metro', 'SF_bay_area', 'LA_metro', 'Boston_metro', 'Seattle_metro', \
                       'Chicago_metro', 'DC_metro', 'Atlanta_metro', 'Denver_metro', 'Austin_metro',\
                       'metro_area'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pydotplus
from sklearn.externals.six import StringIO


#### Tree with no parameter set ######################
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print()
print('The dependent variable is metro_area:')
print('Accuracy score from decision tree prediction is :')
print(metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data , feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("decision_tree_1.pdf")


#### Loop over decision tree max_depth ######################
depth_range = range(1, 200)

scores = []

for d in depth_range:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print()
print('The dependent variable is metro_area:')
print('The range of Accuracy Scores corresponding to different values of max_depth:')
print(scores)


plt.figure()
plt.plot(depth_range, scores)
plt.title('Figure 8. Decision Tree The Accuracy Score over Range of Max_depth')
plt.xlabel('Value of Max Depth')
plt.ylabel('Testing Accuracy for predicting metro_area')
plt.savefig('Tree_Accuracy_Depth.png')
plt.show()




#### Loop over minimum leaf sample ######################
# a leaf is a terminal node
# https://stackoverflow.com/questions/46480457/difference-between-min-samples-split-and-min-samples-leaf-in-sklearn-decisiontre
min_leaf_range = range(1, 30)

scores = []

for l in min_leaf_range:
    clf = DecisionTreeClassifier(min_samples_leaf=l, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print()
print('The dependent variable is metro_area:')
print('The range of Accuracy Scores corresponding to different values of min_sample_leaf')
print(scores)


plt.figure()
plt.plot(min_leaf_range, scores)
plt.title('Figure 9. Decision Tree The Accuracy Score over Range of Min_sample_leaf')
plt.xlabel('Value of Min Samples at each Leaf Node')
plt.ylabel('Testing Accuracy for predicting metro_area')
plt.savefig('Tree_Accuracy_Min_Leaf.png')
plt.show()



#######################################################
#                                                                                                                                                                #
#                                                                                                                                                                #
#        Part IV   Decision Tree and Random Forest Analysis                                                                    #
#                                                                                                                                                                #
#       ---- Random Forest                                                                                                                         #
#                                                                                                                                                                #
#                                                                                                                                                                #
#######################################################
# https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
ax = feat_importances.nlargest(20).plot(kind='bar')
ax.set_title('Figure 10. Random Forest Top 20 Feature Importance')
fig = ax.get_figure()
fig.tight_layout()
fig.savefig('RF_Feature_Importance.png')
