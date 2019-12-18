import json
import pandas as pd
import numpy as np
def file_read(file):
    with open(file,'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df = df.transpose()
            return df
            
#downloading the file and reading it
file = 'stanford professors.json'  
df = file_read(file)

#applying the scores
def Convert_score(df):
    df["score"] = df["score"].astype(float)
    df['score'] = df.score.apply(np.round)
    df['score'] = df.score.astype(int)
    return df
Convert_score(df)

def con(name):
    if name <= 3:
        return 0
    else :
        return 1
df['score'] = df['score'].apply(lambda x: con(x)) 

#removing all columns except names,score and comments
def remove_column(df):
    data = df[['name','score','comments']]
    data = data.reset_index(drop = True)
    return data
data = remove_column(df)

#displaying comments
data.comments

#finding out genders of the professor names
import random 
import nltk
nltk.download('names')

from nltk.corpus import names 
def gender_features(word): 
    return {'last_letter':word[-1]} 

labeled_names = ([(name, 'male') for name in names.words('male.txt')]+
            [(name, 'female') for name in names.words('female.txt')]) 

random.shuffle(labeled_names) 

featuresets = [(gender_features(n), gender) 
                for (n, gender)in labeled_names] 

train_set, test_set = featuresets[500:], featuresets[:500] 

classifier = nltk.NaiveBayesClassifier.train(train_set) 
#print(classifier.classify(gender_features('Adelice'))) 
name= []
gender=[]
for i in data.name:
    #print(i)
    name.append(i)
    gender.append(classifier.classify(gender_features(i)))
    import warnings
    warnings.filterwarnings('ignore')
    
df2= (name,gender)
df3 = pd.DataFrame(df2)
df3 = df3.transpose()
df4 = data[['score','comments']]
data_main = pd.concat([df3,df4],axis=1)
data_main.head()
data_main.columns = ('Name','Gender','score','comments')

data_main.head()
type(data_main)

# displaying the data information
data_main.info()

#loading the gender and scores data to json files
data_main.head()
export_csv = data_main.to_csv (r'C:\Users\Pictures\STANFORD UNIV.csv', index = None, header=True)

data_main.isnull().sum()

#count of the male and female profesesors
data_main['Gender'].value_counts()

#counts of the scores in female dataset
d = data_main.Gender == 'female'
female = data_main[d]
female.head()
female["score"].value_counts()

# plotting the female data in bar graph
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('_classic_test')
sns.countplot(female["score"], palette = 'RdYlGn')
plt.title('Comparison of negative and positive reviews', fontweight = 30)
plt.xlabel('reviews')
plt.ylabel('Count')
plt.show()

#count of counts in male dataset
d1 = data_main.Gender == 'male'
male = data_main[d1]
male.head()
male["score"].value_counts()

#plotting the male data in form of bar graph
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('_classic_test')
sns.countplot(male["score"], palette = 'RdYlGn')
plt.title('Comparison of negative and positive reviews', fontweight = 30)
plt.xlabel('reviews')
plt.ylabel('Count')
plt.show()

#plotting of the genders of male and female data
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('_classic_test')
sns.countplot(data_main['Gender'], palette = 'prism')
plt.title('Comparison of Males and Females', fontweight = 30)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

x = male.comments
x.head()
type(x)

# intializing the stopwords and stemming
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
temp =[]
snow = nltk.stem.SnowballStemmer('english')
for sentence in x:
    sentence = sentence[0].lower()                 
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        
    
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   
    temp.append(words)
    
final_X = temp
sent = []
for row in final_X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

final_X = sent
print(final_X[1])

# showing the top 20 words used in the comments of professors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(final_X)
sum_words = words.sum(axis=0)


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Most Frequently Occuring Words - Top 20")

# Forming of word cloud from comments
from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000).generate_from_frequencies(dict(words_freq))

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Most occur words in the dataset", fontsize = 20)
plt.show()

trainig and test datasets of scores and words
y = data_main.score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(words, y, test_size = 0.3, random_state = 15)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.preprocessing import MaxAbsScaler 

mm = MaxAbsScaler()

# using random forest classifier and calculating the training accuracy
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# using tf-idf vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(final_X)
print(X)

# using SVM classifier and finding best tuned parameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(xtrain, ytrain)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()

pred = clf.predict(xtest)
print(confusion_matrix(ytest,pred))

# displaying the classification report
print(classification_report(ytest,pred))

# calculating the accuracy scores
print(accuracy_score(ytest,pred))
