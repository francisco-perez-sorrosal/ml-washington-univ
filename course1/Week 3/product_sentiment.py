#################################################
# Quiz Theory
#################################################

# 1. The simple threshold classifier for sentiment analysis described in the video (check all that apply):
# - V Must have pre-defined positive and negative attributes
# -V Must either count attributes equally or pre-define weights on attributes
# -X Defines a possibly non-linear decision boundary
# 2. For a linear classifier classifying between “positive” and “negative” sentiment in a review x, Score(x) = 0 implies (check all that apply):
# - X The review is very clearly “negative”
# - V We are uncertain whether the review is “positive” or “negative”
# - X We need to retrain our classifier because an error has occurred
# 3. For which of the following datasets would a linear classifier perform perfectly?
# 4. True or false: High classification accuracy always indicates a good classifier.
# - False
# 5. True or false: For a classifier classifying between 5 classes, there always exists a classifier with accuracy greater than 0.18.
# - True
# 6. True or false: A false negative is always worse than a false positive.
# - false
# 7. Which of the following statements are true? (Check all that apply)
# - V Test error tends to decrease with more training data until a point, and then does not change (i.e., curve flattens out)
# - X Test error always goes to 0 with an unboundedly large training dataset
# - X Test error is never a function of the amount of training data

########################################
# Quiz Assignment
########################################

#1 Out of the 11 words in selected_words, which one is most used in the reviews in the dataset?
#- great

# 2 Out of the 11 words in selected_words, which one is least used in the reviews in the dataset?
#- wow

# 3 Out of the 11 words in selected_words, which one got the most positive weight in the selected_words_model?
#(Tip: when printing the list of coefficients, make sure to use print_rows(rows=12) to print ALL coefficients.)
#- love 1.35

# 4 Out of the 11 words in selected_words, which one got the most negative weight in the selected_words_model?
#(Tip: when printing the list of coefficients, make sure to use print_rows(rows=12) to print ALL coefficients.)
#- horrible -2.25 (terrible)

# 5 Which of the following ranges contains the accuracy of the selected_words_model on the test_data?
#- 0.8463 Selected

# 6 Which of the following ranges contains the accuracy of the sentiment_model in the IPython Notebook from lecture on the test_data?
#- 0.9240 Words

# 7 Which of the following ranges contains the accuracy of the majority class classifier, which simply predicts the majority class on the test_data?
#- 0.8400 Maj Class

# 8 How do you compare the different learned models with the baseline approach where we are just predicting the majority class?
#- The model learned using all words performed much better than the other two. The other two approaches performed about the same.

# 9 Which of the following ranges contains the ‘predicted_sentiment’ for the most positive review for ‘Baby Trend Diaper Champ’, according to the sentiment_model from the IPython Notebook from lecture?
#- 0.9999996

# 10 Consider the most positive review for ‘Baby Trend Diaper Champ’ according to the sentiment_model from the IPython Notebook from lecture. Which of the following ranges contains the predicted_sentiment for this review, if we use the selected_words_model to analyze it?
#- 0.7919

# 11 Why is the value of the predicted_sentiment for the most positive review found using the sentiment_model much more positive than the value predicted using the selected_words_model?
#- None of the selected_words appeared in the text of this review.



import turicreate as graphlab
import matplotlib.pyplot as plt
graphlab.config.set_runtime_config('TURI_NUM_GPUS', 4)
graphlab.visualization.set_target('gui')

# DATA_DIR='dev/ml-washington-univ/course1/Week 3/amazon_baby.gl/'
DATA_DIR='amazon_baby.gl/'
products = graphlab.SFrame(DATA_DIR)
products.head(3)
products.shape

# products['name'].show()
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'] >=4

products.head(3)

amazon_products = products.copy()
amazon_products.shape
amazon_products.head(3)
### Giraffe example

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
len(giraffe_reviews)
giraffe_reviews['rating'].show('Categorical')

### Train/Test sets
train_data,test_data = products.random_split(.8, seed=0)

### Majority class
pos = test_data[test_data['sentiment']==1].shape
neg = test_data[test_data['sentiment']==0].shape
test_data.shape[0] == pos[0] + neg[0]
float(pos[0])/float(test_data.shape[0])

### Selected words
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

def find_selected_word(word, word_count):
    # print(word_count)
    return word_count.get(word, 0)

for word in selected_words:
    amazon_products[word] = amazon_products.apply(
        lambda r: find_selected_word(word, r['word_count']))

amazon_products.shape
amazon_products.head(3)
selected_words
for word in selected_words:
    print(word, amazon_products[word].sum())

### Selected words model

train_data_am,test_data_am = amazon_products.random_split(.8, seed=0)
train_data_am.shape
test_data_am.shape

selected_words_model = graphlab.logistic_classifier.create(train_data_am,
                                                           target='sentiment', features=selected_words, validation_set=test_data_am)
selected_words_model.coefficients.print_rows(15)
selected_words_model.coefficients.sort('value',ascending=False).print_rows(num_rows=15)
selected_words_model.evaluate(test_data_am)

### Selected words model

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target='sentiment',
                                                      features=['word_count'],
                                                      validation_set=test_data)

sentiment_model.evaluate(test_data)
results=sentiment_model.evaluate(test_data, metric='roc_curve')
results
plt.scatter(results['roc_curve']['fpr'], results['roc_curve']['tpr'])

### Diaper review sentiment model

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
len(diaper_champ_reviews)
diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(
    diaper_champ_reviews,
    output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)
diaper_champ_reviews
diaper_champ_reviews[0]

sentiment_model.predict(diaper_champ_reviews[0:1], output_type='probability')

### Diaper review sentiment model

diaper_champ_reviews_am = amazon_products[amazon_products['name'] == 'Baby Trend Diaper Champ']
len(diaper_champ_reviews_am)
diaper_champ_reviews_am['predicted_sentiment'] = selected_words_model.predict(
    diaper_champ_reviews_am,
    output_type='probability')
diaper_champ_reviews_am = diaper_champ_reviews_am.sort('predicted_sentiment', ascending=False)

diaper_champ_reviews_am[0]
diaper_champ_reviews_am[-1]
selected_words_model.predict(diaper_champ_reviews_am[0:1], output_type='probability')

first_model_review = diaper_champ_reviews_am[diaper_champ_reviews_am['review'] == "I originally put this item on my baby registry because friends of mine highly recommended the use of a diaper pail.  I decided to go with the Diaper Champ rather than the more popular Diaper Genie, because I didn't like the idea of having to buy special bags.  Too costly and not efficient if you ask me!THIS PRODUCT IS A LIFESAVER!! I HAVE NEVER HAD AN ODOR PROBLEM!No one has ever come to my house and noticed 'diaper odor.'  In fact, people who have been present while I changed my almost 3 month old son's diaper have been amazed at how the very 'potent' smell of the dirty diaper seems to disappear once I dispose of the diaper in the Diaper Champ.For those who are experiencing odor problems, here are some suggestions which should solve the problem:1. CHANGE BAGS FREQUENTLY!!  I change mine about 1 or 2 times a week.  For those who complain that the pail gets stuck, that is probably a good indication that the BAG IS GETTING TOO FULL!! Also, making sure that the tape on the diapers is secure will prevent that from happening as well.2. YOU STILL HAVE TO CLEAN THE THING!!  Hello, it does house dirty diapers!! All I do is wipe it down with some antibacterial wipes (takes about 10 seconds!!) and then maybe spray some Lysol in it and let it air out for a minute or two.  SIMPLE AND IT WORKS!Also, for those that complain that the Diaper Champ isn't as effective as time passes, keep the following in mind:AS BABIES GET OLDER, THEIR DIETS CHANGES (aka SOLID FOODS!) AND SO DOES THE SMELL OF THEIR DIAPERS!In talking with other parents, I get the same feedback - as babies get older and become toddlers, ALL DIAPER PAILS become gradually less and less effective until they are just no longer useful.  Again, this is because as their diet changes, so does the consistency and odor of their stool, in other words, it gets stinkier!!!All in all, this product is definitely one I would recommend and certainly worth the very reasonable price."]
first_model_review

########################################

from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

tf_vectorizer = CountVectorizer(max_df=0.5,
                                min_df=10,
                                max_features=5000,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(X_df['review'])
len(tf_vectorizer.get_feature_names())

vectorizer = CountVectorizer(min_df=20)
X_df = train_data.to_dataframe()
X_df.sentiment = X_df.sentiment.astype('category')

# X_df['word_count_sk'] = vectorizer.fit_transform(X_df['review'])
tf = vectorizer.fit_transform(X_df['review'])
len(vectorizer.get_feature_names())
for i, col in enumerate(vectorizer.get_feature_names()):
    # print(i, col, tf[:, i].toarray().ravel())
    X_df[col] = pd.SparseSeries(tf[:, i].toarray().ravel(), fill_value=0)
    if i % 500 == 0:
        print(i)

X_df.info()
X_df['word_count_sk'].shape
type(X_df['word_count_sk'][0].todense())
train_data.head(3)

X_df.info()

X = X_df[vectorizer.get_feature_names()]
X.pop('name')
X.pop('review')
X.pop('rating')
X.pop('sentiment')
X.info()
X.head(3)
y = X_df['sentiment']
y.head()

X.shape
y.shape

clf = LogisticRegression(random_state=0, C=1, penalty='l1').fit(X, y)

X_test_df = test_data.to_dataframe()
X_test_df.sentiment = X_test_df.sentiment.astype('category')
y_test = X_test_df['sentiment']

for i, col in enumerate(vectorizer.get_feature_names()):
    # print(i, col, tf[:, i].toarray().ravel())
    X_test_df[col] = pd.SparseSeries(tf[:, i].toarray().ravel(), fill_value=0)
    if i % 500 == 0:
        print(i)

X_test = X_test_df[vectorizer.get_feature_names()]
X_test.pop('name')
X_test.pop('review')
X_test.pop('rating')
X_test.pop('sentiment')

y_pred = clf.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
def plot_fig():
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
plot_fig()
