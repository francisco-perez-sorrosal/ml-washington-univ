# Quiz
# 1) Which of the following statements are true? (Check all that apply)
# -Linear classifiers are never useful, because they cannot represent XOR. X
# -Linear classifiers are useful, because, with enough data, they can represent anything. X
# -Having good non-linear features can allow us to learn very accurate linear classifiers. V
# -None of the above X

# 2) A simple linear classifier can represent which of the following functions? (Check all that apply)
# Hint: If you are stuck, see https://www.coursera.org/learn/ml-foundations/module/nqC1t/discussions/AAIUurrtEeWGphLhfbPAyQ
# -x1 OR x2 OR NOT x3 V
# -x1 AND x2 AND NOT x3 V
# -x1 OR (x2 AND NOT x3) V
# -none of the above X

# 3) Which of the the following neural networks can represent the following function? Select all that apply.
# (x1 AND x2) OR (NOT x1 AND NOT x2)
#Hint: If you are stuck, see https://www.coursera.org/learn/ml-foundations/module/nqC1t/discussions/AAIUurrtEeWGphLhfbPAyQ
# Solution: Image number 4: z1 weights = [x0=-1.5, x1=1, x2=1], z2 weights = [0.5, -1, -1], y weights (for OR) = [z0=-0.5, z1=1, z2=1] 

# 4) Which of the following statements is true? (Check all that apply)
# - Features in computer vision act like local detectors. V
# - Deep learning has had impact in computer vision, because it’s used to combine all the different hand-created features that already exist. X
# - By learning non-linear features, neural networks have allowed us to automatically learn detectors for computer vision. V
# - none of the above X

# 5) If you have lots of images of different types of plankton labeled with their species name, and lots of computational resources, what would you expect to perform better predictions:
# - a deep neural network trained on this data. True
# - a simple classifier trained on this data, using deep features as input, which were trained using ImageNet data. False

# 6) If you have a few images of different types of plankton labeled with their species name, what would you expect to perform better predictions:
# - a deep neural network trained on this data. False
# - simple classifier trained on this data, using deep features as input, which were trained using ImageNet data. True

%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import turicreate as tc
tc.config.set_runtime_config("TURI_NUM_GPUS", 0)
tc.visualization.set_target('gui')
# Load Dataset
image_train=tc.SFrame('image_train_data/')
image_test=tc.SFrame('image_test_data/')
# Explore Dataset
image_train.head(1)
image_train['image'].show
# Train classifier on raw imagess
raw_pixel_model = tc.logistic_classifier.create(image_train, 'label', features=['image_array'])
# Make a prediction
image_test[0:3]
raw_pixel_model.predict(image_test[0:3])
# Evaluate all test data`
raw_pixel_model.evaluate(image_test)
# Improve using transfer learning
deep_learning_model = tc.image_analysis.load_images('imagenet_model') # Not sure how to do this with tc
deep_features_model  = tc.logistic_classifier.create(image_train, 'label', features=['deep_features'])
deep_features_model.predict(image_test[0:3])
deep_features_model.evaluate(image_test)
# Image retrieval
# Train nearest neighbors model for retrieving images using features
knn_model = tc.nearest_neighbors.create(image_train, features=['deep_features'], label='id')


cat = image_train[18:19]
cat['image'].explore()
knn_model.query()
knn_model.query(cat)
def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'], 'id')
cat_neighbors = get_images_from_ids(knn_model.query(cat))
cat_neighbors.explore()
car = image_train[8:9]
car.explore()
car_neighbors = get_images_from_ids(knn_model.query(car))
car_neighbors['image'].explore()
# Lambda to find and show NN images
show_neighbors = lambda i : get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].explore()
show_neighbors(9)
show_neighbors(10)

# Quiz
#1) What’s the least common category in the training data?
# - bird V
# - dog
# - cat
# - automobile
image_train['label'].summary()
dog_train = image_train[image_train['label'] == 'dog']
cat_train = image_train[image_train['label'] == 'cat']
car_train = image_train[image_train['label'] == 'automobile']
bird_train = image_train[image_train['label'] == 'bird']

dog_model = tc.nearest_neighbors.create(dog_train, features=['deep_features'], label='id')
cat_model = tc.nearest_neighbors.create(cat_train, features=['deep_features'], label='id')
car_model = tc.nearest_neighbors.create(car_train, features=['deep_features'], label='id')
bird_model = tc.nearest_neighbors.create(bird_train, features=['deep_features'], label='id')
# 2) Of the images below, which is the nearest ‘cat’ labeled image in the training 
# data to the the first image in the test data (image_test[0:1])?
# - The sixth image

quiz_cat = image_test[0:1]
quiz_cat.explore()

quitz_cats = get_images_from_ids(cat_model.query(quiz_cat))

quitz_cats.explore()
# 3) Of the images below, which is the nearest ‘dog’ labeled image in the
# training data to the the first image in the test data (image_test[0:1])?
# The fourth image
quiz_doggies = get_images_from_ids(dog_model.query(quiz_cat))
quiz_doggies.explore()
# 4) For the first image in the test data, in what range is the mean distance 
# between this image and its 5 nearest neighbors that were labeled ‘cat’ in the
# training data?
# Resp = 36.15

cat_model.query(quiz_cat)['distance'].mean()
# 5) For the first image in the test data, in what range is the mean distance between
# this image and its 5 nearest neighbors that were labeled ‘dog’ in the training data?
# 37.77
dog_model.query(quiz_cat)['distance'].mean()

# 6) On average, is the first image in the test data closer to its 5 nearest neighbors in the ‘cat’ data or in the ‘dog’ data?
# Cat

# 7) In what range is the accuracy of the 1-nearest neighbor classifier at classifying ‘dog’ images from the test set?
# 67.8%
dog_test = image_test[image_test['label'] == 'dog']
cat_test = image_test[image_test['label'] == 'cat']
car_test = image_test[image_test['label'] == 'automobile']
bird_test = image_test[image_test['label'] == 'bird']

dog_cat_neighbors = cat_model.query(dog_test, k=1)
dog_car_neighbors = car_model.query(dog_test, k=1)
dog_bird_neighbors = bird_model.query(dog_test, k=1)
dog_dog_neighbors = dog_model.query(dog_test, k=1)
dog_distances = tc.SFrame({'dog_automobile': dog_car_neighbors['distance'],
                           'dog_bird': dog_bird_neighbors['distance'],
                           'dog_cat': dog_cat_neighbors['distance'],
                           'dog_dog': dog_dog_neighbors['distance']})
dog_distances
def is_dog_correct(row):
    if row['dog_dog'] < row['dog_cat'] and row['dog_dog'] < row['dog_bird'] and row['dog_dog'] < row['dog_automobile']:
        return 1
    else:
        return 0
dog_distances.apply(is_dog_correct).sum()
###############################################################################

# Old stuff


# 2) Of the images below, which is the nearest ‘cat’ labeled image in the training 
# data to the the first image in the test data (image_test[0:1])?
- The sixth image
quiz_cat = image_test[0:1]
quiz_cat.explore()
quitz_cats = get_images_from_ids(knn_model.query(quiz_cat))
quitz_cats.explore()
# 3) Of the images below, which is the nearest ‘dog’ labeled image in the
# training data to the the first image in the test data (image_test[0:1])?
# The fourth image
doggies = get_images_from_ids(knn_model.query(quiz_cat, k=10))
doggies.explore()
# 4) For the first image in the test data, in what range is the mean distance 
# between this image and its 5 nearest neighbors that were labeled ‘cat’ in the
# training data?
# Resp = 36.15
quiz_cats = get_images_from_ids(knn_model.query(quiz_cat))
quiz_cats
quiz_cat
image_train[image_train['id'] == 331]
type(quiz_cat['deep_features'])
quiz_cat
len(list(quiz_cat['deep_features']))
first_cat = image_train[image_train['id'] == str(id)]
len(list(first_cat['deep_features']))
tc.distances.cosine(list(quiz_cat['deep_features']), list(first_cat['deep_features']))
quiz_cats
import numpy as np
quiz_cat_feats = quiz_cat['deep_features']
type(quiz_cat_feats)
mean_dist = 0
for current_cat in quiz_cats:
    # print(type(current_cat['deep_features']))
    # print(type(array(quiz_cat['deep_features'])))
    dist = tc.distances.euclidean(quiz_cat_feats[0],current_cat['deep_features'])
    print(dist)
    mean_dist += dist
print(mean_dist / 5.)
# For the first image in the test data, in what range is the mean distance between
# this image and its 5 nearest neighbors that were labeled ‘dog’ in the training data?
# 39.51
quiz_doggies = get_images_from_ids(knn_model.query(quiz_cat, k=100))

type(quiz_doggies)
quiz_doggies = quiz_doggies[quiz_doggies['label'] == 'dog'].head(5)
quiz_doggies
mean_dist = 0
for current_dog in quiz_doggies:
    # print(type(current_cat['deep_features']))
    # print(type(array(quiz_cat['deep_features'])))
    dist = tc.distances.euclidean(quiz_cat_feats[0],current_dog['deep_features'])
    print(dist)
    mean_dist += dist
print(mean_dist / 5.)
