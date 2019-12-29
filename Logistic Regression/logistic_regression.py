# logistic regression model to predict model

# import basic libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

digits = load_digits()

print("image data shape", digits.data.shape)
print("label data shape", digits.target.shape)

print(plt.figure(figsize=(20,4)))
for index, (image,label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    print(plt.subplot(1,5,index+1))
    print(plt.imshow(np.reshape(image, (8,8)),cmap=plt.cm.gray))
    print(plt.title('Training: %i\n' %label, fontsize=20))

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# remove warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver='liblinear')
logisticRegr.fit(x_train,y_train)

# resturns a numpy array
# predict for one observation(image)
print(logisticRegr.predict(x_test[0].reshape(1,-1)))

print(logisticRegr.predict(x_test[0:10]))

predictions = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test,predictions)
print(cm)

print(plt.figure(figsize=(9,9)))
print(sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r'));
print(plt.ylabel('Actual label'));
print(plt.xlabel('predicted label'));
all_sample_title='Accuracy score: {0}' .format(score)
print(plt.title(all_sample_title, size=15));

index=0
misclassifiedIndex=[]
for predict, actual in zip(predictions, y_test):
    if predict==actual:
        misclassifiedIndex.append(index)
    index +=1
print(plt.figure(figsize=(20,3)))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    print(plt.subplot(1,4,plotIndex +1))
    print(print(plt.imshow(np.reshape(x_test[wrong], (8,8)),cmap=plt.cm.gray)))
    print(plt.title("Predicted: {}, Actual: {}" .format(predictions[wrong], y_test[wrong]), fontsize=20))
