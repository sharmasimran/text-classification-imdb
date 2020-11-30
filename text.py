#Importing the imdb dataset to get the source of texts for my logistic regression classifer to classify the words

import os
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences


max_features = [100,1000,10000,100000]
max_len = 500
batch_size = 32
dataset_dir = "D:/Personal/IMDB dataset/aclImdb"
train = os.path.join(dataset_dir,"train")
text = []
labels =[]
history = []
fscores = []
for label_type in ['pos','neg']:
    dir_name = os.path.join(train,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name,fname),encoding='utf-8')
            text.append(f.read())
            if label_type == 'pos':
                labels.append(1)
            else:
                labels.append(0)



for i in max_features:
    
    tokenizer = Tokenizer(num_words=i)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    labels = np.array(labels)
    wordindexes = tokenizer.word_index
    print("Found %s unique tokens" % len(wordindexes))


    data = pad_sequences(sequences,max_len)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    X_train = data[:20000]
    y_train = labels[:20000]

    x_val = data[20000:]
    y_val = labels[20000:]

#Training on the above dataset using logistic regression
    lr_model = LogisticRegression(penalty="l2", C=1.0,max_iter=5000,verbose=1,solver='newton-cg')
    lr_model.fit(X_train,y_train)
    score = lr_model.score(x_val,y_val)
    print(score)

    history.append(score)
    reviews = lr_model.predict(x_val)

    conf_matrix = confusion_matrix(y_val,reviews)
    print(conf_matrix)

    print('Report : ')
    print(classification_report(y_val,reviews))

    precesion,recall,fscore,support = precision_recall_fscore_support(y_val, reviews, average=None)
    fscores.append(fscore)
