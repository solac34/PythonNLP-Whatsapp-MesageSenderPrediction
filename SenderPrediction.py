import math
import os
import string
from time import time

import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def prepareData(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        senders = []
        messages = []
        msgfile = f.readlines()
        if 'Messages and calls are end-to-end encrypted.' not in msgfile[0]:
            print('''
                                          |--------------------------------------------|
This is not a valid file.Please note that |Messages and calls are end-to-end encrypted.| stamp must be at the beginning of the .txt file!!
                                          |--------------------------------------------|''')
            while 1:
                try:
                    getAnalysis(prepareData(input('Input new file path')))
                    break
                except FileNotFoundError:
                    print('This is not a valid filepath!')
        for i, line in enumerate(msgfile):
            line = line.strip('‎')
            if line[0] != '[':
                messages[len(messages) - 1] += (' ' + line)
                continue
            else:
                senders.append(line.split("]")[1].split(":")[0])
                messages.append(line.split(":")[-1])

    csvpath = fpath.split("/")[-1][0:-4] + ".csv"
    pd.DataFrame({'Sender': senders,
                  'Message': messages}).to_csv(csvpath)
    return csvpath


def getAnalysis(pth):  # pth = csvpath
    language = str(input('Language of this message file:'))
    df = pd.read_csv(pth)
    df.drop('Unnamed: 0', axis=1, inplace=True)

    def prcs(x):
        try:
            return [word for word in [c for c in x if c not in string.punctuation] if
                    word not in stopwords.words(language)]
        except OSError:
            print('Language failed.Try again..')
            getAnalysis(pth)
            quit()

    if str(input('If you want to use SVM, please type SVM or 0 , otherwise, RandomForests will be used!:')).upper() in (
            'SVM', '0'):
        pipeline = Pipeline([('bow', CountVectorizer(analyzer=prcs)),
                             ('tfidf', TfidfTransformer()),
                             ('classifier', GridSearchCV(SVC(), param_grid={
                                 'C': [10, 100, 1000],
                                 'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['rbf', 'sigmoid']}))])
    else:
        pipeline = Pipeline([('bow', CountVectorizer(analyzer=prcs)),
                             ('tfidf', TfidfTransformer()),
                             ('classifier', RandomForestClassifier())])

    X = df['Message'].apply(lambda x: x.strip('\n'))
    y = df['Sender'].apply(lambda x: x[1:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.10)

    if X_train.shape[0] > 25500:
        timeslonger = math.ceil(X_train.shape[0] / 25500)
        print(f'This data is {timeslonger} times longer than usual.This will take more time, please wait!')
        for ftc in range(timeslonger):
            starttime = time()
            try:
                pipeline.fit(X_train[ftc * 25500:25500 * (ftc + 1)], y_train[ftc * 25500:25500 * (ftc + 1)])
            except:
                print('latest phase!')
                pipeline.fit(X_train[ftc * 25500:], y_train[ftc * 25500:])
            print(f'----------------------------------------------------------------------------------------\n',
                  f'preparing and creating the model... phase {ftc + 1}/{timeslonger} completed - runtime: {time() - starttime}sec')
            """even though its in fitting phase, this print label will be more helpful for
                                                   users that has no idea what ML is."""
    else:
        pipeline.fit(X_train, y_train)

    print('Model Accuracy for this data:\t%' + str(
        100 * f1_score(y_test, pipeline.predict(X_test), average='binary', pos_label=df['Sender'].unique()[0][1:]))[:4])

    while str(input('Would you like to make a prediction?')).lower() == 'y':
        print('Result:' + pipeline.predict([str(input('Type a chat msg, then see the prediction!:'))])[0])

    if input('Please type "del" to delete csv file for security.Type anything to save it.').lower() == 'del':
        os.remove(pth)
        print('removed successfully')

    continueApp(input('Would you like to continue? (type y for another analysis, n for quit.)').lower())


def continueApp(c):
    if c == 'y':
        getAnalysis(prepareData(input('File path of txt file that exported from whatsapp:')))
    elif c == 'n':
        quit()
    else:
        continueApp(input('Unaccepted input form.Please only use y or n!'))


continueApp('y')
