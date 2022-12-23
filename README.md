# PythonNLP-Whatsapp-MesageSenderPrediction
#### Predict who sent this message using nlp, with options of SVM and RandomForests!
<li>First, inputs a path where .txt file that exported from whatsapp located.</li> 
<li>Then,open sthe txt file and converts it to csv with editing it. Whatsapp messages includes timestamps and LRMs.Messages that sent with newline
by user is also a problem since they don't have the timestamp.</li>
Note that .txt file needs encryption stamp at the beginning of the file to to prove that it is an exported file from whatsapp.
<li>Saves the csv file to the location where script is located</li>
<li>Analysis function reads the csv file to dataframe, then does the last chores.</li>
<li>Inputs a language to use it in CountVectorizer's analyzer's stopwords.</li>
<li>Creates a pipeline for model.Capacity for fitting is 25500 per fitting.For more datas, it splits the X_train and fits it part by part. </li>
<li>User chooses one of two options: Random Forests or SVM. SVM is mostly better but takes really too much time to fit.</li>
<li>After the model is ready, calculates the f1 score and lets user to predict if user wants to.</li>
<li>User can delete csv file by inputting del if it's a shared computer, or for any other spesific reason.</li>
<li>User predicts his/her requests and sees the results.</li>
<li>If another chat prediction wished, user can do that as much as he/she wants.</li>
<li>Quit from app!</li>
