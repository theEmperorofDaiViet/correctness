<a name="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#api-documentation">API Documentation</li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


# About The Project

This is a part of my Introduction to Data Science's assignment at university. In this part, I tried to write my own module for classification evaluation metrics, based on the sklearn.metrics



## Built With

* [![Numpy][Numpy-shield]][Numpy-url]
* [![Pandas][Pandas-shield]][Pandas-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Getting Started

## Prerequisites
To use this module, your system needs to have:
* numpy
  ```sh
  pip install numpy
  ```
* pandas
  ```sh
  pip install pandas
  ```

## Installation
You can install this module by cloning this repository into your current working directory:
```sh
git clone https://github.com/theEmperorofDaiViet/correctness.git
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# API Documentation
Classification is a type of supervised machine learning problem where the goal is to predict, for one or more observations, the category or class they belong to.

An important element of any machine learning workflow is the evaluation of the performance of the model. This is the process where we use the trained model to make predictions on previously unseen, labelled data. In the case of classification, we then evaluate how many of these predictions the model got right.

In real-world classification problems, it is usually impossible for a model to be 100% correct. When evaluating a model it is, therefore, useful to know, not only how wrong the model was, but in which way the model was wrong.

In the this module, I provide seven different performance metrics and techniques you can use to evaluate a classifier.

## 1. correctness.confusion_matrix

<p style="font-size: 1.17em;"><i>It is a matrix that compares the number of predictions for each class that are correct and those that are incorrect.</i></p>

<p style="font-size: 1.17em;">In a confusion matrix, there are 4 numbers to pay attention to:

<li><b>True Positive:</b> The number of positive observations the model correctly predicted as positive.</li>

<li><b>False Positive:</b> The number of negative observations the model incorrectly predicted as positive.</li>

<li><b>True Negative:</b> The number of negative observations the model correctly predicted as negative.</li>

<li><b>False Negative:</b> The number of positive observations the model incorrectly predicted as negative.</li></p>

<p style="font-size: 1.17em;">Other references may use a different convention for confusion matrix. In <code>correctness</code>'s convention, each row represents the instances in a predicted class, while each column of the matrix represents the instances in an actual class, as follows:</p>

<table align='center'>
  <tr>
    <th></th>
    <th colspan='2'>Actual class</td>
  </tr>
  <tr>
    <th rowspan='2'>Predicted class</th>
    <td>TP</td>
    <td>FP</td>
  </tr>
  <tr>
    <td>FN</td>
    <td>TN</td>
  </tr>
</table>

<p style="text-align:left;">
  <pre><code>confusion_matrix(y_true, y_pred)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L4">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Compute confusion matrix to evaluate the accuracy of a classification.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>y_true: <i>array-like of shape (n_samples)</i></b><br/>
      <p style="margin-left: 2.5%">Ground truth (correct) target values.</p>
      <b>y_pred: <i>array-like of shape (n_samples)</i></b><br/>
      <p style="margin-left: 2.5%">Estimated targets as returned by a classifier.</p>      
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b>C: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 2. correctness.accuracy

<p style="font-size: 1.17em;"><i>The overall <b>accuracy</b> of a model is simply the number of correct predictions divided by the total number of predictions. An accuracy score will give a value between 0 and 1, a value of 1 would indicate a perfect model.</i></p>

<p style="text-align:left;">
  <pre><code>accuracy(cm)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L7">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Accuracy classification score.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>cm: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>     
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b>score: <i>float</i></b><br/>
      <p style="margin-left: 2.5%">The fraction of correctly classified samples.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 3. correctness.precision

<p style="font-size: 1.17em;"><i><b>Precision</b> measures how good the model is at correctly identifying the positive class. In other words out of all predictions for the positive class how many were actually correct?</i></p>

<p align='center'><b><font size='5'>
    precision = TP / (TP + FP)
</font></b></p>

<p style="font-size: 1.17em;">Using alone this metric for optimising a model, we would be minimising the false positives. This might be desirable for our fraud detection example, but would be less useful for diagnosing cancer as we would have little understanding of positive observations that are missed.</p>

<p style="text-align:left;">
  <pre><code>precision(cm, average='binary', pos_label = 0)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L10">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Compute the precision.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>cm: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>
      <b>average: <i>{‘micro’, ‘macro’, ‘weighted’, ‘binary’}</i></b> or <b><i>None, default=’binary’</i></b><br/>
      <p style="margin-left: 2.5%">This parameter is required for multiclass/multilabel targets. If <code>None</code>, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
        <li style="margin-left: 2.5%"><code>'binary'</code>:</li>
        <p style="margin-left: 5%">Only report results for the class specified by <code>pos_label</code>.</p>
        <li style="margin-left: 2.5%"><code>'micro'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics globally by counting the total true positives, false negatives and false positives.</p>
        <li style="margin-left: 2.5%"><code>'macro'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.</p>
        <li style="margin-left: 2.5%"><code>'weighted'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters <code>‘macro’</code> to account for label imbalance; it can result in an F-score that is not between precision and recall.</p> 
      </p>
      <b>pos_label: <i>int, default=0</i></b><br/>
      <p style="margin-left: 2.5%">The class to report if <code>average='binary'</code>. If <code>average != 'binary'</code>, this will be ignored.</p>   
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b>precision: <i>float (if <code>average</code> is not None)</i></b> or <b><i>array of float of shape (n_unique_labels)</i></b><br/>
      <p style="margin-left: 2.5%">Precision of the positive class in binary classification or weighted average of the precision of each class for the multiclass task.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 4. correctness.recall

<p style="font-size: 1.17em;"><i><b>Recall</b> tell us how good the model is at correctly predicting all the positive observations in the dataset.</i></p>

<p align='center'><b><font size='5'>
    recall = TP / (TP + FN)
</font></b></p>

<p style="font-size: 1.17em;">It does not include information about the false positives so would be more useful in the cancer example.</p>

<p style="text-align:left;">
  <pre><code>recall(cm, average='binary', pos_label = 0)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L50">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Compute the recall.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>cm: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>
      <b>average: <i>{‘micro’, ‘macro’, ‘weighted’, ‘binary’}</i></b> or <b><i>None, default=’binary’</i></b><br/>
      <p style="margin-left: 2.5%">This parameter is required for multiclass/multilabel targets. If <code>None</code>, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
        <li style="margin-left: 2.5%"><code>'binary'</code>:</li>
        <p style="margin-left: 5%">Only report results for the class specified by <code>pos_label</code>.</p>
        <li style="margin-left: 2.5%"><code>'micro'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics globally by counting the total true positives, false negatives and false positives.</p>
        <li style="margin-left: 2.5%"><code>'macro'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.</p>
        <li style="margin-left: 2.5%"><code>'weighted'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters <code>‘macro’</code> to account for label imbalance; it can result in an F-score that is not between precision and recall.</p> 
      </p>
      <b>pos_label: <i>int, default=0</i></b><br/>
      <p style="margin-left: 2.5%">The class to report if <code>average='binary'</code>. If <code>average != 'binary'</code>, this will be ignored.</p>   
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b>recall: <i>float (if <code>average</code> is not None)</i></b> or <b><i>array of float of shape (n_unique_labels)</i></b><br/>
      <p style="margin-left: 2.5%">Recall of the positive class in binary classification or weighted average of the recall of each class for the multiclass task.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 5. correctness.f1_score

<p style="font-size: 1.17em;"><i> The <b>F1 score</b> is the harmonic mean of precision and recall.</i></p>

<p align='center'><b><font size='5'>
    F1 = 2 x precision x recall / (precision + recall)
</font></b></p>

<p style="font-size: 1.17em;">The F1 score will give a number between 0 and 1. If the F1 score is 1.0 this indicates perfect precision and recall. If the F1 score is 0 this means that either the precision or the recall is 0.</p>

<p style="text-align:left;">
  <pre><code>f1_score(cm, average='binary', pos_label = 0)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L90">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Compute the F1 score, also known as balanced F-score or F-measure.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>cm: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>
      <b>average: <i>{‘micro’, ‘macro’, ‘weighted’, ‘binary’}</i></b> or <b><i>None, default=’binary’</i></b><br/>
      <p style="margin-left: 2.5%">This parameter is required for multiclass/multilabel targets. If <code>None</code>, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
        <li style="margin-left: 2.5%"><code>'binary'</code>:</li>
        <p style="margin-left: 5%">Only report results for the class specified by <code>pos_label</code>.</p>
        <li style="margin-left: 2.5%"><code>'micro'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics globally by counting the total true positives, false negatives and false positives.</p>
        <li style="margin-left: 2.5%"><code>'macro'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.</p>
        <li style="margin-left: 2.5%"><code>'weighted'</code>:</li>
        <p style="margin-left: 5%">Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters <code>‘macro’</code> to account for label imbalance; it can result in an F-score that is not between precision and recall.</p> 
      </p>
      <b>pos_label: <i>int, default=0</i></b><br/>
      <p style="margin-left: 2.5%">The class to report if <code>average='binary'</code>. If <code>average != 'binary'</code>, this will be ignored.</p>   
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b>f1_score: <i>float (if <code>average</code> is not None)</i></b> or <b><i>array of float of shape (n_unique_labels)</i></b><br/>
      <p style="margin-left: 2.5%">F1 score of the positive class in binary classification or weighted average of the F1 scores of each class for the multiclass task.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 6. correctness.support

<p style="font-size: 1.17em;"><i><b>Support</b> is the number of actual occurrences of the class in the specified dataset.</i></p>

<p style="font-size: 1.17em;">Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing.</p>

<p style="text-align:left;">
  <pre><code>support(cm, average = 'binary', pos_label=0)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L118">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Compute the support.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>cm: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>
      <b>average: <i>{‘micro’, ‘macro’, ‘weighted’, ‘binary’}</i></b> or <b><i>None, default=’binary’</i></b><br/>
      <p style="margin-left: 2.5%">This parameter determines which value would be returned:
        <li style="margin-left: 2.5%"><code>'binary'</code>:</li>
        <p style="margin-left: 5%">Return support of the class specified by <code>pos_label</code>.</p>
        <li style="margin-left: 2.5%"><b>else</b>:</li>
        <p style="margin-left: 5%">Return <code>n_samples</code> of the specified dataset.</p>
      </p>
      <b>pos_label: <i>int, default=0</i></b><br/>
      <p style="margin-left: 2.5%">The class to report if <code>average='binary'</code>. If <code>average != 'binary'</code>, this will be ignored.</p>   
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b>support: <i>int</i></b><br/>
      <p style="margin-left: 2.5%">Support of the specified class or the total number of samples of the dataset.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 7. correctness.report

<p style="text-align:left;">
  <pre><code>report(cm)</code><span style="float:right;">[<a href="https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L124">source</a>]</span></pre>
</p>

<p style="margin-left: 2.5%">Build a text report showing all the classification metrics above.</p>

<table style="width: 97.5%; margin-left: 2.5%">
  <tr>
    <th class='api'>Parameters</th>
    <td>
      <b>cm: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p> 
    </td>
  </tr>
  <tr>
    <th class='api'>Returns</th>
    <td>
      <b><i>None</i></b><br/>
      <p style="margin-left: 2.5%">This is a side effect function.</p>
    </td>
  </tr>
</table>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Usage
<p style="font-size: 1.17em;"><i>Let me illustrate how to use this module to evaluate a classification model.</i></p>

## Example 1:
Inside the module, I already provided a [test case](https://github.com/theEmperorofDaiViet/correctness/blob/master/correctness.py#L160") for it. Since I placed it in the <code>__main__</code> block, you can test it yourself by running the file as a script. I will reintroduce it here:

### Actual values and Predicted values
```python
>>> y_target = ['dog', 'cat', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'cat', 'dog', 'dog', 'dog', 
... 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat']
>>> y_predicted = ['dog', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 
... 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat']
```
### Compute confusion matrix
```python
>>> cm = confusion_matrix(y_target, y_predicted)    
>>> print(cm)
       cat  dog
cat      6    2
dog      1   11
```
### Return classification report
```python
>>> report(cm)
CLASSIFICATION REPORT:
   precision    recall  f1-score  support
0   0.750000  0.857143      0.80        7
1   0.916667  0.846154      0.88       13

          precision    recall  f1-score  support
macro      0.833333  0.851648     0.840       20
micro      0.850000  0.850000     0.850       20
weighted   0.858333  0.850000     0.852       20
accuracy    0.85
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Example 2:
Besides the simple test case above, I will also provide a more objective example by building a classification model and then evaluating it.

More specifically, I will build a Gaussian Naive Bayes model to classify the dry bean dataset from [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset).

### Import libraries, modules and load data
```python
>>> from Naive_Bayes import Gaussian_Naive_Bayes
>>> import correctness
>>> import pandas as pd
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split

>>> df = pd.read_excel('Dry_Bean_Dataset.xlsx')
>>> df.shape
(13611, 17)
```
<p style="margin-left: 2.5%">The <code>Naive_Bayes</code> module I import is my other built-from-scratch module that implements Naive Bayes algorithms. It is a supervised learning method based on applying Bayes’ theorem with strong (naive) feature independence assumptions. You can check it out <a href="https://github.com/theEmperorofDaiViet/naive_bayes">here</a>.</p>

### Preprocess and split data
```python
>>> data = df.drop(['ConvexArea','EquivDiameter','AspectRation','Eccentricity','Class','Area','Perimeter',
... 'ShapeFactor2','ShapeFactor3','ShapeFactor1','ShapeFactor4'],axis = 1)
>>> target = df['Class']

>>> X = np.array(data)
>>> y = np.array(target)

>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Perform classification using this module and evaluate the model performance
```python
>>> nb = Gaussian_Naive_Bayes()
>>> nb.fit(X_train, y_train)
>>> y_pred = nb.predict(X_test)

>>> cm = correctness.confusion_matrix(y_test, y_pred)
>>> scratch = correctness.accuracy(cm)
>>> correctness.report(cm)
CLASSIFICATION REPORT:
   precision    recall  f1-score  support
0   0.853846  0.840909  0.847328      264
1   0.989796  1.000000  0.994872       97
2   0.914773  0.901961  0.908322      357
3   0.914986  0.895628  0.905203      709
4   0.935829  0.951087  0.943396      368
5   0.947368  0.951923  0.949640      416
6   0.834915  0.859375  0.846968      512

          precision    recall  f1-score  support                                                
macro      0.913073  0.914412  0.913676     2723
micro      0.904150  0.904150  0.904150     2723
weighted   0.904404  0.904150  0.904196     2723
accuracy    0.90415
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Contact
You can contact me via:
* [![GitHub][GitHub-shield]][GitHub-url]
* [![LinkedIn][LinkedIn-shield]][LinkedIn-url]
* ![Gmail][Gmail-shield]:&nbsp;<i>Khiet.To.05012001@gmail.com</i>
* [![Facebook][Facebook-shield]][Facebook-url]
* [![Twitter][Twitter-shield]][Twitter-url]

<br/>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- Tech stack -->
[Numpy-shield]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org
[Pandas-shield]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org

<!-- Contact -->
[GitHub-shield]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[GitHub-url]: https://github.com/theEmperorofDaiViet
[LinkedIn-shield]: https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white
[LinkedIn-url]: https://www.linkedin.com/in/khiet-to/
[Gmail-shield]: https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white
[Facebook-shield]: https://img.shields.io/badge/Facebook-%231877F2.svg?style=for-the-badge&logo=Facebook&logoColor=white
[Facebook-url]: https://www.facebook.com/Khiet.To.Official/
[Twitter-shield]: https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white
[Twitter-url]: https://twitter.com/KhietTo

### Style Sheets
Github's markdown processor cannot render ```<style>``` sheets, so you may see it lying here:
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
.api {
  align: left;
  vertical-align: top;
  width: 12%
}
</style>
<br/>
<br/>
You can read this file with the best experience by using other text editor, e.g. <b>Visual Studio Code</b>'s Open Preview mode (Ctrl+Shift+V)