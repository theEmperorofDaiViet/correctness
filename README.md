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

In the this module, I provide five different performance metrics and techniques you can use to evaluate a classifier.

## 1. correctness.confusion_matrix

<p style="font-size: 1.17em;"><i>It is a matrix that compares the number of predictions for each class that are correct and those that are incorrect.</i></p>

<p style="font-size: 1.17em;">In a confusion matrix, there are 4 numbers to pay attention to:

<li><b>True Positives:</b> The number of positive observations the model correctly predicted as positive.</li>

<li><b>False Positive:</b> The number of negative observations the model incorrectly predicted as positive.</li>

<li><b>True Negative:</b> The number of negative observations the model correctly predicted as negative.</li>

<li><b>False Negative:</b> The number of positive observations the model incorrectly predicted as negative.</li></p>

<p style="font-size: 1.17em;">Other references may use a different convention for confusion matrix. In <code>correctness</code>'s convention, each row represents the instances in a predicted class, while each column of the matrix represents the instances in an actual class, as follows:</p>

<table style="margin-left: 37.5%">
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
    <th>Parameters</th>
    <td>
      <b>y_true: <i>array-like of shape (n_samples)</i></b><br/>
      <p style="margin-left: 2.5%">Ground truth (correct) target values.</p>
      <b>y_pred: <i>array-like of shape (n_samples)</i></b><br/>
      <p style="margin-left: 2.5%">Estimated targets as returned by a classifier.</p>      
    </td>
  </tr>
  <tr>
    <th>Returns</th>
    <td>
      <b>C: <i>DataFrame of shape (n_classes, n_classes)</i></b><br/>
      <p style="margin-left: 2.5%">Confusion matrix whose i-th row and j-th column entry indicates the number of samples with predicted label being i-th class and true label being j-th class.</p>
    </td>
  </tr>
</table><br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



# Usage



# Contact



<!-- MARKDOWN LINKS & IMAGES -->
[Numpy-shield]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org
[Pandas-shield]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org

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
