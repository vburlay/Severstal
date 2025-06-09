# Steel Defect Detection

![image](/images/Steel.jpg ) 

> Predicting automobile liability insurance claim payments based on vehicle characteristics.

## Table of Contents 

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)



## General Information

> This project has two main objectives. One was to understand how to effectively predict automobile liability insurance claim payments based on vehicle characteristics, and the other was to determine a best approach to predict automobile liability insurance claim payments using customer data. 
 > Data set: "Allstate Claim Prediction Challenge" comes from Kaggle [_here_](https://www.kaggle.com/c/ClaimPredictionChallenge/overview).

## Technologies Used
- R - version 3.6.1
- Shiny - version 1.7.4

## Features
- tidyverse
- Lineare regression
- Machine Learning ( gbm + random forest + single-hidden-layer neural network)


## Screenshots

![Example screenshot](https://github.com/vburlay/allstate_claim_prediction/raw/main/image/shiny.PNG)

## Setup

It is necessary to install the following R-Packages additionally: 
```r
install.packages(c("DT", "dplyr", "plotly", "shinyjs", "shinyFiles"))
```

## Usage

* Preparation
```r
highCorr <- findCorrelation(correlations, cutoff = .9)
filteredSegData <- df[,-highCorr]
rm(df,highCorr,correlations)
trans <- preProcess(filteredSegData, method = c("BoxCox","center","scale","pca"))
transformed <- predict(trans,filteredSegData)
```
* GBM
```r
ames_gbm1 <- gbm(formula = trainSet$Claim_Amount ~ .,
                 data = trainSet,
                 distribution = "gaussian",
                 n.trees = 500,
                 shrinkage = 0.1,
                 interaction.depth = 3,
                 n.minobsinnode = 10,
                 cv.folds = 5 )
```                 
* Random Forest
```r
model_rf <- randomForest(trainSet$Claim_Amount ~ .,
                          data = trainSet,
                          importance =TRUE,
                          ntree = 110)
```
* NNET
```r
nnetFit <- nnet(predictors, outcome,
                size = 5,
                decay = 0.01,
                linout = TRUE,
                trace = FALSE,
                maxit = 500,
                MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)
predictions <- round(predict(nnetFit, testSet, interval = "predict", level = 0.95),2)
```
## Project Status

Project is: _complete_ 


## Room for Improvement

* By using cross-validation, an increase in the accuracy of the prediction could be achieved. In addition, the amount  of data was not satisfactory for the polynomial and the model was highly prone to overfitting.
* The clustering procedure did not help the linear regression to demonstrate a good result. It is necessary to analyze the other procedures such as Bayes statistics approach 
* The selected feature combinations were not suitable for modeling compared to single values.


## Contact
Created by [Vladimir Burlay](wladimir.burlay@gmail.com) - feel free to contact me!

