Human Activity Recognition - Peer Assignment
========================================================

The original data was obtained from accelerometers on the belt, forearm, arm, and dumbell of 6 participants during dumbbell lifting. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. This report describes how the model was built, how the cross validation was made. Also the model obtained was used to predict 20 different test cases.

So, first, all the library and data needed are called:



```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pmlt=read.csv('/home/cleo/pml-training.csv')
pmltest=read.csv('/home/cleo/pml-testing.csv')
```

## Cleaning the datasets

Cut off the unnecessary columns, like date, username, window and ID sample (X)


```r
pmltrain<-pmlt[,-c(1:7)]
pmltest<-pmltest[,-c(1:7)]
```

And all the variables without readings, specifically those that only has value for rows with the variable 'window=yes'. These columns with 'NA' are removed from the data.


```r
namcol<-colnames(pmltrain)[unlist(lapply(pmltrain, function(x) any(is.na(x))))]
delcol<-which(names(pmltrain) %in% namcol)
pmltrain<-pmltrain[,-delcol]
pmltest<-pmltest[,-delcol]
```

The variables considered 'factor' (except variable 'classe') are unmecessary.


```r
delcol<-c()
for(i in 1:length(names(pmltrain))-1){
        if(is.factor(pmltrain[,i])){
               delcol<-c(delcol,i)
        }
}

pmltrain<-pmltrain[,-delcol]
pmltest<-pmltest[,-delcol]
```

## Preprocess and Analysis

With the cleaned data, it's possible to verify the highly correlated variables. Variables that has a high correlation between them tends to decrease the model performance.


```r
# calculate correlation matrix
correlationMatrix <- cor(pmltrain[,1:ncol(pmltrain)-1])

# find attributes that are highly corrected (ideally >0.7)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7)

#refine the data variables
pmltrain<-cbind(pmltrain[,-highlyCorrelated],pmltrain[,ncol(pmltrain)])
pmltest<-cbind(pmltest[,-highlyCorrelated],pmltest[,ncol(pmltest)])

#mantain the last column name as 'classe'
names(pmltrain)[ncol(pmltrain)]<-'classe'
```

For the model fit, a cross validation is a good choice, with k=10. The default value is 25, which costs so much for computer processing. Folder number setted to 10 is enough for the analysis.
The algorithm choosen is 'random forests'. The final accuracy for this algorithm tends to be satisfatory, although this process cost is high.


```r
set.seed(3)
inTrain<-createDataPartition(pmltrain$classe,p=0.8,list=FALSE)
train<-pmltrain[inTrain,]
test<-pmltrain[-inTrain,]
train_control <- trainControl(method="cv", number=10)
modFit <- train(classe~.,data=train,method="rf",trControl=train_control)
```

```
## Warning in model.matrix.default(Terms, m, contrasts, na.action =
## na.action): a resposta apareceu no lado direito e foi descartada
```

```
## Warning in model.matrix.default(Terms, m, contrasts, na.action =
## na.action): problema com o termo 31 na matriz do modelo: nenhuma coluna foi
## atribuida
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

So, let's verify the results:


```r
# make predictions
predictions <- predict(modFit, test[,-ncol(pmltrain)])
# summarize results
confusionMatrix(predictions, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    9    1    0    0
##          B    0  748    3    0    0
##          C    0    2  679   13    0
##          D    0    0    1  629    0
##          E    0    0    0    1  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9924          
##                  95% CI : (0.9891, 0.9948)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9903          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9855   0.9927   0.9782   1.0000
## Specificity            0.9964   0.9991   0.9954   0.9997   0.9997
## Pos Pred Value         0.9911   0.9960   0.9784   0.9984   0.9986
## Neg Pred Value         1.0000   0.9965   0.9985   0.9957   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1907   0.1731   0.1603   0.1838
## Detection Prevalence   0.2870   0.1914   0.1769   0.1606   0.1840
## Balanced Accuracy      0.9982   0.9923   0.9940   0.9890   0.9998
```

And, finally, construct the array with the answers for the test data.


```r
answers=predict(modFit,pmltest)
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
pml_write_files(answers)
```
