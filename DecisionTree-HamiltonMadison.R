library(RWeka)

fedpapers <- read.csv("fedPapers85.csv")
fedpapers <- fedpapers[-c(66:70),] # removing non-Hamilton/Madison rows
# fedpapers <- fedpapers[,-2] # removing filenames from dataset
str(fedpapers)

trainset <- fedpapers[fedpapers$author!="dispt",-2] # simultaneously remove filenames from dataset
testset <- fedpapers[fedpapers$author=="dispt",-2] # simultaneously remove filenames from dataset

# trainset[sapply(trainset, is.numeric)] <- lapply(trainset[sapply(trainset, is.numeric)], as.factor)
# testset[sapply(testset, is.numeric)] <- lapply(testset[sapply(testset, is.numeric)], as.factor)

NN <- make_Weka_filter("weka/filters/unsupervised/attribute/NumericToNominal")
trainset <- NN(data=trainset, control= Weka_control(R="1-3"), na.action = NULL)
testset <- NN(data=testset, control= Weka_control(R="1,3"), na.action = NULL)
str(trainset)
str(testset)

MS <- make_Weka_filter("weka/filters/unsupervised/attribute/ReplaceMissingValues")

trainset <-MS(data=trainset, na.action = NULL)
testset <-MS(data=testset, na.action = NULL)

str(trainset)

m=J48(author~., data = trainset)
m=J48(author~., data = trainset, control=Weka_control(U=FALSE, M=2, C=0.5))
WOW("J48")

e <- evaluate_Weka_classifier(m, numFolds = 10, seed = 1, class = TRUE)
pred = predict(m, newdata = testset, type = c("class"))
write.csv(pred, file="predict.csv")

# myVars=c("author","all","also","and","any","are","as","at","be","been","but","by","can","down","even","every","for","from","had","has","have","if","in","into","is","it","its","may","more","must","my","no","not","now","of","on","one","only","our","shall","should","so","some","such","than","that","then","there","things","this","up","upon","was","were","what","when","which","who","will","with","would","your")
# newtrain=trainset[myVars,]
# newtest=testset[myVars,]
# m=J48(author~., data = newtrain)
# m=J48(author~., data = newtrain, control=Weka_control(U=FALSE, M=2, C=0.5))
# # e <- evaluate_Weka_classifier(m, numFolds = 10, seed = 1, class = TRUE)
# newpred=predict (m, newdata = newtest, type = c("class"))
# # myids=c("author")
# # id_col=testset[myids,]
# # newpred=cbind(id_col, pred)
# # colnames(newpred)=c("author", "author_predict")
# # View(newpred)
# write.csv(newpred, file="fedpapers-J48-pred.csv", row.names=FALSE)
# InfoGainAttributeEval(author ~ . , data = trainset)
