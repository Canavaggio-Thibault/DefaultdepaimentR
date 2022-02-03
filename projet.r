---
title: "Projet R"
author: "Thibault Canavaggio"
date: "18/04/2021"
output:  
  prettydoc::html_pretty:
    theme: cayman
    highlight: github
editor_options: 
  chunk_output_type: console
---
# Sommaire
## I Initialisation de l'espace de Travail
### 1) Librairies utilisées
### 2) Chargement des données
## II Exploration des données
### 1) Aperçu
### 2) Recodage
### 3) pré-traitement des données
## III Classifieur
-- Arbre de décision 

-- RandomForest

-- K-plus-proches-voisins 

-- Support vector machines 

-- Naive bayes 

-- Reseaux de neuronnes

## IV Resultats


# I Initialisation de l'espace de Travail
```{r}
setwd("C:/Users/visit/OneDrive/Documents/Rprojetc")
# repertoire de travail
```

## 1) Librairies utilisées
```{r,echo=TRUE, warning=FALSE,message=FALSE}
library(ggplot2)
library(rpart)
library(randomForest)
library(kknn)
library(random)
library(C50)
library(tree)
library(e1071)
library(naivebayes)
library(nnet)
library(dplyr)
library(ROCR)

```

## 2) Chargement des données
```{r}
data<-read.csv("C:/Users/visit/OneDrive/Documents/Rprojetc/Data_projet.csv")
```

# II Exploration des données

## 1) Aperçu
```{r echo=TRUE, warning=FALSE,message=FALSE}
str(data) #str: renseigne sur la structure de données de l'objet

# int: entier naturel
# num: nombre à virgule (flottant)
# char: chaine de charactère
```
On observe 3 structures de données différentes :
on remarque que la variable "branch" est encodé en entier, il faut la convertir en variable Catégorielle. Il en va de même pour la variable "ed". Enfin la variable "default" qui est encodé en charactère doit être encodé en booléen.

## 2) Recodage
```{r echo=TRUE, warning=FALSE,message=FALSE}
data$branch <- as.factor(data$branch)
data$ed <- as.factor(data$ed)
for ( i in 1:1200){
  if ( data$default[i] == "Non"){
    data$default[i] = 0
  } else {
    data$default[i] = 1
  }
}
data$default <- as.integer(data$default)
data$default <- as.factor(data$default)

```


```{r}
data <- data[,-3]
```


## 3) pré-traitement des données
```{r echo=TRUE, warning=FALSE,message=FALSE}
data2 <- sample_n(data,size =1200, replace=FALSE ) # permet d'interchanger les lignes du tableau de façon aleatoire
# ENSEMBLE DE TEST #
data_EA <- data2[1:800,]
data_ET <- data2[801:1200,]
# On a cree un ensemble d’apprentissage, comportant les deux premiers tiers de l’ensemble de données, et un ensemble de test composé du tier restant. Ensuite on  compare la representation des classes entre l’ensemble de depart, l’ensemble d’apprentissage et l’ensemble de test.
```

# III Classifieur

## Arbre de décision (rpart)
```{r echo=TRUE, warning=FALSE,message=FALSE, results="hold"}
test1 <- function(arg1,arg2,arg3,arg4){
  classifieur_1 <- rpart(default ~ ., data_EA,parms = list(split = arg1), control = rpart.control(minbucket = arg2))
  test_classifieur1 <- predict(classifieur_1,data_ET, type="class")
  test_classifieur1_prob <- predict(classifieur_1,data_ET, type="prob")
  mc_tree1 <- table(data_ET$default,test_classifieur1)
  roc_pred1 <- prediction(test_classifieur1_prob[,2], data_ET$default)
  roc_perf1 <- performance(roc_pred1,"tnr","fnr")
  plot(roc_perf1, col = arg3,add=arg4, main = "Courbes AUC du classifieur Rpart")
  auc_tree1 <- performance(roc_pred1, "auc")
  #str(auc_tree1)
  cat("AUC = ",as.character(attr(auc_tree1, "y.values")))
  print(mc_tree1)
}

test1("gini",10,"red",FALSE)
test1("gini",5,"blue",TRUE)
test1("information",10,"yellow",TRUE)
test1("information",5,"black",TRUE)

```
#### On remarque que la meilleure configuration pour Rpart donne un indice AUC=0.747547306567265

## RandomForest
```{r echo=TRUE, warning=FALSE,message=FALSE,results="hold"}
test2 <- function(arg1,arg2,arg3,arg4){
  classifieur_2 <- randomForest(default~., data_EA,ntree=arg1,mtry=arg2)
  test_classifieur2 <- predict(classifieur_2,data_ET, type="class")
  test_classifieur2_prob <- predict(classifieur_2,data_ET, type="prob")
  mc_tree2 <- table(data_ET$default,test_classifieur2)
  roc_pred2 <- prediction(test_classifieur2_prob[,2], data_ET$default)
  roc_perf2 <- performance(roc_pred2,"tnr","fnr")
  plot(roc_perf2, col = arg3,add=arg4, main = "Courbes AUC du classifieur RandomForest")
  auc_tree2 <- performance(roc_pred2, "auc")
  #str(auc_tree2)
  cat("AUC =",as.character(attr(auc_tree2, "y.values")))
  print(mc_tree2)
}

test2(500,5,"red",FALSE)
test2(500,3,"blue",TRUE)
test2(300,5,"yellow",TRUE)
test2(300,3,"red",TRUE)

```
#### On observe que la meilleur configuration pour RandomForest donne un indice AUC=0.809686521187647
## kknn

```{r echo=TRUE, warning=FALSE,message=FALSE,results="hold"}
# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_knn <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanes du classifeur de type k-nearest neighbors
  knn <- kknn(default~., data_EA,data_ET, k = arg1, distance = arg2)
  
  # Matrice de confusion
  print(table(data_ET$default, knn$fitted.values))
  
  # Courbe ROC
  knn_pred <- prediction(knn$prob[,2], data_ET$default)
  knn_perf <- performance(knn_pred,"tpr","fpr")
  plot(knn_perf, main = "Classifeurs K-plus-proches-voisins kknn()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  knn_auc <- performance(knn_pred, "auc")
  cat("AUC = ", as.character(attr(knn_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

# K plus proches voisins
test_knn(10, 1, FALSE, "red")
test_knn(10, 2, TRUE, "blue")
test_knn(20, 1, TRUE, "green")
test_knn(20, 2, TRUE, "orange")
```
#### On observe que la meilleure configuration de Kknn donne un indice de AUC= 0.813776500737749


## svm
```{r echo=TRUE, warning=FALSE,message=FALSE,results="hold"}
test_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svm <- svm(default~., data_EA, probability=TRUE, kernel = arg1)
  # Test du classifeur : classe predite
  svm_class <- predict(svm, data_ET, type="response") # Matrice de confusion
  print(table(data_ET$default, svm_class))
  # Test du classifeur : probabilites pour chaque prediction
  svm_prob <- predict(svm, data_ET, probability=TRUE) # Recuperation des probabilites associees aux predictions
  svm_prob <- attr(svm_prob, "probabilities") # Courbe ROC
  svm_pred <- prediction(svm_prob[,1], data_ET$default)
  svm_perf <- performance(svm_pred,"tnr","fnr")
  plot(svm_perf, main = "Support vector machines svm()", add = arg2, col = arg3) # Calcul de l'AUC et affichage par la fonction cat()
  svm_auc <- performance(svm_pred, "auc")
  cat("AUC = ", as.character(attr(svm_auc, "y.values")))
  # Return sans affichage sur la console
  invisible()
}

test_svm("linear", FALSE, "red") 
test_svm("polynomial", TRUE, "blue") 
test_svm("radial", TRUE, "green") 
test_svm("sigmoid", TRUE, "orange")

```
#### On observe que la meilleure configuration de svm donne un indice de AUC=0.19334213455515
## nbb
```{r echo=TRUE, warning=FALSE,message=FALSE,results="hold"}
test_nb <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  nb <- naive_bayes(default~., data_EA, laplace = arg1, usekernel = arg2)
  # Test du classifeur : classe predite
  nb_class <- predict(nb, data_ET, type="class") # Matrice de confusion
  print(table(data_ET$default, nb_class))
  # Test du classifeur : probabilites pour chaque prediction
  nb_prob <- predict(nb, data_ET, type="prob") # Courbe ROC
  nb_pred <- prediction(nb_prob[,2], data_ET$default)
  nb_perf <- performance(nb_pred,"tnr","fnr")
  plot(nb_perf, main = "Classifieurs bayesiens naifes naiveBayes()", add = arg3, col
       = arg4)
  # Calcul de l'AUC et affichage par la fonction cat()
  nb_auc <- performance(nb_pred, "auc")
  cat("AUC = ", as.character(attr(nb_auc, "y.values")))
  # Return sans affichage sur la console
  invisible()
}

test_nb(0, FALSE, FALSE, "red") 
test_nb(20, FALSE, TRUE, "blue") 
test_nb(0, TRUE, TRUE, "green") 
test_nb(20, TRUE, TRUE, "orange")

```
#### On observe que la meilleure configuration de nbb donne un indice de AUC=0.76917501488442
## nnet
```{r echo=TRUE, warning=FALSE,message=FALSE,results="hold"}
test_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermediaires vers un fichier texte
  #sink('output.txt', append=T)
  # Apprentissage du classifeur
  nn <- nnet(default~., data_EA, size = arg1, decay = arg2, maxit=arg3)
  # Reautoriser l'affichage des messages intermediaires
  #sink(file = NULL)
  # Test du classifeur : classe predite
  nn_class <- predict(nn, data_ET, type="class")
  # Matrice de confusion
  print(table(data_ET$default, nn_class))
  # Test des classifeurs : probabilites pour chaque prediction
  nn_prob <- predict(nn, data_ET, type="raw")
  # Courbe ROC
  nn_pred <- prediction(nn_prob[,1], data_ET$default)
  nn_perf <- performance(nn_pred,"tnr","fnr")
  plot(nn_perf, main = "Reseaux de neurones nnet()", add = arg4, col = arg5) # Calcul de l'AUC
  nn_auc <- performance(nn_pred, "auc")
  cat("AUC = ", as.character(attr(nn_auc, "y.values")))
  # Return ans affichage sur la console
  invisible()
}

test_nnet(35, 0.01, 100, F, "red") 
test_nnet(35, 0.01, 300, T, "tomato") 
test_nnet(25, 0.01, 100, T, "blue") 
test_nnet(25, 0.01, 300, T, "purple") 
test_nnet(35, 0.001, 100, T, "green") 
test_nnet(35, 0.001, 300, TRUE, "turquoise") 
test_nnet(25, 0.001, 100, TRUE, "grey") 
test_nnet(25, 0.001, 300, TRUE, "black")

```
#### On observe que la meilleure configuration de nbb donne un indice de AUC= 0.832042698502325

# Résultats: 
On a testé plusieurs algorithmes plusieurs fois pour creer le classifieur avec le plus de précision.Après avoir testé tous les algorithmes vu en cours on a pu observé que le réseaux de neurones est le classifieur qui donne le meilleur résultat pour les défaults de paiement.(Naive bayes donne également de bon résulat)


```{r echo=TRUE, warning=FALSE,message=FALSE,results="hold"}
dataTest <- read.csv("C:/Users/visit/OneDrive/Documents/Rprojetc/Data_projet_new.csv")
str(dataTest)
dataTest$branch<- as.factor(dataTest$branch)
dataTest$ed <- as.factor(dataTest$ed) 
nn_class <- nnet(default~., data_EA, size = 35, decay = 0.001, maxit=300)
prediction_classe <- predict(nn_class, dataTest, type = "class")
str(nn_class)
nn_prob<- predict(nn_class, dataTest, probability=TRUE )



Resultat <- data.frame(dataTest$customer,prediction_classe,nn_prob*100) #on a multiplié la prédiction_classe par 100 pour avoir nos résultats en pourcentage.
names(Resultat) <- c("ID","classe","proba_default_de_paiement")
View(Resultat)
write.csv(Resultat, file= "Resultat.csv")
for ( i in 1:300){
  if ( Resultat$classe[i] == "1"){
    Resultat$classe[i] = "oui"
  } else {
    Resultat$classe[i] = "non"
  }
}
library(questionr)
rmarkdown::paged_table(Resultat)
```
#Bilan: Il faut prendre en compte que tous ces résultats sont génèrés 
#a partir d’un partionnement aléatoire du jeu de donnée, les résultats 
#different légèrement d’un tirage a l’autre.J'ai remarqué que le code Svm peut 
#être le meilleur classifieur comme le pire c'est pourquoi il n'a pas
 été pris comme classifieur à cause de son manque de stabilité.
#On a également remarqué que le classifieur random forest est le plus 
#stable de tous les classifieurs mais le réseaux de neurones offrent de
#meilleurs résultats.