library(mlbench)
library(caret)
library(dplyr)


df <- read.csv( "/Users/vijitchekkala/Desktop/Hydraulics/H.csv")

df_target <- read.csv( "/Users/vijitchekkala/Desktop/Hydraulics/HH_target.csv")

#df <- select(df, -1)
#df_target <- select(df_target, -1)

data_final <- cbind(df,df_target)

#converting dataframe to csv
write.csv(data_final,"/Users/vijitchekkala/Desktop/Hydraulics/H_FINAL.csv", row.names = FALSE)
head(df)
head(data_final)
str(data_final)
summary(data_final)
# calculate correlation matrix
cor_matrix <- cor(data_final[,1:17])

# summarize the correlation matrix
print(cor_matrix)

# find attributes that are highly corrected (ideally >0.75)
high_cor <- findCorrelation(cor_matrix, cutoff=0.5)

# print indexes of highly correlated attributes
print(high_cor)
head(data_final$Coolers)
data_final$Coolers<-as.factor(data_final$Coolers) 
data_final$Valves<-as.factor(data_final$Valves) 
data_final$Pump_Leakage<-as.factor(data_final$Pump_Leakage) 
data_final$Accumulator<-as.factor(data_final$Accumulator)
data_final$Stable<-as.factor(data_final$Stable) 



str(data_final$Coolers)

#Rank Features By Importance
# prepare training scheme
c <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Coolers~. -Valves -Pump_Leakage -Accumulator -Stable, data=data_final, method="lvq", preProcess="scale", trControl=c)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

varImp(model)

#rf to find imoportant variables

modelFit <- train( Coolers~. -Valves -Pump_Leakage -Accumulator -Stable, data=data_final, method="rf" ,importance = TRUE)
varImp(modelFit)



varImpPlot(modelFit)

#random forest #yt bharatendra rai

library(randomForest)

rf<- randomForest(Coolers~. -Valves -Pump_Leakage -Accumulator -Stable, data=data_final)
print(rf)

attributes(rf)

#
library(caret)
p1 <- predict(rf,data_final)
#cheking og and predicited values
head(p1)
head(data_final$Coolers)

confusionMatrix(p1,data_final$Coolers)
#varimp

varImpPlot(rf)

#valve variable importance
rf1<- randomForest(Valves~. -Coolers -Pump_Leakage -Accumulator -Stable, data=data_final)
print(rf1)

predict1 <- predict(rf1,data_final)
head(predict1)
head(data_final$Valves)

#varimp
varImpPlot(rf1)


#pumpleakage
rf2<- randomForest(Pump_Leakage~. -Coolers -Valves -Accumulator -Stable, data=data_final)
print(rf2)

predict2 <- predict(rf2,data_final)
head(predict2)
head(data_final$Pump_Leakage)

#varimp
varImpPlot(rf2)

#accumulator
rf3<- randomForest(Accumulator~. -Coolers -Valves -Pump_Leakage -Stable, data=data_final)
print(rf3)

predict3 <- predict(rf3,data_final)
head(predict3)
head(data_final$Accumulator)

#varimp
varImpPlot(rf3)

#Stable
data_stable <- read.csv( "/Users/vijitchekkala/Desktop/Hydraulics/Data/Stable_final_data.csv")


summary(data_stable)


rf4<- randomForest(Stable~., data=data_stable)
print(rf4)

predict4 <- predict(rf4,data_stable)
head(predict4)
head(data_final$Stable)

#varimp
varImpPlot(rf4)


#exploring data 

cor(df) #Display the correlation matrix
round(df,2)  #Round the correlation matrix with two decimals

#significance level
#install.packages('Hmisc')
library("Hmisc")
data_rcorr <-as.matrix(df[, 1: 17])

mat_2 <-rcorr(data_rcorr)

p_value <-round(mat_2[["P"]], 3)
p_value

#Visualize Correlation Matrix
#install.packages("GGally")
library(GGally)
library(ggplot2)
ggcorr(df, method = c("pairwise", "pearson"),
       nbreaks = NULL, digits = 2, low = "#3B9AB2",
       mid = "#EEEEEE", high = "#F21A00",
       geom = "tile", label = FALSE,
       label_alpha = FALSE)

#Basic heat map
ggcorr(df)

#Add control to the heat map
ggcorr(df,
       nbreaks = 6,
       low = "steelblue",
       mid = "white",
       high = "darkred",
       geom = "circle")

colnames(df)



#chi square

tbl_1 = table(data_final$Coolers,data_final$Stable)
tbl_1
#h0 : Stable is independent of Coolers
#ha : Stable and Coolers are dependent
chisq.test(tbl_1)
#Conclusion- Since p value is greater than 0.05, we cannot reject the Null hypothesis.


tbl_2 = table(data_final$Valves,data_final$Stable)
tbl_2
#h0 : Stable is independent of Valves
#ha : Stable and Valves are dependent
chisq.test(tbl_2)
#  Conclcions - Since p-value is less than 0.1, we  can reject H0 and conclude with 90% confidence interval that Stable and Valves are dependent
# it is statisticall significant and not normally distributed



tbl_3 = table(data_final$Pump_Leakage,data_final$Stable)
tbl_3
#h0 : Stable is independent of Valves
#ha : Stable and Valves are dependent

chisq.test(tbl_3)
#P value less than 0.1, we can reject H0 and conclude with 90% confidence interval that Stable and Pump_Leakage are dependent





tbl_4 = table(data_final$Accumulator,data_final$Stable)
tbl_4
##h0 : Stable is independent of Accumulator
#ha : Stable and Valves are dependent
chisq.test(tbl_4)
#P value less than 0.1, we can reject H0 and conclude with 90% confidence interval that Stable and Accumulator are dependent



#Coolers and Stable
freqCoolers = table(data_final$Coolers)
relfreqCoolers = table(data_final$Coolers)/2205
cbind(freqCoolers,relfreqCoolers)

freqStable = table(data_final$Stable)
relfreqStable = table(data_final$Stable)/2205
cbind(freqStable,relfreqStable)
library(gmodels)

joint = CrossTable(data_final$Coolers,data_final$Stable,prop.chisq = FALSE)
joint

joint_counts = joint$t
barplot(joint_counts,beside=TRUE,ylab='Frequency',xlab='Stable',col=rainbow(3))


#Coolers and Stable
freqValves = table(data_final$Valves)
relfreqValves = table(data_final$Valves)/2205
cbind(freqValves,relfreqValves)

joint_2 = CrossTable(data_final$Valves,data_final$Stable,prop.chisq = FALSE)
joint_2

joint_2_counts = joint_2$t
barplot(joint_2_counts,beside=TRUE,ylab='Valves',xlab='Stable',col=rainbow(4))

#Pump_Leakage and Stable
freqPump_Leakage = table(data_final$Pump_Leakage)
relfreqPump_Leakage = table(data_final$Pump_Leakage)/2205
cbind(freqPump_Leakage,relfreqPump_Leakage)

joint_3 = CrossTable(data_final$Pump_Leakage,data_final$Stable,prop.chisq = FALSE)
joint_3

joint_3_counts = joint_3$t
barplot(joint_3_counts,beside=TRUE,ylab='Pump_Leakage',xlab='Stable',col=rainbow(3))

#Accumulators
freqAccumulator = table(data_final$Accumulator)
relfreqAccumulator = table(data_final$Accumulator)/2205
cbind(freqAccumulator,relfreqAccumulator)

joint_4 = CrossTable(data_final$Accumulator,data_final$Stable,prop.chisq = FALSE)
joint_4

joint_4_counts = joint_4$t
barplot(joint_4_counts,beside=TRUE,ylab='Accumulator',xlab='Stable',col=rainbow(4))

Classification_models = c('LR','XGboost', 'LightGBM','Catboost','RF','ANN')
cooler_condition <- c(51,94,93,93,100,100)
valve_condition <- c(86,96,94,96,94,97)
pump_condition <- c(68,93,95,94,95,94)
accumulator_condition <- c(69,97,98,98,98,92)
stable_condition <- c(75,93,94,94,93,89)
result_height <- c(100,100,100,100,100,100)
result_data <- data.frame(cooler_condition,valve_condition,pump_leakage_,hydraulic_accumulator,stable_condition)
result_data

cooler_bar <- data.frame(Classification_models,cooler_condition)
library(tidyverse)
options(digits=2)
cooler_bar %>% 
        ggplot(aes(Classification_models,cooler_condition,col=topo.colors(6)))+
        geom_col() +
        labs(title="Barplot with labels on bars")+
        geom_text(aes(label = signif(cooler_condition, digits = 3)), nudge_y = 4)

valve_bar <- data.frame(Classification_models,valve_condition)
valve_bar %>% 
        ggplot(aes(Classification_models,valve_condition,col=topo.colors(6)))+
        geom_col() +
        labs(title="Prediciting the valve condition of the hydraulic system",size=20)+
        geom_text(aes(label = signif(valve_condition, digits = 3)), nudge_y = 4)

stable_bar <- data.frame(Classification_models,stable_condition)
stable_bar %>% 
        ggplot(aes(Classification_models,stable_condition,col=topo.colors(6)))+
        geom_col() +
        labs(title="Predicting the stability of the hydraulic system")+
        geom_text(aes(label = signif(stable_condition, digits = 3)), nudge_y = 4)

pump_bar <- data.frame(Classification_models,pump_condition)
pump_bar %>% 
        ggplot(aes(Classification_models,pump_condition,col=topo.colors(6)))+
        geom_col() +
        labs(title="Predicting the Pump condition of the hydraulic system")+
        geom_text(aes(label = signif(pump_condition, digits = 3)), nudge_y = 4)
        
acc_bar <- data.frame(Classification_models,accumulator_condition)
acc_bar %>% 
        ggplot(aes(Classification_models,accumulator_condition,col=topo.colors(6)))+
        geom_col() +
        labs(title="Predicting the Accumulator condition of the hydraulic system")+
        geom_text(aes(label = signif(accumulator_condition, digits = 3)), nudge_y = 4)

