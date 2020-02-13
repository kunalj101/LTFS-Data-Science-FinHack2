library(reticulate)
library(data.table)
library(fpp)  #0.5
library(imputeTS) #3.0
library(dplyr) #0.8.3
library(forecast) #8.10
library(zoo)



########## loading data ###################
train=read.csv("./data/preprocessed_train_for_tbats.csv")
test=read.csv("./data/test.csv")
sub=read.csv("./data/sample_submission.csv")

#create a weekday column
train$weekday<- weekdays(as.Date(train$application_date))

########### preprocessed the data ##########
num2=data.frame("branch_id"=c(),"application_date"=c(),"case_count"=c(),"case_count_preprocessed"=c())
for(i in 1:length(unique(train$branch_id))){
  a=train[train$branch_id==unique(train$branch_id)[i],"case_count"]
  q1=length(a)
  x2=ts(a, frequency = 7)
  x2[!is.na(a)]=cumsum(a[!is.na(a)])
  a=na.trim(a[x2!=0], sides = c("left"), is.na = c("any", "all"))
  q2=length(a)
  if(q2==0) next
  b=train[train$branch_id==unique(train$branch_id)[i],"application_date"]
  b=b[(q1-q2+1):q1]
  num=data.frame(b, a)
  names(num)[1] <- "application_date"
  names(num)[2] <- "case_count"
  num$case_count_preprocessed=c(round(tsclean(ts(a, frequency = 7)),2))
  num$branch_id=unique(train$branch_id)[i]
  num2=rbind(num2,num)
}


num2[num2$case_count_preprocessed<=0,"case_count_preprocessed"]=0
num2[is.na(num2$case_count),"case_count"] = 0


################################ merge and groupby ##########################################################
num3=merge(num2,unique(train[,c("branch_id","segment")]),by="branch_id")
num4=num3 %>%
  dplyr::group_by(segment,application_date) %>%
  dplyr::summarise(case_count_sum = sum(case_count),
            case_count_preprocessed_sum = sum(case_count_preprocessed))

num4_seg1=subset(num4,num4$segment==1)
num4_seg2=subset(num4,num4$segment==2)
num4_seg2$weekday<- weekdays(as.Date(num4_seg2$application_date))

########################## function to calculate mape ##########################################################
mape = function(a,b){
  return(accuracy(b[b!=0],a[b!=0])[5])
}
num4_seg1=num4_seg1[-nrow(num4_seg1),]

###################################### Submission ##########################################################
#For Segment 1
y.msts <- msts(num4_seg1$case_count_preprocessed_sum,seasonal.periods=c(7))
y.msts=tail(y.msts,420)
month=4

fit <- tbats(y.msts, use.box.cox=NULL, use.parallel=TRUE, num.cores = NULL, 
             use.trend=NULL, use.damped.trend=NULL, use.arma.errors=TRUE,                           
             model=NULL)

fc <- forecast(fit,(30*month))

a=seq(as.Date(max(as.Date(num4_seg1$application_date))), as.Date(max(as.Date(sub[sub$segment==1,"application_date"]))), by="days")
a=a[2:length(a)]
b=head(fc$mean,length(a))
sub_seg1=data.frame("application_date"=a,"case_count"=b)
sub_seg1$segment=1

sub_seg1_raw=sub[sub$segment==1,]
sub_seg1_raw$case_count=NULL
sub_seg1_raw$application_date=as.Date(sub_seg1_raw$application_date)


sub5=merge(sub_seg1_raw,sub_seg1,on=c("application_date","segment"))
sub5=sub5[order(sub5$id),]
sub5=sub5[,c("id","application_date","segment","case_count")]
fwrite(sub5,'./data/tbats_seg1.csv',row.names = FALSE)


