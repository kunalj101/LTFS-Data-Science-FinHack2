library(data.table)  # data.table_1.12.2
library(dplyr)       # dplyr_0.8.3
library(xgboost)     # xgboost_0.90.0.2
library(MLmetrics)   # MLmetrics_1.1.1


############# loading data ###############
train_data <- fread('./data/train.csv', na.strings = c('',' ', NA))
test_data <- fread('./data/test.csv' , na.strings = c('',' ', NA))
submission <- fread('./data/sample_submission.csv' , na.strings = c('',' ', NA))

train_data$days <- weekdays(as.Date(train_data$application_date))

holidays_mapping <- fread('./data/holiday_mapping.csv')
holidays_mapping$date <- as.Date(holidays_mapping$date, format = '%d/%m/%Y')
summary(holidays_mapping$date)
holidays_mapping$holiday_flag <- 1
unique(holidays_mapping$type)
sort(table(holidays_mapping$type))
holidays_mapping$type <- ifelse(holidays_mapping$type =='Observance, Christian','Observance', holidays_mapping$type)
holidays_mapping$type <- ifelse(holidays_mapping$type =='Muslim, Common local holiday','local', holidays_mapping$type)

holidays_mapping$type <- tolower(holidays_mapping$type)
holidays_mapping$type <- gsub(' ','_', holidays_mapping$type)

holiday_type_binary_flag <- dcast(setDT(holidays_mapping),  date ~ type, value.var = c("holiday_flag"))
names(holiday_type_binary_flag)

dim(holidays_mapping)
holidays_mapping <- left_join(holidays_mapping, holiday_type_binary_flag)
##########################################


############# branch, state and zone mapping ############
names(train_data)
branch_state_zone_mapping <- unique(train_data[train_data$segment==1,c('branch_id','state','zone')])

state_zone_mapping <- unique(train_data[train_data$segment==1,c('state','zone')])
state_zone_mapping$rm_it_flag <- ifelse(state_zone_mapping$state == 'ORISSA' & state_zone_mapping$zone == 'EAST',1,0)
state_zone_mapping <- state_zone_mapping[state_zone_mapping$rm_it_flag !=1, ]
state_zone_mapping$rm_it_flag <- NULL
#########################################################


####################### Data Preparation #################
# segment 1
seg_1_min_date <- min(train_data$application_date[train_data$segment==1])
seg_1_train_date <- max(train_data$application_date[train_data$segment==1])
seg_1_max_date <- max(submission$application_date[submission$segment==1])
seg_1_dates <- as.character(seq(from=as.Date(seg_1_min_date),to=as.Date(seg_1_max_date),by="1 day"))
seg_1_master_data_set <- merge(unique(train_data$branch_id[train_data$segment==1]), seg_1_dates, all = T)
names(seg_1_master_data_set) <- c('branch_id','application_date')
seg_1_master_data_set$application_date <- as.character(seg_1_master_data_set$application_date)
seg_1_master_data_set$segment <- 1
seg_1_master_data_set <- left_join(seg_1_master_data_set, branch_state_zone_mapping)
sort(colSums(is.na(seg_1_master_data_set)))
seg_1_master_data_set <- left_join(seg_1_master_data_set, train_data)
seg_1_master_data_set$days <- NULL
seg_1_master_data_set <- seg_1_master_data_set[order(seg_1_master_data_set$branch_id, seg_1_master_data_set$application_date), ]
seg_1_master_data_set$case_count[is.na(seg_1_master_data_set$case_count)] <- 0


train_valid_test_split <- unique(data.frame(seg_1_master_data_set[,c('application_date')]))
names(train_valid_test_split)[1] <- 'application_date'
train_valid_test_split$application_date <- as.character(train_valid_test_split$application_date)
train_valid_test_split$data_type <- ifelse(train_valid_test_split$application_date < max(train_data$application_date[train_data$segment==1]), 'train','submission')
dim(train_valid_test_split)
validation_data_seg_1 <- tail(train_valid_test_split[train_valid_test_split$data_type=='train',], n = 87)
validation_data_seg_1$data_type <- ifelse(validation_data_seg_1$application_date < as.Date(min(validation_data_seg_1$application_date))+29, 'validation_M1',
                                 ifelse(validation_data_seg_1$application_date < as.Date(min(validation_data_seg_1$application_date))+58, 'validation_M2','validation_M3'))

table(validation_data_seg_1$data_type)
validation_data_seg_1$segment <- 1

seg_1_master_data_set <- left_join(seg_1_master_data_set, validation_data_seg_1)
seg_1_master_data_set$data_type <- ifelse(seg_1_master_data_set$application_date > seg_1_train_date,'submission',
                                          ifelse(is.na(seg_1_master_data_set$data_type) & seg_1_master_data_set$application_date < seg_1_train_date, 'train', seg_1_master_data_set$data_type ))
seg_1_master_data_set$data_type[is.na(seg_1_master_data_set$data_type)] <- 'Others'


# segment 2
seg_2_min_date <- min(train_data$application_date[train_data$segment==2])
seg_2_train_date <- max(train_data$application_date[train_data$segment==2])
seg_2_max_date <- max(submission$application_date[submission$segment==2])
seg_2_dates <- as.character(seq(from=as.Date(seg_2_min_date),to=as.Date(seg_2_max_date),by="1 day"))
seg_2_master_data_set <- merge(unique(train_data$state[train_data$segment==2]), seg_2_dates, all = T)
names(seg_2_master_data_set) <- c('state','application_date')
seg_2_master_data_set$state <- as.character(seg_2_master_data_set$state)
seg_2_master_data_set$application_date <- as.character(seg_2_master_data_set$application_date)
seg_2_master_data_set$segment <- 2

seg_2_master_data_set <- seg_2_master_data_set[order(seg_2_master_data_set$state, seg_2_master_data_set$application_date), ]
seg_2_master_data_set <- left_join(seg_2_master_data_set, train_data[train_data$segment==2, c('application_date','segment','state','case_count')])
sort(colSums(is.na(seg_2_master_data_set)))
seg_2_master_data_set$case_count[is.na(seg_2_master_data_set$case_count)] <- 0



train_valid_test_split <- unique(data.frame(seg_2_master_data_set[,c('application_date')]))
names(train_valid_test_split)[1] <- 'application_date'
train_valid_test_split$application_date <- as.character(train_valid_test_split$application_date)
train_valid_test_split$data_type <- ifelse(train_valid_test_split$application_date <= max(train_data$application_date[train_data$segment==2]), 'train','submission')
dim(train_valid_test_split)
validation_data_seg_2 <- tail(train_valid_test_split[train_valid_test_split$data_type=='train',], n = 93)
validation_data_seg_2$data_type <- ifelse(validation_data_seg_2$application_date <= as.Date(min(validation_data_seg_2$application_date))+30, 'validation_M1',
                                 ifelse(validation_data_seg_2$application_date <= as.Date(min(validation_data_seg_2$application_date))+61, 'validation_M2','validation_M3'))

table(validation_data_seg_2$data_type)
validation_data_seg_2$segment <- 2

seg_2_master_data_set <- left_join(seg_2_master_data_set, validation_data_seg_2)
seg_2_master_data_set$data_type <- ifelse(seg_2_master_data_set$application_date > seg_2_train_date,'submission',
                                          ifelse(is.na(seg_2_master_data_set$data_type) & seg_2_master_data_set$application_date <= seg_2_train_date, 'train', seg_2_master_data_set$data_type ))

table(seg_2_master_data_set$data_type)

dim(train_data[train_data$segment==1, ]) # 66898
dim(train_data[train_data$segment==2, ]) # 13504

dim(seg_1_master_data_set)               # 75779
dim(seg_2_master_data_set)               # 14992

rm(train_valid_test_split, branch_state_zone_mapping, state_zone_mapping, validation_data_seg_1, validation_data_seg_2)
gc()
###############################################################


###################### date features ###############
date_data <- data.frame(date = unique(c(seg_1_master_data_set$application_date, seg_2_master_data_set$application_date)))
summary(as.Date(date_data$date))
date_data$date <- as.Date(date_data$date)
date_data <- data.table(date_data)

date_data[, ":="( month_alpha = format(date,'%B'),
                  weekdays = weekdays(date),
                  month_week_number = ceiling(as.numeric(format(date, "%d"))/7),
                  weekdays_num = wday(date),
                  month = month(date),
                  quarter = quarter(date),
                  year = year(date),
               
                  month_day = as.numeric(format(date, "%d")),
                  year_week_number = as.numeric(format(date, "%W"))
                )]

names(holidays_mapping)
date_data <- left_join(date_data, holidays_mapping[,c('date','name','type','gazetted_holiday','local','observance','restricted_holiday','season')])
names(date_data)
date_data <- date_data %>% dplyr::group_by(year, month) %>% dplyr::mutate(month_end_date = max(month_day))
date_data$month_end_date <- ifelse(date_data$month_alpha=='October' & date_data$year==2019, 31, date_data$month_end_date)
date_data$temp_key_1 <- ifelse(date_data$month_day <= 7, 'week_1',
                           ifelse(date_data$month_day > 7 & date_data$month_day <= 10, 'week_2_8_10', 
                                   ifelse(date_data$month_day > 10 & date_data$month_day <= 18, 'week_3', 
                                          ifelse(date_data$month_day > 18 & date_data$month_day <= 25, 'week_4','week_5') )))


date_data$custom_date_key <- paste(date_data$temp_key_1, date_data$weekdays, sep = '_')
date_data$custom_date_key <- ifelse(date_data$month_day == date_data$month_end_date, 'month_end', date_data$custom_date_key)
date_data$month_end_date <- NULL
names(date_data)[names(date_data)=='date'] <- 'application_date'
date_data$application_date <- as.character(date_data$application_date)

date_data$gazetted_holiday[is.na(date_data$gazetted_holiday)] <- 0
date_data$local[is.na(date_data$local)] <- 0
date_data$observance[is.na(date_data$observance)] <- 0
date_data$restricted_holiday[is.na(date_data$restricted_holiday)] <- 0
date_data$season[is.na(date_data$season)] <- 0

sort(colSums(is.na(date_data)))
##############################################################

head(date_data)
date_seg_2_master <- seg_2_master_data_set %>% dplyr::group_by(application_date,segment,data_type) %>% dplyr::summarise(case_count = sum(case_count, na.rm = T))
date_seg_2_master <- left_join(date_seg_2_master, date_data)

date_seg_master <- date_seg_2_master
sort(colSums(is.na(date_seg_master)))


##################### creating yearly lag feature (custom_date_key level) #############
names(date_seg_master)
yearly_lag_feature_df <- date_seg_master %>% dplyr::group_by(segment, year, month, custom_date_key) %>% dplyr::summarise(lag_365 = round(mean(case_count)))
yearly_lag_feature_df$year <- yearly_lag_feature_df$year + 1

date_seg_master <- left_join(date_seg_master, yearly_lag_feature_df)
sort(colSums(is.na(date_seg_master)))
######################################################################################


names(date_seg_master)
base_feat <- c('data_type','segment','month_week_number','weekdays_num',"month" , "quarter", "year","month_day", 
                                          "year_week_number","day_of_year", 'case_count')

holiday_flag_feat <- c("gazetted_holiday",   "local",  "observance","restricted_holiday", "season")
lag_feat <- c('custom_date_key','lag_365')

cols_to_select <- unique(c(base_feat, holiday_flag_feat, lag_feat))
date_seg_master <- date_seg_master[, names(date_seg_master) %in% cols_to_select]

date_seg_2_master <- date_seg_master[date_seg_master$segment==2, ]
date_seg_2_master$segment <- NULL
date_seg_2_master$custom_date_key <- as.numeric(as.factor(date_seg_2_master$custom_date_key))



############## MAPE function ################
xgb_mape <- function(preds, dtrain){
  labels <- xgboost::getinfo(dtrain, 'label')
  err <- sum(abs((as.numeric(labels) - as.numeric(preds))/as.numeric(labels)))
  err <- err*100/length(labels)
  return(list(metric='MAPE_custom', value=err))
}
###########################################


########## Model for segment 2 ##################
# 1. CV train 
train_ids <- c('train', 'validation_M1')  
X_train <- date_seg_2_master[date_seg_2_master$data_type %in% train_ids, ]
Y_train <- date_seg_2_master$case_count[date_seg_2_master$data_type %in% train_ids]

X_train$data_type <- NULL
X_train$case_count <- NULL
xgtrain <- xgb.DMatrix(data = as.matrix(X_train), label = Y_train, missing = NA)



# 2. validation
validation_month <- 'validation_M2'
X_test <- date_seg_2_master[date_seg_2_master$data_type %in% validation_month, ]
Y_test <- date_seg_2_master$case_count[date_seg_2_master$data_type %in% validation_month]

X_test$data_type <- NULL
X_test$case_count <- NULL

str(X_test)
xgtest <- xgb.DMatrix(data = as.matrix(X_test), label = Y_test, missing = NA)
watchlist <- list(train = xgtrain, valid = xgtest)


# 3. OOS
validation_month <- 'validation_M3'
valid_X <- date_seg_2_master[date_seg_2_master$data_type %in% validation_month, ]
valid_Y <- date_seg_2_master$case_count[date_seg_2_master$data_type %in% validation_month]

valid_X$data_type <- NULL
valid_X$case_count <- NULL


# 4. final train and test 
train <- date_seg_2_master[date_seg_2_master$data_type=='train' |
                                              date_seg_2_master$data_type %like% 'validation', ]

test <- date_seg_2_master[date_seg_2_master$data_type %like% 'submission', ]

target <- train$case_count
train$case_count <- NULL
test$case_count <- NULL


train$data_type <- NULL
test$data_type <- NULL


####################################### creating the xgb model ########################################
# model hyperparameters
param = list("objective" = "reg:linear",
             # eval_metric =  "rmse" ,
             "eval_metric" =  xgb_mape,     "nthread" = -1,
             "eta" = 0.1, "max_depth" = 10, "subsample" = 0.95, "colsample_bytree" =1,
              "gamma" = 0,"min_child_weight" = 1)

#---------------------------------------- Holdout CV ---------------------------------------------#
set.seed(300)
num_round <- 5000
bst_cv <- xgb.train(param, 
                    data = xgtrain,
                    watchlist = watchlist,
                    num_round, 
                    early_stopping_round = 20,
                    maximize = F,
                    missing = NA)

bst_nrounds <- bst_cv$best_iteration # optimal number of iteration
orig_bst_nrounds <- bst_nrounds

# pred on OOS data 
OOS_pred <- predict (bst_cv, as.matrix(valid_X), missing = NA)
mape_oos <- MAPE(OOS_pred, valid_Y)*100
mape_oos


# training again and then prediction on validation data 
set.seed(500)
bst_model <- xgboost(param=param,
               data =as.matrix(rbind(X_train, X_test)),
               label = c(Y_train,Y_test),
               nrounds=bst_nrounds,  
               missing=NA)

# making predicton on test data 
OOS_pred_2 = predict(bst_model, as.matrix(valid_X),missing=NA)
mape_oos_2 <- MAPE(OOS_pred_2, valid_Y)*100
mape_oos_2
cat('tr score: ',c(bst_cv$evaluation_log[orig_bst_nrounds])[[2]] , 'cv score: ',bst_cv$best_score,'MAPE1: ',mape_oos, 'MAPE2: ', mape_oos_2)



#--------------------------------- Final Model for Test data ----------------------#
# training the final model 
set.seed(500)
bst <- xgboost(param=param,
               data =as.matrix(train),
               label = target,
               nrounds=bst_nrounds/0.8,  
               missing=NA)

# making predicton on test data 
pred = predict(bst, as.matrix(test),missing=NA)
summary(pred)
summary(target)
sum(pred)

# segment 2 submission file
seg_2_subm <- submission[submission$segment==2, ]
seg_2_subm$case_count <- pred
sum(seg_2_subm$case_count)

fwrite(seg_2_subm, './data/final_pred_seg_2.csv')


############## Feature importance plot ##########
names = names(train)
importance_matrix = xgb.importance(names, model=bst)
importance_matrix = xgb.importance(names, model=bst)
gp = xgb.plot.importance(importance_matrix)
print(gp)
#################################################

