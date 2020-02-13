library(data.table)
library(dplyr)

# segment 1 xgb
seg_1_xgb <- fread('./data/pred_seg_1_xgb.csv')
names(seg_1_xgb)[names(seg_1_xgb)=='case_count'] <- 'xgb_case_count'


# segment 1 Tbats
seg_1_tbats <- fread('./data/tbats_seg1.csv')
names(seg_1_tbats)[names(seg_1_tbats)=='case_count'] <- 'tbats_case_count'

seg_1_xgb_tbats <- left_join(seg_1_xgb, seg_1_tbats)
seg_1_xgb_tbats$case_count <- seg_1_xgb_tbats$xgb_case_count*0.2 + seg_1_xgb_tbats$tbats_case_count*0.8


# segment 2 xgb 
seg_2_xgb <- fread('./data/final_pred_seg_2.csv')

final_pred_seg_1_2 <- rbind(seg_1_xgb_tbats[,c('id','application_date','segment','case_count')],
                            seg_2_xgb)

fwrite(final_pred_seg_1_2, './data/final_pred_seg_1_2.csv')


