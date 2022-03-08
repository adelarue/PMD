library(dplyr)
library(readr)
setwd("Dropbox (MIT)/1 - Research/PHD/results/")
df <- read_csv("fakey/all_results.csv")
df <- read_csv("fakey_nmar/all_results.csv")
df <- read_csv("nmar_outliers/all_results.csv")
df <- read_csv("realy/all_results_rev.csv") %>% mutate(kMissing=1) #For realy only

#Claim 1: Mode imputation is detrimental
dataset_list1 <- read_csv("pattern_counts_allfeat.csv") %>% 
  mutate(p_miss_all = p_miss) %>%
  select(Name, p, p_miss_all)
dataset_list2 <- read_csv("pattern_counts_numonly.csv") %>% 
  mutate(p_miss_num = p_miss) %>%
  select(Name, p_miss_num)  
dataset_list <- merge(dataset_list1, dataset_list2, "Name") %>%
  filter(p_miss_all > p_miss_num) %>%
  rename(dataset=Name)

mode_df <- merge(df, dataset_list, on='dataset') %>%
  mutate(keep = (method == "Imp-then-Reg 4") + (method == "Imp-then-Reg 5")) %>%
  filter(keep >= 1) %>%
  mutate(clusterid = paste(dataset,kMissing))


#mode_df %>%
#  group_by(dataset, kMissing, method) %>%
#  summarize(freq = sum(kMissing >= 0)) %>%
#  filter(freq < 10) %>%
#  View()

#Linear model: control for dataset and proportion of missing
model <- lm(osr2 ~ dataset + method +kMissing, data=mode_df)
summary(model)

#Linear model: control for dataset and proportion of missing + cluster SD
library(miceadds)
model <- lm.cluster(osr2 ~ dataset + method , data=mode_df, cluster=mode_df$dataset)
summary(model)

library(reshape2)
mode_df_wide <- dcast( mode_df %>% 
  mutate(treatment = 1*(method=="Imp-then-Reg 5")) %>%
  select(dataset,splitnum,kMissing,treatment,osr2), 
  dataset+splitnum+kMissing ~ treatment, fun.aggregate = mean) 

t.test(mode_df_wide$`1`, mode_df_wide$`0`, paired=TRUE, alternative="less")

wilcox.test(mode_df_wide$`1`, mode_df_wide$`0`, alternative="less", paired=TRUE, conf.int=TRUE)

mean(mode_df_wide$`1`-mode_df_wide$`0`)
median(mode_df_wide$`1`)-median(mode_df_wide$`0`)
