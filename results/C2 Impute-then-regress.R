setwd("~/Dropbox/Work/1 - Research/PHD/results/")
source("setup_script.R")

df <- rbind(
  read_csv("fakey/linear_mar/FINAL_results.csv"),
  read_csv("fakey/linear_nmar/FINAL_results.csv"),
  read_csv("fakey/linear_mar_adv/FINAL_results.csv"),
  
  read_csv("fakey/nn_mar/FINAL_results.csv"),
  read_csv("fakey/nn_nmar/FINAL_results.csv"),
  read_csv("fakey/nn_mar_adv/FINAL_results.csv"), 
  
  read_csv("realy/FINAL_results.csv") %>% mutate(kMissing=1)
)

dataset_list <- read_csv("pattern_counts_numonly.csv") %>% 
  filter(p_miss > 0) %>%
  mutate(dataset=Name) %>%
  select(dataset, p_miss)

################################
##Claim 2: Mean impute not so bad
itr_df <- merge(df, dataset_list, on='dataset') %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(endsWith(method, "best")) %>%
  filter(method < "Imp-then-Reg 5") %>%
  mutate(method = ifelse(method=="Imp-then-Reg 4 - best", "Imp-then-Reg", method))

itr_df <- itr_df %>% mutate(Setting = paste(X_setting, Y_setting, sep="_"))


df %>% select(dataset) %>% unique() %>% nrow()
itr_df %>% select(dataset) %>% unique() %>% nrow()

itr_df %>% 
  group_by(Setting) %>% 
  dplyr::summarize(count_dataset = length(unique(dataset)), 
                   count_k = length(unique(kMissing)), 
                   count_method = length(unique(method))
  ) %>% 
  View()


synmean <- rbind(
  read_csv("synthetic/linear_mar/FINAL_results.csv"),
  read_csv("synthetic/linear_censoring/FINAL_results.csv"),
  read_csv("synthetic/nn_mar/FINAL_results.csv"),
  read_csv("synthetic/nn_censoring/FINAL_results.csv")
) %>% 
  mutate(p_miss=10) %>%
  rename(kMissing = pMissing) %>%
  #mutate(clusterid = paste(dataset,kMissing)) %>%
  mutate(Setting = paste(X_setting, Y_setting, sep="_")) %>%
  select(-muvec) %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(endsWith(method, "best")) %>%
  filter(method < "Imp-then-Reg 5") %>%
  mutate(method = ifelse(method=="Imp-then-Reg 4 - best", "Imp-then-Reg", method)) %>%
  mutate(n = strtoi(gsub("_p_.*","", gsub("n_", "", dataset))) )%>%
  filter(n <= 1000) %>%
  select(-n)

itr_df_save <- itr_df

df %>% select(method) %>% unique() %>% View()

itr_df <- rbind(itr_df_save, 
                synmean[colnames(itr_df)] %>% filter(kMissing < 0.9)
                )

itr_df %>% 
  group_by(Setting) %>% 
  dplyr::summarize(count_dataset = length(unique(dataset)), 
                   count_k = length(unique(kMissing)), 
                   count_method = length(unique(method)),
                   count_obs = length(dataset)) %>% 
  View()

#Paired t- and Wilson-tests V4 vs Vx
method0 = "Imp-then-Reg" #Same as "Imp-then-Reg 4"

for (i in 1:3){
  method1 = paste("Imp-then-Reg", i, "- best")
  
  itr_df_wide <- dcast( itr_df %>% 
                          mutate(treatment = 1*(method==method1), control=1*(method==method0)) %>%
                          filter(treatment+control > 0) %>%
                          select(Setting,dataset,kMissing,splitnum,treatment,osr2), 
                        Setting+dataset+splitnum+kMissing ~ treatment, 
                        fun.aggregate = mean) 
  
  pairedtest_analysis <-itr_df_wide %>% 
    nest(data = -Setting) %>% 
    mutate(ttest.res = map(data, perform_ttest),
           delta_mean = map(ttest.res, function(x) {round((x$estimate), digits=4) }),
           ttest.pvalue = map(ttest.res, function(x) {signif(x$p.value, digits=2) }),
           wtest.res = map(data, perform_wtest),
           delta_median = map(wtest.res, function(x) {round((x$estimate), digits=4) }),
           wtest.pvalue = map(wtest.res, function(x) {signif(x$p.value, digits=2) })
    ) %>% 
    unnest(c(delta_mean,ttest.pvalue,delta_median,wtest.pvalue)) %>%
    select(Setting, delta_mean,ttest.pvalue, delta_median,wtest.pvalue)
  
  pairedtest_analysis <- merge(pairedtest_analysis, 
                               itr_df %>% select(Setting, X_setting, Y_setting) %>% unique(), 
                               all.X = T,
                               by = 'Setting')
  
  write_csv(pairedtest_analysis %>% 
              select(Y_setting, X_setting, delta_mean, ttest.pvalue, delta_median, wtest.pvalue) %>%
              arrange(Y_setting,X_setting), 
            paste("ImputeThenReg_",i,"vs4_TestAnalysis.csv", sep=""))
}



itr_df <- itr_df %>% mutate(Setting = paste(X_setting, Y_setting, kMissing, sep="_"))
for (i in 2:2){
  method1 = paste("Imp-then-Reg", i, "- best")
  
  itr_df_wide <- dcast( itr_df %>% 
                          mutate(treatment = 1*(method==method1), control=1*(method==method0)) %>%
                          filter(treatment+control > 0) %>%
                          select(Setting,dataset,kMissing,splitnum,treatment,osr2), 
                        Setting+dataset+splitnum+kMissing ~ treatment, 
                        fun.aggregate = mean) 
  
  pairedtest_analysis <-itr_df_wide %>% 
    nest(data = -Setting) %>% 
    mutate(ttest.res = map(data, perform_ttest),
           delta_mean = map(ttest.res, function(x) {round((x$estimate), digits=4) }),
           ttest.pvalue = map(ttest.res, function(x) {signif(x$p.value, digits=2) }),
           wtest.res = map(data, perform_wtest),
           delta_median = map(wtest.res, function(x) {round((x$estimate), digits=4) }),
           wtest.pvalue = map(wtest.res, function(x) {signif(x$p.value, digits=2) })
    ) %>% 
    unnest(c(delta_mean,ttest.pvalue,delta_median,wtest.pvalue)) %>%
    select(Setting, delta_mean,ttest.pvalue, delta_median,wtest.pvalue)
  
  pairedtest_analysis <- merge(pairedtest_analysis, 
                               itr_df %>% select(Setting, X_setting, Y_setting, kMissing) %>% unique(), 
                               all.X = T,
                               by = 'Setting')
  
  write_csv(pairedtest_analysis %>% 
              select(Y_setting, X_setting, kMissing, delta_mean, ttest.pvalue, delta_median, wtest.pvalue) %>%
              arrange(Y_setting,X_setting,kMissing), 
            paste("ImputeThenReg_",i,"vs4_TestAnalysis_perMissingLevel.csv", sep=""))
}



################################
##Claim 2: Implementation of mice-impute
synmean <- rbind(
  read_csv("synthetic/linear_mar/FINAL_results.csv"),
  read_csv("synthetic/linear_censoring/FINAL_results.csv"),
  read_csv("synthetic/nn_mar/FINAL_results.csv"),
  read_csv("synthetic/nn_censoring/FINAL_results.csv")
) %>% 
  mutate(p_miss=10) %>%
  rename(kMissing = pMissing) %>%
  select(-muvec) %>%
  filter(method_cat == "Imp-then-Reg")

itr_df <- rbind( merge(df, dataset_list, on='dataset'), synmean) %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(endsWith(method, "best")) %>%
  filter(method < "Imp-then-Reg 4") %>%
  mutate(method = ifelse(method=="Imp-then-Reg 2 - best", "Imp-then-Reg", method))

itr_df <- itr_df %>% mutate(Setting = paste(X_setting, Y_setting, sep="_"))

itr_df %>% 
  group_by(Setting) %>% 
  dplyr::summarize(count_dataset = length(unique(dataset)), 
                   count_k = length(unique(kMissing)), 
                   count_method = length(unique(method))
  ) %>% 
  View()


#Linear model: control for dataset and proportion of missing
get_coeff <- function(x,n="1"){
  round(summary(x)[paste('methodImp-then-Reg',n,'- best'),'Estimate'], digits=4)
}
get_se <- function(x,n="1"){
  ceiling( (1e4)*summary(x)[paste('methodImp-then-Reg',n,'- best'),'Std. Error']) / (1e4)
}
get_pvalue <- function(x,n="1"){
  signif(summary(x)[paste('methodImp-then-Reg',n,'- best'),'Pr(>|t|)'], digits=2 )
}
get_r2 <- function(x){
  round(summary(x)$adj.r.squared, digits=4)
}

lm.model = lm(osr2 ~ dataset + method +kMissing, data=itr_df)
summary(lm.model)

linearreg_analysis <- itr_df %>% 
  nest(data = -Setting) %>% 
  mutate(lm.model = map(data, ~lm(osr2 ~ dataset + method +kMissing, data=.)),
         model = map(data, ~lm.cluster(osr2 ~ dataset + method +kMissing, data=., cluster="dataset")),
         coef1 = map(model, function(x){get_coeff(x,n="1")}),
         se1 = map(model, function(x){get_se(x,n="1")}),
         pvalue1 = map(model, function(x){get_pvalue(x,n="1")}),
         
         # coef2 = map(model, function(x){get_coeff(x,n="2")}),
         # se2 = map(model, function(x){get_se(x,n="2")}),
         # pvalue2 = map(model, function(x){get_pvalue(x,n="2")}),         
         
         coef3 = map(model, function(x){get_coeff(x,n="3")}),
         se3 = map(model, function(x){get_se(x,n="3")}),
         pvalue3 = map(model, function(x){get_pvalue(x,n="3")}),
         
         r2 = map(lm.model, get_r2)
  ) %>% 
  unnest(c(coef1,se1,pvalue1, coef3,se3,pvalue3, r2)) %>%
  arrange(Setting) %>%
  select(Setting, coef1,se1,pvalue1, coef3,se3,pvalue3, r2)

linearreg_analysis <- merge(linearreg_analysis, 
                             itr_df %>% select(Setting, X_setting, Y_setting) %>% unique(), 
                             all.X = T,
                             by = 'Setting')

linearreg_analysis %>% View()

write_csv(linearreg_analysis %>% 
            select(Y_setting, X_setting, coef1,se1,pvalue1, coef3,se3,pvalue3, r2) %>%
            arrange(Y_setting,X_setting), 
          "ImputeThenReg_RegAnalysis.csv")


#Computational time
library(ggplot2)
library(stringr)

rbind(df, synmean %>% select(colnames(df))) %>%
  left_join(dataset_list) %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(method < "Imp-then-Reg 5") %>%
  filter(endsWith(method, "- best")) %>%
  #filter(splitnum <= 10) %>%
  mutate(Method = str_replace(method, "- best", "")) %>%
  mutate(Method = str_replace(Method, "Imp-then-Reg ", "V")) %>%
  mutate(Signal = startsWith(Y_setting, "real_Y")) %>%
  mutate(Xse = startsWith(X_setting, "real_X")) %>%
  mutate(Setting = case_when(Signal & Xse ~ "Real",
                              Xse ~ "Semi-synthetic",
                              TRUE ~ "Synthetic")) %>%
  mutate(Method = case_when(Method == "V4 " ~ "Mean-impute",
                            TRUE ~ Method)) %>%
  group_by(Method, Setting) %>%
  dplyr::summarize(time = mean(time)) %>%
  ggplot() + aes(x=Setting, y=time, fill=Method, color=Method) + 
  geom_col(position="dodge",alpha=0.7) +
  labs(x="Setting", y = "Average time (in seconds)") +
  scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1") +
  theme(
    panel.border = element_blank(),
    panel.background = element_blank(),
    panel.grid.major = element_line(colour = "grey", linetype="dashed"),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    #legend.box.margin = margin(0, 0, 0, 0),
    legend.spacing.y = unit(2, "line"))

ggsave("ImputeThenReg_Time.png",  width = 7, height = 5, dpi = 300)




##################

#Paired t- and Wilson-tests V2 vs V3
library(reshape2)
method0 = "Imp-then-Reg 3"
method1 = "Imp-then-Reg 2"
itr_df_wide <- dcast( itr_df %>% 
                        mutate(treatment = 1*(method==method1), control=1*(method==method0)) %>%
                        filter(treatment+control > 0) %>%
                        select(Setting,dataset,kMissing,splitnum,treatment,osr2), 
                      Setting+dataset+splitnum+kMissing~ treatment, 
                      fun.aggregate = mean) 

pairedtest_analysis <-itr_df_wide %>% 
  nest(data = -Setting) %>% 
  mutate(ttest.res = map(data, perform_ttest),
         delta_mean = map(ttest.res, function(x) {round((x$estimate), digits=4) }),
         ttest.pvalue = map(ttest.res, function(x) {signif(x$p.value, digits=2) }),
         wtest.res = map(data, perform_wtest),
         delta_median = map(wtest.res, function(x) {round((x$estimate), digits=4) }),
         wtest.pvalue = map(wtest.res, function(x) {signif(x$p.value, digits=2) })
  ) %>% 
  unnest(c(delta_mean,ttest.pvalue,delta_median,wtest.pvalue)) %>%
  select(Setting, delta_mean,ttest.pvalue, delta_median,wtest.pvalue)

pairedtest_analysis %>% View()
write_csv(pairedtest_analysis, "ImputeThenReg_2vs3_TestAnalysis.csv")




