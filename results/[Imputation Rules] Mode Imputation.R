setwd("~/Dropbox/Work/1 - Research/PHD/results/")
source("setup_script.R")

df <- rbind(
  read_csv("aistats-rev/fakey/linear_mar/FINAL_results.csv"),
  read_csv("aistats-rev/fakey/linear_nmar/FINAL_results.csv"),
  read_csv("aistats-rev/fakey/linear_mar_adv/FINAL_results.csv"),
  
  read_csv("aistats-rev/fakey/nn_mar/FINAL_results.csv"),
  read_csv("aistats-rev/fakey/nn_nmar/FINAL_results.csv"),
  read_csv("aistats-rev/fakey/nn_mar_adv/FINAL_results.csv"), 
  
  read_csv("aistats-rev/realy/FINAL_results.csv") %>% mutate(kMissing=0)
)

#df %>% select(method) %>% unique() %>% View()

#Claim 1: Mode imputation is detrimental
dataset_list1 <- read_csv("pattern_counts_allfeat.csv") %>% 
  mutate(p_miss_all = p_miss) %>%
  select(Name, p, p_miss_all)
dataset_list2 <- read_csv("pattern_counts_numonly.csv") %>% 
  mutate(p_miss_num = p_miss) %>%
  select(Name, p_miss_num)  
dataset_list <- merge(dataset_list1, dataset_list2, "Name") %>%
  filter(p_miss_all > p_miss_num) %>% #Keep datasets where some discrete/categorical features are missing
  rename(dataset=Name)

mode_df <- merge(df, dataset_list, on='dataset') %>%
  mutate(keep = (startsWith(method, "Imp-then-Reg 4")) + startsWith(method, "Imp-then-Reg 5")) %>%
  filter(keep >= 1) %>%
  mutate(clusterid = paste(dataset,kMissing))

mode_df <- merge(df, dataset_list, on='dataset') %>%
  mutate(keep = (method == "Imp-then-Reg 4 - best") + (method == "Imp-then-Reg 5 - best") ) %>%
  filter(keep >= 1) %>%
  mutate(clusterid = paste(dataset,kMissing))

# merge(df, dataset_list, on='dataset') %>%
#   mutate(keep = (method == "Imp-then-Reg 4 - best") + (method == "Imp-then-Reg 5 - best") ) %>%
#   filter(keep >= 1) %>%
#   mutate(clusterid = paste(dataset,kMissing)) %>%
#   View()


mode_df <- mode_df %>% mutate(Setting = paste(X_setting, Y_setting, sep="_"))


df %>% select(dataset) %>% unique() %>% nrow()
mode_df %>% select(dataset) %>% unique() %>% nrow()


# mode_df %>% 
#   group_by(Setting) %>% 
#   dplyr::summarize(count_dataset = length(unique(dataset)), 
#                    count_k = length(unique(kMissing)), 
#                    count_method = length(unique(method))
#                    ) %>% 
#   View()
  

synmode <- rbind(
  read_csv("aistats-rev/synthetic_discrete/linear_mar/FINAL_results.csv"),
  read_csv("aistats-rev/synthetic_discrete/linear_censoring/FINAL_results.csv"),
  read_csv("aistats-rev/synthetic_discrete/nn_mar/FINAL_results.csv"),
  read_csv("aistats-rev/synthetic_discrete/nn_censoring/FINAL_results.csv")
) %>% 
  mutate(p = 10, p_miss_all=10, p_miss_num=0) %>%
  rename(kMissing = pMissing) %>%
  mutate(keep = 1) %>%
  mutate(clusterid = paste(dataset,kMissing)) %>%
  mutate(Setting = paste(X_setting, Y_setting, sep="_")) %>%
  select(-muvec) %>%
  filter(endsWith(method, "best")) %>%
  mutate(n = strtoi(gsub("_p_.*","", gsub("n_", "", dataset))) )%>%
  filter(n > 20) %>%
  select(-n) %>%
  filter(kMissing < 0.9)
  


mode_df <- rbind(mode_df, synmode[colnames(mode_df)])

# mode_df %>% 
#   group_by(Setting) %>% 
#   dplyr::summarize(count_dataset = length(unique(dataset)), 
#                    count_k = length(unique(kMissing)), 
#                    count_method = length(unique(method)),
#                    count_obs = length(dataset)) %>% 
#   View()

#Linear model: control for dataset and proportion of missing --> Get adjusted R2
#Linear model: control for dataset and proportion of missing + cluster SD --> Get coeff estimates + SE + p-values
#Functions to extract output from regression 
get_coeff <- function(x){
  round(summary(x)['methodImp-then-Reg 5 - best','Estimate'], digits=4)
}
get_se <- function(x){
  ceiling( (1e4)*summary(x)['methodImp-then-Reg 5 - best','Std. Error']) / (1e4)
}
get_pvalue <- function(x){
  signif(summary(x)['methodImp-then-Reg 5 - best','Pr(>|t|)'], digits=2 )
}
get_r2 <- function(x){
  round(summary(x)$adj.r.squared, digits=4)
}

linearreg_analysis <- mode_df %>% 
  nest(data = -Setting) %>% 
  dplyr::mutate(lm.model = map(data, ~lm(osr2 ~ dataset + method +kMissing, data=.)),
         model = map(data, ~lm.cluster(osr2 ~ dataset + method +kMissing, data=., cluster="clusterid")),
         coef = map(model, get_coeff),
         se = map(model, get_se),
         pvalue = map(model, get_pvalue),
         r2 = map(lm.model, get_r2)
         ) %>% 
  unnest(c(coef,se,pvalue,r2)) %>%
  arrange(Setting) %>%
  select(Setting, coef, se, pvalue,r2)

linearreg_analysis %>% View()
write_csv(linearreg_analysis, "../figures/imputation_rules/mode_impute/ModeImpute_RegAnalysis.csv")

#Paired Wilcoxon and t-test
mode_df_wide <- dcast( 
  mode_df %>% 
    mutate(treatment = 1*(method=="Imp-then-Reg 5 - best")) %>%
    select(Setting,dataset,splitnum,kMissing,treatment,osr2), 
  Setting+dataset+splitnum+kMissing ~ treatment, 
  fun.aggregate = mean) 

pairedtest_analysis <-mode_df_wide %>% 
  nest(data = -Setting) %>% 
  mutate(ttest.res = map(data, perform_ttest),
         delta_mean = map(ttest.res, function(x) {round((x$estimate), digits=4) }),
         ttest.pvalue = map(ttest.res, function(x) {signif(x$p.value, digits=2) }),
         wtest.res = map(data, perform_wtest),
         delta_median = map(wtest.res, function(x) {round((x$estimate), digits=4) }),
         wtest.pvalue = map(wtest.res, function(x) {signif(x$p.value, digits=2) }),
         ncomp = map(data, nrow)
  ) %>% 
  unnest(c(delta_mean,ttest.pvalue,delta_median,wtest.pvalue,ncomp)) %>%
  select(Setting, delta_mean,ttest.pvalue, delta_median,wtest.pvalue,ncomp)

#pairedtest_analysis %>% View()

pairedtest_analysis <- merge(pairedtest_analysis, 
     mode_df %>% select(Setting, X_setting, Y_setting) %>% unique(), 
     all.X = T,
     by = 'Setting')

#pairedtest_analysis %>% View()

# Define a function to convert p-values to stars
p_to_stars <- function(p){
  if (p < 1e-20) {
    return("***")
  } else if (p < 1e-10) {
    return("**")
  } else if (p < 1e-5) {
    return("*")
  } else {
    return(as.character(round(p, digits = 3)))
  }
}
pairedtest_analysis <- pairedtest_analysis %>%
    mutate( ttest.pvalue.star = map(ttest.pvalue, p_to_stars),
            wtest.pvalue.star = map(wtest.pvalue, p_to_stars))

write_csv(pairedtest_analysis %>% 
            select(Y_setting, X_setting, ncomp, delta_mean, ttest.pvalue.star, ttest.pvalue, delta_median, wtest.pvalue.star, wtest.pvalue) %>%
            arrange(-ncomp,Y_setting,X_setting), 
            "../figures/imputation_rules/mode_impute/ModeImpute_TestAnalysis.csv"
          )

write_delim(pairedtest_analysis %>% 
            select(Y_setting, X_setting, ncomp, delta_mean, ttest.pvalue.star, ttest.pvalue, delta_median, wtest.pvalue.star, wtest.pvalue) %>%
            arrange(-ncomp,Y_setting,X_setting), "../figures/imputation_rules/mode_impute/ModeImpute_TestAnalysis.txt", delim = " & "
            )

pairedtest_analysis %>% 
  select(Y_setting, X_setting,  ncomp, delta_mean, ttest.pvalue.star, ttest.pvalue, delta_median, wtest.pvalue.star, wtest.pvalue) %>%
  arrange(-ncomp,Y_setting,X_setting) %>%
  View()

itr_df_wide %>% filter(Setting == "real_X_nmar_syn_Y_linear") %>% 
  mutate(d = `1` - `0`) %>%
  View()
