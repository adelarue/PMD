setwd("Dropbox (MIT)/1 - Research/PHD/results/")
source("setup_script.R")

df <- rbind(
  read_csv("fakey_mar/FINAL_results.csv") %>% mutate(Setting = "1 - Syn-MAR"),
  read_csv("fakey_nmar/FINAL_results.csv") %>% mutate(Setting = "2 - Syn-NMAR"),
  read_csv("fakey_mar_adv/FINAL_results.csv") %>% mutate(Setting = "3 - Syn-NMAR adv"),
  read_csv("realy/FINAL_results.csv") %>% mutate(SNR = 2, k = 10, kMissing=1, Setting = "4 - Real") 
)

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

df %>% select(dataset) %>% unique() %>% nrow()
mode_df %>% select(dataset) %>% unique() %>% nrow()


#Linear model: control for dataset and proportion of missing --> Get adjusted R2
#Linear model: control for dataset and proportion of missing + cluster SD --> Get coeff estimates + SE + p-values
#Functions to extract output from regression 
get_coeff <- function(x){
  round(summary(x)['methodImp-then-Reg 5','Estimate'], digits=4)
}
get_se <- function(x){
  ceiling( (1e4)*summary(x)['methodImp-then-Reg 5','Std. Error']) / (1e4)
}
get_pvalue <- function(x){
  signif(summary(x)['methodImp-then-Reg 5','Pr(>|t|)'], digits=2 )
}
get_r2 <- function(x){
  round(summary(x)$adj.r.squared, digits=4)
}

linearreg_analysis <- mode_df %>% 
  nest(data = -Setting) %>% 
  mutate(lm.model = map(data, ~lm(osr2 ~ dataset + method +kMissing, data=.)),
         model = map(data, ~lm.cluster(osr2 ~ dataset + method +kMissing, data=., cluster="clusterid")),
         coef = map(model, get_coeff),
         se = map(model, get_se),
         pvalue = map(model, get_pvalue),
         r2 = map(lm.model, get_r2)
         ) %>% 
  unnest(c(coef,se,pvalue,r2)) %>%
  arrange(Setting) %>%
  select(Setting, coef, se, pvalue,r2)

write_csv(linearreg_analysis, "ModeImpute_RegAnalysis.csv")

#Paired Wilcoxon and t-test
mode_df_wide <- dcast( 
  mode_df %>% 
    mutate(treatment = 1*(method=="Imp-then-Reg 5")) %>%
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
  ) %>% 
  unnest(c(delta_mean,ttest.pvalue,delta_median,wtest.pvalue)) %>%
  select(Setting, delta_mean,ttest.pvalue, delta_median,wtest.pvalue)

write_csv(pairedtest_analysis, "ModeImpute_TestAnalysis.csv")
