setwd("Dropbox (MIT)/1 - Research/PHD/results/")
source("setup_script.R")

df <- rbind(
  read_csv("nonlinear/fakey_mar/FINAL_results.csv") %>% mutate(Setting = "1 - Syn-MAR"),
  read_csv("nonlinear/fakey_nmar/FINAL_results.csv") %>% mutate(Setting = "2 - Syn-NMAR"),
  read_csv("nonlinear/fakey_mar_adv/FINAL_results.csv") %>% mutate(Setting = "3 - Syn-NMAR adv"),
  #read_csv("realy/FINAL_results.csv") %>% mutate(SNR = 2, k = 10, kMissing=1, Setting = "4 - Real")
)

#Claim 2: Mean impute not so bad
dataset_list <- read_csv("pattern_counts_numonly.csv") %>% 
  filter(p_miss > 0) %>%
  mutate(dataset=Name) %>%
  select(dataset, p_miss)

itr_df <- merge(df, dataset_list, on='dataset') %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(method <= "Imp-then-Reg 4") %>%
  mutate(method = ifelse(method=="Imp-then-Reg 4", "Imp-then-Reg", method))


df %>% select(dataset) %>% unique() %>% nrow()
itr_df %>% select(dataset) %>% unique() %>% nrow()


#Linear model: control for dataset and proportion of missing

get_coeff <- function(x,n="1"){
  round(summary(x)[paste('methodImp-then-Reg',n),'Estimate'], digits=4)
}
get_se <- function(x,n="1"){
  ceiling( (1e4)*summary(x)[paste('methodImp-then-Reg',n),'Std. Error']) / (1e4)
}
get_pvalue <- function(x,n="1"){
  signif(summary(x)[paste('methodImp-then-Reg',n),'Pr(>|t|)'], digits=2 )
}
get_r2 <- function(x){
  round(summary(x)$adj.r.squared, digits=4)
}

linearreg_analysis <- itr_df %>% 
  nest(data = -Setting) %>% 
  mutate(lm.model = map(data, ~lm(osr2 ~ dataset + method +kMissing, data=.)),
         model = map(data, ~lm.cluster(osr2 ~ dataset + method +kMissing, data=., cluster="dataset")),
         coef1 = map(model, function(x){get_coeff(x,n="1")}),
         se1 = map(model, function(x){get_se(x,n="1")}),
         pvalue1 = map(model, function(x){get_pvalue(x,n="1")}),
         
         coef2 = map(model, function(x){get_coeff(x,n="2")}),
         se2 = map(model, function(x){get_se(x,n="2")}),
         pvalue2 = map(model, function(x){get_pvalue(x,n="2")}),         
         
         coef3 = map(model, function(x){get_coeff(x,n="3")}),
         se3 = map(model, function(x){get_se(x,n="3")}),
         pvalue3 = map(model, function(x){get_pvalue(x,n="3")}),
         
         r2 = map(lm.model, get_r2)
  ) %>% 
  unnest(c(coef1,se1,pvalue1, coef2,se2,pvalue2, coef3,se3,pvalue3, r2)) %>%
  arrange(Setting) %>%
  select(Setting, coef1,se1,pvalue1, coef2,se2,pvalue2, coef3,se3,pvalue3, r2)

linearreg_analysis %>% View()
write_csv(linearreg_analysis, "ImputeThenReg_RegAnalysis.csv")


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


#Paired t- and Wilson-tests V4 vs Vx
method0 = "Imp-then-Reg" #Same as "Imp-then-Reg 4"

for (i in 1:3){
  method1 = paste("Imp-then-Reg", i)
  
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
  
  
  write_csv(pairedtest_analysis, paste("ImputeThenReg_",i,"vs4_TestAnalysis.csv", sep=""))
}

#Computational time
library(ggplot2)
library(stringr)

df %>%
  left_join(dataset_list) %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(method < "Imp-then-Reg 5") %>%
  #filter(splitnum <= 10) %>%
  mutate(Method = str_replace(method, "Imp-then-Reg ", "V")) %>%
  group_by(Method, Setting) %>%
  summarize(time = mean(time)) %>%
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
