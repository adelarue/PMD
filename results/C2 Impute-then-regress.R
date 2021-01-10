library(dplyr)
library(readr)
setwd("Dropbox (MIT)/1 - Research/PHD/results/")
df <- read_csv("fakey/all_results.csv")
df <- read_csv("fakey_nmar/all_results.csv")
df <- read_csv("nmar_outliers/all_results.csv")
df <- read_csv("realy/all_results.csv") %>%  mutate(kMissing = 1)

#Claim 2: Mean impute not so bad
dataset_list <- read_csv("pattern_counts_numonly.csv") %>% 
  filter(p_miss > 0) %>%
  mutate(dataset=Name) %>%
  select(dataset, p_miss)

itr_df <- merge(df, dataset_list, on='dataset') %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(method <= "Imp-then-Reg 4") %>%
  mutate(method = ifelse(method=="Imp-then-Reg 4", "Imp-then-Reg", method))


#itr_df %>%
#  group_by(dataset, kMissing, method) %>%
#  summarize(freq = sum(kMissing >= 0)) %>%
#  filter(freq < 10) %>%
#  View()

#Linear model: control for dataset and proportion of missing
model <- lm(osr2 ~ dataset + kMissing + method , data=itr_df)
summary(model)

#Linear model: control for dataset and proportion of missing + cluster SD
library(miceadds)
model <- lm.cluster(osr2 ~ dataset + kMissing + method , data=itr_df, cluster=itr_df$dataset)
summary(model)

library(reshape2)
method0 = "Imp-then-Reg 3"
method1 = "Imp-then-Reg 2"
itr_df_wide <- dcast( itr_df %>% 
                        mutate(treatment = 1*(method==method1), control=1*(method==method0)) %>%
                        filter(treatment+control > 0) %>%
                        select(dataset,kMissing,splitnum,treatment,osr2), 
                       dataset+splitnum+kMissing~ treatment, fun.aggregate = mean) 

t.test(itr_df_wide$`1`, itr_df_wide$`0`, paired=TRUE)
mean(itr_df_wide$`1`-itr_df_wide$`0`, na.rm=T)

wilcox.test(itr_df_wide$`1`, itr_df_wide$`0`, paired=TRUE, conf.int=TRUE)



#Computational time
df <- read_csv("fakey/all_results.csv") %>%  
  mutate(setting = "1 - Syn MAR") %>% 
  select(-SNR,-k,-kMissing,-pMissing)
df <- rbind(df, read_csv("fakey_nmar/all_results.csv") %>%  
  mutate(setting = "2 - Syn NMAR") %>% 
  select(-SNR,-k,-kMissing,-pMissing))
df <- rbind(df, read_csv("nmar_outliers/all_results.csv") %>%  
  mutate(setting = "3 - Syn MAR adv") %>% 
  select(-SNR,-k,-kMissing,-pMissing))
df <- rbind(df, read_csv("realy/all_results.csv") %>%  mutate(setting = "4 - Real"))

library(ggplot2)
library(stringr)

df %>%
  left_join(dataset_list) %>%
  filter(method_cat == "Imp-then-Reg") %>%
  filter(method < "Imp-then-Reg 5") %>%
  #filter(splitnum <= 10) %>%
  mutate(Method = str_replace(method, "Imp-then-Reg ", "V")) %>%
  group_by(Method, setting) %>%
  summarize(time = mean(time)) %>%
  ggplot() + aes(x=setting, y=time, fill=Method, color=Method) + 
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
