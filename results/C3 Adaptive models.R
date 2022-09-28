setwd("Dropbox (MIT)/1 - Research/PHD/results/")
source("setup_script.R")

library(RColorBrewer)

df <- rbind(
  read_csv("fakey/linear_mar/FINAL_results.csv"),
  read_csv("fakey/linear_nmar/FINAL_results.csv"),
  read_csv("fakey/linear_mar_adv/FINAL_results.csv"),
  
  read_csv("fakey/nn_mar/FINAL_results.csv"),
  read_csv("fakey/nn_nmar/FINAL_results.csv"),
  read_csv("fakey/nn_mar_adv/FINAL_results.csv"), 
  
  read_csv("realy/FINAL_results.csv") %>% mutate(kMissing=1)
)

patterns = read_csv("pattern_counts_numonly.csv") %>% 
  rename(dataset=Name) %>%  
  filter(p_miss > 0) %>%
  select(dataset)


df %>% select(method) %>% unique() %>% View()

df %>% filter(X_setting == "real_X") %>%select(dataset) %>% unique() %>% nrow()
df %>% filter(X_setting != "real_X") %>%select(dataset) %>% unique() %>% nrow()


#Plot 3: Win rate
library(Hmisc)
#mlist = c("Imp-then-Reg 4 - best", "Adaptive LR - best")
mlist = c("Imp-then-Reg 4 - best", "Joint Imp-then-Reg - best")

df_winrate <- df %>%
  left_join(patterns) %>%
  filter(method %in% mlist) %>%
  group_by(X_setting, Y_setting, dataset, k, kMissing, splitnum) %>%
  select(method, osr2) %>%
  spread(key = method, value=osr2) %>%
  mutate(win = max(`Joint Imp-then-Reg - best`) > max(`Imp-then-Reg 4 - best`)) %>%
  #mutate(win = max(`Adaptive LR - best`) > max(`Imp-then-Reg 4 - best`)) %>%
  mutate(pMissing = kMissing / k) %>% #k - kMissing) %>%
  group_by(X_setting, Y_setting, dataset, pMissing) %>%
  dplyr::summarize(winpct = sum(win)/n()) %>%
  ungroup() %>%
  select(X_setting, Y_setting, dataset, winpct)

df_winrate %>%
  filter(Y_setting != "real_Y") %>%
  ggplot() + aes(winpct, group=X_setting, color=X_setting, fill=X_setting) + 
  geom_density(alpha=0.4) +
  labs(x="Average % of wins", y = "Density") +
  ylim(0, 2.3) +
  geom_vline(xintercept=0.5, size=.8, color="darkgrey") +
  geom_text(aes(x=0.45,y=1.75,label="Majority: \n 0.5"), color="darkgrey", size=6.5) +
  theme(#legend.position = c(0.86, 0.67), 
    #legend.text = element_text(size=22),
    legend.title = element_blank(), axis.title = element_text(size=22), 
    axis.text = element_text(size=22),
    legend.background = element_blank(),
    legend.text = element_text(size=16),
    #legend.box.background = element_rect(color="black"),
    panel.border = element_blank(),
    panel.background = element_blank(),
    panel.grid.major = element_line(colour = "gray", linetype="dashed"),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    #legend.box.margin = margin(0, 0, 0, 0),
    legend.spacing.y = unit(2, "line"))

ggsave("win-rate-syn.png",  width = 12, height = 7, dpi = 300)


df_winrate %>%
  filter(Y_setting == "real_Y") %>%
  ggplot() + aes(winpct, group=X_setting, color=X_setting, fill=X_setting) + 
  geom_density(alpha=0.4) +
  labs(x="Average % of wins", y = "Density") +
  ylim(0, 2.3) +
  geom_vline(xintercept=0.5, size=.8, color="darkgrey") +
  geom_text(aes(x=0.45,y=1.75,label="Majority: \n 0.5"), color="darkgrey", size=6.5) +
  theme(#legend.position = c(0.86, 0.67), 
    #legend.text = element_text(size=22),
    legend.title = element_blank(), axis.title = element_text(size=22), 
    axis.text = element_text(size=22),
    legend.background = element_blank(),
    legend.text = element_text(size=16),
    #legend.box.background = element_rect(color="black"),
    panel.border = element_blank(),
    panel.background = element_blank(),
    panel.grid.major = element_line(colour = "gray", linetype="dashed"),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    #legend.box.margin = margin(0, 0, 0, 0),
    legend.spacing.y = unit(2, "line"))

ggsave("win-rate-realy.png",  width = 12, height = 7, dpi = 300)





## OUTPUT 1: 

mlist = c("Imp-then-Reg 4 - best",
          "Joint Imp-then-Reg - best", "Adaptive LR - best")

merge(df, patterns, on='dataset') %>%
  #inner_join(df %>% filter(X_setting == "real_X") %>%select(dataset) %>% unique()) %>%
  filter(method %in% mlist) %>%
  filter((osr2 < -10)) %>%
  select(method, dataset, X_setting, osr2) %>% View()


merge(df, patterns, on='dataset') %>%
  #inner_join(df %>% filter(X_setting == "real_X") %>%select(dataset) %>% unique()) %>%
  filter(method %in% mlist) %>%
  filter(!is.na(osr2)) %>%
  group_by(method, X_setting) %>%
  summarize(osr2_mean = mean(osr2), osr2_se =  sd(osr2)/sqrt(length(osr2))) %>%
  ggplot() + aes(x=X_setting, y =osr2_mean, fill=method) + 
  geom_col(position='dodge') 


+
  scale_y_continuous(limits=c(0.5,0.7))
#  geom_errorbar(aes(x=X_setting, ymin=osr2_mean-osr2_se, ymax=osr2_mean+osr2_se, fill=method))
  

View()


  
plot_data = df %>%
  left_join(patterns) %>%
  filter(!is.na(p)) %>%
  filter(method %in% c("Affine", "Static", "Finite", "Oracle XM", "Imp-then-Reg 4", "Complete Features")) %>%
  mutate(fraction = kMissing) %>%
  group_by(Setting, method, kMissing) %>%
  dplyr::summarize(meanr2 = mean(r2), stdr2 = sd(r2) / sqrt(n()),
            q1r2 = quantile(osr2, 0.25, na.rm=T), 
            q3r2 = quantile(osr2, 0.75, na.rm=T),
            meanosr2 = median(osr2, na.rm=T), stdosr2 = (q3r2 - q1r2)/2)
#            meanosr2 = mean(osr2), stdosr2 = sd(osr2) / sqrt(n()))

plot_data$method = recode_factor(plot_data$method,
                                 `Oracle XM` = "Oracle",
                                 `Complete Features` = "Compl. feat.",
                                 `Imp-then-Reg 4` = "Mean-imp.",
                                 Affine = "Affine",
                                 Static = "Static",
                                 Finite = "Finite")

synset = c("1 - Syn-MAR", "2 - Syn-NMAR", "3 - Syn-NMAR adv")
for (s in synset) {
  #Plot 2: Out-of-sample R2
  plot_data %>%
    filter(Setting == s) %>%
    filter(kMissing > 0) %>%
    ggplot() +
    aes(x = kMissing, y = meanosr2, fill=method, group=method, color=method, linetype=method) +
    geom_line(size=2) +
    #geom_ribbon(aes(x = kMissing, y = meanosr2, ymin=meanosr2-stdosr2, ymax=meanosr2+stdosr2), alpha=0.3) +
    scale_fill_brewer(palette = "Set2", name = "Method") +
    scale_color_brewer(palette = "Set2", name = "Method") +
    labs(linetype = "Method", x = "Missing features in signal",
         y = "Out-of-sample R2") +
    scale_x_continuous(breaks = c(0, 2, 4, 6, 8, 10)) +
    theme(
          legend.position = "bottom", 
          legend.text = element_text(size=32),
          legend.title = element_text(size=32), 
          axis.title = element_text(size=32), 
          axis.text = element_text(size=32),
          axis.line = element_line(colour = "black"),
          #legend.background = element_rect(color="white"),
          panel.border = element_blank(),
          panel.background = element_blank(),
          panel.grid.major = element_line(colour = "gray", linetype="dashed"),
          panel.grid.minor = element_blank(),
          legend.box.margin = margin(0, 0, 0, 0),
          legend.box.background = element_rect(color="black"),
          #legend.spacing.y = unit(1, "line"),
          legend.key.height = unit(2, "line"),
          legend.key.width = unit(2, "line"))
  
  ggsave(paste("out-of-sample-",s,".png",sep=""),  width = 20, height = 15, dpi = 300)
}


