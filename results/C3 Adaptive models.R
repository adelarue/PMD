library(tidyverse)
library(RColorBrewer)
library(readr)
setwd("Dropbox (MIT)/1 - Research/PHD/results/")
df <- read_csv("fakey/all_results.csv")
df <- read_csv("fakey_nmar/all_results.csv")
df <- read_csv("nmar_outliers/all_results.csv")
df <- read_csv("realy/all_results_rev.csv") %>%  mutate(kMissing = 1)


patterns = read_csv("pattern_counts_numonly.csv") %>% rename(dataset=Name) %>%  filter(p_miss > 0)
  
plot_data = df %>%
  left_join(patterns) %>%
  filter(!is.na(p)) %>%
  filter(method %in% c("Affine", "Static", "Finite", "Oracle XM", "Imp-then-Reg 4", "Complete Features")) %>%
  mutate(fraction = kMissing) %>%
  group_by(method, kMissing) %>%
  summarize(meanr2 = mean(r2), stdr2 = sd(r2) / sqrt(n()),
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


#Plot 2: Out-of-sample R2
plot_data %>%
  #filter(method != "Compl. feat.") %>% filter(method != "Oracle") %>%
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

ggsave("out-of-sample-mar.png",  width = 20, height = 15, dpi = 300)

ggsave("out-of-sample-nmar.png",  width = 20, height = 15, dpi = 300,)
ggsave("out-of-sample-nmar-outliers.png",  width = 20, height = 15, dpi = 300,)

#Plot 3: Win rate
df <- read_csv("fakey/all_results_new.csv") %>%  
  mutate(setting = "1 - Syn MAR")
df <- rbind(df, read_csv("fakey_nmar/all_results_new.csv") %>%  
              mutate(setting = "2 - Syn NMAR") )
df <- rbind(df, read_csv("nmar_outliers/all_results_new.csv") %>%  
              mutate(setting = "3 - Syn MAR adv") )
df_syn <- df %>%
  left_join(patterns) %>%
  filter(!is.na(p)) %>%
  group_by(setting, dataset, k, kMissing, splitnum) %>%
  select(method, osr2) %>%
  spread(key = method, value=osr2) %>%
  filter(!is.na(Affine)) %>%
  filter(!is.na(Finite)) %>%
  #mutate(win = max(Finite,Affine, Static) > max(`Imp-then-Reg 1`,`Imp-then-Reg 2`,`Imp-then-Reg 3`,`Imp-then-Reg 4`)) %>%
  mutate(win = max(Finite,Affine, Static) > max(`Imp-then-Reg 2`, `Imp-then-Reg 4`)) %>%
  mutate(pMissing = kMissing / k) %>% #k - kMissing) %>%
  group_by(setting, dataset, pMissing) %>%
  summarize(winpct = sum(win)/n()) %>%
  ungroup() %>%
  select(setting, dataset, winpct)


df <- read_csv("realy/all_results_rev.csv") %>%  
  mutate(setting = "4 - Real") 
df_real <- df %>%
  left_join(patterns) %>%
  filter(!is.na(p)) %>%
  group_by(setting, dataset, splitnum) %>%
  select(method, osr2) %>%
  spread(key = method, value=osr2) %>%
  filter(!is.na(Affine)) %>%
  filter(!is.na(Finite)) %>%
  #mutate(win = max(Finite,Affine, Static) > max(`Imp-then-Reg 1`,`Imp-then-Reg 2`,`Imp-then-Reg 3`,`Imp-then-Reg 4`)) %>%
  mutate(win = max(Finite,Affine, Static) > max(`Imp-then-Reg 2`, `Imp-then-Reg 4`)) %>%
  group_by(setting, dataset) %>%
  summarize(winpct = sum(win)/n()) %>%
  ungroup() %>%
  select(setting, dataset, winpct)


rbind(df_syn,df_real) %>%

df_real %>%
rbind(df_syn,df_real) %>%
  ggplot() + aes(winpct, group=setting, color=setting, fill=setting) + 
  geom_density(alpha=0.4) +
  labs(x="Average % of wins", y = "Density") +
  geom_vline(xintercept=0.5, size=.8, color="darkgrey") +
  geom_text(aes(x=0.45,y=1.75,label="Majority: \n 0.5"), color="darkgrey", size=6.5) +
  theme(#legend.position = c(0.86, 0.67), 
      #legend.text = element_text(size=22),
      legend.title = element_blank(), axis.title = element_text(size=22), 
      axis.text = element_text(size=22),
      legend.background = element_blank(),
      #legend.box.background = element_rect(color="black"),
      panel.border = element_blank(),
      panel.background = element_blank(),
      panel.grid.major = element_line(colour = "gray", linetype="dashed"),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = "black"),
      #legend.box.margin = margin(0, 0, 0, 0),
      legend.spacing.y = unit(2, "line"))
#ggsave("win-rate-all.png",  width = 17, height = 10, dpi = 300,)
ggsave("win-rate-synonly.png",  width = 17, height = 10, dpi = 300,)
ggsave("win-rate-realonly.png",  width = 17, height = 10, dpi = 300,)


#Plot 1: In-sample R2
plot_data %>%
  #filter(method != "Compl. feat.") %>%
  ggplot() +
  aes(x = kMissing, y = meanr2, fill=method, group=method, color=method, linetype=method) +
  geom_line(size=2) +
  geom_ribbon(aes(x = kMissing, y = meanr2,
                  ymin=meanr2-stdr2, ymax=meanr2+stdr2),
              alpha=0.3) +
  scale_fill_brewer(palette = "Set2", name = "Method") +
  scale_color_brewer(palette = "Set2", name = "Method") +
  labs(linetype = "Method", x = "Missing features in signal",
       y = "In-sample R2") +
  scale_x_continuous(breaks = c(0, 2, 4, 6, 8, 10)) +
  theme(#legend.position = c(0.15, 0.27), 
    legend.text = element_text(size=22),
    legend.title = element_text(size=22), axis.title = element_text(size=22), 
    axis.text = element_text(size=22),
    #legend.background = element_rect(color="white"),
    legend.box.background = element_rect(color="black"),
    panel.border = element_blank(),
    panel.background = element_blank(),
    panel.grid.major = element_line(colour = "gray", linetype="dashed"),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    legend.box.margin = margin(0, 0, 0, 0),
    #legend.spacing.y = unit(1, "line"),
    legend.key.height = unit(2, "line"),
    legend.key.width = unit(2, "line"))

#ggsave("in-sample-nmar.png")
