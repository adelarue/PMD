setwd("~/Documents/MIT/Grad/PHD/notebooks")
library(tidyverse)
library(RColorBrewer)

# read data
data = read_csv("../results/fakey_nmar/arrhythmia_SNR_2_nmiss_0_1.csv") %>%
  head(0)
for (filename in list.files("../results/fakey_nmar/")) {
  data = data %>%
    rbind(read_csv(paste("../results/fakey_nmar/", filename, sep="")))
}
# write data to single file
# data %>% write_csv("../results/fakey_nmar.csv")

# on subsequent runs, load from file
data = read_csv("../results/fakey_nmar.csv")

patterns = read_csv("../results/pattern_counts_new.csv") %>%
  rename(dataset=Name)

patterns %>%
  filter(p_miss > 0) %>%
  View()

data %>%
  left_join(patterns) %>%
  filter(p_miss > 0) %>%
  filter(Num_Patterns > 50) %>%
  # filter(dataset == "thyroid-disease-sick") %>%
  group_by(method, kMissing) %>%
  summarize(meanosr2 = mean(osr2), stdosr2 = sd(osr2) / sqrt(n())) %>%
  ggplot() +
  aes(x = kMissing, y = meanosr2, color=method, group=method, linetype=method) +
  geom_line() #+
  # geom_ribbon(aes(x = kMissing, y = meanosr2, ymin=meanosr2-stdosr2, ymax=meanosr2+stdosr2,
  #                 fill=method),
  #             alpha=0.1)

# Round 2 - nmar_outliers

# read data
data = read_csv("../results/nmar_outliers/thyroid-disease-allrep_k_10_kmiss_5_iter_4.csv") %>%
  head(0)
for (filename in list.files("../results/nmar_outliers/")) {
  data = data %>%
    rbind(read_csv(paste("../results/nmar_outliers/", filename, sep="")))
}
# write data to single file
# data %>% write_csv("../results/nmar_outliers.csv")

# on subsequent runs, load from file
data = read_csv("../results/nmar_outliers.csv")

data %>%
  filter(k == 10) %>%
  filter(kMissing == 10) %>%
  select(dataset) %>%
  unique()

data %>%
  filter(dataset == "ozone-level-detection-eight") %>%
  View()

data %>%
  group_by(dataset, k) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  group_by(dataset) %>%
  summarize(count = n()) %>%
  View()

data %>%
  filter(method %in% c("Affine", "Static", "Oracle XM", "Imp-then-Reg 4")) %>%
  filter(osr2 < -0.1) %>%
  filter(kMissing == 5) %>%
  group_by(dataset) %>%
  summarize(count = n(), osr2 = mean(osr2))

data %>%
  filter(dataset == "thyroid-disease-allhyper") %>%
  filter(k == 5 & kMissing == 5) %>%
  # filter(osr2 < 0) %>%
  View()

data %>%
  filter(method %in% c("Affine", "Static", "Oracle XM", "Imp-then-Reg 4",
                       "Complete Features")) %>%
  left_join(patterns) %>%
  filter(Good_Turing_Prob > -5) %>%
  # filter(k > 5) %>%
  # filter((dataset %in% c("sleep", "thyroid-disease-thyroid-0387",
  #                         "thyroid-disease-allbp",
  #                         "thyroid-disease-allhyper",
  #                         "thyroid-disease-allhypo",
  #                         "thyroid-disease-allrep",
  #                         "thyroid-disease-dis",
  #                         "thyroid-disease-sick"))) %>%
  mutate(fraction = kMissing) %>%
  group_by(method, kMissing) %>%
  summarize(meanr2 = mean(r2), stdr2 = sd(r2) / sqrt(n()),
            q1r2 = quantile(osr2, 0.25), q3r2 = quantile(osr2, 0.75),
            meanosr2 = mean(osr2), stdosr2 = sd(osr2) / sqrt(n())) %>%
  ggplot() +
  aes(x = kMissing, y = meanr2, fill=method, group=method, color=method, linetype=method) +
  geom_line() +
  geom_ribbon(aes(x = kMissing, y = meanr2,
                  #ymin=q1r2, ymax=q3r2
                  ymin=meanr2-stdr2, ymax=meanr2+stdr2
                  ),
              alpha=0.3)

data %>%
  group_by(dataset, k, kMissing, splitnum) %>%
  select(method, osr2) %>%
  spread(key = method, value=osr2) %>%
  filter(!is.na(Affine)) %>%
  mutate(win = max(Affine, Static) > max(`Imp-then-Reg 1`, `Imp-then-Reg 2`, `Imp-then-Reg 3`,
                                         `Imp-then-Reg 4`, `Imp-then-Reg 5`)) %>%
  mutate(kNonMissing = k - kMissing) %>%
  group_by(kMissing, kNonMissing) %>%
  summarize(winpct = sum(win)/n()) %>%
  ggplot() +
  aes(x = kMissing, y = kNonMissing, fill=winpct) +
  scale_fill_gradient2(low = "red", mid = "yellow", high = "green", midpoint=0.5,
                       breaks = c(0, 0.5, 1), labels = c("0%", "50%", "100%"),
                       name = c("Adaptive win\npercentage")) +
  geom_tile(color="black") +
  scale_x_continuous(breaks = c(0, 2, 4, 6, 8, 10), name = "Missing features in signal") +
  scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10), name = "Non-missing features in signal") +
  theme(legend.position = c(0.86, 0.67), legend.text = element_text(size=22),
        legend.title = element_text(size=22), axis.title = element_text(size=22), 
        axis.text = element_text(size=22),
        legend.background = element_blank(),
        legend.box.background = element_rect(color="black"),
        panel.border = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_line(colour = "gray", linetype="dashed"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.box.margin = margin(0, 0, 10, 0),
        legend.spacing.y = unit(2, "line"))

ggsave("~/Desktop/winrate.png")


