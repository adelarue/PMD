library(tidyverse)

data = read_csv("../results/pattern_counts.csv")

data_new %>%
  view()

data %>%
  filter(Num_Patterns > 1) %>%
  filter(Num_Patterns < 10) %>%
  view()

data_new = read_csv("../results/pattern_counts_new.csv")

data_new %>%
  summary

data_new %>%
  filter(Num_Patterns > 1) %>%
  view()

data_new %>%
  #filter(!(Name %in% data$Name)) %>%
  ggplot() +
  aes(x = n, y = p, size = Good_Turing_Prob) +
  geom_point() +
  scale_x_log10() +
  scale_y_log10()

data_new %>%
  filter(Good_Turing_Prob > 0) %>%
  ggplot() +
  aes(x = Good_Turing_Prob) +
  geom_histogram() +
  scale_x_log10()


