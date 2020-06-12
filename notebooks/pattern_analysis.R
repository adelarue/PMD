library(tidyverse)

data = read_csv("../results/pattern_counts.csv")

data %>%
  filter(Num_Patterns > 1) %>%
  filter(Num_Patterns < 10) %>%
  view()
