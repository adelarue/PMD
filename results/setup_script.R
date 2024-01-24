library(dplyr)
library(readr)

library(miceadds)
library(broom)
library(tidyverse)

library(reshape2)

#Functions to extract output from statistical tests
perform_ttest <- function(x){
  t.test(x$`1`, x$`0`, paired=TRUE, alternative="two.sided")
}
perform_wtest <- function(x){
  wilcox.test(x$`1`, x$`0`, paired=TRUE, alternative="two.sided", conf.int=TRUE)
}
