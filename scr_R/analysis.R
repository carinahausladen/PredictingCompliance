rm(list=ls())
library("ggpubr")
library(tidyverse)
library(rstatix)
library(stargazer)

setwd('../../scr_R')
source("combine_data.R")
setwd('../../scr_R')

###############################
########## Beliefs ############
###############################

ggscatter(df, y = "player.hours_stated", x = "player.belief", 
          add = "reg.line",
          ylab = "Hours stated", xlab = "Beliefs about hours stated")

ggqqplot(df$player.hours_stated, ylab = "Hours stated")
ggqqplot(df$player.belief, ylab = "Beliefs about hours stated")

###############################
####### Socio demographics ####
###############################
mean(df$player.age)
paste('share females:',
      round(table(df$player.gender)[2]/sum(table(df$player.gender)), digits = 3))
paste('share Fachbereich Wirtschaftswisseschaft:',
      round(table(df$player.study)[3]/sum(table(df$player.study)), digits = 2))
paste('share Bachelor:',
      round(table(df$player.study_level)[1]/sum(table(df$player.study_level)), digits = 2))
table(df$player.econ)  # 1= JA, 2 = Nein

###############################
# Moral ####
###############################
base_theme <- theme(axis.line = element_line(colour = "black"),
                    panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_blank(),
                    axis.title.y = element_blank())

gfreude <- ggplot(df, aes(x=player.moral_freude)) +
  geom_histogram(position="identity", bins = 12) +
  scale_x_continuous(name="joy", limits=c(0, 10), breaks = c(1, 5, 10)) +  
  base_theme

garger <- ggplot(df, aes(x=player.moral_aerger)) +
  geom_histogram(position="identity", bins = 12) +
  scale_x_continuous(name="anger", limits=c(0, 10), breaks = c(1, 5, 10)) +  
  base_theme

gangst <- ggplot(df, aes(x=player.moral_angst)) +
  geom_histogram(position="identity", bins = 12) +
  scale_x_continuous(name="fear", limits=c(0, 10), breaks = c(1, 5, 10)) +  
  base_theme

gscham <- ggplot(df, aes(x=player.moral_scham)) +
  geom_histogram(position="identity", bins = 12) +
  scale_x_continuous(name="shame", limits=c(0, 10), breaks = c(1, 5, 10)) +  # Modified this line
  base_theme

gschuld <- ggplot(df, aes(x=player.moral_schuld)) +
  geom_histogram(position="identity", bins = 12) +
  scale_x_continuous(name="guilt", limits=c(0, 10), breaks = c(1, 5, 10)) +  
  base_theme

p_all <- ggarrange(gfreude, garger, gangst, gscham, gschuld,
                   widths = rep(1.342, 5),
                   ncol=5, nrow=1, common.legend = TRUE, legend="bottom")
p_all
ggsave(p_all, file="../../figures/moral.pdf", device="pdf", width = 6.71, height = 1.342)


###############################
# other ####
###############################
gbelief<-ggplot(df, aes(x=player.belief)) +
  geom_histogram(position="identity", bins = 20) +
  scale_x_continuous(name="beliefs", limits=c(0, 101)) +
  base_theme
grisk<-ggplot(df, aes(x=player.risk)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="risk attitude", limits=c(0, 11), breaks = c(1:10)) +
  base_theme
glie<-ggplot(df, aes(x=player.lie)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="lying attitude", limits=c(0, 11), breaks = c(1:10)) +
  base_theme
gpol<-ggplot(df, aes(x=player.pol)) +
  geom_histogram(position="identity", bins = 12) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="shame", limits=c(0, 11), breaks = c(1:10)) +
  base_theme
glab<-ggplot(df, aes(x=player.lab_exp)) +
  geom_histogram(position="identity", bins = 10) +
  scale_x_continuous(name="lab experience", limits=c(0, 6), breaks = c(1:5)) +
  base_theme
gcomplex<-ggplot(df, aes(x=player.complex)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="complexity experienced", limits=c(0, 11), breaks = c(1:10)) +
  base_theme
gincome<-ggplot(df, aes(x=player.income)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="income", limits=c(0, 3500)) +
  base_theme
greligion<-ggplot(df, aes(x=player.pray)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="religiosity", limits=c(0, 5.5), breaks = c(1:5)) +
  base_theme

p_all<-ggarrange(gbelief,grisk,glie,gpol, glab,gcomplex,gincome,greligion,
                 widths = rep(6.71/4, 4),
                 ncol=4, nrow=2, 
                 common.legend = TRUE, legend="bottom")
ggsave(p_all, file="../../figures/other.pdf", device="pdf", height = 5)

#######################
# experimenter demand #
######################

paste(round(table(df$player.exp_demand_chat)[1]/table(df$player.exp_demand_chat)[2]*100, digits = 2),
      "% stated that they changed their chat behavior due to the anticipated presence of the experimenter")
df %>% 
  filter(df$player.exp_demand_chat==1,
         player.exp_demand_chat_open != "",
         nchar(player.exp_demand_chat_open) > 2) %>%
  select(player.exp_demand_chat_open) %>%
  mutate(annotation = c("more honest", "spelling/grammar/style", "less text", 
                        "spelling/grammar/style", "spelling/grammar/style", 
                        "behavior", "more text", "more honest", "less text", 
                        "less text", "less text", "less text", "less text")) #annotation generated by first author

#######################
### number of words ###
#######################
df_chat_hours <- read.csv("../../data/df_chat_hours.csv")
df_chat_hours_word_count <- df_chat_hours %>%
  group_by(participant.code) %>%
  summarise(total_words = sum(sapply(strsplit(as.character(Chat_subject), " "), length)))
df = df %>%
  left_join(df_chat_hours_word_count, by = "participant.code")


#######################
####### regression ####
#######################
model1<-lm(player.hours_stated~ player.moral_freude+player.risk+player.belief+player.lab_exp+total_words+player.lie+player.pol+player.econ, data = df)
names_map <- c(
  "(Intercept)" = "(Intercept)",  # If you don't want to rename the intercept, you can keep it as is
  "player.moral_freude" = "Joy",
  "player.risk" = "Risk Attitude",
  "player.belief" = "Belief",
  "player.lab_exp" = "Lab Experience",
  "player.lie" = "Lying Attitude",
  "player.pol" = "Political Orientation",
  "player.econ" = "Econ Classes",
  "total_words" = "Total Words"
)
attr(model1$terms, "term.labels") <- names_map[attr(model1$terms, "term.labels")]
names(model1$coefficients) <- names_map[names(model1$coefficients)]
summary(model1)
stargazer(model1, out = "../../figures/reg.tex", type = "latex")

# checking assumptions ####
mean(model1$residuals) # II: mean of residuals is approx zero
par(mfrow=c(2,2))
plot(model1) # III: Homoscedasticity of residuals: all lines are pretty flat!

# IV: The X variables and residuals are uncorrelated: Yes! pvalues is high: H0 that true corr is 0 can not be rejected
cor.test(df$player.econ, model1$residuals)[["p.value"]]
cor.test(df$player.belief, model1$residuals)[["p.value"]]
cor.test(df$player.moral_freude, model1$residuals)[["p.value"]]
cor.test(df$player.risk, model1$residuals)[["p.value"]]
cor.test(df$player.lie, model1$residuals)[["p.value"]]
cor.test(df$player.pol, model1$residuals)[["p.value"]]
cor.test(df$player.lab_exp, model1$residuals)[["p.value"]]

library(car)
vif(model1) # no perfect multicollinearity

library(gvlma)  #checking all assumptions
gvlma::gvlma(model1)

df %>% shapiro_test(player.hours_stated,
                    player.econ,
                    player.belief,
                    player.moral_freude,
                    player.risk,
                    player.lie,
                    player.pol,
                    player.lab_exp)  # all normally distributed!
