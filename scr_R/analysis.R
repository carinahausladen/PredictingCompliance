## Check Understanding
rm(list=ls())
library("ggpubr")
library(tidyverse)
library(rstatix)

setwd('../../scr_R')
source("combine_data.R")
setwd('../../scr_R')

###############################
########## Beliefs ############
###############################

ggscatter(df, x = "player.hours_stated", y = "player.belief", 
          add = "reg.line",
          xlab = "Hours stated", ylab = "Beliefs about hours stated")
ggsave("../figures/cor.pdf")

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

# Moral ####
gfreude<-ggplot(df, aes(x=player.moral_freude)) +
  geom_histogram(position="identity", bins = 12) +
  scale_x_continuous(name="joy", limits=c(0, 10), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
                      panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank(),
                      panel.border = element_blank(),
                      panel.background = element_blank()) 
garger<-ggplot(df, aes(x=player.moral_aerger)) +
  geom_histogram(position="identity", bins = 12) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="anger", limits=c(0, 10), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
gangst<-ggplot(df, aes(x=player.moral_angst)) +
  geom_histogram(position="identity", bins = 12) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="fear", limits=c(0, 10), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
gscham<-ggplot(df, aes(x=player.moral_scham)) +
  geom_histogram(position="identity", bins = 12) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="shame", limits=c(0, 10), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
gschuld<-ggplot(df, aes(x=player.moral_schuld)) +
  geom_histogram(position="identity", bins = 12) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="guilt", limits=c(0, 10), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 

p_all<-ggarrange(gfreude,garger,gangst,gscham, gschuld,
                 ncol=5, nrow=1, common.legend = TRUE, legend="bottom")
ggsave(p_all, file="../figures/moral.eps", device="eps", height = 3)

###
plot(df$player.moral_freude, df$player.hours_stated)
ggscatter(df, x = "player.hours_stated", y = "player.moral_freude", 
          add = "reg.line",
          xlab = "Hours stated", ylab = "Joy experienced ")
ggsave("../figures/joy.pdf")

# other ####
gbelief<-ggplot(df, aes(x=player.belief)) +
  geom_histogram(position="identity", bins = 20) +
  scale_x_continuous(name="beliefs", limits=c(0, 101)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
grisk<-ggplot(df, aes(x=player.risk)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="risk attitude", limits=c(0, 11), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
glie<-ggplot(df, aes(x=player.lie)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="lying attitude", limits=c(0, 11), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
gpol<-ggplot(df, aes(x=player.pol)) +
  geom_histogram(position="identity", bins = 12) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="shame", limits=c(0, 11), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
glab<-ggplot(df, aes(x=player.lab_exp)) +
  geom_histogram(position="identity", bins = 10) +
  scale_x_continuous(name="lab experience", limits=c(0, 6), breaks = c(1:5)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
gcomplex<-ggplot(df, aes(x=player.complex)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="complexity experienced", limits=c(0, 11), breaks = c(1:10)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
gincome<-ggplot(df, aes(x=player.income)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="income", limits=c(0, 3500)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
greligion<-ggplot(df, aes(x=player.pray)) +
  geom_histogram(position="identity", bins = 15) +
  theme(axis.title.y = element_blank() )+
  scale_x_continuous(name="religiosity", limits=c(0, 5.5), breaks = c(1:5)) +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
p_all<-ggarrange(gbelief,grisk,glie,gpol, glab,gcomplex,gincome,greligion,
                 ncol=4, nrow=2, common.legend = TRUE, legend="bottom")
ggsave(p_all, file="../figures/other.eps", device="eps", height = 5)

#######################
# experimenter demand #
######################

paste(table(df$player.exp_demand_chat)[1]/table(df$player.exp_demand_chat)[2]*100,
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


# regression ####
model1<-lm(player.hours_stated~ player.econ + player.belief+ player.moral_freude + player.risk + player.lie + player.pol + player.lab_exp, data = df)
summary(model1)

# checking assumptions ####
mean(model1$residuals) # II: mean of residuals is approx zero
par(mfrow=c(2,2))
plot(model1) # III: Homoscedasticity of residuals: all lines are pretty flat!

# IV: The X variables and residuals are uncorrelated: Yes! pvalues is high: H0 that ture corr is 0 can not be rejected
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

library(stargazer)
stargazer(model1, out = "../figures/reg.tex", type = "latex")

# correlation analysis ####
# checking for normality first 

df %>% shapiro_test(player.hours_stated,
                    player.econ,
                    player.belief,
                    player.moral_freude,
                    player.risk,
                    player.lie,
                    player.pol,
                    player.lab_exp)  # all normally distributed!

variables<-c()
estimates<-c()
p_value<-c()
counter<-0
for (i in c(6,12,17,18,21,23,11)) {
  counter<-counter+1
  c<-cor.test(df[,4], df[,i], method=c("pearson"))
  variables[counter]<-colnames(df)[i]
  estimates[counter]<-c$estimate
  p_value[counter]<-c$p.value
}

variables<-c("Belief", "Joy", "Risk Attitude", "Lying Attitude", "Political Orientation", "Lab Experience", "Econ Classes")
all<-as.data.frame(cbind(variables, estimates, p_value))
rownames(all)<-all$variables
all$estimates<-as.numeric(as.character(all$estimates))
all$p_value<-as.numeric(as.character(all$p_value))

library(xtable)
print(xtable(all[,2:3],
             type = "latex",
             digits = 3),
     file="../data/corr.tex", 
    only.contents = TRUE,
    )

