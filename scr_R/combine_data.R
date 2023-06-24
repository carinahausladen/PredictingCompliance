rm(list=ls())

library(dplyr)
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library('pander')

setwd('../data/raw_data')

# apps
#s1; includes hours; s2 incldues the other variables
s1_11_26 <- read.csv("11_26/stage_1_2019-11-26.csv")
s1_11_26 <- s1_11_26[s1_11_26$session.code=='lz7dyuw7',]
names(s1_11_26)[names(s1_11_26) == 'player.hours_stated_player'] <- 'player.hours_stated'
plot(sort(s1_11_26$player.hours_stated))  # in this session we had 15 as truth

s1_12_05 <- read.csv("12_05/stage_1_2019-12-05.csv")
s1_12_05 <- s1_12_05[s1_12_05$session.code=='advbl96w',]
plot(sort(s1_12_05$player.hours_stated))  

s1_12_12 <- read.csv("12_12/stage_1_2019-12-12.csv")
s1_12_12 <- s1_12_12[s1_12_12$session.code=='yut5khvr' & !s1_12_12$participant.code=='150ktt1n' & !s1_12_12$participant.code=='u1e8d4fp' ,]
plot(sort(s1_12_12$player.hours_stated))  

s1_12_17 <- read.csv("12_17/stage_1_2019-12-17.csv")
s1_12_17 <- s1_12_17[s1_12_17$session.code=='ooopruyi',]
plot(sort(s1_12_17$player.hours_stated))  

s1_05<-read.csv("05_26_13/hausladen_FHM_2020-05-26.csv")
s1_05<-s1_05[s1_05$participant.label!="" & !s1_05$participant.label=="hausladen_coll_mpg_de" & !s1_05$participant.label=="saral_coll_mpg_de",]
plot(sort(s1_05$player.hours_stated))  # why are there 2 people with 0??
real_participants<-unique(s1_05$participant.code)

colnames(s1_05)

s1<-rbind(s1_11_26[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_12_05[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_12_12[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_12_17[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_05[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")])

hist(s1$player.hours_stated)
write.csv(s1,"hours_stated.csv", row.names = FALSE)

# chat
chat_11_26 <- read.csv("11_26/Chat messages (accessed 2019-11-26).csv")
chat_11_26 <- chat_11_26[chat_11_26$participant__session__code=='lz7dyuw7',]

chat_12_05 <- read.csv("12_05/Chat messages (accessed 2019-12-05).csv")
chat_12_05 <- chat_12_05[chat_12_05$participant__session__code=='advbl96w',]

chat_12_12 <- read.csv("12_12/Chat messages (accessed 2019-12-12).csv")
chat_12_12 <- chat_12_12[chat_12_12$participant__session__code=='yut5khvr' & !chat_12_12$participant__code=='150ktt1n' & !chat_12_12$participant__code=='u1e8d4fp' ,]

chat_12_17 <- read.csv("12_17/Chat messages (accessed 2019-12-17).csv")
chat_12_17 <- chat_12_17[chat_12_17$participant__session__code=='ooopruyi',]

chat_05 <- read.csv("05_26_13/Chat messages (accessed 2020-05-26).csv")
chat_05 <- chat_05[chat_05$participant__code %in% real_participants, ]

chat<-rbind(chat_11_26[,c("participant__code","channel", "body", "timestamp")],
            chat_12_05[,c("participant__code","channel", "body", "timestamp")],
            chat_12_12[,c("participant__code","channel", "body", "timestamp")],
            chat_12_17[,c("participant__code","channel", "body", "timestamp")],
            chat_05[,c("participant__code","channel", "body", "timestamp")])
write.csv(chat,"chat.csv", row.names = FALSE)

# s2 includes the other variables stated
s2_11_26 <- read.csv("11_26/stage_2_2019-11-26.csv")
s2_11_26 <- s2_11_26[s2_11_26$session.code=='lz7dyuw7',]
names(s2_11_26)[names(s2_11_26) == 'player.hours_stated_player'] <- 'player.hours_stated'

s2_12_05 <- read.csv("12_05/stage_2_2019-12-05.csv")
s2_12_05 <- s2_12_05[s2_12_05$session.code=='advbl96w',]

s2_12_12 <- read.csv("12_12/stage_2_2019-12-12.csv")
s2_12_12 <- s2_12_12[s2_12_12$session.code=='yut5khvr' & !s2_12_12$participant.code=='150ktt1n' & !s2_12_12$participant.code=='u1e8d4fp' ,]
s2_12_12<-merge(x = s2_12_12, y = s1_12_12[,c("participant.code", "player.control_correct")], by = "participant.code", all = TRUE)

s2_12_17 <- read.csv("12_17/stage_2_2019-12-17.csv")
s2_12_17 <- s2_12_17[s2_12_17$session.code=='ooopruyi',]
s2_12_17 <-merge(x = s2_12_17, y = s1_12_17[,c("participant.code", "player.control_correct")], by = "participant.code", all = TRUE)

