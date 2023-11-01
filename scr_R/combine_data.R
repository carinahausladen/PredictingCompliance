rm(list=ls())

library(dplyr)
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library('pander')

setwd('../data/raw_data')

# the experiment was conducted in two stages: the first stage was in November and December 2019 at FU Berlin.
# the second stage was in May 2020 at MPI Bonn where the laboratory was larger and therefore a larger session could be conducted.

#s1; includes hours; s2 includes the other variables; 
#however, s1_05 was a version of the app where only one DF was exported so there is no s1 and s2

#########################################
# s1
#########################################

s1_11_26 <- read.csv("11_26/stage_1_2019-11-26.csv")
s1_11_26 <- s1_11_26[s1_11_26$session.code=='lz7dyuw7',] # in this session we had 50 as truth
names(s1_11_26)[names(s1_11_26) == 'player.hours_stated_player'] <- 'player.hours_stated'

s1_12_05 <- read.csv("12_05/stage_1_2019-12-05.csv")
s1_12_05 <- s1_12_05[s1_12_05$session.code=='advbl96w',]

s1_12_12 <- read.csv("12_12/stage_1_2019-12-12.csv")
s1_12_12 <- s1_12_12[s1_12_12$session.code=='yut5khvr' & !s1_12_12$participant.code=='150ktt1n' & !s1_12_12$participant.code=='u1e8d4fp' ,]

s1_12_17 <- read.csv("12_17/stage_1_2019-12-17.csv")
s1_12_17 <- s1_12_17[s1_12_17$session.code=='ooopruyi',]

s1_05<-read.csv("05_26_13/hausladen_FHM_2020-05-26.csv")

s1<-rbind(s1_11_26[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_12_05[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_12_12[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_12_17[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")],
          s1_05[,c("session.code", "group.id_in_subsession", "participant.code", "player.hours_stated")])

#########################################
# s2 includes the other variables stated
#########################################

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


myvars<-c(
  "session.code", "group.id_in_subsession", "participant.code", "player.hours_stated",
  "player.control_correct",
  "player.belief",
  "player.age",
  "player.gender",
  "player.study",
  "player.study_level",
  "player.econ",
  "player.moral_freude",
  "player.moral_aerger",
  "player.moral_angst",
  "player.moral_scham",
  "player.moral_schuld",
  "player.risk",
  "player.lie", 
  "player.complex",
  "player.income",
  "player.pol", 
  "player.pray",
  "player.lab_exp", 
  "player.exp_demand_open",
  "player.exp_demand_chat",
  "player.exp_demand_chat_open"
)

df<-rbind(s2_11_26[myvars],
          s2_12_05[myvars],
          s2_12_12[myvars],
          s2_12_17[myvars],
          s1_05[myvars])
df %>%
  filter(!is.na(player.age)) %>%
  filter(!player.hours_stated==0) %>%
  filter(!is.na(player.hours_stated)) -> df
write.csv(df,"/Users/carinah/Documents/GitHub/PredictingCompliance/data/df.csv", row.names = FALSE)


s1 %>%
  filter(!player.hours_stated==0) %>%
  filter(participant.code %in% df$participant.code) %>%
  filter(!is.na(player.hours_stated))-> s1
write.csv(s1,"/Users/carinah/Documents/GitHub/PredictingCompliance/data/hours_stated.csv", row.names = FALSE)



#########################################
# chat
#########################################

chat_11_26 <- read.csv("11_26/Chat messages (accessed 2019-11-26).csv")
chat_11_26 <- chat_11_26[chat_11_26$participant__session__code=='lz7dyuw7',]

chat_12_05 <- read.csv("12_05/Chat messages (accessed 2019-12-05).csv")
chat_12_05 <- chat_12_05[chat_12_05$participant__session__code=='advbl96w',]

chat_12_12 <- read.csv("12_12/Chat messages (accessed 2019-12-12).csv")
chat_12_12 <- chat_12_12[chat_12_12$participant__session__code=='yut5khvr' & !chat_12_12$participant__code=='150ktt1n' & !chat_12_12$participant__code=='u1e8d4fp' ,]

chat_12_17 <- read.csv("12_17/Chat messages (accessed 2019-12-17).csv")
chat_12_17 <- chat_12_17[chat_12_17$participant__session__code=='ooopruyi',]

chat_05 <- read.csv("05_26_13/Chat messages (accessed 2020-05-26).csv")

chat<-rbind(chat_11_26[,c("participant__code","channel", "body", "timestamp")],
            chat_12_05[,c("participant__code","channel", "body", "timestamp")],
            chat_12_12[,c("participant__code","channel", "body", "timestamp")],
            chat_12_17[,c("participant__code","channel", "body", "timestamp")],
            chat_05[,c("participant__code","channel", "body", "timestamp")])

length(unique(chat$participant__code))
length(unique(df$participant.code))
chat %>% 
  filter(participant__code %in% df$participant.code) -> chat
write.csv(chat,"/Users/carinah/Documents/GitHub/PredictingCompliance/data/chat.csv", row.names = FALSE)

# df and s1
objs_to_keep <- c("chat", "df", "s1")
objs_to_remove <- setdiff(ls(), objs_to_keep)
rm(list = objs_to_remove)

