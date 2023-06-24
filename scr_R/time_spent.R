library(dplyr)
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library('pander')

setwd('/Users/carinahausladen/FHM')


time <- read.csv("expmt_orga/sessions/TimeSpent (accessed 2020-05-15).csv")
time<-time[!time$participant__id_in_session==14,]  # I needed to play as 14th participant!


plot(sort(time[time$page_name=="intro_gen","seconds_on_page"]))
plot(sort(time[time$page_name=="intro_spec","seconds_on_page"]))
plot(sort(time[time$page_name=="check_understanding_0","seconds_on_page"]))
plot(sort(time[time$page_name=="check_understanding","seconds_on_page"]))

##### Session 18.Mai
time <- read.csv("expmt_orga/sessions/05_18/TimeSpent (accessed 2020-05-18).csv")
df<- read.csv("expmt_orga/sessions/05_18/hausladen_FHM_2020-05-18.csv")
chat<- read.csv("expmt_orga/sessions/05_18/Chat messages (accessed 2020-05-18).csv")

# delete test rounds
df<-df[!df$participant.label=="hausladen_coll_mpg_de"|df$participant.label=="saral_coll_mpg_de",]
df<-df[!df$participant.visited==0,]

p_dropped <- df[df$participant._current_page_name == "outro_alternative", "participant.code"]
time_dropped <- time[time$participant__code %in% p_dropped, c("participant__code", "page_name", "seconds_on_page")]

group_4 <- time_dropped[time_dropped$participant__code=="wipyck3j"|time_dropped$participant__code=="79d67ni1",]  # got stuck on chat
chat_group4 <- chat[chat$participant__code=="wipyck3j"|chat$participant__code=="79d67ni1", "body"]

group_5 <- time_dropped[time_dropped$participant__code=="2590my8r"|time_dropped$participant__code=="9k34mngn",]  # got stuck on chat
chat_group5 <- chat[chat$participant__code=="2590my8r"|chat$participant__code=="9k34mngn", "body"]


# both groups got lost because they exceeded time limit on chat