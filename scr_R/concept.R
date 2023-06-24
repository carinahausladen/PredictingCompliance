'analysing the chat texts wrt belief, risk and joy. For the latter, Rauhs dictionary is used.'
rm(list=ls())
library(tm)
library(ggplot2)
library(stringr)
library("ggpubr")

setwd('../../scr_R')

# load data
df_chat_hours <- read.csv("../data/chat_hours.csv")
chat <- read.csv("../data/chat.csv")
chat<-chat[sapply(strsplit(as.character(chat$body)," "),length)>2,]

# checking for concepts
docs <- Corpus(VectorSource(chat$body))
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("german"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument)

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 6)

#### positvity score
library("quanteda")
corp <- corpus(as.character(df_chat_hours$Chat_subject))

df_risiko<-as.data.frame(kwic(corp, pattern = "risik*", window = 3, valuetype = "glob"))
df_risiko<-as.data.frame(kwic(corp, pattern = "siche*", window = 3, valuetype = "glob"))

kwic(corp, pattern = "freud*", window = 3, valuetype = "glob")
df_risiko<-as.data.frame(kwic(corp, pattern = ":*", window = 3, valuetype = "glob"))
kwic(corp, pattern = "schön", window = 3, valuetype = "glob")
kwic(corp, pattern = "cool*", window = 3, valuetype = "glob")


#source("create_Rauh.R") # to create data_dictionary_Rauh
positive_words <- data_dictionary_Rauh$positive

df_chat_hours <- df_chat_hours %>%
  mutate(positive_word_count = str_count(Chat_subject, paste(positive_words, collapse="|"))) %>%
  mutate(total_word_count = str_count(Chat_subject, "\\S+")) %>%
  mutate(positive = (positive_word_count / total_word_count) * 100)

df_chat_hours2<-df_chat_hours[!df_chat_hours$positive %in% boxplot.stats(df_chat_hours$positive)$out,] #removing outliers


# plot
g<-ggplot(df_chat_hours2, aes(x=positive, y=player.hours_stated)) + 
  geom_point()+
  geom_smooth(method=lm)+ 
  xlab("positivity score")+
  ylab("hours stated")+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 
ggsave(g, file="../figures/positivity_hours.pdf", height = 4)

kwic(corp, pattern = "andere*", window = 3, valuetype = "glob")
kwic(corp, pattern = "pol*", window = 3, valuetype = "glob")

# correlation coefficient 
cor(df_chat_hours2$positive, df_chat_hours2$player.hours_stated, use="complete.obs",
    method = c("pearson", "kendall", "spearman"))
cor.test(df_chat_hours2$positive, df_chat_hours2$player.hours_stated, method=c("pearson", "kendall", "spearman"))
