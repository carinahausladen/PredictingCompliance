library("quanteda")
library("dplyr")

# load dictionary dataframes (downloaded here: https://github.com/kbenoit/quanteda.dictionaries/tree/master/sources/Rauh)
load("/Users/carinah/Downloads/Rauh_SentDictionaryGerman.Rdata")
load("/Users/carinah/Downloads/Rauh_SentDictionaryGerman_Negation.Rdata")

# new column where NOT and word are divided with a space
neg.sent.dictionary <- neg.sent.dictionary %>% 
  mutate(word = gsub("NOT_", "NOT ", feature)) %>% 
  mutate(sentiment = ifelse(sentiment == 1, "neg_negative", "neg_positive"))

sent.dictionary <- sent.dictionary %>% 
  mutate(word = feature) %>% 
  mutate(sentiment = ifelse(sentiment == -1, "negative", "positive"))

# bind both dataframes
sent_dictionary_rauh <- bind_rows(sent.dictionary, neg.sent.dictionary)

# save as quanteda dictionary (word and sentiment column)
data_dictionary_Rauh <- quanteda::as.dictionary(sent_dictionary_rauh)
