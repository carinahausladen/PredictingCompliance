require(quanteda)
require(lubridate)
corp_news <- download("data_corpus_guardian")
corp_news$year <- year(corp_news$date)
corp_news$month <- month(corp_news$date)
corp_news$week <- week(corp_news$date)

corp_news <- corpus_subset(corp_news, "year" >= 2016)
toks_news <- tokens(corp_news, remove_punct = TRUE)

toks_news_lsd <- tokens_lookup(toks_news, dictionary =  data_dictionary_LSD2015[1:2])
head(toks_news_lsd, 2)