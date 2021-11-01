library(ggplot2)

data1 <- read.csv("data/original/heart.csv")

plot(data1)
hist(data1$п.їage)
hist(data1$sex, breaks = 2)

ggplot(data = data1, aes(x = factor(order), fill = sex)) + 
  geom_bar(subset=.(sex==0)) + 
  geom_bar(subset=.(sex==1),aes(y=..count..*(-1))) + 
  coord_flip() +
  theme(text = element_text(size=16)) +
  scale_y_continuous(breaks=seq(-40,40,10),labels=abs(seq(-40,40,10))) + 
  scale_fill_brewer(palette="Dark2") +
  facet_wrap(~year)

p <- ggplot(data=data1, aes(x=data$sex)) + geom_histogram()
