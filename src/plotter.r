filename = "orb_vectors.csv"
print(filename)
library(cluster)

data <- read.csv(file=filename, head=TRUE, sep=',')

c <- data.frame()

for (i in 1:length(data$x1)) {
	if(data[i, 5] < 20){
		c <- rbind(c, data.frame(data[i,]))
	}
}

km <- kmeans(c, 12)
dissE <- daisy(c)
dE2 <- dissE^2
sk <- silhouette(km$cl, dE2)

summary(sk)

vectors = data.frame()

for (i in 1:length(c$x1)) {
	v <- sk[i, 3]
	if(v > 0.6){
		vectors <- rbind(vectors, data.frame(x1 = c[i, 1], y1 = c[i, 2], x2 = c[i, 3], y2 = c[i, 4]))
	}
}

head(vectors, n=100)

chart = "vector_map.png"

png(chart, width=500, height=500)

plot(NA, xlim=c(0, 180), ylim=c(0, 180), main="Mapa de vetores de nuvens", xlab="x", ylab="y")
arrows(vectors$x1, vectors$y1, vectors$x2, vectors$y2, length=0.09)

dev.off()
