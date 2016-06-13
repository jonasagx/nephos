filename = "orb_vectors.csv"
# print(filename)
library(cluster)

d <- read.csv(file=filename, head=TRUE, sep=',')

km <- kmeans(d, 12)
dissE <- daisy(d)
sk <- silhouette(km$cl, dissE)

vectors = data.frame()
v <- 1

for (i in 1:length(d$x1)) {
	v <- sk[i, 3]
	if(v > 0.51){
		# cat(v)
		vectors <- rbind(vectors, data.frame(x1 = d[i, 1], y1 = d[i, 2], x2 = d[i, 3], y2 = d[i, 4]))
		# vectors$x1[v] <- d[i, 1]
		# vectors$y1[v] <- d[i, 2]
		# vectors$x2[v] <- d[i, 3]
		# vectors$y2[v] <- d[i, 4]
		# v = v + 1
	}
}

head(vectors, n=100)

chart = "vector_map.png"

png(chart, width=500, height=500)

plot(NA, xlim=c(0, 200), ylim=c(0, 200), main="Wind vectors' map", xlab="latitude", ylab="longitude")
arrows(vectors$x1, vectors$y1, vectors$x2, vectors$y2, length=0.09)

dev.off()
