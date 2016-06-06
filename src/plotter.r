filename = "orb_vectors.csv"
# print(filename)

d <- read.csv(file=filename, head=TRUE, sep=',')

chart = "vector_map.png"

png(chart, width=500, height=500)

plot(NA, xlim=c(0, 200), ylim=c(0, 200), main="Wind vectros' map", xlab="latitude", ylab="longitude")
arrows(d$x1, d$y1, d$x2, d$y2, length=0.09)

dev.off()
