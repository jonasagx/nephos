filename = "vector_map.csv"
print(filename)

d <- read.csv(file=filename, head=TRUE, sep=',')

png(paste("chart_", filename, sep=''), width=550, height=550)

plot(NA, xlim=c(0, 500), ylim=c(0, 500), main="Wind vectros' map", xlab="latitude", ylab="longitude")
arrows(d$x1, d$y1, d$x2, d$y2, length=0.09)

dev.off()
