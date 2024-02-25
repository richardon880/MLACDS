grep 'Average 0' res.dat > average_0.dat
grep 'Average 1' res.dat > average_1.dat

cat average_0.dat | awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }'
cat average_1.dat | awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }'
