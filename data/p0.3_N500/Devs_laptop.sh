for i in {1..10}
do
   echo "**************** RUN $i ***********************"
   ./r_all.sh
   ./clean.sh .
done

