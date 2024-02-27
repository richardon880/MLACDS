for i in {1..50}
do
   echo "**************** RUN $i ***********************"
   ./r_all.sh
   ./clean.sh .
done

