for i in {1..100}
do
   echo "**************** RUN $i ***********************"
   ./r_all.sh
   ./clean.sh .
done

