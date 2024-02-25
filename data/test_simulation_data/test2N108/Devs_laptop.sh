for i in {1..25}
do
   echo "**************** RUN $i ***********************"
   ./r_all.sh
   ./clean.sh .
done

