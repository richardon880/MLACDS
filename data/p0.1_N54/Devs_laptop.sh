for i in {1..200}
do
   echo "**************** RUN $i ***********************"
   ./r_all.sh
   ./clean.sh .
done

