x=170
echo $x
while [ $x -le 1000 ]
do
    python3 nll.py --Mode 4 --StocValue 100 --StocGrad True --TrainLen 1000 --TestLen 500 --ThreshTau 0 --ThreshCollectTill 0.6 --Budget $x --Seed 0 >> mode_4.txt
    x=$(( $x + 5 ))
done