x=60
echo $x
while [ $x -le 1000 ]
do
    python3 nll.py --Mode 2 --StocValue 100 --StocGrad True --TrainLen 1000 --TestLen 500 --ThreshTau 0 --ThreshCollectTill 1.0 --Budget $x --Seed 0 >> mode_2.txt
    x=$(( $x + 5 ))
done