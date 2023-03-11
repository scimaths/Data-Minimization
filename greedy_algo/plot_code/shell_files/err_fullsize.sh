#!/bin/bash
for expName in err_fullsize
do
    for mode in 1 2
    do
        for stoc in True
        do
            for val in 60
            do 
                for trainlen in 100 200 500 1000 2000 5000 10000 
                do
                    for testlen in 10
                    do
                        for tau in 0
                        do
                            for coll in 0.5
                            do
                                python3 nll.py --ExpName $expName --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll
                                # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll
                                echo $trainlen done
                            done
                        done
                    done
                done
            done
        done
    done
done