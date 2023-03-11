#!/bin/bash
for expName in minsize_tau
do
    for mode in 2
    do
        for stoc in True
        do
            for val in 60
            do 
                for trainlen in 1000
                do
                    for testlen in 10
                    do
                        for tau in 0 0.2 0.5 1.0 1.5 2.0 2.5 5.0
                        do
                            for coll in 0.5
                            do
                                python3 nll.py --ExpName $expName --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll
                                # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll
                                echo $tau done
                            done
                        done
                    done
                done
            done
        done
    done
done