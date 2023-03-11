#!/bin/bash
for expName in minsize_collectTill
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
                        for tau in 0
                        do
                            for coll in 0.2 0.4 0.6 0.8 1.0
                            do
                                python3 nll.py --ExpName $expName --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll
                                # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll
                                echo $coll done
                            done
                        done
                    done
                done
            done
        done
    done
done