#!/bin/bash
for mode in 2
do
    for stoc in True
    do
        for val in 60
        do 
            for trainlen in 1200 
            do
                for testlen in 500
                do
                    for tau in 0
                    do
                        for coll in 0.5 0.6 0.8 0.9 1.0
                        do
                            python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll
                            # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll
                        done
                    done
                done
            done
        done
    done
done
