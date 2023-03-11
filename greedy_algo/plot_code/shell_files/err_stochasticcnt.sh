#!/bin/bash
for expName in err_stochasticcnt
do
    for mode in 1 2
    do
        for stoc in True
        do
            for val in 10 50 100 200 500 1000
            do 
                for trainlen in 1000
                do
                    for testlen in 200
                    do
                        for tau in 0
                        do
                            for coll in 0.5
                            do
                                python3 nll.py --ExpName $expName --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll
                                # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll
                                echo $val done
                            done
                        done
                    done
                done
            done
        done
    done
done