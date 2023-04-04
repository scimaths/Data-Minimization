#!/bin/bash
for seed in 0
do
    for tau in 0
    do
        for stoc in True
        do
            for val in 100
            do 
                for trainlen in 1000
                do
                    for testlen in 500
                    do
                        for mode in 3
                        do
                            for bud in 0.2
                            do
                                for coll in 0.25
                                do
                                    # python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud --Seed $seed
                                    echo "nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud"
                                    # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll --Budget $bud
                                    # echo "nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll --Budget $bud"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done