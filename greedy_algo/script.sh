#!/bin/bash
i=0
for tau in 0
do
    for stoc in True
    do
        for val in 50 100
        do 
            for trainlen in 400 600 800 1000 1200 1400 1600
            do
                for testlen in 500
                do
                    for mode in 2 1
                    do
                        for bud in 0.2 0.4 0.6 0.8 1.0
                        do
                            for coll in 0.25 0.5 0.75 1.0
                            do
                                python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud
                                echo "nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud"
                                python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll --Budget $bud
                                echo "nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll --Budget $bud"
                            done
                        done
                    done
                done
            done
        done
    done
done
