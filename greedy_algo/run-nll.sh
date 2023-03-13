#!/bin/bash
for mode in 2
do
    for stoc in False True
    do
        for val in 50 60 70 80
        do 
            for trainlen in 800 900 1000 1100 1200 1300 1400 1500 1600
            do
                for testlen in 600
                do
                    for tau in 0 0.2 0.5 0.7 1.0 1.2 1.5
                    do
                        for coll in 0.4 0.5 0.6 0.7 0.8 0.9 1.0
                        do
                            for bud in 400 500 600 700 800
                            do
                                python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud
                                # python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll
                            done
                        done
                    done
                done
            done
        done
    done
done
