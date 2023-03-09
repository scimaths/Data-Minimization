#!/bin/bash
for mode in 2
do
    for stoc in True
    do
        for val in 60
        do 
            for trainlen in 400 600 800 1000 1200 1400 
            do
                for testlen in 500
                do
                    for tau in 5
                    do
                        for coll in 0.8
                        do
                            python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll
                        done
                    done
                done
            done
        done
    done
done
