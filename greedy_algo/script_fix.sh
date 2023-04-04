#!/bin/bash
cd ~/../ashwinr/ashwinr-abirde/Data-Minimization/greedy_algo/logs
x="$(ls | wc -l)"
echo $x
for tau in 0
do
    for stoc in True
    do
        for val in 50 100
        do 
            for trainlen in 1200
            do
                for testlen in 500
                do
                    for mode in 2 1
                    do
                        for bud in 0.2 0.4 0.6 0.8 1.0
                        do
                            for coll in 0.25 0.5 0.75 1.0
                            do
                                x="$(ls | grep reverse_mode-$mode-stochastic_gradient-$stoc-stochastic_value-$val-threshRemoveTill-$coll-budget-$bud-threshTau-$tau-train_len-$trainlen-test_len-$testlen.json | wc -l)"
                                if [ $x -eq 0 ]
                                then
                                    echo python3 nll_reverse.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshRemoveTill $coll --Budget $bud
                                fi
                                x="$(ls | grep mode-$mode-stochastic_gradient-$stoc-stochastic_value-$val-threshCollectTill-$coll-budget-$bud-threshTau-$tau-train_len-$trainlen-test_len-$testlen.json | wc -l)"
                                if [ $x -eq 0 ]
                                then
                                    echo python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud
                                fi
                                # python3 nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud
                                # echo "nll.py --Mode $mode --StocValue $val --StocGrad $stoc --TrainLen $trainlen --TestLen $testlen --ThreshTau $tau --ThreshCollectTill $coll --Budget $bud"
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
