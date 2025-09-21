CONFIG=configs/geofon_seismoe.json
WEIGHTS=weights/geofon_seismoe

# train
CUDA_VISIBLE_DEVICES=0 python benchmark/train.py --config $CONFIG

# eval
CUDA_VISIBLE_DEVICES=0 python benchmark/eval.py $WEIGHTS targets/stead
CUDA_VISIBLE_DEVICES=0 python benchmark/eval.py $WEIGHTS targets/geofon

# collect results
python benchmark/collect_results.py pred results.csv
python benchmark/collect_results.py pred_cross --cross results_cross.csv