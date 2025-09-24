# 训练

对应 configs/ 底下有对应需要训练的eqtransformer的配置，可以把目前集群上有保存的原始数据集开始训练
```python
python train.py --config config/ethz_eqtransformer.json
```

训练完后，会得到一个 weights/{name}/{version} 的文件夹，里面有模型权重和日志

# 评估

评估脚本在 benchmark/eval.py，使用方式如下
```python
python benchmark/eval.py weights/ethz_eqtransformer/0 targets/stead
```
其中targets是作者将数据集整理得到的材料，可以通过[text](https://dcache-demo.desy.de:2443/Helmholtz/HelmholtzAI/SeisBench/auxiliary/pick-benchmark/targets/)下载，或者benchmark/genrate_eval_targets.py脚本生成

对应 in-domain 的 测试，结果会保存在pred/中
对应 cross-domain 的测试，结果会保存在 pred_cross/ 中

# 结果收集

评估完后，可以使用 benchmark/collect_results.py 脚本收集结果，生成csv文件
```python
python collect_results.py pred results.csv
python collect_results.py pred_cross results_cross.csv
```