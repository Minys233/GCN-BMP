## DDI based on graph representation learning

### Train from scratch

```python
python train_ddi_modify_eval2.py --method=ggnn --epoch=200 --learning-rate=0.001 --exp-shift-rate=0.5 --weight-tying=False --fp-hidden-dim=32 --conv-layers=8 --net-hidden-dims= --out=result_total_ggnn_epoch200_lr0.001_exp0.5_wt0_fph32_conv8_nh_hole_x44_0816_2 --gpu=2 --train-datafile=dataset/interaction/isc/ddi_ib_isc35000_train.csv --valid-datafile=dataset/interaction/isc/ddi_ib_isc35000_valid.csv --sim-method=hole --augment=True
```


The pretrained model and associated results are stored in: ```ddi/output/best```

