*NOTE: Our poposed dataset is not included in the repository*

PLEASE REFER TO OUR [WEBSITE](https://moviepuzzle.github.io/)

*Updates*

- [2024-03-06] We provide the `train_new.pt/val_new.pt/test_in_domain_new.pt/test_out_domain_new.pt` and processed image features in `pt/` directory to match our code. You can now download from [BaiduNetDisk](https://pan.baidu.com/s/1sIg-Vjp5S2xEu0PnMm-xqQ?pwd=qgum). **We still recommend that you process the original image on your own**.
- [2023-08-31] We provide the `train.pt/val.pt` files which are used to accelerate the data processing. You can now download from [BaiduNetDisk](https://pan.baidu.com/s/1sIg-Vjp5S2xEu0PnMm-xqQ?pwd=qgum).

```
# format of *.pt
{
	VID:{
		"img_id": tensor([2, 3, 11, ...]), # shuffled img ids for reorder
		"shot_id": tensor([200, 212, 321, ...]), # corresponding shot id
		"scene_id": tensor([2, 2, 2, ...]), # corresponding scene id
		"feature": TENSOR(size: LENGTH*1*FEATURE_DIM) # image feature
		},
}
```
