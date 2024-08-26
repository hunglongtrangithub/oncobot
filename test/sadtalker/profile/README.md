`make_animation` input tensors:

```python
source_image.shape torch.Size([batch_size, 3, 256, 256])
source_semantics.shape torch.Size([batch_size, 70, 27])
target_semantics.shape torch.Size([batch_size, 8, 70, 27])
```

`KPDetector` output tensors:

```python
kp_canonical.keys() dict_keys(['value'])
value torch.Size([batch_size, 15, 3])
```

`MappingNet` output tensors:

```python
he_source.keys() dict_keys(['yaw', 'pitch', 'roll', 't', 'exp'])
yaw torch.Size([batch_size, 66])
pitch torch.Size([batch_size, 66])
roll torch.Size([batch_size, 66])
t torch.Size([batch_size, 3])
exp torch.Size([batch_size, 45])
```

`DenseMotionNetwork` input tensors:

```python
feature.shape torch.Size([batch_size, 32, 16, 64, 64])
kp_driving['value'].shape torch.Size([batch_size, 15, 3])
kp_source['value'].shape torch.Size([batch_size, 15, 3])
```

`keypoint_transformation` output tensor:

```python
kp_source["value"].shape torch.Size([batch_size, 15, 3])
```
