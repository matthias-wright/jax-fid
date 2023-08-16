# FID computation in Jax/Flax

This is a port of [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid), which is a port of the original FID implementation ([bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)).

The parameters for the [InceptionV3](https://arxiv.org/abs/1512.00567) network are taken from [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid). The FID scores are almost identical (absolute difference around 1e-7).  
The only difference is that [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) [resizes](https://github.com/mseitzer/pytorch-fid/blob/d042ab8a9f8e4b388c21bc7b38d9599c5fbcfe7b/src/pytorch_fid/inception.py#L146) the images to 299x299 by default. In this implementation, the images are not resized by default. You can resize the images using the `--img_size` argument. 


## Installation
You will need Python 3.7 or later.
 
1. For GPU usage, follow the <a href="https://github.com/google/jax#installation">Jax</a> installation with CUDA.
2. Then install:
   ```sh
   > pip install jax-fid
   ```
For CPU-only you can skip step 1.

## Usage

### Compute FID score
```python
> CUDA_VISIBLE_DEVICES=N python -m jax_fid --path1 /path/to/dataset1 --path2 /path/to/dataset2
```
where `N` is the GPU index.

### Pre-compute statistics for image directory
```python
> CUDA_VISIBLE_DEVICES=N python -m jax_fid --precompute --img_dir /path/to/dataset --out_dir /path/to/stats
```

### Arguments
`--path1` - Path to image directory or .npz file containing pre-computed statistics.  
`--path2` - Path to image directory or .npz file containing pre-computed statistics.  
`--batch_size` - Batch size per device for computing the Inception activations.  
`--img_size` - Resize images to this size. The format is (height, width).  
`--precompute` - If True, pre-compute statistics for given image directory.  
`--img_dir` - Path to image directory for pre-computing statistics.   
`--out_dir` - Path where pre-computed statistics are stored.   
`--mmap` - If True, use mmap to compute statistics.
`--mmap_file` - Name of mmap file. Only used if mmap is True.


## License
Apache-2.0 License
