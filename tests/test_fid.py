import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from jax_fid import fid


def test_inception_output():
    mu1 = np.load('tests/aux_files/mu1.npy')
    mu2 = np.load('tests/aux_files/mu2.npy')
    sigma1 = np.load('tests/aux_files/sigma1.npy')
    sigma2 = np.load('tests/aux_files/sigma2.npy')
    fid_target = np.load('tests/aux_files/fid_target.npy')
    
    fid_score = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2)
    
    diff = np.abs(fid_score - fid_target)

    assert diff < 0.01
