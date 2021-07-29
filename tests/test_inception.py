import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from jax_fid import inception


def test_inception_output():
    img = Image.open('tests/aux_files/elefant.jpg')
    x = np.array(img, dtype=np.float32) / 255.0
    x = 2 * x - 1
    x = np.expand_dims(x, axis=0)

    rng = jax.random.PRNGKey(0)
    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))
    out = model.apply(params, x, train=False)

    out_target = np.load('tests/aux_files/elefant_inception_output.npy')

    diff = np.mean(np.abs(out - out_target))
    
    assert diff < 1e-5
