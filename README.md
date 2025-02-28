# PyTorch Transformer Code examples

Functional Transformer code examples that can be used as starting point for your Transformer-based projects.

The scripts are careful to differentiate teacher forcing-based loss/accuracy during training from an autoregressive, generative accuracy used in evaluation.

- **eval_toy_data1.py**: we generate a sequence of 10 tokens, and the model must learn to shift the sequence right by 1 token.
- **eval_toy_data2.py**: we generate a sequence of up to 15 tokens, and the model must learn to repeat each token twice in a row.
- **eval_grid_autoencoding.py**: we generate 5x5 or 6x6 ARC-like grids (i.e. each cell corresponds to a pixel color between 0 and 9 inclusively), and it must reconstruct the input grid.
- **eval_grid_rot180.py**: we generate 5x5 or 6x6 ARC-like grids, and it must rotate the grid by 180 degrees.
- **eval_grid_rot90.py**: we generate 5x5 or 6x6 ARC-like grids, and it must rotate the grid by 90 degrees.
- **eval_grid_rot_class.py**: we generate pairs of 5x5 or 6x6 ARC-like grids, the second one being either a 90 degree or a 180 degree rotation of the first. The transformer must classify whether it was a 90-degree rotation or a 180-degree rotation.
- **eval_grid_flip_class.py**: similar to eval_grid_rot_class.py, but it's horizontal and vertical flips instead of rotations.
- **eval_grid_flipcolor_class.py**: similar to eval_grid_flip_class.py, but we add an extra step of changing the non-zero pixels' colors to generate the target grid. Still must distinguish between horizontal and vertical flip (it must ignore the fact that the pixel colors changed).
- **eval_transform_classification.py**: loads data from .json files. Instructions on how to generate these data files soon to be added.
