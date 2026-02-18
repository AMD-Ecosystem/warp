# ROCm Warp

ROCm Warp is a port of the [Warp](https://github.com/ROCm/warp) library and adds support for AMD instinct GPUs. This project is in active development. 

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for [spatial computing](https://en.wikipedia.org/wiki/Spatial_computing)
and comes with a rich set of primitives that make it easy to write
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch, JAX and Paddle.

## Requirements
* Python 3.12
* ROCm 7.0 or higher (for HIP builds)
* [Git LFS](https://git-lfs.github.com/) installed

## GPU and ROCm Support

**Supported GPU:** gfx942 (CDNA3 architecture)

**Supported ROCm version:** 6.4.1, 7.x

##  Installing
Python version 3.9 or newer is required. ROCm Warp is currently supported on AMD Instinct GPUs with ROCm 7.x

HIP/ROCm is auto-detected just like CUDA. Ensure ROCm 7.x is installed
and `hipcc`, `hipconfig` are on your `PATH` or under `ROCM_PATH`. 
If you're using TheRock (e.g., `rocm`/`rocm-sdk` wheels), locate the install root with
`rocm-sdk path --bin` and set `ROCM_PATH` to its parent directory so the toolchain and headers resolve
correctly. 

Clone the repository
```
git clone https://github.com/ROCm/warp.git
```

We can then build and install warp using
```
cd warp/
python build_lib.py
pip install -e .
```

The build script will automatically detect and enable HIP if ROCm is found.
You can also specify a custom ROCm path with `--rocm-path="..."`.

#### Tips
- To target a specific AMD GPU architecture, pass `--hip-arch="gfx942"`.
- For a non-fat build, building for the default architecture (gfx942) pass `--quick`.
- To build in debug mode, pass `--mode=debug`.



## Running Examples

The [warp/examples](https://github.com/ROCm/warp/tree/amd-integration/warp/examples) directory contains a number of scripts categorized under subdirectories
that show how to implement various simulation methods using the Warp API.
Most examples will generate USD files containing time-sampled animations in the current working directory.
Before running examples, users should ensure that the ``usd-core``, ``matplotlib``, and ``pyglet`` packages are installed using:

```text
pip install warp-lang[extras]
```

These dependencies can also be manually installed using:

```text
pip install usd-core matplotlib pyglet
```

Examples can be run from the command-line as follows:

```text
python -m warp.examples.<example_subdir>.<example>
```

To browse the example source code, you can open the directory where the files are located like this:

```text
python -m warp.examples.browse
```

Most examples can be run on either the CPU or a CUDA-capable device, but a handful require a CUDA-capable device. These are marked at the top of the example script.


### warp/examples/core

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_dem.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_dem.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_fluid.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_fluid.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_graph_capture.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_graph_capture.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_marching_cubes.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_marching_cubes.png"></a></td>
        </tr>
        <tr>
            <td align="center">dem</td>
            <td align="center">fluid</td>
            <td align="center">graph capture</td>
            <td align="center">marching cubes</td>
        </tr>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_mesh.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_mesh.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_nvdb.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_nvdb.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_raycast.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_raycast.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_raymarch.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_raymarch.png"></a></td>
        </tr>
        <tr>
            <td align="center">mesh</td>
            <td align="center">nvdb</td>
            <td align="center">raycast</td>
            <td align="center">raymarch</td>
        </tr>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_sample_mesh.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_sample_mesh.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_sph.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_sph.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_torch.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_torch.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/core/example_wave.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/core_wave.png"></a></td>
        </tr>
        <tr>
            <td align="center">sample mesh</td>
            <td align="center">sph</td>
            <td align="center">torch</td>
            <td align="center">wave</td>
        </tr>
    </tbody>
</table>

### warp/examples/fem

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_diffusion_3d.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_diffusion_3d.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_mixed_elasticity.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_mixed_elasticity.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_apic_fluid.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_apic_fluid.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_streamlines.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_streamlines.png"></a></td>
        </tr>
        <tr>
            <td align="center">diffusion 3d</td>
            <td align="center">mixed elasticity</td>
            <td align="center">apic fluid</td>
            <td align="center">streamlines</td>
        </tr>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_distortion_energy.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_distortion_energy.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_navier_stokes.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_navier_stokes.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_burgers.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_burgers.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_magnetostatics.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_magnetostatics.png"></a></td>
        </tr>
        <tr>
            <td align="center">distortion energy</td>
            <td align="center">navier stokes</td>
            <td align="center">burgers</td>
            <td align="center">magnetostatics</td>
        </tr>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_adaptive_grid.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_adaptive_grid.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_nonconforming_contact.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_nonconforming_contact.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_darcy_ls_optimization.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_darcy_ls_optimization.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/fem/example_elastic_shape_optimization.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/fem_elastic_shape_optimization.png"></a></td>
        </tr>
        <tr>
            <td align="center">adaptive grid</td>
            <td align="center">nonconforming contact</td>
            <td align="center">darcy level-set optimization</td>
            <td align="center">elastic shape optimization</td>
        </tr>
    </tbody>
</table>

### warp/examples/optim

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/optim/example_diffray.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/optim_diffray.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/optim/example_fluid_checkpoint.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/optim_fluid_checkpoint.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/optim/example_particle_repulsion.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/optim_particle_repulsion.png"></a></td>
            <td></td>
        </tr>
        <tr>
            <td align="center">diffray</td>
            <td align="center">fluid checkpoint</td>
            <td align="center">particle repulsion</td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

### warp/examples/tile

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/tile/example_tile_mlp.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/tile_mlp.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/tile/example_tile_nbody.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/tile_nbody.png"></a></td>
            <td><a href="https://github.com/ROCm/warp/tree/amd-integration/warp/examples/tile/example_tile_mcgp.py"><img src="https://media.githubusercontent.com/media/ROCm/warp/refs/heads/amd-integration/docs/img/examples/tile_mcgp.png"></a></td>
            <td></td>
        </tr>
        <tr>
            <td align="center">mlp</td>
            <td align="center">nbody</td>
            <td align="center">mcgp</td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

## Support

Problems, questions, and feature requests can be opened on [GitHub Issues](https://github.com/ROCm/warp/issues).

## License

Warp is provided under the Apache License, Version 2.0.
Please see [LICENSE.md](https://github.com/ROCm/warp/blob/amd-integration/LICENSE.md) for full license text.

This project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use.

## Contributing

Contributions and pull requests from the community are welcome.
Please setup `pre-commit` hooks using

```
pip install pre-commit
```
And then in the source directory of the project
```
pre-commit install
```

## Citation

To cite Warp itself in your own publications, please use the following BibTeX entry:

```bibtex
@misc{warp2022,
  title        = {Warp: A High-performance Python Framework for GPU Simulation and Graphics},
  author       = {Miles Macklin},
  month        = {March},
  year         = {2022},
  note         = {NVIDIA GPU Technology Conference (GTC)},
  howpublished = {\url{https://github.com/nvidia/warp}}
}
```
