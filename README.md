# Scalable End-to-End Autonomous Vehicle Testing via Rare-event Simulation
Coming soon!
<p align="center">
  <img src="assets/RareSim.gif"/>
</p>


This is the reference implementation for our paper:

[PDF](https://arxiv.org/abs/1811.00145)

[Matthew O'Kelly*](http://www.mokelly.net/), [Aman Sinha*](http://amansinha.org), [Hongseok Namkoong*](https://web.stanford.edu/~hnamk/), [John Duchi](http://stanford.edu/~jduchi/), [Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)

<em><b>Abstract:</b></em> While recent developments in autonomous vehicle technology highlight substantial progress, we lack tools for rigorous and scalable testing. Real-world testing, the de facto evaluation environment, places the public in danger, and, due to the rare nature of accidents, will require billions of miles in order to statistically validate performance claims. We implement a simulation framework that can test an entire modern autonomous driving system, including, in particular, systems that employ deep-learning based perception and control algorithms. Using adaptive importance-sampling methods to accelerate rare-event probability evaluation, we estimate the probability of an accident under a base distribution (learned from real-world data) governing standard traffic behavior. We demonstrate our framework on a highway scenario, our evaluation is 2-20 faster than naive Monte Carlo sampling methods and 10-300 times (where P is the number of processors) faster than real-world testing.

#### Citing

If you find this code useful in your work, please consider citing:

```
@inproceedings{okelly2018,
  title={Scalable End-to-End Autonomous Vehicle Testing via Rare-event Simulation},
  author={O'Kelly*, Matthew and Sinha*, Aman and Namkoong*, Hongseok and Tedrake, Russ and Duchi, John},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}

```

# Dependencies
Requires:
* docker
* nvidia-docker2
* A recent Nvidia GPU *e.g.* GTX980 or better.

# Installation
Coming soon.
