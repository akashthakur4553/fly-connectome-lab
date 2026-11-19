# Fly Connectome Lab

A real fruit-fly brain, given a body and a world.

The male fruit-fly connectome (MaleCNS v1.0, Janelia FlyEM + Google Research, 2026: 167,106 neurons, 25.6 million
connections) is simulated as spiking neurons on an Apple M2 laptop and plugged into a car, a Pong game, a
physics-simulated fly body and a smell maze. The wiring is never edited. What learns is a small interface around it,
and in the memory experiment only the mushroom body's Kenyon cell → output synapses, which is where real flies store
these memories.

**Results page:** `site/` (static, deploys on Vercel as is). It has every video, the numbers and their controls.

## Results

| Experiment | Result | Control |
|---|---|---|
| Car, on a track it never trained on | 90% crash-free | its own steering neurons 5%, eyes covered 10% |
| Car, training track | 100% crash-free | 15% / 15% |
| Pong | 100% of balls returned (40/40) | eyes covered 9%, paddle frozen 49% |
| Two copies of the brain playing each other | 30+ hit rally, no point lost in 30 s | |
| Balance on a ±25° rocking floor (flybody, MuJoCo) | 81% of body tilt cancelled | senses cut: worse than legs locked |
| Same brain in 100 random bodies | best ~82%, worst ~35% | long front legs hurt |
| Evolution of the body, brain fixed | 87%: shorter middle legs, longer hind legs | |
| Learning to love petrol (T-maze) | 62% → 100% chose petrol | without dopamine: 62% → 75% |
| Remembering people by smell | 19% → 100% chose the rewarded person | without dopamine: 62% |
| Remembering faces | not learned: in this model the memory centre barely responds to vision | |

Raw numbers are in `results/`.

## What is real and what is not

- **Real:** the wiring (MaleCNS v1.0), the retina geometry, the neuron model (leaky integrate-and-fire after
  Shiu et al. 2024, via flybrain), the flybody physics model, the dopamine-gated learning rule.
- **Hand-built and labelled in the code:** the driving instructor and Pong coach used for imitation learning, sensor
  encodings, the muscle → joint table, the leveling reflex, MBON valence signs and the T-maze choice rule.
- **Approximate:** the petrol and body-odour smells are stand-in glomerulus blends, not measured receptor data.
- **Not attempted:** 1,000 interacting flies (about 70 hours per simulated minute on this laptop).

## Layout

```
site/              the results page: index.html, videos/, images/ (Vercel output directory)
experiments/       exp_car.py, exp_pong.py, exp_balance.py, exp_memory.py and the flylab/ package
results/           JSON results of every test, training run and control
vercel.json        static deploy settings (serves site/)
```

## Running the experiments

The experiments need two open-source projects cloned next to `experiments/` (they are not part of this repository):

```sh
git clone https://github.com/HEREISCB/flybrain.git      # connectome simulator (MIT)
git clone https://github.com/TuragaLab/flybody.git      # fly body for MuJoCo (Apache-2.0)
cd flybrain && sh "Install FlyBrain (macOS).command"     # Python env + 1.1 GB of MaleCNS data + cache
uv pip install --python .venv mujoco dm_control h5py imageio-ffmpeg && uv pip install --python .venv --no-deps -e ../flybody
cd ../experiments
../flybrain/.venv/bin/python -I exp_car.py train2 && ../flybrain/.venv/bin/python -I exp_car.py test2
```

Each script's docstring lists its commands. Speed on an M2: the full brain runs at about 0.24× real time; the
nerve-cord subset used for the physics experiments runs 13–33× faster.

## Deploying the page

Import the repository in Vercel. `vercel.json` sets the output directory to `site/` with no build step, so the page
and its videos are served as they are.

## Credits

- MaleCNS v1.0 connectome: Janelia FlyEM and Google Research (CC BY 4.0).
- flybrain simulator (MIT), flybody fly model (Apache-2.0), MuJoCo (Apache-2.0).
- Neuron model after Shiu et al., Nature 2024.
