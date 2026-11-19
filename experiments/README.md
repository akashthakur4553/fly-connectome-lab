# fly-connectome-lab experiments: a real fruit-fly connectome in bodies, cars and games

All experiments run the **MaleCNS v1.0 connectome** (Janelia FlyEM + Google Research, Sept 2026; 167,106 neurons,
25.6M connections) as leaky integrate-and-fire neurons through [flybrain](../flybrain) on an Apple M2 laptop GPU.

**Ground rule: the wiring is never edited.** What learns is always a small, named interface: a linear readout, a
handful of gains, or (in the memory experiment only) the Kenyon cell → MBON synapses, which is where real flies
store associative memories.

| # | Experiment | Brain | Body | What learns |
|---|---|---|---|---|
| 1 | `exp_car.py`: fly drives a car around a track | full 167,106 neurons, real retina | car kinematics (eyes 2 cm above the track) | linear steering readout from 11,080 visual-projection + descending neurons (DAgger) |
| 2 | `exp_pong.py`: fly plays Pong, plus fly vs fly | full 167,106 neurons | paddle | linear paddle readout (DAgger) |
| 3 | `exp_balance.py train/eval`: gimbal fly on a rocking platform | 24,073 VNC + descending + ascending + leg-proprioceptor neurons | **flybody** (MuJoCo, 67 parts, adhesive claws) | 24 interface gains (ES); muscles → joints by MaleCNS motor-neuron type |
| 4 | `exp_balance.py bodies/evolve`: one brain, 100 bodies + evolution | same | flybody with leg-length / abdomen / stance genes | nothing in the brain: bodies evolve |
| 5 | `exp_memory.py`: love the smell of petrol; remember people by smell | 53,446 neurons (all but the optic lobes) | T-maze choice | dopamine-gated KC → MBON plasticity only |

Run everything with the flybrain environment, e.g. `../flybrain/.venv/bin/python -I exp_car.py train`.
Outputs (videos, JSON results, logs) go to `out/<experiment>/`.

## Honest notes
- **Speed on M2:** full brain ≈ 0.24× real time (42 ms per 10 ms of fly time); the VNC-only brain runs 13–33 fly-seconds
  per second at batch 32–128; the no-eyes brain ≈ 3.6 fly-seconds per second at batch 32.
- **Fly Civilization (1,000 full connectomes): not attempted.** That would take about 70 h per simulated minute here.
- **Face memory: not supported by this model.** Kenyon cells stay at 0 Hz for visual input, so memory is done with smells.
- **Petrol / people smells are stand-ins:** glomerulus sets were chosen as plausible (aromatic hydrocarbons; sweat
  acids/amines; esters). They are not measured receptor responses to those mixtures.
- **Hand-written pieces** are labelled in each file: the driving instructor and Pong coach (teachers for DAgger),
  sensor encodings (load/angle → Hz), the muscle → joint table, the MBON valence sign rule (opponent compartments),
  and the T-maze choice rule.
- Car and Pong rows in a batch share one wiring diagram but have independent neural states.

## Final results (Nov 19, 2026, Apple M2)
| Experiment | Result | Controls |
|---|---|---|
| Car, unseen bean track | 90% crash-free | own steering neurons 5%, blind 10% |
| Car, training oval | 100% crash-free | own steering 15%, blind 15% |
| Pong | 100% returned (40/40) | blind 9%, frozen paddle 49% |
| Fly vs fly Pong | longest rally 30, no misses in 30 s | |
| Balance (±25° rocking floor) | 81% of tilt removed (3.1° vs 16.5°) | senses cut: worse than legs locked |
| 100 bodies, same brain | best ~82%, worst ~35%; long front legs hurt (r=+0.52) | |
| Evolution (12 gens × 48) | 87% (2.2°): mid legs 0.7×, hind legs 1.6× | |
| Petrol memory (T-maze) | petrol chosen 100% / 88% / 94% vs banana / Sam / Alex | sham: 75% / 62% / 50% |
| People memory | Sam over Alex: 19% → 100% | sham: 62% |

Web-ready videos: `out/linkedin/`. Full-quality videos: `out/<experiment>/`.
