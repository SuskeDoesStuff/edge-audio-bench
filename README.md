# edge-audio-bench

Taking a small audio model from prototype toward real-time inference on
constrained hardware, and measuring how it holds up as conditions get noisy.

A deliberate miniature of the deployment step in real edge-audio systems:
the model is the easy part; making it run fast enough, small enough, and
robustly enough to survive the real world is the work.

## The deliverable

One table, the kind that actually decides whether a system ships:

| SNR (dB) | accuracy | fp32 latency | int8 latency |
|---------:|---------:|-------------:|-------------:|
| 40 (clean) | … | … | … |
| 20 | … | … | … |
| 10 | … | … | … |
| 5 | … | … | … |
| 0 | … | … | … |
| -5 | … | … | … |

Accuracy as the environment degrades, alongside the latency cost of running
in real time and the saving from quantisation.

## Status

Working now:
- `src/noise.py` — SNR-controlled noise mixing, verified exact (target SNR
  reproduces to 2 dp). The robustness primitive the whole sweep stands on.
- `src/latency.py` — latency benchmark reporting the full distribution
  (mean/p50/p95/p99), because tail latency is what breaks real-time.
- `src/model.py` — compact CNN keyword spotter + int8 dynamic quantisation.
- `src/evaluate.py` — accuracy-vs-noise sweep, wired to the noise pipeline.

Active next step:
- Hook the eval to the Speech Commands set + a trained `TinyKWS`, extract
  log-mel features, and fill the table with real numbers.
- Benchmark fp32 vs int8 latency and report the trade-off.

## Run the working pieces

```bash
pip install -r requirements.txt
python src/noise.py        # verify SNR mixing is exact
python src/latency.py      # benchmark a stand-in workload
python src/evaluate.py     # run the sweep (noise pipeline live, model TODO)
```

## Why this exists

Built as a focused study of the prototype-to-edge deployment loop for
real-time audio: quantise, profile latency, and stress-test under noise.

## Results

TinyKWS (14.5k params) reaches 0.873 validation accuracy on the 10-word task.
The point of the project is what happens next: how it holds up under noise, and
what it costs to run in real time.

### Robustness to noise

![accuracy vs noise](results/accuracy_vs_noise.png)

| SNR (dB) | accuracy |
|---------:|---------:|
| 40 (~clean) | 0.861 |
| 20 | 0.781 |
| 10 | 0.612 |
| 5  | 0.466 |
| 0  | 0.302 |
| -5 | 0.175 |

Accuracy holds up to ~20 dB, degrades roughly linearly through the mid-SNR
range, and collapses toward chance (0.10) around 0 dB, the point where noise
power equals speech power. Breaking down exactly at signal/noise parity is the
check that the sweep behaves physically.

### Real-time latency, and the quantization finding

![latency](results/latency.png)

Batch-1 CPU inference: fp32 runs at 0.40 ms mean / 0.50 ms p99 (~2,500
inferences/sec), far inside any real-time budget.

Dynamic int8 quantization is slightly *slower* here (0.43 ms mean, 0.75 ms
p99), and that is the intended finding, not a failure. Dynamic quantization
only converts `Linear` layers, and TinyKWS is convolution-dominated with one
small linear head, so it quantizes ~2% of the compute while adding
quant/dequant overhead that surfaces in the tail. The real lever for a
conv-dominated model is static quantization with calibration, or an exported
runtime such as ONNX Runtime, which is the natural next step.

### Caveats
- Noise is additive white Gaussian. Structured real-world noise (babble, wind,
  reverberation) is harder and is the next robustness axis to test.
- The quantization result is a deliberate diagnostic of head-only dynamic
  quant, not a claim of a speedup.
