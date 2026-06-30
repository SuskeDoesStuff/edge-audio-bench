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
