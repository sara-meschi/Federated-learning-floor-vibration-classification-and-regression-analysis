# Sensor layout and channel map

Source: instrumentation drawing at `docs/figures/sensor_layout.png`. **Verify the corridor ordering in §3 against the drawing before relying on it.**

## 1. Full channel map (20 channels, 16 sensors)

| Channel | Sensor | Type | SN | Sensitivity (mV/g) | In hallway? | Used? |
|---|---|---|---|---|---|---|
| 1 | 1 | PCB393B31 (uniaxial) | 51815 | 9890.00 | yes | **yes** |
| 2 | 2 | PCB393B31 | 51819 | 9710.00 | yes | **yes** |
| 3 | 3 | PCB393B31 | 72538 | 9940.00 | yes | **yes** |
| 4 | 4 | PCB393B31 | 51835 | 9770.00 | yes | **yes** |
| 5 | 5 | PCB393B31 | 51836 | 9730.00 | yes | **yes** |
| 6 | 6 | PCB393B31 | 61292 | 9910.00 | yes | **yes** |
| 7 | 7 | PCB393B31 | 61321 | 9700.00 | yes | **yes** |
| 8 | 8x | PCB393A03 (uniaxial, x) | 67865 | 975.1810 | yes | **yes** |
| 9 | 8y | PCB393A03 (uniaxial, y) | 67866 | 997.2150 | yes | no |
| 10 | 8z | PCB393A03 (uniaxial, z) | 68542 | 1004.5919 | yes | **yes** |
| 11 | 9x | PCB393A03 (uniaxial, x) | 68578 | 977.3876 | no | no |
| 12 | 9y | PCB393A03 (uniaxial, y) | 68588 | 988.2226 | no | no |
| 13 | 9z | PCB393A03 (uniaxial, z) | 68595 | 983.7188 | no | no |
| 14 | 10 | PCB393B31 | 61596 | 9620.00 | no | no |
| 15 | 11 | PCB393B31 | 61603 | 9820.00 | no | no |
| 16 | 12 | PCB393B31 | 61605 | 9580.00 | no | no |
| 17 | 13 | PCB393B31 | 72955 | 9770.00 | no | no |
| 18 | 14 | PCB393B31 | 72956 | 9640.00 | no | no |
| 19 | 15 | PCB393B31 | 61462 | 9770.00 | no | no |
| 20 | 16 | PCB393B31 | 61593 | 9620.00 | no | no |

## 2. The selected 9 channels cover 8 distinct hallway positions

`channels = [1, 2, 3, 4, 5, 6, 7, 8, 10]`:

- Channels 1–7: Sensors 1–7, PCB393B31 uniaxial, one unit at each of seven points down the corridor.
- Channels 8, 9, 10 (`8x`, `8y`, `8z`): **three separate uniaxial units, co-located at a single eighth point**, oriented in the x, y, and z directions. They are physically distinct sensors with distinct serial numbers — both PCB393B31 (~10 V/g) and PCB393A03 (~1 V/g) are single-axis seismic accelerometers, so this is not one triaxial package — but they occupy the same position.
- Channel 9 (`8y`) is excluded. Channels 8 (x) and 10 (z) are retained.

**Consequence for federated partitioning:** the partition unit is the **position**, and channels 8 and 10 are assigned together. They could in principle be split, since they are separate hardware, but they observe the same point — a client holding only channel 8 gains no independent spatial coverage there, and splitting the position breaks the "each site instruments one stretch of hallway" argument that makes the spatial-block scheme defensible. **Define the overlap ratio ρ over positions, not channels.**

```python
POSITION_TO_CHANNELS = {1: [1], 2: [2], 3: [3], 4: [4],
                        5: [5], 6: [6], 7: [7], 8: [8, 10]}   # 1-indexed
HALLWAY_POSITIONS = [1, 2, 3, 4, 5, 6, 7, 8]
NUMPY_INDICES     = [0, 1, 2, 3, 4, 5, 6, 7, 9]
```

**Log positions and channels separately in the audit.** A client holding position 8 receives 3 channels from 2 positions — less spatial coverage than the channel count implies.

**Orientation is not uniform.** Channel 8 measures x (horizontal); the other eight channels are vertical or vertically mounted. Footfall energy in floor vibration is predominantly vertical, so whichever client holds position 8 under a disjoint partition observes an extra, qualitatively different view of the same footstep. Report across ≥3 random position assignments rather than one fixed layout, log which client holds position 8, and give this a sentence in the paper.

## 3. Spatial ordering along the corridor

In the drawing, positions 1 → 8 run monotonically down the corridor, with position 8 at the end of the line. **Contiguous position blocks are therefore contiguous corridor segments**, which is what makes the spatial-block partition scheme physically meaningful ("each site instruments one stretch of hallway") rather than an arbitrary grouping. Confirmed against the layout.

K=3 disjoint (ρ=0) assignment:

| Client | Positions | Channels (1-indexed) | Channels (numpy) | Corridor segment |
|---|---|---|---|---|
| 0 | 1, 2, 3 | 1, 2, 3 | 0, 1, 2 | upper |
| 1 | 4, 5, 6 | 4, 5, 6 | 3, 4, 5 | middle |
| 2 | 7, 8 | 7, 8, 10 | 6, 7, 9 | lower |

Three channels per client, but **2 positions for client 2 versus 3 for the others** — record both counts. Client 2 also holds the only horizontal-axis channel (8), so its results are not exchangeable with the other two.

## 4. Excluded channels — justification for the paper

Sensors 9–16 (channels 11–20) were not located in the hallway where the walking tests were conducted and therefore carry no direct gait signal. **This selection was made a priori from physical sensor placement, before any modelling, and was not driven by model performance.** State this explicitly in Section III alongside the layout figure; it is the answer to "why 9 of 20 channels?", and it is a strong answer only if it is visible.

For the figure: keep the excluded room sensors in the drawing, greyed rather than deleted. Showing what was excluded is what makes the argument legible.

## 5. Two checks to run

**Sensitivity mismatch.** The PCB393B31 units are ~9,700–9,940 mV/g; the PCB393A03 units are ~975–1,005 mV/g — roughly a 10× difference. If the pipeline converts raw signal to engineering units with a single global sensitivity, channels 8 and 10 are wrong by an order of magnitude. Per-channel normalization would absorb a constant scale factor so training may be unaffected, but any physical-units claim would be wrong. Assert that per-channel sensitivity is applied, or document that the pipeline is scale-invariant by construction.

**Axis mixing.** Channel 8 is the only horizontal-axis channel retained; the other eight are vertical or vertically mounted. Either justify keeping channel 8 or run a cheap 8-channel all-vertical robustness check (drop channel 8) and report it in one sentence. Under disjoint channel partitions, log which client holds channel 8 — its results are not exchangeable with the others.
