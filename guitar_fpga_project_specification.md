# Project Specification: Hardware-Aware Optimisation of Guitar String/Fret Classification

**Project type:** Independent DSP/ML research and FPGA hardware acceleration  
**Target duration:** 4 focused working days, subject to board access  
**Primary languages:** Python, SystemVerilog  
**Tools:** cocotb, Verilator, FPGA synthesis and implementation tools  
**Hardware:** Available FPGA development board (exact model to be selected)  
**Primary application:** Imperial College London UROP — Professor George Constantinides  
**Secondary value:** Future RTL, FPGA, quant hardware and low-latency systems applications

---

## 1. Project Overview

### Background

The existing classical guitar string/fret localisation project investigates whether different physical positions producing the same pitch can be distinguished through acoustic features.

It includes audio signal processing, physically motivated feature engineering, machine-learning classification, cross-validation, later-session evaluation and analysis of limitations.

The current classification pipeline uses per-note trained models, with logistic regression providing a suitable starting point for fixed-point hardware inference.

### Research Question

**How do numerical representation and hardware architecture affect the classification accuracy, resource utilisation and performance of an FPGA-based guitar string/fret localisation classifier?**

The investigation has four dimensions:

1. Numerical precision versus classification accuracy.
2. Hardware resource utilisation.
3. Latency, throughput and timing.
4. Agreement between software simulation, RTL simulation and physical FPGA execution.

### Intended Contribution

A reproducible, end-to-end experimental study connecting an independently developed DSP/ML application to working FPGA hardware.

The project does not claim a new ML algorithm or state-of-the-art accelerator performance.

---

## 2. Scope

### Required

- Reuse existing DSP features, trained models and evaluation datasets.
- Build a deterministic fixed-point Python reference.
- Evaluate multiple numerical precisions.
- Implement synthesizable SystemVerilog inference hardware.
- Verify RTL using cocotb against the bit-exact reference.
- Synthesise and place-and-route the design for a real FPGA target.
- **Deploy the classifier onto a physical FPGA board.**
- Feed known feature vectors into the FPGA and retrieve classifications.
- Compare board predictions against the software reference.
- Measure or characterise implementation resources, timing and inference latency.
- Document methods, results and limitations.

### Explicitly out of scope

- Real-time microphone input.
- FPGA-based FFT or full DSP feature extraction.
- Training a new deep-learning model.
- Recreating LUT-based neural network research.
- Designing floating-point inference hardware.
- Implementing an entirely new classifier algorithm.
- Building a standalone commercial-quality device.

**The FPGA receives extracted features, not raw audio.**

---

## 3. Existing Classifier Integration

### Task 3.1 — Audit the original implementation

Identify:

- Exact feature vector and feature ordering.
- Preprocessing and standardisation.
- Trained weights and biases.
- Per-note classifier selection.
- Training and evaluation splits.
- Existing accuracy measurements.

Reuse the trained models and original data without silently modifying the evaluation methodology.

### Task 3.2 — Export the inference model

For a multiclass linear classifier:

\[
s_j=b_j+\sum_iw_{ji}x_i
\]

\[
\hat y=\arg\max_j s_j
\]

Use the actual trained model's decision rule, including any binary-classifier special cases.

Where appropriate, fold standardisation into coefficients:

\[
w'_{ji}=\frac{w_{ji}}{\sigma_i}
\]

\[
b'_j=b_j-\sum_i\frac{w_{ji}\mu_i}{\sigma_i}
\]

Verify equivalence with the original floating-point pipeline before introducing quantisation.

**Deliverable:** Deterministic exported model and reference inference implementation.

---

## 4. Numerical Representation Study

### Objective

Determine how reduced numerical precision affects classifier predictions and implementation cost.

### Required implementation

Build a Python fixed-point reference using explicit integer arithmetic.

Specify:

- Signed feature representation.
- Signed coefficient representation.
- Bias scaling.
- Product widths.
- Accumulator width.
- Rounding or truncation rules.
- Overflow handling.
- Output comparison and tie-breaking.

Use the same numerical semantics in Python and SystemVerilog.

### Precision sweep

Begin with candidate widths such as 8, 12, 16 and 24 bits, adjusting fractional widths to accommodate actual data ranges.

Evaluate:

- Accuracy against labelled recordings.
- Agreement with floating-point predictions.
- Prediction changes.
- Numerical overflow or saturation.
- Sensitivity to precision.

Maintain consistent evaluation splits and report limitations of the existing dataset.

**Deliverable:** Precision-versus-accuracy and reference-agreement results.

---

## 5. SystemVerilog Accelerator

### Objective

Implement fixed-point classifier inference in synthesizable RTL.

### Minimum architecture

Use one MAC-based classifier datapath, with:

- Feature input registers.
- Trained coefficient storage.
- Signed multiplication.
- Fixed-point accumulation.
- Bias addition.
- Control FSM or counters.
- Class-score comparison.
- `start`, `busy`, `done` and classification outputs.

A resource-shared MAC is a reasonable baseline, although the exact architecture should follow the model's actual dimensions and FPGA target.

### Requirements

- Correct signed arithmetic.
- Explicit intermediate widths.
- Defined overflow and rounding semantics.
- Deterministic tie-breaking.
- Correct reset behaviour.
- Reliable transaction protocol.
- Synthesizable implementation.

Support at least one trained note-specific classifier. Multiple classifier parameter sets are desirable if inexpensive, but not essential.

**Deliverable:** Synthesizable fixed-point inference RTL.

---

## 6. RTL Verification

### Tools

Use cocotb with Verilator.

### Required tests

- Recorded feature vectors.
- Random valid numerical inputs.
- Positive, negative and zero operands.
- Accumulator boundaries.
- Near-equal class scores.
- Exact score ties.
- Reset and restart behaviour.
- Back-to-back transactions.
- Correct `busy` and `done` sequencing.

Compare numerical score outputs where practical, rather than checking classifications alone.

The Python fixed-point implementation must serve as the bit-exact reference.

**Deliverable:** Automated verification suite with reproducible results.

---

## 7. FPGA Synthesis and Implementation

### Objective

Implement the accelerator for a real FPGA and evaluate hardware cost.

### Required tasks

1. Select an available FPGA development board.
2. Establish the target FPGA part and vendor toolchain.
3. Create a top-level hardware integration module.
4. Apply clock and I/O constraints.
5. Synthesise the accelerator.
6. Run place-and-route.
7. Generate the FPGA configuration bitstream.
8. Inspect timing and resource reports.
9. Program the physical board.

### Required measurements

- LUT or logic-element utilisation.
- Flip-flop utilisation.
- DSP block utilisation.
- Block RAM utilisation, if applicable.
- Critical path and timing slack.
- Maximum supported clock frequency, where established.
- Clock cycles per inference.
- Estimated or measured inference latency.

For a design requiring \(C\) cycles:

\[
T_{\text{inference}}=\frac{C}{f_{\text{clk}}}
\]

Use the actual board clock frequency for the deployed design.

Clearly distinguish post-route timing estimates from physical runtime measurements.

**Deliverable:** Timing-clean FPGA implementation and reproducible bitstream-generation process.

---

## 8. Physical FPGA Deployment

### Objective

Demonstrate that the classifier executes correctly on physical hardware.

### Minimum system

Use the following dataflow:

**Python host → FPGA interface → fixed-point classifier → FPGA interface → Python host**

The Python application should:

1. Load or generate a previously extracted feature vector.
2. Quantise it using the reference implementation.
3. Send the numerical features to the FPGA.
4. Start inference.
5. Retrieve the predicted classification.
6. Compare it with the fixed-point Python reference.

### Communication interface

Select the simplest practical interface supported by the development board.

Potential approaches:

- UART using an onboard USB-to-serial bridge.
- A board-supported host communication interface.
- An existing lightweight FPGA control interface.

Do not implement PCIe, Ethernet or a custom high-speed link merely to transport a small feature vector.

### Hardware integration

Implement only the supporting logic necessary to transfer feature vectors, trigger computation and retrieve results.

Possible modules:

- Host communications interface.
- Input feature register bank.
- Accelerator control logic.
- Classification result register.
- Output response interface.

Use a clearly defined packet or transaction format.

### Required demonstration

Run a batch of known feature vectors through the physical FPGA.

For every vector, compare:

- Fixed-point Python prediction.
- RTL reference prediction.
- Physical FPGA prediction.

Report exact agreement counts.

Also test multiple consecutive transactions without reprogramming the board.

### Timing experiment

If practical, expose a hardware cycle counter around inference.

Measure the number of clock cycles between starting computation and result availability.

The core's cycle count and the host-to-host transaction latency must be treated separately.

UART or host-software overhead should not be presented as accelerator computation time.

### Success criteria

- Bitstream programmed successfully.
- FPGA receives numerical input features.
- Accelerator completes inference.
- Classification returned to the host.
- Outputs agree with the bit-exact reference for the tested vectors.
- Repeated transactions work reliably.

**Deliverable:** Working FPGA demonstration, host control script and documented execution results.

---

## 9. Experimental Analysis

Address the following questions.

### Q1 — What precision does the classifier require?

Identify the lowest tested precision that meets the selected classification-agreement target.

### Q2 — Does reduced precision save FPGA resources?

Compare actual synthesis results rather than assuming a proportional relationship between bit width and area.

### Q3 — What is the inference latency?

Report clock cycles, deployed clock frequency and computational latency.

### Q4 — Does physical hardware agree with simulation?

Compare physical classifications to the bit-exact numerical reference.

Investigate any mismatches.

### Q5 — What are the limitations?

Discuss:

- Small, controlled acoustic dataset.
- Restricted note and instrument coverage.
- Software-based feature extraction.
- FPGA target dependence.
- Resource and latency measurement limitations.
- Lack of a complete streaming-audio system.

### Optional investigation

Evaluate a second architecture or mixed-precision representation if the core system is already complete.

---

## 10. Revised Four-Day Roadmap

### Day 1 — Numerical reference and setup

**Objectives**

- Audit existing classifier.
- Export model and feature vectors.
- Implement fixed-point inference.
- Run initial precision sweep.
- Select the FPGA board and toolchain.
- Verify that board programming and basic host communication are feasible.

**Milestone:** Working numerical reference and confirmed board development environment.

### Day 2 — RTL and verification

**Objectives**

- Implement classifier RTL.
- Build cocotb tests.
- Resolve arithmetic discrepancies.
- Establish the accelerator transaction protocol.
- Prepare FPGA integration top-level.

**Milestone:** Verified synthesizable accelerator.

### Day 3 — FPGA implementation

**Objectives**

- Integrate clock, control and communications.
- Run synthesis and place-and-route.
- Generate bitstream.
- Program board.
- Run first physical inference transaction.

**Milestone:** At least one correct classification executed on the FPGA.

### Day 4 — Physical validation and results

**Objectives**

- Execute batch inference tests.
- Compare FPGA output against Python reference.
- Record timing and resource measurements.
- Compare at least two precision configurations where feasible.
- Generate summary tables and figures.
- Document setup and reproducibility.
- Finalise repository README.

**Milestone:** Complete FPGA-backed experimental study.

---

## 11. Priority Order and Contingencies

### Priority 1 — Correct numerical model

The reference and quantisation study must work before hardware implementation.

### Priority 2 — Verified RTL

The accelerator must match the reference precisely.

### Priority 3 — Physical FPGA execution

Deploy and validate at least one numerical configuration.

### Priority 4 — Hardware measurements

Collect timing, utilisation and cycle-count data.

### Priority 5 — Additional optimisation

Second architecture, mixed precision, further synthesis sweeps or interface enhancements.

Board deployment is part of the required project scope, not an optional research direction.

However, it depends on timely access to a usable FPGA board and the required tools. If board access or toolchain problems make completion within four days impractical, preserve the verified RTL and synthesis work, report the blocker accurately and do not sacrifice unrelated academic commitments.

---

## 12. Documentation and Repository Deliverables

The completed project should include:

- Original research context.
- Numerical reference implementation.
- Precision exploration.
- RTL architecture diagram.
- Synthesizable SystemVerilog source.
- Automated cocotb verification.
- FPGA top-level integration.
- FPGA constraints and build instructions.
- Host communication script.
- Synthesis and timing reports.
- Physical hardware test results.
- Results tables and plots.
- Limitations and future work.

Keep the repository reproducible and accessible.

If practical, include a short video showing the board running inference and the host receiving classifications.

The demo should show actual execution rather than merely presenting a successful simulation.

---

## 13. Completion Criteria

The core project is complete when:

- Fixed-point reference inference works.
- Numerical precision has been investigated.
- At least one RTL accelerator is verified.
- FPGA synthesis and implementation succeed.
- A bitstream has been deployed to a physical board.
- Feature vectors can be sent to the FPGA.
- FPGA classifications match the reference on tested examples.
- Hardware timing and resource results are recorded.
- Results and limitations are documented.

Do not expand into real-time audio processing, advanced communications or elaborate accelerator microarchitectures to satisfy these criteria.

---

## 14. Intended Outcome

The final project should demonstrate the complete path:

**Acoustic signal → DSP features → trained ML model → fixed-point representation → SystemVerilog datapath → physical FPGA inference → experimental evaluation**

This connects experience in:

- Signal processing.
- Machine learning.
- Software engineering.
- Numerical representation.
- RTL and digital hardware design.
- Verification.
- FPGA implementation.
- HW/SW integration.
- Experimental performance evaluation.

For the UROP application, emphasise hardware-aware numerical optimisation and the independent research process.

For future RTL and quant hardware applications, emphasise arithmetic design, implementation constraints, measured latency and successful physical FPGA deployment.

The aim is a compact but complete engineering investigation, not a feature-heavy demonstration.
