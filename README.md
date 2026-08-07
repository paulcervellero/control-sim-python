# ControlSim

A Python command-line PID control-system simulator for analyzing the closed-loop response of a first-order dynamic plant.

ControlSim provides configurable PID gains, plant parameters, simulation settings, performance metrics, parameter sweeps, and visualization tools for exploring fundamental feedback-control behavior.

---

## Overview

ControlSim models a PID-controlled first-order plant described by:

```text
dy/dt = -a*y + b*u
```

where:

- `y` is the plant output
- `u` is the controller output
- `a` controls the natural plant decay
- `b` controls the plant response to the input

The PID controller calculates the control input using proportional, integral, and derivative feedback.

The simulator numerically propagates the system through time and visualizes the resulting closed-loop response.

---

## Features

- Simulate a PID-controlled first-order plant
- Configure proportional, integral, and derivative gains
- Configure simulation setpoint
- Configure simulation time step and duration
- Configure first-order plant parameters
- Apply actuator saturation
- Use conditional integration for basic anti-windup protection
- Calculate closed-loop performance metrics
- Perform proportional-gain parameter sweeps
- Display interactive response plots
- Export plots to image files
- Validate simulation parameters
- Return useful command-line errors for invalid inputs

---

## System Model

The simulated plant follows the first-order differential equation:

```text
dy/dt = -a*y + b*u
```

The PID controller uses:

```text
u = Kp*e + Ki*integral(e) + Kd*de/dt
```

where:

```text
e = setpoint - y
```

The control signal is limited to:

```text
-100 <= u <= 100
```

to model actuator saturation.

---

## Simulation Architecture

```text
                Setpoint
                   |
                   v
              Error Signal
                   |
                   v
          +----------------+
          | PID Controller |
          |                |
          | Kp             |
          | Ki             |
          | Kd             |
          +----------------+
                   |
                   v
          Control Saturation
                   |
                   v
          +----------------+
          | First-Order    |
          | Plant          |
          |                |
          | dy/dt=-ay+bu   |
          +----------------+
                   |
                   v
             Plant Output
                   |
                   +----------> Feedback
```

---

## Performance Metrics

For single simulations, ControlSim calculates:

### Steady-State Error

The absolute difference between the setpoint and final simulated plant output.

### Overshoot

The percentage by which the plant output exceeds the requested setpoint.

### Settling Time

The first time at which the response enters and remains within the defined tolerance around the setpoint.

Example output:

```text
Performance Metrics
-------------------
Steady-state error: 0.0517
Overshoot: 0.00%
Settling time: not reached
```

---

## Technologies

- Python 3
- NumPy
- Matplotlib
- argparse

---

## Installation

Clone the repository:

```bash
git clone git@github.com:paulcervellero/control-sim-python.git
cd control-sim-python
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

---

## Usage

### Default Simulation

```bash
python3 pid_sim.py
```

Default controller parameters:

```text
Kp = 2.0
Ki = 0.5
Kd = 0.1
```

Default simulation parameters:

```text
Setpoint = 1.0
dt       = 0.01 s
Duration = 10.0 s
a        = 1.0
b        = 1.0
```

---

## Custom PID Gains

```bash
python3 pid_sim.py \
  --kp 3.0 \
  --ki 0.5 \
  --kd 0.05
```

---

## Custom Plant

The first-order plant coefficients can also be modified:

```bash
python3 pid_sim.py \
  --a 1.5 \
  --b 0.8
```

---

## Custom Simulation

```bash
python3 pid_sim.py \
  --setpoint 2.0 \
  --dt 0.005 \
  --duration 15
```

---

## Save a Response Plot

```bash
python3 pid_sim.py --save response.png
```

The generated image is saved locally and excluded from version control.

---

## Parameter Sweep

ControlSim can compare multiple proportional-gain values while keeping `Ki` and `Kd` constant.

Example:

```bash
python3 pid_sim.py \
  --sweep \
  --kp-range 0.5 4.0 6
```

This generates six simulations with proportional gains distributed between `0.5` and `4.0`.

The sweep can also be exported:

```bash
python3 pid_sim.py \
  --sweep \
  --kp-range 0.5 4.0 6 \
  --save sweep.png
```

---

## Command-Line Options

| Option | Description |
| --- | --- |
| `--kp` | Proportional gain |
| `--ki` | Integral gain |
| `--kd` | Derivative gain |
| `--setpoint` | Desired plant output |
| `--dt` | Simulation time step |
| `--duration` | Simulation duration |
| `--a` | Plant decay coefficient |
| `--b` | Plant input coefficient |
| `--save` | Save the generated plot |
| `--sweep` | Run a proportional-gain sweep |
| `--kp-range` | Define sweep minimum, maximum, and count |

Display command help with:

```bash
python3 pid_sim.py --help
```

---

## Program Structure

```text
main()
 |
 +-- parse_args()
 |
 +-- simulate_pid()
 |
 +-- calculate_metrics()
 |
 +-- print_metrics()
 |
 +-- plot_response()
 |
 +-- plot_sweep()
```

The program separates simulation, analysis, visualization, and command-line configuration into focused functions.

---

## Project Structure

```text
control-sim-python/
├── .gitignore
├── LICENSE
├── README.md
├── pid_sim.py
└── requirements.txt
```

Generated plots are excluded from version control.

---

## Input Validation

ControlSim validates important simulation parameters before running.

For example:

```bash
python3 pid_sim.py --dt -1
```

returns:

```text
ERROR: Time step must be greater than zero.
```

The simulator also validates:

- Simulation duration
- Plant decay coefficient
- Plant input coefficient
- Parameter-sweep size
- Parameter-sweep bounds

---

## Verification

The current implementation has been manually verified for:

- Dependency installation
- Command-line argument handling
- Default PID simulation
- Custom PID gains
- Performance metric calculation
- Plot generation
- Plot export
- Proportional-gain parameter sweeps
- Invalid time-step rejection
- Generated-file exclusion through `.gitignore`

---

## What I Learned

This project strengthened my understanding of:

- PID feedback control
- First-order dynamic systems
- Numerical simulation
- Controller tuning
- Actuator saturation
- Integral windup
- Closed-loop performance analysis
- NumPy-based numerical computation
- Matplotlib visualization
- Command-line application design
- Input validation and error handling

---

## Future Improvements

Potential extensions include:

- Integral absolute error metrics
- Rise-time calculation
- Control-effort visualization
- Additional plant models
- Automated PID tuning
- Ziegler-Nichols tuning tools
- Automated unit tests
- CSV simulation export
- Comparison of multiple PID configurations
- Interactive visualization

---

## License

This project is licensed under the MIT License.

See `LICENSE` for details.

---

## Author

Paul Cervellero

Computer Engineering  
University of South Carolina

Portfolio:  
https://paulcervellero.github.io
