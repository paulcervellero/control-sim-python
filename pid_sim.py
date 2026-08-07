#!/usr/bin/env python3
"""
ControlSim

PID controller simulation for a first-order plant.

Plant model:
    dy/dt = -a*y + b*u

Examples:
    python3 pid_sim.py
    python3 pid_sim.py --kp 3.0 --ki 0.5 --kd 0.05
    python3 pid_sim.py --save response.png
    python3 pid_sim.py --sweep --kp-range 0.5 4.0 6
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


CONTROL_LIMIT = 100.0


def simulate_pid(
    kp,
    ki,
    kd,
    setpoint=1.0,
    dt=0.01,
    duration=10.0,
    a=1.0,
    b=1.0,
):
    """
    Simulate a PID-controlled first-order plant.

    Plant:
        dy/dt = -a*y + b*u

    Returns:
        time, output, control
    """

    if dt <= 0:
        raise ValueError("Time step must be greater than zero.")

    if duration <= 0:
        raise ValueError("Simulation duration must be greater than zero.")

    if a <= 0:
        raise ValueError("Plant parameter 'a' must be greater than zero.")

    if b == 0:
        raise ValueError("Plant parameter 'b' cannot be zero.")

    steps = int(duration / dt) + 1

    time = np.arange(steps) * dt
    output = np.zeros(steps)
    control = np.zeros(steps)

    y = 0.0
    integral = 0.0
    previous_error = setpoint - y

    for i in range(steps):
        error = setpoint - y

        derivative = (
            (error - previous_error) / dt
            if i > 0
            else 0.0
        )

        candidate_integral = integral + error * dt

        unsaturated_control = (
            kp * error
            + ki * candidate_integral
            + kd * derivative
        )

        u = np.clip(
            unsaturated_control,
            -CONTROL_LIMIT,
            CONTROL_LIMIT,
        )

        # Simple conditional-integration anti-windup:
        # accept the new integral only when the controller
        # output is not being limited.
        if u == unsaturated_control:
            integral = candidate_integral

        y += dt * (-a * y + b * u)

        output[i] = y
        control[i] = u

        previous_error = error

    return time, output, control


def calculate_metrics(time, output, setpoint):
    """Calculate basic closed-loop response metrics."""

    error = setpoint - output

    steady_state_error = abs(error[-1])

    if setpoint != 0:
        overshoot = max(
            0.0,
            (np.max(output) - setpoint)
            / abs(setpoint)
            * 100.0,
        )
    else:
        overshoot = 0.0

    tolerance = max(0.02 * abs(setpoint), 0.02)

    settling_time = None

    for i in range(len(output)):
        if np.all(np.abs(error[i:]) <= tolerance):
            settling_time = time[i]
            break

    return {
        "steady_state_error": steady_state_error,
        "overshoot_percent": overshoot,
        "settling_time": settling_time,
    }


def print_metrics(metrics):
    """Print simulation performance metrics."""

    print("\nPerformance Metrics")
    print("-------------------")

    print(
        f"Steady-state error: "
        f"{metrics['steady_state_error']:.4f}"
    )

    print(
        f"Overshoot: "
        f"{metrics['overshoot_percent']:.2f}%"
    )

    settling_time = metrics["settling_time"]

    if settling_time is None:
        print("Settling time: not reached")
    else:
        print(f"Settling time: {settling_time:.2f} s")


def plot_response(
    time,
    output,
    setpoint,
    label=None,
    save=None,
):
    """Plot a single PID response."""

    plt.figure(figsize=(9, 5))

    plt.plot(
        time,
        output,
        label=label or "Plant output",
    )

    plt.axhline(
        setpoint,
        linestyle="--",
        label="Setpoint",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Plant Output")
    plt.title("PID Closed-Loop Response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
        print(f"Saved response plot to {save}")
    else:
        plt.show()


def plot_sweep(
    time,
    results,
    labels,
    setpoint,
    save=None,
):
    """Plot multiple proportional-gain responses."""

    plt.figure(figsize=(9, 5))

    for output, label in zip(results, labels):
        plt.plot(time, output, label=label)

    plt.axhline(
        setpoint,
        linestyle="--",
        label="Setpoint",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Plant Output")
    plt.title("PID Proportional-Gain Sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
        print(f"Saved sweep plot to {save}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PID controller simulation for "
            "a first-order plant."
        )
    )

    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--ki", type=float, default=0.5)
    parser.add_argument("--kd", type=float, default=0.1)

    parser.add_argument(
        "--setpoint",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation time step.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds.",
    )

    parser.add_argument(
        "--a",
        type=float,
        default=1.0,
        help="First-order plant decay coefficient.",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=1.0,
        help="First-order plant input coefficient.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save plot to the specified file.",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a proportional-gain parameter sweep.",
    )

    parser.add_argument(
        "--kp-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "N"),
        default=[0.5, 4.0, 5],
        help="Kp sweep range: minimum maximum count.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.sweep:
            kp_min, kp_max, kp_count = args.kp_range

            count = int(kp_count)

            if count < 2:
                raise ValueError(
                    "Kp sweep requires at least two values."
                )

            if kp_max <= kp_min:
                raise ValueError(
                    "Kp sweep maximum must exceed minimum."
                )

            gains = np.linspace(
                kp_min,
                kp_max,
                count,
            )

            results = []
            labels = []
            time = None

            for kp in gains:
                time, output, _ = simulate_pid(
                    kp,
                    args.ki,
                    args.kd,
                    setpoint=args.setpoint,
                    dt=args.dt,
                    duration=args.duration,
                    a=args.a,
                    b=args.b,
                )

                results.append(output)
                labels.append(f"Kp={kp:.2f}")

            plot_sweep(
                time,
                results,
                labels,
                args.setpoint,
                save=args.save,
            )

            return 0

        time, output, _ = simulate_pid(
            args.kp,
            args.ki,
            args.kd,
            setpoint=args.setpoint,
            dt=args.dt,
            duration=args.duration,
            a=args.a,
            b=args.b,
        )

        metrics = calculate_metrics(
            time,
            output,
            args.setpoint,
        )

        print_metrics(metrics)

        plot_response(
            time,
            output,
            args.setpoint,
            label=(
                f"Kp={args.kp}, "
                f"Ki={args.ki}, "
                f"Kd={args.kd}"
            ),
            save=args.save,
        )

        return 0

    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
