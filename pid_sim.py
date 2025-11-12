#!/usr/bin/env python3
"""
pid_sim.py

Simple PID controller simulation of a first-order plant.

Usage examples:
  python pid_sim.py
  python pid_sim.py --kp 3.0 --ki 0.5 --kd 0.05 --save response.png
  python pid_sim.py --sweep --kp-range 0.5 4.0 6
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

def simulate_pid(kp, ki, kd, setpoint=1.0, dt=0.01, T=10.0, a=1.0, b=1.0):
    """
    Simulate a closed-loop PID controller on a first-order plant:
      plant: dy/dt = -a*y + b*u
    Returns: (t_array, y_array, u_array)
    """
    n = int(T / dt)
    y = 0.0
    integral = 0.0
    prev_error = 0.0
    ys = np.zeros(n)
    us = np.zeros(n)
    ts = np.linspace(0, T, n)
    for i in range(n):
        error = setpoint - y
        integral += error * dt
        derivative = (error - prev_error) / dt if i > 0 else 0.0
        u = kp * error + ki * integral + kd * derivative
        u = max(min(u, 100.0), -100.0)  # saturate control
        y += dt * (-a * y + b * u)
        ys[i] = y
        us[i] = u
        prev_error = error
    return ts, ys, us

def plot_single(ts, ys, title="PID response", label=None, save=None):
    plt.figure(figsize=(9,4))
    plt.plot(ts, ys, label=label or "response")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title(title)
    plt.grid(True)
    if label:
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved plot to {save}")
    else:
        plt.show()

def plot_sweep(ts, results, labels, title="PID sweep"):
    plt.figure(figsize=(9,5))
    for ys, lab in zip(results, labels):
        plt.plot(ts, ys, label=lab)
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def parse_args():
    p = argparse.ArgumentParser(description="PID control simulation (first-order plant)")
    p.add_argument("--kp", type=float, default=2.0)
    p.add_argument("--ki", type=float, default=0.5)
    p.add_argument("--kd", type=float, default=0.1)
    p.add_argument("--setpoint", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--T", type=float, default=10.0)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--kp-range", nargs=3, type=float, metavar=("MIN","MAX","N"), default=[0.5, 4.0, 5])
    return p.parse_args()

def main():
    args = parse_args()
    if args.sweep:
        kp_min, kp_max, kp_n = args.kp_range
        ks = np.linspace(kp_min, kp_max, int(kp_n))
        results = []
        labels = []
        ts = None
        for k in ks:
            ts, ys, us = simulate_pid(k, args.ki, args.kd, setpoint=args.setpoint, dt=args.dt, T=args.T)
            results.append(ys)
            labels.append(f"kp={k:.2f}")
        plot_sweep(ts, results, labels)
        return
    ts, ys, us = simulate_pid(args.kp, args.ki, args.kd, setpoint=args.setpoint, dt=args.dt, T=args.T)
    plot_single(ts, ys, label=f"kp={args.kp} ki={args.ki} kd={args.kd}", save=args.save)

if __name__ == "__main__":
    main()
