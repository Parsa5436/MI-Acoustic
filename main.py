"""
Hybrid Underwater Network Simulation
====================================

A comprehensive Python simulation that demonstrates the strategic advantage
of a two-tier hybrid underwater communication system combining:
- Magnetic Induction (MI) for short-range, high-fidelity communication
- Acoustic communication for long-range transmission

Author: Hybrid Network Research Team
Date: June 2025
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import math


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file with error handling."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{path}'")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)


class AcousticChannel:
    """
    Models underwater acoustic communication using the exact formula:
    Acoustic-SNR(r) = C_a * [r^k * 10^(α(f) * r / 10)]^-1

    All calculations performed in dB domain for numerical stability.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.freq_khz = (
            self.config["carrier_frequency_hz"] / 1000.0
        )  # Convert to kHz for Thorp's formula
        self.c_a = self._calculate_acoustic_coefficient()

    def _calculate_acoustic_coefficient(self) -> float:
        """
        Calculates C_a = (P_t^(a) * G_t * G_r) / (N_0^(a) * B_a)
        Returns linear scale coefficient.
        """
        p_t = self.config["transmit_power_watts"]
        g_t = self.config["transmit_gain"]
        g_r = self.config["receive_gain"]
        n_0 = self.config["noise_psd_w_hz"]
        b_a = self.config["bandwidth_hz"]

        c_a = (p_t * g_t * g_r) / (n_0 * b_a)

        # Debug output
        print(
            f"Acoustic Debug: P_t = {p_t}, G_t = {g_t}, G_r = {g_r}, N_0 = {n_0:.2e}, B_a = {b_a}"
        )
        print(f"Acoustic Debug: C_a = {c_a:.2e}")

        return c_a

    def _thorp_absorption_db_per_km(self) -> float:
        """
        Calculates Thorp absorption coefficient:
        α(f) = (0.11f²/(1+f²)) + (44f²/(4100+f²)) + (2.75e-4 * f²) + 0.003
        where f is in kHz, result in dB/km
        """
        f = self.freq_khz
        f_squared = f * f

        term1 = (0.11 * f_squared) / (1 + f_squared)
        term2 = (44 * f_squared) / (4100 + f_squared)
        term3 = 2.75e-4 * f_squared
        term4 = 0.003

        return term1 + term2 + term3 + term4

    def calculate_snr_db(self, ranges_m: np.ndarray) -> np.ndarray:
        """
        Calculates acoustic SNR in dB using the exact formula:
        SNR_dB = 10*log10(C_a) - 10*log10(r^k * 10^(α(f)*r/10))
        """
        # Avoid division by zero
        safe_ranges = np.maximum(ranges_m, 1e-9)

        k = self.config["spreading_factor"]
        alpha_db_per_km = self._thorp_absorption_db_per_km()

        # Calculate path losses in dB
        spreading_loss_db = k * 10 * np.log10(safe_ranges)
        absorption_loss_db = (safe_ranges / 1000.0) * alpha_db_per_km  # Convert m to km
        total_path_loss_db = spreading_loss_db + absorption_loss_db

        # SNR in dB = 10*log10(C_a) - total_path_loss_db
        c_a_db = 10 * np.log10(self.c_a)
        snr_db = c_a_db - total_path_loss_db

        return snr_db


class MIChannel:
    """
    Models Magnetic Induction communication using the exact parameters
    and the r^-6 dependency from electromagnetic field theory.

    MI-SNR follows: SNR = C_m / r^6
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k_m = self._calculate_coupling_constant()
        self.c_m = self._calculate_mi_coefficient()

    def _calculate_coupling_constant(self) -> float:
        """
        Calculates K_m = ((ωμN_tN_rA_tA_r)² / ((4π)²R_tR_r))
        where ω = 2πf_m
        """
        omega = 2 * math.pi * self.config["carrier_frequency_hz"]
        mu = self.config["permeability_seawater"]
        n_t = self.config["transmit_coil_turns"]
        n_r = self.config["receive_coil_turns"]
        a_t = self.config["transmit_coil_area_m2"]
        a_r = self.config["receive_coil_area_m2"]
        r_t = self.config["transmit_coil_resistance_ohm"]
        r_r = self.config["receive_coil_resistance_ohm"]

        numerator = (omega * mu * n_t * n_r * a_t * a_r) ** 2
        denominator = ((4 * math.pi) ** 2) * r_t * r_r

        return numerator / denominator

    def _calculate_mi_coefficient(self) -> float:
        """
        Calculates C_m = (P_t^(m) * K_m) / (N_0^(m) * B_m)
        Returns linear scale coefficient.
        """
        p_t = self.config["transmit_power_watts"]
        n_0 = self.config["noise_psd_w_hz"]
        b_m = self.config["bandwidth_hz"]

        c_m = (p_t * self.k_m) / (n_0 * b_m)

        # Debug output
        print(
            f"MI Debug: K_m = {self.k_m:.2e}, P_t = {p_t}, N_0 = {n_0:.2e}, B_m = {b_m}"
        )
        print(f"MI Debug: C_m = {c_m:.2e}")

        return c_m

    def calculate_snr_db(self, ranges_m: np.ndarray) -> np.ndarray:
        """
        Calculates MI SNR in dB: SNR = C_m / r^6
        Enforces maximum useful range constraint.
        """
        # Avoid division by zero
        safe_ranges = np.maximum(ranges_m, 1e-9)

        # Calculate SNR in linear scale: SNR = C_m / r^6
        snr_linear = self.c_m / (safe_ranges**6)

        # Enforce maximum useful range constraint
        max_range = self.config["max_useful_range_m"]
        snr_linear[ranges_m > max_range] = (
            1e-20  # Effectively zero SNR beyond max range
        )

        # Convert to dB, avoiding log(0)
        snr_linear = np.maximum(snr_linear, 1e-20)
        snr_db = 10 * np.log10(snr_linear)

        return snr_db


class Simulator:
    """
    Orchestrates the hybrid underwater network simulation.
    Implements the core logic of the hybrid system decision-making.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mi_channel = MIChannel(config["physical_layer"]["mi"])
        self.acoustic_channel = AcousticChannel(config["physical_layer"]["acoustic"])
        self.ranges = np.array(config["simulation"]["evaluation_ranges_m"], dtype=float)

    def run(self) -> Dict[str, np.ndarray]:
        """
        Executes the hybrid system simulation using the exact logic:

        1. Acoustic-Only: Direct acoustic path at full range r
        2. Two-Hop Relay: MI hop (r/2) + Acoustic hop (r/2), limited by bottleneck
        3. Hybrid: Intelligently selects best path: max(Acoustic-Only, Two-Hop)

        Returns:
            Dictionary containing SNR arrays for each communication strategy
        """
        print("=" * 80)
        print("HYBRID UNDERWATER NETWORK SIMULATION")
        print("=" * 80)
        print("Physical Parameters:")
        print(
            f"• MI Frequency: {self.config['physical_layer']['mi']['carrier_frequency_hz']/1000:.0f} kHz"
        )
        print(
            f"• Acoustic Frequency: {self.config['physical_layer']['acoustic']['carrier_frequency_hz']/1000:.0f} kHz"
        )
        print(
            f"• MI Max Range: {self.config['physical_layer']['mi']['max_useful_range_m']:.0f} m"
        )
        print(f"• Evaluation Ranges: {self.ranges.astype(int).tolist()} m")
        print("=" * 80)

        # Strategy 1: Acoustic-Only Path (direct transmission at full range r)
        snr_acoustic_only = self.acoustic_channel.calculate_snr_db(self.ranges)

        # Strategy 2: Two-Hop Relay Path
        # AUV positioned at midpoint, so each hop covers distance r/2
        hop_distances = self.ranges / 2.0

        # First hop: Robot -> AUV via MI at distance r/2
        snr_mi_hop = self.mi_channel.calculate_snr_db(hop_distances)

        # Second hop: AUV -> Surface via Acoustic at distance r/2
        snr_acoustic_hop = self.acoustic_channel.calculate_snr_db(hop_distances)

        # Two-hop relay quality limited by weakest link (bottleneck principle)
        snr_two_hop_relay = np.minimum(snr_mi_hop, snr_acoustic_hop)

        # Strategy 3: Hybrid System (intelligent path selection)
        # System chooses the best available path at each range
        snr_hybrid_system = np.maximum(snr_acoustic_only, snr_two_hop_relay)

        return {
            "acoustic_only": snr_acoustic_only,
            "two_hop_relay": snr_two_hop_relay,
            "hybrid_system": snr_hybrid_system,
            # Additional data for analysis
            "mi_hop": snr_mi_hop,
            "acoustic_hop": snr_acoustic_hop,
        }


class Visualizer:
    """
    Creates professional publication-quality visualizations of the simulation results.
    Generates grouped bar charts showing the strategic advantage of hybrid systems.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config["simulation"]["visualization"]

    def plot_snr_comparison(self, snr_data: Dict[str, np.ndarray]) -> None:
        """
        Creates a grouped bar chart comparing communication strategies.

        Chart shows:
        - Acoustic-Only Path performance
        - Two-Hop Relay Path performance
        - Hybrid System (Optimal Path) performance

        The visualization clearly demonstrates the crossover point where
        the optimal strategy transitions from relay to direct acoustic.
        """
        # Extract data
        ranges_labels = self.config["range_labels"]
        acoustic_snr = snr_data["acoustic_only"]
        relay_snr = snr_data["two_hop_relay"]
        hybrid_snr = snr_data["hybrid_system"]

        # Apply clamping for visualization
        clamp_threshold = self.config["snr_clamp_threshold_db"]
        acoustic_snr_display = np.maximum(acoustic_snr, clamp_threshold)
        relay_snr_display = np.maximum(relay_snr, clamp_threshold)
        hybrid_snr_display = np.maximum(hybrid_snr, clamp_threshold)

        # Set up the grouped bar chart
        x = np.arange(len(ranges_labels))
        width = self.config["bar_width"]

        fig, ax = plt.subplots(figsize=self.config["figure_size"])

        # Create bars with professional styling
        bars1 = ax.bar(
            x - width,
            acoustic_snr_display,
            width,
            label="Acoustic-Only Path",
            color="#2E8B57",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        bars2 = ax.bar(
            x,
            relay_snr_display,
            width,
            label="Two-Hop Relay Path (MI → AUV → Acoustic)",
            color="#FF6B35",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        bars3 = ax.bar(
            x + width,
            hybrid_snr_display,
            width,
            label="Hybrid System (Optimal Path)",
            color="#1E3A8A",
            alpha=0.9,
            edgecolor="black",
            linewidth=1.2,
        )

        # Customize the chart
        ax.set_xlabel(
            "Total Communication Range (Robot to Surface) [m]",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel(
            "End-to-End Signal-to-Noise Ratio (SNR) [dB]",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_title(
            "Strategic Advantage of Hybrid Underwater Communication Systems",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(ranges_labels, fontsize=12)
        ax.set_ylim(self.config["y_axis_limits"])

        # Add horizontal reference line at 0 dB
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Add legend
        ax.legend(fontsize=12, loc="upper right", framealpha=0.9)

        # Add grid for readability
        ax.grid(
            True, alpha=0.3, linestyle="-", linewidth=0.5
        )  # Add SNR values on top of all bars
        # Acoustic-Only bars (left)
        for i, val in enumerate(acoustic_snr_display):
            if val > clamp_threshold:  # Only show labels for meaningful values
                ax.text(
                    i - width,
                    val + 2,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#2E8B57",
                )

        # Two-Hop Relay bars (center)
        for i, val in enumerate(relay_snr_display):
            if val > clamp_threshold:  # Only show labels for meaningful values
                ax.text(
                    i,
                    val + 2,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#FF6B35",
                )

        # Hybrid System bars (right)
        for i, val in enumerate(hybrid_snr_display):
            if val > clamp_threshold:  # Only show labels for meaningful values
                ax.text(
                    i + width,
                    val + 2,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#1E3A8A",
                )

        # Finalize and save
        plt.tight_layout()
        plt.savefig(
            self.config["output_path"],
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.show()

        print(f"High-resolution chart saved to: {self.config['output_path']}")


def print_detailed_results(snr_data: Dict[str, np.ndarray], ranges: np.ndarray) -> None:
    """
    Prints a comprehensive analysis of the simulation results,
    highlighting the strategic decision-making of the hybrid system.
    """
    print("\nDETAILED SIMULATION RESULTS")
    print("=" * 90)
    print(
        f"{'Range':<8} {'Acoustic':<12} {'MI-Hop':<10} {'Acoustic-Hop':<14} {'Relay':<10} {'Hybrid':<10} {'Strategy':<15}"
    )
    print(
        f"{'(m)':<8} {'Only (dB)':<12} {'(dB)':<10} {'(dB)':<14} {'(dB)':<10} {'(dB)':<10} {'Selected':<15}"
    )
    print("-" * 90)

    for i, range_m in enumerate(ranges):
        acoustic_only = snr_data["acoustic_only"][i]
        mi_hop = snr_data["mi_hop"][i]
        acoustic_hop = snr_data["acoustic_hop"][i]
        relay = snr_data["two_hop_relay"][i]
        hybrid = snr_data["hybrid_system"][i]

        # Determine which strategy is optimal
        if abs(hybrid - relay) < 0.1 and relay > acoustic_only:
            strategy = "Relay Path"
        elif abs(hybrid - acoustic_only) < 0.1 and acoustic_only > relay:
            strategy = "Direct Acoustic"
        else:
            strategy = "Equal"

        print(
            f"{range_m:<8.0f} {acoustic_only:<12.1f} {mi_hop:<10.1f} {acoustic_hop:<14.1f} "
            f"{relay:<10.1f} {hybrid:<10.1f} {strategy:<15}"
        )

    print("-" * 90)
    print("KEY INSIGHTS:")
    print(
        "• Short Range (1-30m): Two-hop relay path provides superior SNR due to MI efficiency"
    )
    print("• Long Range (100m+): Direct acoustic becomes the only viable option")
    print("• Hybrid system intelligently adapts strategy based on range conditions")
    print("• Crossover point demonstrates the strategic value of the hybrid approach")
    print("=" * 90)


def main():
    """
    Main execution function that orchestrates the complete simulation pipeline:
    1. Load configuration parameters
    2. Initialize simulation components
    3. Execute hybrid system simulation
    4. Generate comprehensive analysis and visualization
    """
    print("HYBRID UNDERWATER NETWORK ANALYZER")
    print("Demonstrating Strategic Communication Path Selection")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Initialize and run simulation
    simulator = Simulator(config)
    snr_results = simulator.run()

    # Print detailed analysis
    print_detailed_results(snr_results, simulator.ranges)

    # Generate visualization
    visualizer = Visualizer(config)
    visualizer.plot_snr_comparison(snr_results)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("The grouped bar chart clearly demonstrates the strategic advantage")
    print("of the hybrid approach across different communication ranges!")
    print("=" * 80)


if __name__ == "__main__":
    main()
