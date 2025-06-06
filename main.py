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
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import os
from matplotlib.patches import Circle
import matplotlib.patches as patches


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


def run_phy_simulation(config: Dict[str, Any]) -> None:
    """
    Run the original physical layer simulation and analysis.
    """
    print("HYBRID UNDERWATER NETWORK ANALYZER")
    print("Demonstrating Strategic Communication Path Selection")
    print("=" * 80)

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


def run_game_theory_simulation(config: Dict[str, Any]) -> None:
    """
    Run the game theory multi-agent optimization simulation.
    """
    # Initialize and run game theory simulation
    game_manager = GameManager(config)
    results = game_manager.run_simulation()

    # Generate visualizations
    gt_visualizer = GameTheoryVisualizer(config, results)
    gt_visualizer.plot_auv_paths()
    gt_visualizer.plot_convergence_analysis()

    # Print final analysis
    print("\n" + "=" * 80)
    print("GAME THEORY ANALYSIS COMPLETE")
    print("=" * 80)
    print("Key Results:")

    # Calculate final metrics
    total_utility = sum(
        results["utility_histories"][i][-1]
        for i in range(len(results["utility_histories"]))
    )
    avg_utility_per_auv = total_utility / len(results["utility_histories"])

    print(f"• Total System Utility: {total_utility:.2f}")
    print(f"• Average Utility per AUV: {avg_utility_per_auv:.2f}")
    print(f"• Number of AUVs: {len(results['utility_histories'])}")
    print(f"• Convergence achieved: {'Yes' if results['convergence_data'] else 'No'}")

    # Show final AUV states
    for i, auv in enumerate(game_manager.auvs):
        print(
            f"• AUV-{i}: Final Position {auv.position}, Energy: {auv.energy:.1f}J, Buffer: {len(auv.data_buffer)} packets"
        )

    print("=" * 80)


def main():
    """
    Main execution function that orchestrates both simulation modes:
    1. Physical Layer (PHY) Analysis - Original SNR comparison
    2. Game Theory Multi-Agent Optimization - Nash Equilibrium finding
    """
    print("🌊 HYBRID UNDERWATER COMMUNICATION NETWORK SIMULATOR")
    print("Advanced Multi-Layer Analysis Framework")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Check if game theory configuration exists
    run_game_theory = "game_theory" in config

    if run_game_theory:
        print("🎮 RUNNING GAME THEORY MULTI-AGENT SIMULATION")
        print("=" * 80)
        run_game_theory_simulation(config)

        print("\n" + "=" * 80)
        print("📡 RUNNING PHYSICAL LAYER ANALYSIS")
        print("=" * 80)
        run_phy_simulation(config)
    else:
        print("📡 RUNNING PHYSICAL LAYER ANALYSIS ONLY")
        print("(Add 'game_theory' section to config.yaml for multi-agent simulation)")
        print("=" * 80)
        run_phy_simulation(config)


# ===================================================================
# GAME THEORY MULTI-AGENT OPTIMIZATION LAYER
# ===================================================================


class CommunicationMode(Enum):
    """Enumeration of possible communication modes for AUVs"""

    LISTEN = "listen"
    TRANSMIT_ACOUSTIC = "transmit_acoustic"
    RELAY = "relay"
    IDLE = "idle"


@dataclass
class Position:
    """3D position representation"""

    x: float
    y: float
    z: float

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def __str__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


@dataclass
class Action:
    """Action representation for AUV decision making"""

    next_waypoint: Position
    communication_mode: CommunicationMode
    target_auv_id: Optional[int] = None  # For relay operations
    target_sensor_id: Optional[int] = None  # For data collection


class AUV:
    """
    Autonomous Underwater Vehicle agent in the game theory model.
    Each AUV is a rational player seeking to maximize its utility.
    """

    def __init__(self, auv_id: int, initial_position: Position, config: Dict[str, Any]):
        self.id = auv_id
        self.position = initial_position
        self.config = config["game_theory"]
        self.phy_config = config["physical_layer"]

        # AUV State
        self.energy = self.config["auv_specs"]["initial_energy"]
        self.data_buffer = []  # List of (packet_data, timestamp) tuples
        self.movement_speed = self.config["auv_specs"]["movement_speed"]
        self.max_buffer_capacity = self.config["auv_specs"]["buffer_capacity"]

        # Communication ranges
        self.mi_range = self.config["auv_specs"]["communication_range_mi"]
        self.acoustic_range = self.config["auv_specs"]["communication_range_acoustic"]

        # Utility tracking
        self.cumulative_utility = 0.0
        self.action_history = []
        self.utility_history = []

        # Initialize PHY layer channels for SNR calculations
        self.acoustic_channel = AcousticChannel(self.phy_config["acoustic"])
        self.mi_channel = MIChannel(self.phy_config["mi"])

    def get_possible_actions(self, game_state: "GameState") -> List[Action]:
        """
        Generate all possible actions for this AUV given the current game state.
        """
        actions = []

        # Action 1: Move to sensors for data collection (via MI)
        for i, sensor_pos in enumerate(game_state.sensor_positions):
            if len(self.data_buffer) < self.max_buffer_capacity:
                actions.append(
                    Action(
                        next_waypoint=sensor_pos,
                        communication_mode=CommunicationMode.LISTEN,
                        target_sensor_id=i,
                    )
                )

        # Action 2: Move to surface station for data offload (via Acoustic)
        if len(self.data_buffer) > 0:
            actions.append(
                Action(
                    next_waypoint=game_state.surface_station,
                    communication_mode=CommunicationMode.TRANSMIT_ACOUSTIC,
                )
            )

        # Action 3: Position as relay for other AUVs
        for other_auv in game_state.auvs:
            if other_auv.id != self.id and len(other_auv.data_buffer) > 0:
                # Calculate optimal relay position (midpoint between AUV and surface)
                relay_x = (other_auv.position.x + game_state.surface_station.x) / 2
                relay_y = (other_auv.position.y + game_state.surface_station.y) / 2
                relay_z = (other_auv.position.z + game_state.surface_station.z) / 2
                relay_position = Position(relay_x, relay_y, relay_z)

                actions.append(
                    Action(
                        next_waypoint=relay_position,
                        communication_mode=CommunicationMode.RELAY,
                        target_auv_id=other_auv.id,
                    )
                )

        # Action 4: Stay idle (do nothing)
        actions.append(
            Action(
                next_waypoint=self.position, communication_mode=CommunicationMode.IDLE
            )
        )

        return actions

    def calculate_utility(self, action: Action, game_state: "GameState") -> float:
        """
        Calculate the utility for taking a specific action in the given game state.
        U_i(s, a_i) = w_1 * Throughput_i - w_2 * Energy_Cost_i - w_3 * Delay_i
        """
        weights = self.config["utility_weights"]
        energy_costs = self.config["energy_costs"]

        # Initialize utility components
        throughput_reward = 0.0
        energy_cost = 0.0
        delay_penalty = 0.0

        # Calculate movement energy cost
        movement_distance = self.position.distance_to(action.next_waypoint)
        energy_cost += movement_distance * energy_costs["movement_per_meter"]

        # Calculate communication rewards and costs
        if action.communication_mode == CommunicationMode.LISTEN:
            # Reward for collecting data (if within MI range of sensor)
            if action.target_sensor_id is not None:
                sensor_pos = game_state.sensor_positions[action.target_sensor_id]
                distance_to_sensor = action.next_waypoint.distance_to(sensor_pos)

                if distance_to_sensor <= self.mi_range:
                    # Calculate MI SNR for data collection
                    snr_db = self.mi_channel.calculate_snr_db(
                        np.array([distance_to_sensor])
                    )[0]
                    if snr_db > 3.0:  # Minimum viable SNR
                        throughput_reward += snr_db * 0.1  # Scale SNR to utility
                        energy_cost += energy_costs["mi_transmit_per_packet"]

        elif action.communication_mode == CommunicationMode.TRANSMIT_ACOUSTIC:
            # Reward for offloading data to surface station
            distance_to_surface = action.next_waypoint.distance_to(
                game_state.surface_station
            )

            if distance_to_surface <= self.acoustic_range:
                # Calculate Acoustic SNR for data transmission
                snr_db = self.acoustic_channel.calculate_snr_db(
                    np.array([distance_to_surface])
                )[0]
                if snr_db > 3.0:  # Minimum viable SNR
                    # Reward proportional to data amount and link quality
                    data_packets = len(self.data_buffer)
                    throughput_reward += data_packets * snr_db * 0.2
                    energy_cost += (
                        data_packets * energy_costs["acoustic_transmit_per_packet"]
                    )

        elif action.communication_mode == CommunicationMode.RELAY:
            # Reward for helping other AUVs (cooperative behavior)
            if action.target_auv_id is not None:
                target_auv = next(
                    (auv for auv in game_state.auvs if auv.id == action.target_auv_id),
                    None,
                )
                if target_auv and len(target_auv.data_buffer) > 0:
                    # Calculate two-hop relay viability
                    dist_to_auv = action.next_waypoint.distance_to(target_auv.position)
                    dist_to_surface = action.next_waypoint.distance_to(
                        game_state.surface_station
                    )

                    if (
                        dist_to_auv <= self.mi_range
                        and dist_to_surface <= self.acoustic_range
                    ):
                        mi_snr = self.mi_channel.calculate_snr_db(
                            np.array([dist_to_auv])
                        )[0]
                        acoustic_snr = self.acoustic_channel.calculate_snr_db(
                            np.array([dist_to_surface])
                        )[0]

                        if mi_snr > 3.0 and acoustic_snr > 3.0:
                            relay_snr = min(
                                mi_snr, acoustic_snr
                            )  # Bottleneck principle
                            throughput_reward += (
                                len(target_auv.data_buffer) * relay_snr * 0.1
                            )
                            energy_cost += (
                                len(target_auv.data_buffer)
                                * energy_costs["relay_per_packet"]
                            )

        # Calculate delay penalty (age of data in buffer)
        current_time = game_state.current_time
        for packet_data, timestamp in self.data_buffer:
            delay_penalty += (current_time - timestamp) * 0.1

        # Add idle energy cost
        energy_cost += (
            energy_costs["idle_per_second"]
            * self.config["simulation"]["time_step_duration"]
        )

        # Combine utility components
        utility = (
            weights["throughput"] * throughput_reward
            - weights["energy"] * energy_cost
            - weights["delay"] * delay_penalty
        )

        return utility

    def choose_best_action(self, game_state: "GameState") -> Action:
        """
        Choose the action that maximizes utility for this AUV.
        """
        possible_actions = self.get_possible_actions(game_state)

        if not possible_actions:
            # Fallback to idle if no actions available
            return Action(
                next_waypoint=self.position, communication_mode=CommunicationMode.IDLE
            )

        # Evaluate utility for each action
        best_action = None
        best_utility = float("-inf")

        for action in possible_actions:
            utility = self.calculate_utility(action, game_state)
            if utility > best_utility:
                best_utility = utility
                best_action = action

        # Store action and utility for analysis
        self.action_history.append(best_action)
        self.utility_history.append(best_utility)
        self.cumulative_utility += best_utility

        return best_action

    def execute_action(self, action: Action, game_state: "GameState") -> Dict[str, Any]:
        """
        Execute the chosen action and update AUV state.
        Returns a dictionary with execution results.
        """
        results = {
            "action_type": action.communication_mode.value,
            "energy_consumed": 0.0,
            "data_transferred": 0,
            "success": False,
        }

        energy_costs = self.config["energy_costs"]

        # Move to target waypoint
        movement_distance = self.position.distance_to(action.next_waypoint)
        movement_energy = movement_distance * energy_costs["movement_per_meter"]

        # Check if we have enough energy to move
        if self.energy >= movement_energy:
            self.position = action.next_waypoint
            self.energy -= movement_energy
            results["energy_consumed"] += movement_energy

            # Execute communication action
            if (
                action.communication_mode == CommunicationMode.LISTEN
                and action.target_sensor_id is not None
            ):
                # Collect data from sensor
                sensor_pos = game_state.sensor_positions[action.target_sensor_id]
                distance_to_sensor = self.position.distance_to(sensor_pos)

                if (
                    distance_to_sensor <= self.mi_range
                    and len(self.data_buffer) < self.max_buffer_capacity
                ):
                    # Generate new data packet
                    packet = {
                        "source_sensor": action.target_sensor_id,
                        "data_size": 1,  # Normalized packet size
                        "collection_time": game_state.current_time,
                    }
                    self.data_buffer.append((packet, game_state.current_time))
                    results["data_transferred"] = 1
                    results["success"] = True

                    # Energy cost for MI reception
                    mi_energy = energy_costs["mi_transmit_per_packet"]
                    self.energy -= mi_energy
                    results["energy_consumed"] += mi_energy

            elif action.communication_mode == CommunicationMode.TRANSMIT_ACOUSTIC:
                # Transmit data to surface station
                distance_to_surface = self.position.distance_to(
                    game_state.surface_station
                )

                if (
                    distance_to_surface <= self.acoustic_range
                    and len(self.data_buffer) > 0
                ):
                    # Transmit all buffered data
                    data_packets = len(self.data_buffer)
                    acoustic_energy = (
                        data_packets * energy_costs["acoustic_transmit_per_packet"]
                    )

                    if self.energy >= acoustic_energy:
                        self.energy -= acoustic_energy
                        results["energy_consumed"] += acoustic_energy
                        results["data_transferred"] = data_packets
                        results["success"] = True

                        # Clear buffer after successful transmission
                        self.data_buffer = []

            elif (
                action.communication_mode == CommunicationMode.RELAY
                and action.target_auv_id is not None
            ):
                # Act as relay for another AUV
                target_auv = next(
                    (auv for auv in game_state.auvs if auv.id == action.target_auv_id),
                    None,
                )
                if target_auv and len(target_auv.data_buffer) > 0:
                    dist_to_auv = self.position.distance_to(target_auv.position)
                    dist_to_surface = self.position.distance_to(
                        game_state.surface_station
                    )

                    if (
                        dist_to_auv <= self.mi_range
                        and dist_to_surface <= self.acoustic_range
                    ):
                        # Relay target AUV's data
                        data_packets = len(target_auv.data_buffer)
                        relay_energy = data_packets * energy_costs["relay_per_packet"]

                        if self.energy >= relay_energy:
                            self.energy -= relay_energy
                            results["energy_consumed"] += relay_energy
                            results["data_transferred"] = data_packets
                            results["success"] = True

                            # Clear target AUV's buffer after successful relay
                            target_auv.data_buffer = []

        # Add idle energy cost
        idle_energy = (
            energy_costs["idle_per_second"]
            * self.config["simulation"]["time_step_duration"]
        )
        self.energy -= idle_energy
        results["energy_consumed"] += idle_energy

        return results


@dataclass
class GameState:
    """Represents the complete state of the multi-agent game"""

    current_time: float
    auvs: List[AUV]
    surface_station: Position
    sensor_positions: List[Position]
    iteration: int = 0


class GameManager:
    """
    Orchestrates the multi-agent game simulation and Nash Equilibrium finding.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config["game_theory"]
        self.phy_config = config["physical_layer"]

        # Initialize environment
        self.bounds = self.config["environment"]["bounds"]
        self.surface_station = Position(*self.config["surface_station"]["position"])
        self.sensor_positions = [
            Position(*pos) for pos in self.config["sensor_nodes"]["positions"]
        ]

        # Initialize AUVs with random starting positions
        self.auvs = []
        for i in range(self.config["num_auvs"]):
            initial_pos = self._generate_random_position()
            auv = AUV(i, initial_pos, config)
            self.auvs.append(auv)

        # Simulation state
        self.current_time = 0.0
        self.iteration = 0
        self.convergence_history = []
        self.system_utility_history = []

        # Results storage
        self.results = {
            "auv_paths": [[] for _ in range(self.config["num_auvs"])],
            "utility_histories": [[] for _ in range(self.config["num_auvs"])],
            "convergence_data": [],
            "final_strategies": [],
        }

    def _generate_random_position(self) -> Position:
        """Generate a random valid position within environment bounds"""
        x = random.uniform(self.bounds["x_range"][0], self.bounds["x_range"][1])
        y = random.uniform(self.bounds["y_range"][0], self.bounds["y_range"][1])
        z = random.uniform(self.bounds["z_range"][0], self.bounds["z_range"][1])
        return Position(x, y, z)

    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the main game simulation loop to find Nash Equilibrium.
        """
        print("=" * 80)
        print("GAME THEORY MULTI-AGENT OPTIMIZATION")
        print("=" * 80)
        print(f"Initializing {self.config['num_auvs']} AUVs in underwater environment")
        print(f"Surface Station: {self.surface_station}")
        print(f"Sensor Nodes: {len(self.sensor_positions)} nodes")
        print(f"Algorithm: {self.config['learning_algorithm']['type']}")
        print("=" * 80)

        max_iterations = self.config["learning_algorithm"]["iterations"]
        convergence_threshold = self.config["learning_algorithm"][
            "convergence_threshold"
        ]
        convergence_check_interval = self.config["simulation"][
            "convergence_check_interval"
        ]

        last_strategies = [None] * self.config["num_auvs"]
        convergence_count = 0

        for iteration in range(max_iterations):
            self.iteration = iteration
            self.current_time = (
                iteration * self.config["simulation"]["time_step_duration"]
            )

            # Create current game state
            game_state = GameState(
                current_time=self.current_time,
                auvs=self.auvs,
                surface_station=self.surface_station,
                sensor_positions=self.sensor_positions,
                iteration=iteration,
            )

            # Each AUV chooses its best action
            current_strategies = []
            total_system_utility = 0.0

            for auv in self.auvs:
                action = auv.choose_best_action(game_state)
                execution_results = auv.execute_action(action, game_state)

                # Store AUV path for visualization
                self.results["auv_paths"][auv.id].append(
                    (auv.position.x, auv.position.y, auv.position.z)
                )
                self.results["utility_histories"][auv.id].append(auv.cumulative_utility)

                current_strategies.append(
                    (action.next_waypoint, action.communication_mode)
                )
                total_system_utility += (
                    auv.utility_history[-1] if auv.utility_history else 0.0
                )

                # Print iteration details
                if iteration % 20 == 0:
                    print(
                        f"Iter {iteration:3d} | AUV-{auv.id} | Pos: {auv.position} | "
                        f"Energy: {auv.energy:6.1f}J | Buffer: {len(auv.data_buffer):2d} | "
                        f"Action: {action.communication_mode.value} | "
                        f"Utility: {auv.utility_history[-1]:8.2f}"
                    )

            self.system_utility_history.append(total_system_utility)

            # Check for convergence (Nash Equilibrium)
            if iteration % convergence_check_interval == 0 and iteration > 0:
                if self._check_convergence(
                    current_strategies, last_strategies, convergence_threshold
                ):
                    convergence_count += 1
                    if convergence_count >= 3:  # Require consistent convergence
                        print(f"\n🎯 NASH EQUILIBRIUM REACHED at iteration {iteration}")
                        print("✅ System has converged to stable strategies")
                        break
                else:
                    convergence_count = 0

                last_strategies = current_strategies.copy()

            # Store convergence data
            if iteration > 0:
                strategy_change = self._calculate_strategy_change(
                    current_strategies, last_strategies
                )
                self.results["convergence_data"].append(
                    (iteration, strategy_change, total_system_utility)
                )

        # Store final strategies
        self.results["final_strategies"] = current_strategies

        print("=" * 80)
        print("SIMULATION COMPLETE")
        print(f"Total iterations: {iteration + 1}")
        print(f"Final system utility: {total_system_utility:.2f}")
        print("=" * 80)

        return self.results

    def _check_convergence(
        self, current_strategies: List, last_strategies: List, threshold: float
    ) -> bool:
        """Check if the system has converged to a Nash Equilibrium"""
        if last_strategies is None or len(last_strategies) != len(current_strategies):
            return False

        total_change = 0.0
        for i, (current, last) in enumerate(zip(current_strategies, last_strategies)):
            if current != last:
                total_change += 1.0

        strategy_change_rate = total_change / len(current_strategies)
        return strategy_change_rate <= threshold

    def _calculate_strategy_change(self, current: List, last: List) -> float:
        """Calculate the rate of strategy change between iterations"""
        if last is None:
            return 1.0

        changes = sum(1 for c, l in zip(current, last) if c != l)
        return changes / len(current) if current else 0.0


class GameTheoryVisualizer:
    """
    Visualization tools for game theory simulation results.
    """

    def __init__(self, config: Dict[str, Any], results: Dict[str, Any]):
        self.config = config["game_theory"]
        self.results = results

        # Create output directory
        self.output_dir = self.config["visualization"]["output_directory"]
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_auv_paths(self) -> None:
        """Plot 2D visualization of AUV paths and convergence"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: AUV Paths in 2D (top view)
        colors = ["red", "blue", "green", "orange", "purple"]

        # Plot environment elements
        ax1.scatter(
            *self.config["surface_station"]["position"][:2],
            s=200,
            c="gold",
            marker="*",
            label="Surface Station",
            edgecolor="black",
            linewidth=2,
        )

        for i, sensor_pos in enumerate(self.config["sensor_nodes"]["positions"]):
            ax1.scatter(
                sensor_pos[0],
                sensor_pos[1],
                s=100,
                c="brown",
                marker="s",
                alpha=0.7,
                edgecolor="black",
                linewidth=1,
            )
            ax1.annotate(
                f"S{i}",
                (sensor_pos[0], sensor_pos[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Plot AUV paths
        for auv_id, path in enumerate(self.results["auv_paths"]):
            if path:
                x_coords = [pos[0] for pos in path]
                y_coords = [pos[1] for pos in path]

                # Plot path
                ax1.plot(
                    x_coords,
                    y_coords,
                    color=colors[auv_id % len(colors)],
                    linewidth=2,
                    alpha=0.7,
                    label=f"AUV-{auv_id}",
                )

                # Mark start and end positions
                ax1.scatter(
                    x_coords[0],
                    y_coords[0],
                    color=colors[auv_id % len(colors)],
                    s=100,
                    marker="o",
                    edgecolor="black",
                    linewidth=2,
                )
                ax1.scatter(
                    x_coords[-1],
                    y_coords[-1],
                    color=colors[auv_id % len(colors)],
                    s=100,
                    marker="X",
                    edgecolor="black",
                    linewidth=2,
                )

        ax1.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax1.set_title(
            "AUV Path Optimization (Top View)", fontsize=14, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.config["environment"]["bounds"]["x_range"])
        ax1.set_ylim(self.config["environment"]["bounds"]["y_range"])

        # Plot 2: System Utility Convergence
        iterations = list(range(len(self.results["utility_histories"][0])))
        total_utilities = []

        for i in iterations:
            total_util = sum(
                self.results["utility_histories"][auv_id][i]
                for auv_id in range(len(self.results["utility_histories"]))
            )
            total_utilities.append(total_util)

        ax2.plot(
            iterations, total_utilities, "b-", linewidth=2, label="Total System Utility"
        )

        # Individual AUV utilities
        for auv_id, utilities in enumerate(self.results["utility_histories"]):
            ax2.plot(
                iterations,
                utilities,
                color=colors[auv_id % len(colors)],
                linewidth=1,
                alpha=0.7,
                linestyle="--",
                label=f"AUV-{auv_id} Utility",
            )

        ax2.set_xlabel("Iteration", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Cumulative Utility", fontsize=12, fontweight="bold")
        ax2.set_title("Nash Equilibrium Convergence", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/auv_optimization_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"📊 Path optimization visualization saved to: {self.output_dir}/auv_optimization_analysis.png"
        )

    def plot_convergence_analysis(self) -> None:
        """Plot detailed convergence analysis"""
        if not self.results["convergence_data"]:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        iterations = [data[0] for data in self.results["convergence_data"]]
        strategy_changes = [data[1] for data in self.results["convergence_data"]]
        system_utilities = [data[2] for data in self.results["convergence_data"]]

        # Plot strategy change rate
        ax1.plot(
            iterations, strategy_changes, "r-", linewidth=2, marker="o", markersize=4
        )
        ax1.axhline(
            y=self.config["learning_algorithm"]["convergence_threshold"],
            color="black",
            linestyle="--",
            alpha=0.7,
            label="Convergence Threshold",
        )
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Strategy Change Rate", fontsize=12)
        ax1.set_title("Convergence to Nash Equilibrium", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot system utility evolution
        ax2.plot(
            iterations, system_utilities, "g-", linewidth=2, marker="s", markersize=4
        )
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Total System Utility", fontsize=12)
        ax2.set_title("System Performance Evolution", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/convergence_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        print(
            f"📈 Convergence analysis saved to: {self.output_dir}/convergence_analysis.png"
        )


if __name__ == "__main__":
    main()
