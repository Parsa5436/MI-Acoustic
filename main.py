import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file.

    Args:
        path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{path}'")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit()


class AcousticChannel:
    """
    Models the acoustic communication channel based on physics principles.
    """

    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        """
        Initializes the AcousticChannel with parameters from the config.
        """
        self.freq_khz = config["center_frequency_khz"]
        self.power_w = config["default_transmit_power_watts"]
        self.bandwidth_hz = config["bandwidth_hz"]
        self.spreading_k = config["spreading_factor"]
        self.wind_speed_ms = env_config["wind_speed_ms"]
        self.shipping_factor = env_config["shipping_factor"]
        self.source_level = 170.8 + 10 * np.log10(self.power_w)
        self.noise_level = self._calculate_total_noise_level()

    def _thorp_absorption_db_per_km(self) -> float:
        """Calculates absorption loss using Thorp's formula."""
        f2 = self.freq_khz**2
        alpha = (
            (0.11 * f2 / (1 + f2)) + (44 * f2 / (4100 + f2)) + (2.75e-4 * f2) + 0.003
        )
        return alpha

    def _urick_ambient_noise_db_per_hz(self) -> float:
        """Calculates ambient noise spectral level using Urick's model."""
        f2 = self.freq_khz**2
        noise_turbulence = 17 - 30 * np.log10(self.freq_khz)
        noise_shipping = (
            40
            + 20 * (self.shipping_factor - 0.5)
            + 26 * np.log10(self.freq_khz)
            - 60 * np.log10(self.freq_khz + 0.03)
        )
        noise_wind = (
            50
            + 7.5 * np.sqrt(self.wind_speed_ms)
            + 20 * np.log10(self.freq_khz)
            - 40 * np.log10(self.freq_khz + 0.4)
        )
        noise_thermal = -15 + 20 * np.log10(self.freq_khz)

        total_noise_linear = sum(
            10 ** (n / 10)
            for n in [noise_turbulence, noise_shipping, noise_wind, noise_thermal]
        )
        return 10 * np.log10(total_noise_linear)

    def _calculate_total_noise_level(self) -> float:
        """Calculates the total noise level across the signal bandwidth."""
        noise_spectrum_level = self._urick_ambient_noise_db_per_hz()
        return noise_spectrum_level + 10 * np.log10(self.bandwidth_hz)

    def calculate_snr(self, ranges_m: np.ndarray) -> np.ndarray:
        """
        Calculates the Signal-to-Noise Ratio (SNR) for given ranges.

        Args:
            ranges_m (np.ndarray): An array of distances in meters.

        Returns:
            np.ndarray: The calculated SNR in dB for each range.
        """
        safe_ranges = np.maximum(ranges_m, 1e-9)  # Avoid log(0)

        spreading_loss = self.spreading_k * 10 * np.log10(safe_ranges)
        absorption_loss = (safe_ranges / 1000.0) * self._thorp_absorption_db_per_km()
        transmission_loss = spreading_loss + absorption_loss

        snr = self.source_level - transmission_loss - self.noise_level
        return snr


class MIChannel:
    """
    Models the Magnetic Induction (MI) communication channel.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MIChannel and calibrates it based on config parameters.
        """
        calib_config = config["calibration"]
        self.calibration_constant_db = self._calibrate(
            calib_config["max_range_m"], calib_config["snr_at_max_range_db"]
        )

    def _calibrate(self, max_range_m: float, snr_at_max_range_db: float) -> float:
        """
        Calculates the system constant C_m based on a known performance point.
        The model is SNR_dB = C_dB - 60 * log10(r).
        """
        return snr_at_max_range_db + 60 * np.log10(max_range_m)

    def calculate_snr(self, ranges_m: np.ndarray) -> np.ndarray:
        """
        Calculates the Signal-to-Noise Ratio (SNR) for given ranges.

        Args:
            ranges_m (np.ndarray): An array of distances in meters.

        Returns:
            np.ndarray: The calculated SNR in dB for each range.
        """
        safe_ranges = np.maximum(ranges_m, 1e-9)  # Avoid log(0)
        snr = self.calibration_constant_db - 60 * np.log10(safe_ranges)
        return snr


class Simulator:
    """
    Runs the simulation to calculate SNR values for different models.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mi_channel = MIChannel(config["physical_layer"]["mi"])
        self.acoustic_channel = AcousticChannel(
            config["physical_layer"]["acoustic"], config["environment"]
        )
        self.evaluation_ranges = np.array(config["simulation"]["evaluation_ranges_m"])

    def run(self) -> Dict[str, np.ndarray]:
        """
        Executes the simulation and returns the SNR results.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing SNR values for
                                   'mi', 'acoustic', and 'hybrid' models.
        """
        snr_mi = self.mi_channel.calculate_snr(self.evaluation_ranges)
        snr_acoustic = self.acoustic_channel.calculate_snr(self.evaluation_ranges)
        snr_hybrid = np.maximum(snr_mi, snr_acoustic)

        return {"mi": snr_mi, "acoustic": snr_acoustic, "hybrid": snr_hybrid}


class Visualizer:
    """
    Handles the plotting and visualization of simulation results.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        plt.style.use(config["style"])

    def plot_snr_comparison(self, snr_data: Dict[str, np.ndarray]):
        """
        Generates and displays a grouped bar chart comparing the SNR of the models.

        Args:
            snr_data (Dict[str, np.ndarray]): The SNR results from the simulator.
        """
        labels = self.config["range_labels"]
        x = np.arange(len(labels))
        width = self.config["bar_width"]
        clamp_val = self.config["snr_clamp_threshold_db"]

        snr_mi = np.maximum(snr_data["mi"], clamp_val)
        snr_acoustic = np.maximum(snr_data["acoustic"], clamp_val)
        snr_hybrid = np.maximum(snr_data["hybrid"], clamp_val)

        fig, ax = plt.subplots(figsize=self.config["figure_size"])

        ax.bar(x - width, snr_mi, width, label="MI-Only", color="#ff6666", alpha=0.9)
        ax.bar(
            x, snr_acoustic, width, label="Acoustic-Only", color="#6666ff", alpha=0.9
        )
        ax.bar(
            x + width,
            snr_hybrid,
            width,
            label="Hybrid System (Operational)",
            color="green",
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_ylabel(
            "Signal-to-Noise Ratio (SNR) (dB)", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Communication Range", fontsize=14, fontweight="bold")
        ax.set_title(
            "Comparative SNR of Underwater Communication Models",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.legend(fontsize=12)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylim(self.config["y_axis_limits"])

        # Annotations for clarity
        ax.axvspan(-0.5, 2.5, facecolor="red", alpha=0.05, zorder=0)
        ax.text(
            1,
            95,
            "MI Tier Dominates\n(High Bandwidth)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1),
        )
        ax.axvspan(2.5, 3.5, facecolor="gray", alpha=0.2, zorder=0)
        ax.text(
            3,
            60,
            "Communication Gap\n(Bridged by AUV)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
        )
        ax.axvspan(3.5, len(labels) - 0.5, facecolor="blue", alpha=0.05, zorder=0)
        ax.text(
            5.5,
            95,
            "Acoustic Tier Dominates\n(Long Range)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1),
        )

        # Add data labels on top of the hybrid bars
        for i, val in enumerate(snr_hybrid):
            ax.text(
                i + width,
                val + 1.5,
                f"{val:.1f} dB",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        fig.tight_layout()
        plt.savefig(self.config["output_path"], dpi=300)
        plt.show()


def main():
    """
    Main function to run the simulation and visualization pipeline.
    """
    config = load_config()
    simulator = Simulator(config)
    snr_results = simulator.run()
    visualizer = Visualizer(config["simulation"]["visualization"])
    visualizer.plot_snr_comparison(snr_results)
    print(f"Chart saved to {config['simulation']['visualization']['output_path']}")


if __name__ == "__main__":
    main()
