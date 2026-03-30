"""Plotting utilities for recursive benchmarking results."""

import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Any

class ResultsPlotter:
    """Generates trajectory plots for recursive experiments."""
    
    @staticmethod
    def plot_regime_trajectories(regime_results: Dict[str, List[Dict[str, Any]]], 
                                 output_dir: str):
        """
        Plots Accuracy and Perplexity over generations for all regimes.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Accuracy Plot (Grounded Tasks)
        plt.figure(figsize=(10, 6))
        for regime, metrics in regime_results.items():
            gens = [m["generation"] for m in metrics]
            # Average accuracy across families that have it
            accs = []
            for m in metrics:
                family_accs = [v for k, v in m.items() if "_accuracy" in k]
                accs.append(sum(family_accs) / len(family_accs) if family_accs else 0.0)
            
            plt.plot(gens, accs, marker='o', label=regime)
            
        plt.title("Grounded Accuracy over Recursive Generations")
        plt.xlabel("Generation")
        plt.ylabel("Mean Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "accuracy_trajectory.png"))
        plt.close()

        # 2. Perplexity Plot
        plt.figure(figsize=(10, 6))
        for regime, metrics in regime_results.items():
            gens = [m["generation"] for m in metrics]
            # Use wikitext as the pure fluency proxy
            ppls = [m.get("fluency_wiki_perplexity", 0.0) for m in metrics]
            plt.plot(gens, ppls, marker='s', linestyle='--', label=regime)
            
        plt.title("OOD Fluency (Wikitext Perplexity) over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "perplexity_trajectory.png"))
        plt.close()
        
        # 3. Early Warning Gap (Acc Drop vs PPL Rise)
        plt.figure(figsize=(10, 6))
        for regime, metrics in regime_results.items():
            if not metrics: continue
            gens = [m["generation"] for m in metrics]
            
            base_accs = [v for k, v in metrics[0].items() if "_accuracy" in k]
            base_acc = sum(base_accs) / len(base_accs) if base_accs else 0.0
            base_ppl = metrics[0].get("fluency_wiki_perplexity", 1.0)

            gaps = []
            for m in metrics:
                curr_accs = [v for k, v in m.items() if "_accuracy" in k]
                curr_acc = sum(curr_accs) / len(curr_accs) if curr_accs else 0.0
                curr_ppl = m.get("fluency_wiki_perplexity", base_ppl)
                
                acc_drop = (base_acc - curr_acc)
                ppl_rise = (curr_ppl - base_ppl) / base_ppl # normalized rise
                gaps.append(acc_drop - ppl_rise)
                
            plt.plot(gens, gaps, marker='^', label=regime)
            
        plt.axhline(0, color='black', linewidth=1)
        plt.title("Early Warning Gap (Accuracy Drop - Normalized PPL Rise)")
        plt.xlabel("Generation")
        plt.ylabel("Gap (Positive = Predicted Signature)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "early_warning_gap.png"))
        plt.close()
        
        print(f"Plots saved to {output_dir}")

    @staticmethod
    def plot_prism_trajectory(
        regime_results: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
    ) -> None:
        """Plot PRISM spectral entropy and effective dimension trajectories.

        Skipped silently if no regime has PRISM data (spectral_entropy key absent).

        Args:
            regime_results: Same format as plot_regime_trajectories — dict of
                            regime -> list of per-generation metric dicts.
                            Dicts should contain "spectral_entropy",
                            "effective_dimension", and optionally "viability_score"
                            and "phase_coherence".
            output_dir:     Directory to write PNG files.
        """
        # Check any PRISM data exists
        has_prism = any(
            any("spectral_entropy" in m for m in metrics)
            for metrics in regime_results.values()
        )
        if not has_prism:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Spectral Entropy Trajectory
        plt.figure(figsize=(10, 6))
        for regime, metrics in regime_results.items():
            se_data = [(m["generation"], m["spectral_entropy"])
                       for m in metrics if "spectral_entropy" in m]
            if se_data:
                gens, ses = zip(*se_data)
                plt.plot(gens, ses, marker="o", label=regime)
        plt.title("PRISM: Spectral Entropy over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Mean Spectral Entropy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "prism_spectral_entropy.png"))
        plt.close()

        # 2. Effective Dimension Trajectory
        plt.figure(figsize=(10, 6))
        for regime, metrics in regime_results.items():
            ed_data = [(m["generation"], m["effective_dimension"])
                       for m in metrics if "effective_dimension" in m]
            if ed_data:
                gens, eds = zip(*ed_data)
                plt.plot(gens, eds, marker="s", linestyle="--", label=regime)
        plt.title("PRISM: Effective Dimension over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Effective Dimension")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "prism_effective_dimension.png"))
        plt.close()

        # 3. Combined PRISM health panel (entropy + dim + viability + quant hostility)
        has_hostility = any(
            any("quantization_hostility" in m for m in metrics)
            for metrics in regime_results.values()
        )
        n_subplots = 4 if has_hostility else 3
        fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots), sharex=True)

        for regime, metrics in regime_results.items():
            prism_metrics = [m for m in metrics if "spectral_entropy" in m]
            gens = [m["generation"] for m in prism_metrics]
            ses = [m.get("spectral_entropy", 0.0) for m in prism_metrics]
            eds = [m.get("effective_dimension", 0.0) for m in prism_metrics]
            vss = [m.get("viability_score", 0.0) for m in prism_metrics]

            if gens:
                axes[0].plot(gens, ses, marker="o", label=regime)
                axes[1].plot(gens, eds, marker="s", linestyle="--", label=regime)
                axes[2].plot(gens, vss, marker="^", linestyle=":", label=regime)

                if has_hostility:
                    qhs = [m.get("quantization_hostility", 0.0) for m in prism_metrics]
                    axes[3].plot(gens, qhs, marker="D", linestyle="-.", label=regime)

        axes[0].set_ylabel("Spectral Entropy")
        axes[0].set_title("PRISM Geometric Health")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Effective Dimension")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel("Viability Score")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        if has_hostility:
            axes[3].axhspan(0.7, 1.0, color="red", alpha=0.08, label="Hostile zone (>0.7)")
            axes[3].axhline(0.7, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
            axes[3].set_ylabel("Quant. Hostility")
            axes[3].set_ylim(0.0, 1.05)
            axes[3].set_xlabel("Generation")
            axes[3].legend(fontsize=8)
            axes[3].grid(True, alpha=0.3)
        else:
            axes[2].set_xlabel("Generation")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prism_health_panel.png"))
        plt.close()

        # 4. Standalone quantization hostility plot (only when data present)
        if has_hostility:
            ResultsPlotter.plot_quantization_hostility(regime_results, output_dir)

        print(f"PRISM plots saved to {output_dir}")

    @staticmethod
    def plot_quantization_hostility(
        regime_results: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
    ) -> None:
        """Standalone quantization-hostility trajectory with hostile-zone shading.

        Plots mean_quantization_hostility per generation. Annotates each data
        point with the worst_layer_idx when available. The red band (0.7–1.0)
        marks the region where low-bit quantization (Q4/Q5) is unreliable.

        Args:
            regime_results: Same format as plot_regime_trajectories.
            output_dir:     Directory to write PNG file.
        """
        has_data = any(
            any("quantization_hostility" in m for m in metrics)
            for metrics in regime_results.values()
        )
        if not has_data:
            return

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Hostile zone shading
        ax.axhspan(0.7, 1.0, color="red", alpha=0.08)
        ax.axhline(0.7, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
                   label="Hostile threshold (0.7)")

        for regime, metrics in regime_results.items():
            pts = [
                (m["generation"], m["quantization_hostility"],
                 m.get("worst_layer_idx", -1))
                for m in metrics if "quantization_hostility" in m
            ]
            if not pts:
                continue
            gens, qhs, worst_layers = zip(*pts)
            line, = ax.plot(gens, qhs, marker="D", linestyle="-.", label=regime)

            # Annotate worst_layer_idx on each point
            for g, qh, wl in zip(gens, qhs, worst_layers):
                if wl >= 0:
                    ax.annotate(
                        f"L{wl}",
                        xy=(g, qh),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=6,
                        color=line.get_color(),
                        alpha=0.75,
                    )

        ax.set_title("PRISM: Quantization Hostility over Generations\n"
                     "(>0.7 = low-bit quant unreliable; annotated: worst layer)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Quantization Hostility [0–1]")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prism_quantization_hostility.png"))
        plt.close()
