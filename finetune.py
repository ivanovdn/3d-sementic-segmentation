import numpy as np
import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from shapely import MultiPoint, MultiPolygon, Polygon
from simplification.cutil import simplify_coords_vw
from tqdm import tqdm

from detect_floor import *
from detect_walls import *
from get_boundary import *
from get_dimentions import *


def finetune_params_with_ransac(
    vertical_points,
    floor_height,
    ceiling_height,
    distance_threshold,
    cluster_eps,
    concave_ratio,
    simplification_method,
    tolerance,
):
    try:
        walls = detect_walls_ransac_clustering(
            vertical_points,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            distance_threshold=distance_threshold,
            cluster_eps=cluster_eps,
            min_cluster_points=50,
            min_wall_height=1.8,
            min_wall_length=0.10,
            max_wall_thickness=0.25,
            min_height_ratio=0.60,
            max_iterations=30,
        )
        wall_points = get_room_wall_points(walls)
        wall_points_2d = project_points_to_2d(wall_points)
        median_thickness = compute_median_wall_thickness(walls)

        boundary_polygon = extract_room_boundary_from_walls(
            wall_points_2d, method="concave_hull", concave_ratio=concave_ratio
        )

        # Check if boundary is valid
        if boundary_polygon is None or boundary_polygon.is_empty:
            return np.nan, np.nan

        offset_boundary = offset_boundary_inward(
            boundary_polygon, offset_distance=median_thickness / 2
        )

        # Handle MultiPolygon case - take the largest polygon
        if isinstance(offset_boundary, MultiPolygon):
            # If we get multiple polygons, take the largest one by area
            offset_boundary = max(offset_boundary.geoms, key=lambda p: p.area)

        # Check if the result is valid
        if offset_boundary.is_empty or not isinstance(offset_boundary, Polygon):
            return np.nan, np.nan

        if simplification_method == "dp":
            simplified_boundary = offset_boundary.simplify(
                tolerance, preserve_topology=True
            )
        else:
            simplified_points = simplify_coords_vw(
                np.array(offset_boundary.exterior.coords), tolerance
            )

            # Check if we have enough points to form a polygon (at least 4 including closing point)
            if len(simplified_points) < 4:
                return np.nan, np.nan

            simplified_boundary = Polygon(simplified_points)

        # Check if simplified boundary is valid and has enough points
        if simplified_boundary.is_empty or len(simplified_boundary.exterior.coords) < 4:
            return np.nan

        return simplified_boundary.length, simplified_boundary.area

    except (ValueError, Exception) as e:
        # Catch any geometry errors and return NaN
        return np.nan, np.nan


def finetune_params(
    wall_points_2d,
    method,
    median_thickness,
    # median_thickness_ratio,
    concave_ratio,
    simplification_method,
    tolerance,
):
    try:

        boundary_polygon = extract_room_boundary_from_walls(
            wall_points_2d, method=method, concave_ratio=concave_ratio
        )

        # Check if boundary is valid
        if boundary_polygon is None or boundary_polygon.is_empty:
            return np.nan, np.nan

        offset_boundary = offset_boundary_inward(
            boundary_polygon, offset_distance=median_thickness
        )

        # Handle MultiPolygon case - take the largest polygon
        if isinstance(offset_boundary, MultiPolygon):
            # If we get multiple polygons, take the largest one by area
            offset_boundary = max(offset_boundary.geoms, key=lambda p: p.area)

        # Check if the result is valid
        if offset_boundary.is_empty or not isinstance(offset_boundary, Polygon):
            return np.nan, np.nan

        if simplification_method == "dp":
            simplified_boundary = offset_boundary.simplify(
                tolerance, preserve_topology=True
            )
        else:
            simplified_points = simplify_coords_vw(
                np.array(offset_boundary.exterior.coords), tolerance
            )

            # Check if we have enough points to form a polygon (at least 4 including closing point)
            if len(simplified_points) < 4:
                return np.nan, np.nan

            simplified_boundary = Polygon(simplified_points)

        # Check if simplified boundary is valid and has enough points
        if simplified_boundary.is_empty or len(simplified_boundary.exterior.coords) < 4:
            return np.nan

        return simplified_boundary.length, simplified_boundary.area

    except (ValueError, Exception) as e:
        # Catch any geometry errors and return NaN
        return np.nan, np.nan


def run_grid(
    wall_points_2d, method, median_thickness, grid, optimize, ground_truth, split=0.7
):
    """
    Grid search with proper error calculation

    Parameters:
    -----------
    grid : list of dict
        Parameter combinations
    optimize : str
        'area', 'perimeter', or 'both'
    ground_truth : dict
        {'perimeter': float, 'area': float}
    split : float
        Weight for perimeter when optimize='both'
        Default 0.7 = 70% perimeter, 30% area

    Returns:
    --------
    results : dict
        Best configuration and all results
    """

    best_score = np.inf
    best_pred_perimeter = 0.0
    best_pred_area = 0.0
    best_params = {}
    all_results = []

    print(f"Optimization: {optimize}")
    if optimize == "both":
        print(f"Weights: Perimeter={split:.1%}, Area={1-split:.1%}")

    for params in tqdm(grid):
        # Extract boundary with current params
        pred_perimeter, pred_area = finetune_params(
            wall_points_2d=wall_points_2d,
            method=method,
            median_thickness=median_thickness,
            median_thickness_ratio=params["median_thickness_ratio"],
            concave_ratio=params["concave_ratio"],
            simplification_method=params["simplification"][0],
            tolerance=params["simplification"][1],
        )

        # Skip invalid results
        if np.isnan(pred_perimeter) or np.isnan(pred_area):
            # print(f"[{i}/{len(grid)}] SKIPPED (invalid result)")
            continue

        # Calculate percentage errors for each metric
        perimeter_error_pct = (
            abs(pred_perimeter - ground_truth["perimeter"])
            / ground_truth["perimeter"]
            * 100
        )
        area_error_pct = (
            abs(pred_area - ground_truth["area"]) / ground_truth["area"] * 100
        )

        # Calculate score based on optimization target
        if optimize == "area":
            score = area_error_pct

        elif optimize == "perimeter":
            score = perimeter_error_pct

        elif optimize == "both":
            # Weighted combination of percentage errors
            score = split * perimeter_error_pct + (1 - split) * area_error_pct

        else:
            raise ValueError(f"Unknown optimize: {optimize}")

        # Store result
        result = {
            "params": params,
            "pred_perimeter": pred_perimeter,
            "pred_area": pred_area,
            "perimeter_error_pct": perimeter_error_pct,
            "area_error_pct": area_error_pct,
            "combined_error": score,
        }
        all_results.append(result)

        # Print progress
        simp_str = f"{params['simplification'][0]}"
        if params["simplification"][0] != "none":
            simp_str += f"({params['simplification'][1]})"

        # Update best
        if score < best_score:
            best_score = score
            best_pred_perimeter = pred_perimeter
            best_pred_area = pred_area
            best_params = params

    best_perim_error = (
        abs(best_pred_perimeter - ground_truth["perimeter"])
        / ground_truth["perimeter"]
        * 100
    )
    best_area_error = (
        abs(best_pred_area - ground_truth["area"]) / ground_truth["area"] * 100
    )

    return {
        "best_score": best_score,
        "best_pred_perimeter": best_pred_perimeter,
        "best_pred_area": best_pred_area,
        "best_params": best_params,
        # "all_results": all_results,
    }


# ========================================================================
# OPTUNA


class EarlyStoppingCallback:
    """
    Early stopping based on RELATIVE improvement

    Stops if (best_old - best_new) / best_old < min_relative_improvement
    """

    def __init__(self, patience=100, min_relative_improvement=0.01):
        """
        Parameters:
        -----------
        patience : int
            Number of trials to wait
        min_relative_improvement : float
            Minimum relative improvement (e.g., 0.01 = 1%)
        """
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.best_value = float("inf")
        self.trials_without_improvement = 0

    def __call__(self, study, trial):
        current_best = study.best_value

        # Calculate relative improvement
        if self.best_value < float("inf"):
            relative_improvement = (self.best_value - current_best) / self.best_value
        else:
            relative_improvement = 1.0

        if relative_improvement > self.min_relative_improvement:
            # Significant relative improvement
            print(
                f"\n  Trial {trial.number}: Improved by {relative_improvement*100:.2f}% "
                f"({self.best_value:.4f} → {current_best:.4f})"
            )
            self.best_value = current_best
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        if self.trials_without_improvement >= self.patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING")
            print(f"{'='*70}")
            print(
                f"No relative improvement > {self.min_relative_improvement*100:.1f}% "
                f"for {self.patience} trials"
            )
            print(f"Best value: {self.best_value:.4f}%")
            study.stop()


def create_objective_function(
    wall_points_2d, method, median_thickness, ground_truth, optimize="both", split=0.7
):
    """
    Create Optuna objective function

    This is a closure that captures your data and returns
    the objective function that Optuna will optimize
    """

    def objective(trial):
        """
        Optuna objective function

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object that suggests parameters

        Returns:
        --------
        score : float
            Error to minimize (lower is better)
        """

        concave_ratio = trial.suggest_float("concave_ratio", 0.05, 0.70)

        # Suggest simplification method
        simplification_method = trial.suggest_categorical(
            "simplification_method", ["vw", "dp"]
        )

        if simplification_method == "vw":
            # Log scale for VW (better for exploring wide range)
            tolerance = trial.suggest_float("tolerance", 0.0001, 0.5, log=True)
        else:  # dp
            tolerance = trial.suggest_float("tolerance", 0.01, 0.4, log=True)

        # Run your evaluation function
        try:
            pred_perimeter, pred_area = finetune_params(
                wall_points_2d=wall_points_2d,
                method=method,
                median_thickness=median_thickness,
                concave_ratio=concave_ratio,
                simplification_method=simplification_method,
                tolerance=tolerance,
            )

            # Handle invalid results
            if np.isnan(pred_perimeter) or np.isnan(pred_area):
                # Return high penalty for invalid configs
                return 1e6

            # Calculate errors
            perimeter_error_pct = (
                abs(pred_perimeter - ground_truth["perimeter"])
                / ground_truth["perimeter"]
                * 100
            )
            area_error_pct = (
                abs(pred_area - ground_truth["area"]) / ground_truth["area"] * 100
            )

            # Calculate score based on optimization target
            if optimize == "area":
                score = area_error_pct
            elif optimize == "perimeter":
                score = perimeter_error_pct
            elif optimize == "both":
                score = split * perimeter_error_pct + (1 - split) * area_error_pct
            else:
                raise ValueError(f"Unknown optimize: {optimize}")

            # Store additional info for later analysis
            trial.set_user_attr("pred_perimeter", pred_perimeter)
            trial.set_user_attr("pred_area", pred_area)
            trial.set_user_attr("perimeter_error_pct", perimeter_error_pct)
            trial.set_user_attr("area_error_pct", area_error_pct)

            return score

        except Exception as e:
            # Return high penalty for failed trials
            print(f"Trial failed: {e}")
            return 1e6

    return objective


def run_optuna_optimization(
    wall_points_2d,
    method,
    median_thickness,
    ground_truth,
    optimize="both",
    split=0.7,
    n_trials=100,
    timeout=None,
    study_name="boundary_optimization",
):
    """
    Run Bayesian optimization with Optuna

    Parameters:
    -----------
    wall_points_2d : np.ndarray
        2D wall points
    median_thickness : float
        Wall thickness
    ground_truth : dict
        {'perimeter': float, 'area': float}
    optimize : str
        'area', 'perimeter', or 'both'
    split : float
        Weight for perimeter when optimize='both'
    n_trials : int
        Number of trials to run (default 100)
        More trials = better optimization but slower
    timeout : float or None
        Maximum time in seconds (None = no limit)
    study_name : str
        Name for the study (for saving/loading)

    Returns:
    --------
    study : optuna.Study
        Completed study object with all results
    best_params : dict
        Best parameters found
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n{'='*70}")
    print(f"BAYESIAN OPTIMIZATION WITH OPTUNA")
    print(f"{'='*70}")
    print(f"Optimization: {optimize}")
    if optimize == "both":
        print(f"Weights: Perimeter={split:.1%}, Area={1-split:.1%}")
    print(
        f"Ground truth: Perimeter={ground_truth['perimeter']:.2f}m, Area={ground_truth['area']:.2f}m²"
    )
    print(f"Number of trials: {n_trials}")
    if timeout:
        print(f"Timeout: {timeout}s")

    # Create objective function
    objective = create_objective_function(
        wall_points_2d, method, median_thickness, ground_truth, optimize, split
    )

    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(),
    )

    # Run optimization with progress bar
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )

    # Print results
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(
        f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )

    # Best trial
    best_trial = study.best_trial

    best_params = {
        "concave_ratio": best_trial.params["concave_ratio"],
        "simplification": (
            best_trial.params["simplification_method"],
            best_trial.params.get("tolerance", 0.0),
        ),
    }

    return study, best_params


def run_optuna_with_early_stopping(
    wall_points_2d,
    method,
    median_thickness,
    ground_truth,
    optimize="both",
    split=0.7,
    n_trials=300,
    patience=100,
    min_relative_improvement=0.01,
):
    """
    Run Optuna with early stopping
    """

    import optuna
    from tqdm import tqdm

    # Silence Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n{'='*70}")
    print(f"BAYESIAN OPTIMIZATION WITH EARLY STOPPING")
    print(f"{'='*70}")
    print(f"Max trials: {n_trials}")
    print(
        f"Early stopping: patience={patience}, min_relative_improvement={min_relative_improvement}"
    )
    print(
        f"Ground truth: Perimeter={ground_truth['perimeter']:.2f}m, Area={ground_truth['area']:.2f}m²\n"
    )

    # Create objective
    objective = create_objective_function(
        wall_points_2d, median_thickness, ground_truth, optimize, split
    )

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )

    # Create early stopping callback
    early_stopping = EarlyStoppingCallback(
        patience=patience, min_relative_improvement=min_relative_improvement
    )

    # Progress bar
    with tqdm(total=n_trials, desc="Optimizing", unit="trial") as pbar:

        def progress_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix(
                {
                    "best": f"{study.best_value:.2f}%",
                    "current": f"{trial.value:.2f}%" if trial.value < 1e5 else "failed",
                    "no_improve": early_stopping.trials_without_improvement,
                }
            )

        # Optimize with both callbacks
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[early_stopping, progress_callback],
            show_progress_bar=False,
        )

    # Print results
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best error: {study.best_value:.2f}%")

    if early_stopping.trials_without_improvement >= patience:
        print(f"Stopped early after {len(study.trials)} trials")
    else:
        print(f"Completed all {n_trials} trials")

    # Extract best params
    best_trial = study.best_trial
    best_params = {
        "median_thickness_ratio": best_trial.params["median_thickness_ratio"],
        "concave_ratio": best_trial.params["concave_ratio"],
        "simplification": (
            best_trial.params["simplification_method"],
            best_trial.params.get("tolerance", 0.0),
        ),
    }

    return study, best_params


def visualize_optuna_results(study):
    """
    Visualize Optuna optimization results
    """

    import matplotlib.pyplot as plt

    print(f"\nGenerating visualizations...")

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Optimization history
    ax1 = plt.subplot(2, 3, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
    ax1.set_title("Optimization History", fontweight="bold")

    # Plot 2: Parameter importances
    ax2 = plt.subplot(2, 3, 2)
    try:
        optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
        ax2.set_title("Parameter Importances", fontweight="bold")
    except:
        ax2.text(0.5, 0.5, "Not enough trials", ha="center", va="center")
        ax2.set_title("Parameter Importances", fontweight="bold")

    # Plot 3: Parallel coordinate
    ax3 = plt.subplot(2, 3, 3)
    try:
        optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=ax3)
        ax3.set_title("Parallel Coordinate Plot", fontweight="bold")
    except:
        ax3.text(0.5, 0.5, "Cannot plot", ha="center", va="center")
        ax3.set_title("Parallel Coordinate Plot", fontweight="bold")

    # Plot 4: Error over trials
    ax4 = plt.subplot(2, 3, 4)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    errors = [t.value for t in trials]
    ax4.plot(errors, "b-", alpha=0.6, linewidth=1)
    ax4.plot(np.minimum.accumulate(errors), "r-", linewidth=2, label="Best so far")
    ax4.set_xlabel("Trial", fontweight="bold")
    ax4.set_ylabel("Error (%)", fontweight="bold")
    ax4.set_title("Error Progress", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Parameter distribution (concave_ratio)
    ax5 = plt.subplot(2, 3, 5)
    alphas = [t.params["concave_ratio"] for t in trials]
    errors_by_alpha = [t.value for t in trials]
    scatter = ax5.scatter(
        alphas, errors_by_alpha, c=errors_by_alpha, s=50, alpha=0.6, cmap="RdYlGn_r"
    )
    ax5.set_xlabel("Concave Ratio", fontweight="bold")
    ax5.set_ylabel("Error (%)", fontweight="bold")
    ax5.set_title("Error vs Concave Ratio", fontweight="bold")
    plt.colorbar(scatter, ax=ax5)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Perimeter vs Area error
    ax6 = plt.subplot(2, 3, 6)
    perim_errors = [t.user_attrs.get("perimeter_error_pct", np.nan) for t in trials]
    area_errors = [t.user_attrs.get("area_error_pct", np.nan) for t in trials]
    valid_mask = ~(np.isnan(perim_errors) | np.isnan(area_errors))

    if np.any(valid_mask):
        scatter = ax6.scatter(
            np.array(perim_errors)[valid_mask],
            np.array(area_errors)[valid_mask],
            c=np.array(errors)[valid_mask],
            s=50,
            alpha=0.6,
            cmap="RdYlGn_r",
        )

        # Mark best
        best = study.best_trial
        ax6.scatter(
            best.user_attrs["perimeter_error_pct"],
            best.user_attrs["area_error_pct"],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="Best",
            zorder=10,
        )

        ax6.set_xlabel("Perimeter Error (%)", fontweight="bold")
        ax6.set_ylabel("Area Error (%)", fontweight="bold")
        ax6.set_title("Perimeter vs Area Error", fontweight="bold")
        plt.colorbar(scatter, ax=ax6, label="Combined Error (%)")
        ax6.legend()

    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optuna_optimization_results.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved: optuna_optimization_results.png")
    plt.show()


def save_optuna_results(study, filename="optuna_results.csv"):
    """
    Save Optuna results to CSV
    """

    import pandas as pd

    # Extract trial data
    data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                "trial_number": trial.number,
                "value": trial.value,
                **trial.params,
                **trial.user_attrs,
            }
            data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values("value")

    df.to_csv(filename, index=False)
    print(f"✓ Saved {len(df)} trials to: {filename}")

    return df
