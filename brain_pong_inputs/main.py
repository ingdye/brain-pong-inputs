from __future__ import annotations

import os
import yaml
from pathlib import Path
import argparse

from psychopy import visual, core

from brain_pong_inputs.block import TrialBlock
from brain_pong_inputs.utils import load_defaults_params, get_game_params, load_defaults_levels

WIN_SIZE = [1200, 800]


def handle_game_config(config: str | Path | None):
    params = load_defaults_params()
    if config is None:
        return params

    with open(config, "r") as f:
        custom = yaml.safe_load(f)

    params.update(custom)
    return params


def handle_levels(levels: str | Path | None):
    level_input = load_defaults_levels()
    if not levels:
        return level_input

    with open(levels, "r") as f:
        level_configs = yaml.safe_load(f)
    
    level_input.update(level_configs)
    return level_input


def run_experiment(
    save_dir: str | Path,
    input_method: str,
    n_blocks: int,
    n_trials: int,
    levels: str | Path | None = None,
    config: str | Path | None = None,
    overwrite: bool = False,
):
    """Top-level function that runs the pong task in its entirety

    Args:
        save_dir: Output directory for experiment data.
        n_blocks: Number of trial blocks to run.
        n_trials: Number of trials per block.
        input_method: input method for response
        levels: Path to YAML file with level configurations.
        config: Path to YAML file with runtime parameter overrides.
        overwrite: Allow overwriting existing output directory.
    """
    blocks = range(n_blocks)

    params = handle_game_config(config)
    levels = handle_levels(levels)

    os.makedirs(save_dir, exist_ok=True)

    game_params = get_game_params(params)
    game_params["input_method"] = input_method   # ← 여기!

    win = visual.Window(
        size=params["win_size"],
        fullscr=False,
        winType="pyglet",
        color="black",
        units="pix",
    )

    current_level = params["init_level"]
    total_points = params["init_points"]
    for b in blocks:
        trial_block = TrialBlock(
            b,
            n_trials,
            save_dir,
            levels,
            current_level,
            total_points,
            params["block_pre_delay"],
            params["performance_criteria"],
            win=win,
            game_params=game_params,
            use_aggressive_leveling=params["use_aggressive_leveling"],
        )
        completed = trial_block.run()

        if not completed:
            print("Experiment ended early by user")
            break

        total_points = trial_block.total_points
        current_level = trial_block.current_level

    win.close()
    core.quit()
    print("DONE!")


def handler(args: argparse.Namespace) -> int:

    params = load_defaults_params()
    game_params = get_game_params(params)

    game_params["input_method"] = args.input_method

    run_experiment(
        save_dir=args.save_dir,
        input_method=args.input_method,
        n_blocks=args.blocks,
        n_trials=args.trials,
        levels=args.levels,
        config=args.config,
        overwrite=args.overwrite,
    )
    return 0


def create_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:

    #parser = argparse.ArgumentParser(...)
    #subparsers = parser.add_subparsers()
    
    parser = parser or argparse.ArgumentParser()
    #positional = _parser.add_argument_group("positional arguments")
    #required = _parser.add_argument_group("required parameters")
    #optional = _parser.add_argument_group("optional parameters")

    #run = subparsers.add_parser("run", help="run experiment")
   
    parser.add_argument(
        "save_dir", type=Path, metavar="OUT_DIR", help="Output directory"
    )

    parser.add_argument(
        "--input_method", type=str, required = True, help = "mouse or press or wheel or brain"
    )

    parser.add_argument(
        "--blocks", "-b", type=int, required=True, help="Set the number of trial blocks"
    )

    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        required=True,
        help="Set the number of trials per block",
    )

    parser.add_argument(
        "--levels",
        "-l",
        type=str,
        help="Path to a YAML file containing level configurations. If not "
        "provided, then the experiments a single constant level across trials. "
        "Will overwrite paddle_size and points_increment if used.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to a YAML file containing runtime parameter configurations "
        "Any key in the file overwrites default configuration. If not provided, "
        "then default configuration is used",
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Allow output overwriting"
    )

    parser.set_defaults(handler=handler)
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    if hasattr(args, "handler"):
        return args.handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    main()
