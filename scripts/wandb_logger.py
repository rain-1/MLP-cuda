#!/usr/bin/env python3
"""
Simple wandb logging interface for C++ training scripts.
Reads metrics from a JSON file and logs to wandb.
"""

import wandb
import json
import sys
import time
from pathlib import Path

class WandbLogger:
    def __init__(self, project_name="transformer-training", run_name=None, config=None):
        """Initialize wandb run."""
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config or {}
        )

    def log(self, metrics, step=None):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)

    def log_text(self, key, text, step=None):
        """Log text samples to wandb."""
        wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self):
        """Finish wandb run."""
        wandb.finish()


def watch_file_and_log(metrics_file, project_name="transformer-training", run_name=None, config_file=None):
    """
    Watch a metrics file and continuously log to wandb.

    Expected JSON format:
    {
        "step": 123,
        "metrics": {"loss": 1.23, "lr": 0.001},
        "samples": {"prompt": "once upon", "output": "once upon a time..."}
    }

    Special commands:
    {"command": "init", "config": {...}}  - Initialize wandb
    {"command": "finish"}                  - Finish wandb run
    """

    # Load config if provided
    config = {}
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    logger = WandbLogger(project_name, run_name, config)
    print(f"Wandb logger initialized: {project_name}/{run_name or 'auto'}")

    metrics_path = Path(metrics_file)
    last_size = 0

    try:
        while True:
            if metrics_path.exists():
                current_size = metrics_path.stat().st_size

                # Only read if file has grown
                if current_size > last_size:
                    with open(metrics_file, 'r') as f:
                        f.seek(last_size)
                        new_data = f.read()
                        last_size = current_size

                        # Process each line as a JSON object
                        for line in new_data.strip().split('\n'):
                            if not line.strip():
                                continue

                            try:
                                entry = json.loads(line)

                                # Handle commands
                                if 'command' in entry:
                                    if entry['command'] == 'finish':
                                        logger.finish()
                                        print("Wandb run finished")
                                        return
                                    elif entry['command'] == 'init':
                                        # Reinitialize with new config
                                        if 'config' in entry:
                                            wandb.config.update(entry['config'])

                                # Handle metrics
                                if 'metrics' in entry:
                                    step = entry.get('step', None)
                                    logger.log(entry['metrics'], step=step)

                                # Handle text samples
                                if 'samples' in entry:
                                    step = entry.get('step', None)
                                    for key, value in entry['samples'].items():
                                        logger.log_text(key, value, step=step)

                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse JSON line: {e}")
                                continue

            time.sleep(0.5)  # Check every 0.5 seconds

    except KeyboardInterrupt:
        print("\nStopping wandb logger...")
        logger.finish()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wandb_logger.py <metrics_file> [project_name] [run_name] [config_file]")
        sys.exit(1)

    metrics_file = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else "transformer-training"
    run_name = sys.argv[3] if len(sys.argv) > 3 else None
    config_file = sys.argv[4] if len(sys.argv) > 4 else None

    watch_file_and_log(metrics_file, project_name, run_name, config_file)
