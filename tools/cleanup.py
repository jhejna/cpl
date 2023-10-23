# This script cleans up all the temporary files used by the research codebase.

import os
import shutil

if __name__ == "__main__":
    base_path = "/tmp/"

    job_scripts_removed = 0
    replay_buffers_removed = 0
    sweeper_configs_removed = 0

    for name in os.listdir(base_path):
        path = os.path.join(base_path, name)
        try:
            if name.startswith("job_"):
                os.remove(path)
                job_scripts_removed += 1
            elif name.startswith("config_"):
                os.remove(path)
                sweeper_configs_removed += 1
            elif name.startswith("replay_buffer_"):
                shutil.rmtree(path)
                replay_buffers_removed += 1
        except OSError:
            continue

    print("Finished Cleanup.")
    print("Removed", job_scripts_removed, "job scripts.")
    print("Removed", sweeper_configs_removed, "sweeper configs.")
    print("Removed", replay_buffers_removed, "replay buffers.")
