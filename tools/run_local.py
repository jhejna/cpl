import os
import subprocess
import time

import utils

"""
A script for launching jobs on a local machine. This is designed to mimic the design as the SLURM launch script.
"""

if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument("--cpus", "-c", type=int, default=None, help="Number of CPUs per job instance")
    parser.add_argument("--gpus", "-g", type=int, default=None, help="Number of GPUs per job instance")
    parser.add_argument("--valid-gpus", type=int, nargs="+", default=None, help="Specifies which GPUS to use.")
    parser.add_argument("--valid-cpus", type=str, nargs="+", default=None)
    parser.add_argument(
        "--use-taskset", action="store_true", default=False, help="Whether or not to CPU load balance with taskset"
    )

    # Add Taskset and GPU arguments
    args = parser.parse_args()
    assert isinstance(args.valid_gpus, list) or args.valid_gpus is None, "Valid GPUs must be a list of ints or None."
    assert isinstance(args.valid_cpus, list) or args.valid_cpus is None, "Valid CPUs must be a list"
    if args.gpus is not None:
        assert (
            isinstance(args.valid_gpus, list) and len(args.valid_gpus) >= args.gpus
        ), "If GPU, must provide valid gpus >= num gpus"

    scripts = utils.get_scripts(args)

    if args.valid_cpus is None:
        args.valid_cpus = ["0-" + str(os.cpu_count())]

    cpu_list = []  # Populate CPU list with a list of all valid CPU cores [1,2,3,4,5,6, ...] etc.
    for cpu_item in args.valid_cpus:
        if isinstance(cpu_item, str) and "-" in cpu_item:
            # We have a CPU range
            cpu_min, cpu_max = cpu_item.split("-")
            cpu_min, cpu_max = int(cpu_min), int(cpu_max)
            cpu_list.extend(list(range(cpu_min, cpu_max)))
        else:
            cpu_list.append(int(cpu_item))
    assert (
        len(cpu_list) >= args.cpus or args.cpus is None
    ), "Must have more valid CPUs than cpus per script, otherwise nothing can launch"

    gpu_list = [] if args.valid_gpus is None else args.valid_gpus

    job_list = []

    try:
        while len(scripts) > 0:
            # Check on existing proceses
            finished_jobs = []
            for i, (processes, job_cpus, job_gpus) in enumerate(finished_jobs):
                if all([process.poll() is not None for process in processes]):
                    cpu_list.extend(job_cpus)
                    gpu_list.extend(job_gpus)
                    finished_jobs.append(i)
            for i in reversed(finished_jobs):
                del job_list[i]

            # Next, check to see if we can launch a process
            have_sufficient_cpus = args.cpus is None or len(cpu_list) >= args.cpus
            have_sufficient_gpus = args.gpus is None or len(gpu_list) >= args.gpus
            if have_sufficient_cpus and have_sufficient_cpus:
                # we have the resources to launch a job, so launch it
                job_cpus = cpu_list[: args.cpus] if args.cpus is not None else []
                job_gpus = gpu_list[: args.gpus] if args.gpus is not None else []
                job_scripts = scripts[: args.scripts_per_job]
                processes = []
                for entry_point, script_args in job_scripts:
                    command_list = []
                    if args.use_taskset:
                        command_list.extend(["taskset", "-c", ",".join(job_cpus)])
                    command_list.extend(["python", entry_point])
                    for arg_name, arg_value in script_args.items():
                        command_list.append("--" + arg_name)
                        command_list.append(str(arg_value))
                    if job_gpus is not None:
                        env = os.environ
                        env["CUDA_VISIBLE_DEVICES"] = ",".join(job_gpus)
                    else:
                        env = None

                    print("[Local Sweeper] launching script on gpu:", job_gpus, "and cpus:", job_cpus)
                    proc = subprocess.Popen(command_list, env=env)
                    processes.append(proc)

                # After all the jobs have launched, updated the set of available resources and remaining scripts
                cpu_list = cpu_list[len(job_cpus) :]
                gpu_list = gpu_list[len(job_gpus) :]
                scripts = scripts[len(job_scripts) :]
                # Append to the set of currently running jobs
                job_list.append((processes, job_cpus, job_gpus))

            else:
                # If we were unable to launch a job, sleep for a while.
                time.sleep(10)

        # We have launched all the scripts, now wait for the remaining ones to complete.
        all_processes = []
        for processes, _, _ in job_list:
            all_processes.extend(processes)
        exit_codes = [p.wait() for p in all_processes]
        print("[Local Sweeper] Completed.")

    except KeyboardInterrupt:
        # If we detect a keyboard interrupt, manually send a kill signal to all subprocesses.
        all_processes = []
        for processes, _, _ in job_list:
            all_processes.extend(processes)

        for p in processes:
            try:
                p.terminate()
            except OSError:
                pass
            p.wait()
