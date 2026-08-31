import requests
import json
from .ssh_manager import ConnectionManager

class JobDispatcher:
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager

    def submit_job(self, config, target_cluster):
        if target_cluster.mode == 'daemon':
            tunnel_port = getattr(target_cluster, 'tunnel_port', 8080)
            url = f"http://localhost:{tunnel_port}/jobs"
            response = requests.post(url, json=config)
            response.raise_for_status()
            return response.json()
        elif target_cluster.mode == 'slurm':
            job_name = config.get("job_name", "tensorspec_job")
            script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err

{config.get('command', 'echo "No command specified"')}
"""
            remote_script_path = f"/tmp/{job_name}.sh"
            
            create_script_cmd = f"cat << 'EOF' > {remote_script_path}\n{script}\nEOF"
            self.connection_manager.execute_command(create_script_cmd)
            stdout, stderr = self.connection_manager.execute_command(f"sbatch {remote_script_path}")
            return {"stdout": stdout, "stderr": stderr}
        else:
            raise ValueError(f"Unknown cluster mode: {target_cluster.mode}")
