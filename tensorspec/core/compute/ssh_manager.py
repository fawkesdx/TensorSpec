import paramiko
import subprocess
import os

class ConnectionManager:
    def __init__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.sftp = None
        self.port_forward_process = None

    def connect(self, host, user, password=None, key=None):
        connect_kwargs = {
            "hostname": host,
            "username": user
        }
        if password:
            connect_kwargs["password"] = password
        if key:
            connect_kwargs["key_filename"] = key
            
        self.ssh.connect(**connect_kwargs)
        self.sftp = self.ssh.open_sftp()

    def sftp_put(self, local, remote):
        if not self.sftp:
            raise Exception("Not connected")
        self.sftp.put(local, remote)

    def sftp_get(self, remote, local):
        if not self.sftp:
            raise Exception("Not connected")
        self.sftp.get(remote, local)

    def execute_command(self, cmd):
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')

    def forward_port(self, local_port, remote_port, host, user, key=None):
        cmd = ["ssh", "-N", "-L", f"{local_port}:localhost:{remote_port}", f"{user}@{host}"]
        if key:
            cmd.extend(["-i", key])
        self.port_forward_process = subprocess.Popen(cmd)

    def close(self):
        if self.sftp:
            self.sftp.close()
        self.ssh.close()
        if self.port_forward_process:
            self.port_forward_process.terminate()
