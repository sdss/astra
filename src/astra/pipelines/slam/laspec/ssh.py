import paramiko

ssh = paramiko.SSHClient()
known_host = paramiko.AutoAddPolicy()
ssh.set_missing_host_key_policy(known_host)
ssh.connect(hostname="10.0.9.114", username="cham")


scp = paramiko.SFTPClient()
