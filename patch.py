with open("tensorspec/gui/components/arpes_panel.py", "r") as f:
    content = f.read()

content = content.replace(
    "export OPENBLAS_NUM_THREADS=1\n\n/home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python",
    "export OPENBLAS_NUM_THREADS=1\n\n/home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/pip install chinook psutil\n/home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python"
)

content = content.replace(
    "export OPENBLAS_NUM_THREADS=1 && nohup /home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python",
    "export OPENBLAS_NUM_THREADS=1 && /home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/pip install chinook psutil >> sys.out.full 2>&1 && nohup /home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python"
)

with open("tensorspec/gui/components/arpes_panel.py", "w") as f:
    f.write(content)
