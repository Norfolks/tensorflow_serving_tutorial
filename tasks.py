import os
import subprocess

from invoke import task



@task
def run(ctx):
    subprocess.Popen(['/bin/bash', '-c', 'tensorflow_model_server --port=9000 --model_name=resnet50 --model_base_path=model' ])
    ctx.run('python rest_service.py')
    

