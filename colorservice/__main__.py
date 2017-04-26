# Hacky way to import python-core
import os
import sys
import SalNet
path = os.path.abspath(".")
sys.path.insert(0, path)
from pythoncore import Constants, WorkerService


def handle_task(task_input, task_token):
    ep_id = task_input["epId"]
    sn = SalNet.SalNet(ep_id)
    sn.task_token = task_token
    sn.run()

if __name__ == '__main__':
    thisTask = Constants.TASK_ARNS['CV_GET_SALIENCY_MASK']

    WorkerService.start(thisTask, handle_task)



