# Hacky way to import python-core
import ColorCatcher
from pythoncore import Constants, WorkerService


def handle_task(task_input, task_token):
    ep_id = task_input["epId"]
    hit_id = task_input["hitId"]
    cc = ColorCatcher.ColorCatcher(ep_id, hit_id, task_token)
    cc.run()

if __name__ == '__main__':
    # handle_task({"epId": 356, "hitId": 792}, "asdf")
    thisTask = Constants.TASK_ARNS['CV_GET_COLORS']

    WorkerService.start((thisTask, handle_task, 8))
