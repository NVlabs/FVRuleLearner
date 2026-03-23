#!/opt/conda/bin/python

from gen_plan import gen_plan
from gen_KG_graphRAG import build_KG

from config import FLAGS
from saver import saver
from utils import OurTimer, load_replace_flags, get_root_path, slack_notify
import traceback
import sys

sys.path.insert(0, '../')
# sys.path.insert(0, '../../../../fv_eval')


timer = OurTimer()


def main():
    if FLAGS.task == 'gen_plan':
        gen_plan()
    elif FLAGS.task == 'build_KG':
        build_KG()
    else:
        raise NotImplementedError()


if __name__ == '__main__':

    timer = OurTimer()

    try:
        main()
        status = 'Complete'
    except:
        traceback.print_exc()
        s = '\n'.join(traceback.format_exc(limit=-1).split('\n')[1:])
        saver.log_info(traceback.format_exc(), silent=True)
        saver.save_exception_msg(traceback.format_exc())
        status = 'Error'

    tot_time = timer.time_and_clear()
    saver.log_info(f'Total time: {tot_time}')
    saver.close()
