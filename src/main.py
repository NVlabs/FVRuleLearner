#!/opt/conda/bin/python

from config import FLAGS
from saver import saver
from utils import OurTimer, load_replace_flags, get_root_path, slack_notify
import traceback
import sys
from pathlib import Path
import time

sys.path.insert(0, '../')
# sys.path.insert(0, '../../../../fv_eval')

timer = OurTimer()


def main():
    if FLAGS.global_task in ['inference', 'train']:

        if FLAGS.model == 'agent':
            start_time = time.time()
            from fv_harwareagent_example import FVProcessor
            processor = FVProcessor()
            processor.main_fv()
            saver.save_stats('overall_runtime', time.time()-start_time)

        elif FLAGS.model == 'timing_agent':
            from hardware_agent.examples.timing_summary_hardwareagent_example import main
            main()
        else:
            raise NotImplementedError()

        if FLAGS.global_task == 'inference':
            from eval import eval
            eval(Path(saver.logdir))

    elif FLAGS.global_task == "eval":
        from eval import eval
        eval(FLAGS.folder_to_eval)
        
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
