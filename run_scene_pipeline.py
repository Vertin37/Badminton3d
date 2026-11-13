"""Latest supported entry point; heavy backends are imported only by workers."""
import argparse,sys
from pathlib import Path
sys.dont_write_bytecode=True

if __name__=='__main__':
    if '--worker' in sys.argv:
        parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True);parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
        from src.scene_pipeline.worker import run
        run(args.worker,args.job)
    else:
        from src.scene_pipeline.orchestrator import main
        main()
