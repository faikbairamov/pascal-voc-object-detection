"""Profile the selected CPU deployment with fixed decoded development tensors."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .inference import DEFAULT_CHECKPOINT, load_detector, sha256
from .project import load_config, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--images',type=int,default=80)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('reports/verification/selected_latency.json'))
    args=parser.parse_args()
    data,_=make_loader(load_config(),'dev')
    if not 1 <= args.images <= len(data) or args.repeats < 1:
        parser.error('Choose positive image and repeat counts within the dev split.')
    indices=np.linspace(0,len(data)-1,args.images,dtype=int)
    images=[data[int(index)][0] for index in indices]
    model,metadata=load_detector(args.checkpoint)
    durations=[]
    with torch.inference_mode():
        for image in images[:5]:model([image])
        for repeat in range(args.repeats):
            for image in images:
                start=time.perf_counter();model([image]);durations.append(1000*(time.perf_counter()-start))
    result={'checkpoint_sha256':sha256(args.checkpoint),'images':args.images,'repeats':args.repeats,
            'batch_size':1,'warmup_images':min(5,args.images),'torch_threads':torch.get_num_threads(),
            'torch':torch.__version__,'platform':platform.platform(),
            'inference_config':{'min_size':model.transform.min_size[0],'max_size':model.transform.max_size,
                                'rpn_proposals':model.rpn._post_nms_top_n['testing']},
            'scope':'CPU model forward; decoded tensors; no service concurrency or drawing/encoding',
            'mean_ms':float(np.mean(durations)),'median_ms':float(np.median(durations)),
            'p95_ms':float(np.percentile(durations,95)),
            'image_ids':[data.image_ids[int(index)] for index in indices],'samples_ms':durations}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in {'samples_ms','image_ids'}})


if __name__=='__main__':main()
