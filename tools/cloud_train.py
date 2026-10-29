"""CUDA continuation for a free Colab runtime; fit/dev data only, checkpoint every epoch."""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from voc_detection.dataset import VOCDetectionDataset, VOC_CLASSES, detection_collate_fn
from voc_detection.metrics import evaluate_category


def score(model, loader, data):
    records=[]
    model.eval()
    with torch.inference_mode():
        for images,targets in loader:
            output=model([image.cuda() for image in images])
            for prediction,target in zip(output,targets):
                records.append({'image_id':data.image_ids[len(records)],'target':target,
                                'prediction':{k:v.cpu() for k,v in prediction.items()}})
    per_class={name:evaluate_category(records,index)['ap'] for index,name in enumerate(VOC_CLASSES,1)}
    return float(np.mean(list(per_class.values()))),per_class


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--lr',type=float,default=.0001)
    args=parser.parse_args()
    assert torch.cuda.is_available(),'Select a GPU runtime.'
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=True
    root=Path(__file__).resolve().parent
    dataset=root/'data/VOC2012'
    train=VOCDetectionDataset(dataset,root/'splits/train.txt',horizontal_flip_prob=.5)
    dev=VOCDetectionDataset(dataset,root/'splits/dev.txt')
    generator=torch.Generator().manual_seed(42)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=2,
                            pin_memory=True,collate_fn=detection_collate_fn,generator=generator)
    dev_loader=DataLoader(dev,batch_size=4,shuffle=False,num_workers=2,pin_memory=True,
                          collate_fn=detection_collate_fn)
    checkpoint=torch.load(root/'source.pt',map_location='cpu',weights_only=True)
    model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,21)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.transform.min_size=(384,);model.transform.max_size=768
    model.rpn._pre_nms_top_n['testing']=300;model.rpn._post_nms_top_n['testing']=300
    model.cuda()
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=args.lr,momentum=.9,weight_decay=.0005)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for group in optimizer.param_groups:group['lr']=args.lr
    scaler=torch.amp.GradScaler('cuda')
    output=root/'cloud_results';output.mkdir(exist_ok=True)
    config={**checkpoint['config'],'device':'cuda','batch_size':args.batch_size,
            'learning_rate':args.lr,'horizontal_flip_prob':.5,'training_min_sizes':[384],
            'training_max_size':768,'amp':True,'shuffle_policy':'restart seed 42 on CUDA',
            'source_checkpoint':checkpoint['mlflow_run_id'],'torch_version':torch.__version__,
            'torchvision_version':__import__('torchvision').__version__,'python_version':platform.python_version(),
            'gpu':torch.cuda.get_device_name(0),'evaluation_min_size':384,'evaluation_max_size':768,
            'evaluation_rpn_proposals':300}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('GPU',config['gpu'],'torch',torch.__version__,'train',len(train),'dev',len(dev),flush=True)
    history=[]
    for epoch in range(checkpoint['epoch']+1,checkpoint['epoch']+1+args.epochs):
        model.train();weighted=0.;seen=0;start=time.perf_counter()
        for batch,(images,targets) in enumerate(train_loader,1):
            images=[x.cuda(non_blocking=True) for x in images]
            targets=[{k:t[k].cuda(non_blocking=True) for k in ['boxes','labels']} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                loss=sum(model(images,targets).values())
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            weighted+=loss.detach().item()*len(images);seen+=len(images)
            if batch==1 or batch%100==0:print('Epoch',epoch,'batch',batch,'/',len(train_loader),'loss',loss.item(),flush=True)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-start
        state={'epoch':epoch,'model_state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
               'optimizer_state_dict':optimizer.state_dict(),'config':config,'mlflow_run_id':f'colab_t4_epoch_{epoch}',
               'torch_rng_state':torch.get_rng_state(),'loader_rng_state':generator.get_state(),
               'cuda_rng_state':torch.cuda.get_rng_state_all(),'scaler_state_dict':scaler.state_dict(),
               'inference_config':{'min_size':384,'max_size':768,'rpn_proposals':300}}
        torch.save(state,output/f'epoch_{epoch}.pt')
        map50,per_class=score(model,dev_loader,dev)
        record={'epoch':epoch,'train_loss':weighted/seen,'train_seconds':train_seconds,
                'dev_map50':map50,'per_category':per_class}
        history.append(record)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('COMPLETE',json.dumps(record),flush=True)
        # Package every completed result so retrieval never depends on the last epoch finishing.
        import zipfile
        with zipfile.ZipFile(output/'results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in output.iterdir():
                if path.suffix in {'.pt','.json'}:archive.write(path,path.name)


if __name__=='__main__':main()
