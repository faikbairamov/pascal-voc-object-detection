"""Import retrieved Colab history and its best full checkpoint into local MLflow."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mlflow
import torch
from voc_detection.project import PROJECT_ROOT
from voc_detection.tracking import configure_tracking


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    history=json.loads((args.directory/'history.json').read_text())
    config=json.loads((args.directory/'config.json').read_text())
    best=max(history,key=lambda r:r['dev_map50'])
    path=args.directory/f"epoch_{best['epoch']}.pt"
    if not path.exists():raise SystemExit(f'Retrieve the best checkpoint first: {path}')
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    # This is our own retrieved checkpoint. Older Torch may pickle its version as
    # TorchVersion; allow only that known string subclass, then replace config
    # with the plain JSON provenance before re-exporting.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    checkpoint['config']=config
    configure_tracking()
    with mlflow.start_run(run_name='colab-t4-384-amp') as run:
        mlflow.log_params(config)
        mlflow.set_tag('evaluation_device','cuda')
        mlflow.set_tag('retrieved_checkpoint_sha256',digest)
        mlflow.set_tag('source_checkpoint',str(config['source_checkpoint']))
        for record in history:
            epoch=record['epoch']
            mlflow.log_metrics({'train_loss':record['train_loss'],'epoch_seconds':record['train_seconds'],
                                'dev_map50':record['dev_map50']},step=epoch)
            mlflow.log_metrics({f'dev_ap50_{name}':value for name,value in record['per_category'].items()},step=epoch)
        mlflow.log_artifact(str(args.directory/'history.json'),artifact_path='cloud')
        mlflow.log_artifact(str(args.directory/'config.json'),artifact_path='cloud')
        checkpoint['cloud_checkpoint_id']=checkpoint['mlflow_run_id']
        checkpoint['mlflow_run_id']=run.info.run_id
        checkpoint['retrieved_sha256']=digest
        destination=PROJECT_ROOT/'mlflow_local/checkpoints'/f"{run.info.run_id}_epoch_{best['epoch']}.pt"
        torch.save(checkpoint,destination)
        mlflow.log_artifact(str(destination),artifact_path='checkpoints')
        print(json.dumps({'checkpoint':str(destination),'epoch':best['epoch'],
                          'cloud_dev_map50':best['dev_map50'],'source_sha256':digest},indent=2))


if __name__=='__main__':main()
