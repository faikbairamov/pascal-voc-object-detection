"""Short device feasibility benchmark; updates only temporary in-memory models."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from voc_detection.model import create_model
from voc_detection.project import load_config, make_loader, seed_everything

checkpoint = torch.load('mlflow_local/checkpoints/1e1ca85634974b3289a6dac12ea3df12_epoch_3.pt',
                        map_location='cpu', weights_only=True)
data, loader = make_loader(load_config(), 'train')
images, targets = next(iter(loader))
results = {}
for name in ['cpu', 'mps']:
    seed_everything(42)
    model = create_model(pretrained=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(name).train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.0001, momentum=.9)
    photo = [x.to(name) for x in images]
    truth = [{k: t[k].to(name) for k in ['boxes','labels']} for t in targets]
    durations = []
    for step in range(8):
        if name == 'mps': torch.mps.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad()
        loss = sum(model(photo, truth).values())
        loss.backward()
        optimizer.step()
        if name == 'mps': torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        print(name, step, float(loss.detach().cpu()), elapsed, flush=True)
        if step >= 3: durations.append(elapsed)
    results[name] = {'batch_size': len(photo), 'timed_steps': len(durations),
                     'mean_step_seconds': sum(durations)/len(durations), 'samples_seconds': durations}
    del optimizer, model, photo, truth, loss
    if name == 'mps': torch.mps.empty_cache()
Path('reports/optimization/training_device.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
