"""BOBCAT retraining for this repo, adapted from the original run script (author code, author defaults, fold 1, seed 221).
Not a sealed file. The checkpoint is not redistributed (it is trained on ASSISTments data). Training may not be bit-identical
across machines; scripts/bobcat/check_checkpoint.py compares the result with the original run.
"""
import os, sys, time, json, hashlib, random, shutil
from pathlib import Path
os.environ.update({k:'6' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')})
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'published_cat_reference_v1'))
from bobcat_support import ROOT, RUN, atomic_json

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    # (public) The original read RUN/freeze.json and checked files on the original machine; here the settings come from
    # reports/receipts/bobcat-run-settings.json and RUN/freeze.json is written from them.
    freeze=json.loads((Path(__file__).resolve().parents[2]/'reports/receipts/bobcat-run-settings.json').read_text())
    RUN.mkdir(parents=True,exist_ok=True)
    if not (RUN/'freeze.json').exists(): (RUN/'freeze.json').write_text(json.dumps(freeze,indent=1,sort_keys=True))
    sys.path.insert(0,str(ROOT/'arghosh-BOBCAT'))
    import train as t
    from utils.configuration import create_parser, initialize_seeds
    from utils.utils import data_split
    from dataset import Dataset, collate_fn
    from model import MAMLModel
    from policy import StraightThrough
    sys.argv=['train.py','--dataset','assist2009','--model','biirt-biased','--fold','1','--seed','221','--n_query','5']
    p=create_parser();torch.set_num_threads(6);initialize_seeds(p.seed)
    assert vars(p)==freeze['settings']
    t.params=p;t.sampling='biased';t.config={};t.num_workers=0
    t.model=MAMLModel(sampling='biased',n_query=p.n_query,n_question=p.n_question,question_dim=1)
    t.meta_params=[torch.Tensor(1,1).normal_(-1.,1.).requires_grad_()]
    t.optimizer=torch.optim.Adam(t.model.parameters(),lr=p.lr,weight_decay=1e-8)
    t.meta_params_optimizer=torch.optim.SGD(t.meta_params,lr=p.meta_lr,weight_decay=2e-6,momentum=.9)
    t.st_policy=StraightThrough(p.n_question,p.n_question,p.policy_lr,(.9,.999))
    receipt=json.loads((ROOT/'dataset-receipt.json').read_text())
    parts=data_split(receipt['file'],p.fold,p.seed)
    t.train_dataset,t.valid_dataset,t.test_dataset=[Dataset(x) for x in parts]
    t.collate_fn=collate_fn(p.n_question)
    t.train_loader=torch.utils.data.DataLoader(t.train_dataset,collate_fn=t.collate_fn,batch_size=p.train_batch_size,num_workers=0,shuffle=True,drop_last=True)
    state_names=['best_epoch','best_val_score','best_val_auc','best_test_score','best_test_auc']
    latest=RUN/'latest.pt';start_epoch=0
    if latest.exists():
        c=torch.load(latest,weights_only=False,map_location='cpu')
        assert c['freeze_sha256']==digest(RUN/'freeze.json')
        t.model.load_state_dict(c['model']);t.st_policy.policy.load_state_dict(c['policy'])
        t.meta_params[0].data.copy_(c['meta'])
        t.optimizer.load_state_dict(c['optimizer']);t.meta_params_optimizer.load_state_dict(c['meta_optimizer']);t.st_policy.optimizer.load_state_dict(c['policy_optimizer'])
        for k in state_names:setattr(t,k,c[k])
        random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng'])
        start_epoch=c['epoch']+1
    original_run=t.run_biased;original_test=t.test_model;active={}
    def recorded_run(batch,config):
        output=original_run(batch,config)
        if active:
            mask=batch['output_mask'].numpy()==1
            rows,items=np.nonzero(mask)
            ir,ii=np.nonzero(batch['input_mask'].numpy())
            sr,si=np.nonzero(config['train_mask'].detach().cpu().numpy())
            # Save continuous probabilities before author's accuracy mutates its copy.
            dest=RUN/('current-validation' if active['split']=='val' else 'best-test')
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f"repeat-{active['repeat']}-batch-{active['batch']}.npz",
                epoch=np.array(t.epoch),repeat=np.array(active['repeat']),row=rows,item=items,
                prediction=output[mask].copy(),target=batch['output_labels'].numpy()[mask],
                input_row=ir,input_item=ii,input_label=batch['input_labels'].numpy()[ir,ii],
                selected_row=sr,selected_item=si)
            active['batch']+=1
        return output
    def recorded_test(id_,split='val'):
        active.update(repeat=id_,split=split,batch=0)
        try:return original_test(id_,split)
        finally:active.clear()
    t.run_biased=recorded_run;t.test_model=recorded_test
    for epoch in range(start_epoch,p.n_epoch):
        t.epoch=epoch;begin=time.monotonic();old_best=t.best_epoch
        t.train_model()
        state={k:getattr(t,k) for k in state_names}
        state.update(epoch=epoch,model=t.model.state_dict(),policy=t.st_policy.policy.state_dict(),meta=t.meta_params[0].detach(),optimizer=t.optimizer.state_dict(),meta_optimizer=t.meta_params_optimizer.state_dict(),policy_optimizer=t.st_policy.optimizer.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),freeze_sha256=digest(RUN/'freeze.json'))
        temp=RUN/'latest.pt.tmp';torch.save(state,temp);os.replace(temp,latest)
        if t.best_epoch!=old_best:
            shutil.copyfile(latest,RUN/'best.pt.tmp');os.replace(RUN/'best.pt.tmp',RUN/'best.pt')
            dest=RUN/'best-validation';dest.mkdir(exist_ok=True)
            for f in (RUN/'current-validation').glob('*.npz'):shutil.copyfile(f,dest/f.name)
        record={k:state[k] for k in state_names};record.update(epoch=epoch,epoch_seconds=time.monotonic()-begin,paid_usd=0)
        with (RUN/'epochs.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        atomic_json(RUN/'progress.json',record)
        print(json.dumps(record),flush=True)
        if epoch >= t.best_epoch+p.wait:break
    atomic_json(RUN/'completion.json',dict(record,completed=True,stop='author patience' if epoch<p.n_epoch-1 else 'author maximum',scope='one fold, author defaults; not five-fold paper-table replication'))
if __name__=='__main__':main()
