"""Prospective, frozen-checkpoint selection contrast. Never trains on test data."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='6'
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
pass  # (public) the original added a local site-packages path for scikit-learn; install requirements.txt
from bobcat_support import ROOT,RUN,atomic_json

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup():
    sys.path.insert(0,str(ROOT/'arghosh-BOBCAT'))
    import train as t
    from model import MAMLModel
    from policy import StraightThrough
    from utils.configuration import create_parser,initialize_seeds
    sys.argv=['train.py','--dataset','assist2009','--model','biirt-biased','--fold','1','--seed','221','--n_query','5']
    p=create_parser();initialize_seeds(p.seed);torch.set_num_threads(6)
    c=torch.load(RUN/'best.pt',weights_only=False,map_location='cpu')
    assert c['epoch']==c['best_epoch']==105
    t.params=p;t.config={};t.sampling='biased'
    t.model=MAMLModel(n_question=p.n_question,question_dim=1,sampling='biased',n_query=5)
    t.model.load_state_dict(c['model']);t.model.eval()
    t.meta_params=[c['meta'].detach().clone().requires_grad_()]
    t.st_policy=StraightThrough(p.n_question,p.n_question,p.policy_lr,(.9,.999))
    t.st_policy.policy.load_state_dict(c['policy']);t.st_policy.policy.eval()
    t.optimizer=torch.optim.Adam(t.model.parameters(),lr=p.lr)
    t.meta_params_optimizer=torch.optim.SGD(t.meta_params,lr=p.meta_lr)
    return t,p

def adapt(t,batch,mask):
    """Same author's five-step, sum-loss adaptation from fixed learned prior."""
    config={'mode':'test','train_mask':mask}
    params=t.clone_meta_params(batch)
    t.inner_algo(batch,config,params)
    return config['meta_param'].detach()

def trajectory(t,batch,policy,max_budget,seed):
    """Only previous revealed input answers enter choice; targets never enter."""
    available=batch['input_mask'].clone();mask=torch.zeros_like(available)
    labels=batch['input_labels'];b,n=mask.shape
    gen=torch.Generator().manual_seed(seed)
    # Independent fixed random order gives nested prefixes and common RNG controls.
    random_order=torch.rand((b,n),generator=gen).masked_fill(available==0,-1)
    theta=t.meta_params[0].expand(b,-1).detach().clone()
    snapshots={};actions=[]
    for k in range(max_budget+1):
        if k:
            if policy=='learned':
                with torch.no_grad(): _,action=t.st_policy.policy((labels.float()-.5)*2*mask,available)
            elif policy=='random':action=random_order.masked_fill(available==0,-1).argmax(dim=1)
            elif policy=='uncertainty':action=t.model.pick_uncertain_sample(theta,available)
            else:raise ValueError('unknown policy')
            assert torch.all(available[torch.arange(b),action]==1)
            available[torch.arange(b),action]=0;mask[torch.arange(b),action]=1
            actions.append(action.numpy().copy())
            theta=adapt(t,batch,mask)
        snapshots[k]=theta.numpy().copy()
    return snapshots,np.asarray(actions)

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--profile',action='store_true');args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    t,p=setup()
    from dataset import Dataset,collate_fn
    from utils.utils import data_split
    receipt=json.loads((ROOT/'dataset-receipt.json').read_text())
    parts=data_split(receipt['file'],p.fold,p.seed)
    if args.profile:
        ds=Dataset(parts[0][:16],seed=100);batch=collate_fn(p.n_question)([ds[i] for i in range(len(ds))])
        start=time.monotonic()
        for policy in ('learned','random','uncertainty'):trajectory(t,batch,policy,8,2026102441)
        atomic_json(out/'profile.json',dict(seconds=time.monotonic()-start,learners=16,split='training only',policies=3,budgets=8))
        return
    reg=json.loads((out/'registration.json').read_text())
    for path,digest in reg['freeze_files'].items():assert sha(path)==digest,path
    budgets=reg['budgets'];targets=[];rows=[];preds=[];saved=[];fidelity=[]
    # Ten original partition repeats; each person remains one statistical cluster.
    for rep in range(100,110):
        ds=Dataset(parts[2],seed=rep);batch=collate_fn(p.n_question)([ds[i] for i in range(len(ds))])
        assert not torch.any(batch['input_mask']*batch['output_mask']), 'input/target overlap'
        assert batch['input_mask'].sum(1).min()>=max(budgets), 'budget unsupported; no silent exclusions'
        row,item=np.nonzero(batch['output_mask'].numpy());target=batch['output_labels'].numpy()[row,item]
        predictions=[];allactions=[]
        for policy in reg['policies']:
            theta,actions=trajectory(t,batch,policy,max(budgets),reg['seed']+rep)
            difficulty=t.model.question_difficulty.detach().numpy()[0]
            for budget in budgets:
                prob=1/(1+np.exp(-(theta[budget][row,0]-difficulty[item])))
                predictions.append(prob)
            allactions.append(actions)
        # Native algorithm parity on the original five-question checkpoint.
        native=t.run_biased(batch,{'mode':'test'})[row,item]
        learned5=predictions[budgets.index(5)]
        error=float(np.max(np.abs(native-learned5)));assert error<1e-6,error
        frozen=np.load(RUN/'best-test'/f'repeat-{rep}-batch-0.npz')
        assert np.array_equal(row,frozen['row']) and np.array_equal(item,frozen['item'])
        assert np.array_equal(target,frozen['target'])
        saved_error=float(np.max(np.abs(native-frozen['prediction'])));assert saved_error<1e-6,saved_error
        fidelity.append(dict(repeat=rep,native_max_error=error,saved_max_error=saved_error))
        np.savez_compressed(out/f'repeat-{rep}.npz',row=row,item=item,target=target,prediction=np.array(predictions),actions=np.array(allactions))
        print(json.dumps(dict(completed_repeat=rep,targets=len(target))),flush=True)
    atomic_json(out/'fidelity.json',dict(checks=fidelity,checkpoint_epoch=105,split_sizes=list(map(len,parts)),completed=True))

if __name__=='__main__':main()
