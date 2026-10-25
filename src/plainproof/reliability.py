"""Reporting only: decisions never infer truth from assistance or confidence."""
import math

def matched_coverage(records,coverage):
    if not 0<coverage<=1 or not records:raise ValueError('coverage/population')
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate decision ID')
    for r in records:
        if not 0<=r['p_mastery']<=1 or r['truth'] not in (0,1):raise ValueError('invalid decision')
    # Confidence and stable public ID only. Truth cannot affect the selected set.
    ranked=sorted(records,key=lambda r:(-max(r['p_mastery'],1-r['p_mastery']),r['id']))
    n=math.ceil(len(records)*coverage);chosen=ranked[:n]
    fm=sum(r['p_mastery']>=.5 and r['truth']==0 for r in chosen)
    fg=sum(r['p_mastery']<.5 and r['truth']==1 for r in chosen)
    return dict(total=len(records),decisions=n,inconclusive=len(records)-n,coverage=n/len(records),
                false_mastery=fm,false_gap=fg,error_given_decision=(fm+fg)/n,
                false_mastery_all=fm/len(records),false_gap_all=fg/len(records),
                selected_ids=[r['id'] for r in chosen])

def first_stop(history,threshold):
    """One target's actual first passage. Never selects the best realized endpoint."""
    if not .5<threshold<=1:raise ValueError('confidence threshold')
    previous=-1
    for row in history:
        if row['budget']<=previous or not 0<=row['p_mastery']<=1:raise ValueError('bad trajectory')
        previous=row['budget']
        if max(row['p_mastery'],1-row['p_mastery'])>=threshold:return row
    return None

def crossing(budgets,errors,target):
    if len(budgets)!=len(errors) or not budgets or any(a>=b for a,b in zip(budgets,budgets[1:])):raise ValueError('budget grid')
    if not all(math.isfinite(x) for x in [*errors,target]):raise ValueError('nonfinite error')
    return next((b for b,e in zip(budgets,errors) if e<=target),None)
