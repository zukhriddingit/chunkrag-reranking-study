import json, hashlib, re, string, itertools, datetime
from pathlib import Path
import numpy as np
import pandas as pd
assert not Path('/dev/nvidia0').exists()
assert (PSO/'analysis_specification_v1.md').exists()
reads={}
def read(rel):
 p=PS4/rel; reads[str(p)]={'sha256':psha(p),'bytes':p.stat().st_size}
 return json.loads(p.read_text())
def rows(rel):
 p=PS4/rel; reads[str(p)]={'sha256':psha(p),'bytes':p.stat().st_size}
 return [json.loads(l) for l in p.open() if l.strip()]
def norm(s):
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',str(s).lower().translate(str.maketrans('','',string.punctuation))).split())
def summ(v):
 a=np.asarray(list(v),dtype=float); a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'min':float(a.min()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p95':float(np.quantile(a,.95)) if len(a) else None,'max':float(a.max()) if len(a) else None}
c=rows('evaluation/common_candidates.jsonl'); r=rows('evaluation/rankings.jsonl'); g=rows('evaluation/generations_recovery_v2.jsonl'); m=rows('analysis/private_generation_metrics.jsonl'); cases=rows('private_evaluation/cases.jsonl'); cells=rows('evaluation/common_candidate_cells.jsonl')
assert (len(c),len(r),len(g),len(m),len(cases),len(cells))==(21840,8736,8736,8736,273,1092)
key=lambda x:(x['document_id'],x['chunker'],x['system_instance'])
R={key(x):x for x in r}; G={key(x):x for x in g}; M={key(x):x for x in m}; C={x['document_id']:x for x in cases}
assert len(R)==len(G)==len(M)==8736 and R.keys()==G.keys()==M.keys()
pools={}
for x in c: pools.setdefault((x['document_id'],x['chunker']),[]).append(x)
assert len(pools)==1092 and all(len(v)==20 for v in pools.values())
D=[]
for k,x in R.items():
 gg=G[k]; mm=M[k]; pool=pools[k[:2]]; top=x['top_k']; spans=sorted(set(filter(None,map(norm,C[k[0]]['grounding_span_texts'])))); texts=[norm(t['text']) for t in top]; packed=norm(gg['packed_context']); target=norm(C[k[0]]['target_agent_response'])
 avail=sum(t['evidence_grade']==2 for t in pool); selected=sum(t['evidence_grade']==2 for t in top)
 selected_spans={s for s in spans if any(s in t for t in texts)}; packed_spans={s for s in spans if s in packed}; pool_spans={s for s in spans if any(s in norm(t['text']) for t in pool)}
 dd={a:mm[a] for a in ['document_id','domain','family_key','chunker','system_instance','system','seed']}
 for a in ['agent_response_f1','agent_response_exact_match','grounding_span_f1','corrected_common_pool_evidence_ndcg_at_4','answer_visibility_at_4','gold_document_coverage_at_4']: dd[a]=mm[a]
 dd.update(pool_grade2=avail,pool_grade_ge1=sum(t['evidence_grade']>=1 for t in pool),selected_grade2=selected,grade2_candidate_recall=selected/avail if avail else None,annotated_span_count=len(spans),pool_literal_span_count=len(pool_spans),selected_literal_span_count=len(selected_spans),packed_literal_span_count=len(packed_spans),selected_spans_lost_in_packing=len(selected_spans-packed_spans),selected_any_literal_span=bool(selected_spans),packed_any_literal_span=bool(packed_spans),pool_any_literal_span=bool(pool_spans),target_literal_selected=bool(target) and any(target in t for t in texts),target_literal_packed=bool(target) and target in packed)
 for a in ['generated_tokens','full_prompt_tokens','used_prompt_tokens','context_truncated','generation_length_capped']: dd[a]=gg[a]
 dd['empty_normalized_output']=not gg['normalized_output'].strip(); D.append(dd)
f=pd.DataFrame(D)
metrics=['agent_response_f1','agent_response_exact_match','grounding_span_f1','corrected_common_pool_evidence_ndcg_at_4','answer_visibility_at_4','gold_document_coverage_at_4','grade2_candidate_recall','selected_grade2','pool_any_literal_span','selected_any_literal_span','packed_any_literal_span','target_literal_selected','target_literal_packed']
out={'classification':'new exploratory descriptive diagnostics; frozen metrics reused without resampling','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'matrix':{'documents':273,'families':235,'cells':1092,'generations':8736},'system_means':f.groupby('system')[metrics].mean().reset_index().to_dict('records'),'by_domain':f.groupby(['domain','system'])[metrics].mean().reset_index().to_dict('records'),'by_chunker':f.groupby(['chunker','system'])[metrics].mean().reset_index().to_dict('records'),'by_seed':f[f.system.isin(['B','E'])].groupby(['seed','system'])[metrics].mean().reset_index().to_dict('records')}
poolf=f.drop_duplicates(['document_id','chunker']); out['candidate_availability']={'n_cells':len(poolf),'zero_grade2':int((poolf.pool_grade2==0).sum()),'zero_grade_ge1':int((poolf.pool_grade_ge1==0).sum()),'grade2_count':summ(poolf.pool_grade2),'grade_ge1_count':summ(poolf.pool_grade_ge1),'cells_with_literal_annotated_span':int(poolf.pool_any_literal_span.sum())}
ov=[]
for doc,ch in pools:
 for seed in [20260904,20260905,20260906]:
  bs=[z for z in r if z['document_id']==doc and z['chunker']==ch and z['system']=='B' and z['seed']==seed]; es=[z for z in r if z['document_id']==doc and z['chunker']==ch and z['system']=='E' and z['seed']==seed]
  assert len(bs)==len(es)==1
  b=[t['chunk_id'] for t in bs[0]['top_k']]; e=[t['chunk_id'] for t in es[0]['top_k']]; inter=len(set(b)&set(e))
  ov.append({'document_id':doc,'chunker':ch,'seed':seed,'intersection_fraction':inter/4,'jaccard':inter/len(set(b)|set(e)),'identical_order':b==e,'identical_set':set(b)==set(e)})
o=pd.DataFrame(ov); out['BE_top4_overlap']={a:summ(o[a]) for a in ['intersection_fraction','jaccard','identical_order','identical_set']}; out['full_ranking_correlation']='unavailable: saved rankings contain only top four; no full permutation reconstructed'
out['generation']={}
for sys,z in f.groupby('system'):
 out['generation'][sys]={'n':len(z),'lengths':{a:summ(z[a]) for a in ['generated_tokens','full_prompt_tokens','used_prompt_tokens']},'flags':{a:int(z[a].sum()) for a in ['context_truncated','generation_length_capped','empty_normalized_output']},'selected_span_instances_lost':int(z.selected_spans_lost_in_packing.sum()),'records_with_span_loss':int((z.selected_spans_lost_in_packing>0).sum()),'grade2_recall_undefined_no_support':int(z.grade2_candidate_recall.isna().sum())}
doc=f.groupby(['document_id','system'])[metrics+['generated_tokens','generation_length_capped']].mean(); b=doc.xs('B',level='system'); e=doc.xs('E',level='system'); delta=e-b; delta['top4_overlap']=o.groupby('document_id').intersection_fraction.mean(); out['document_associations']=[]
for a in ['corrected_common_pool_evidence_ndcg_at_4','grade2_candidate_recall','selected_grade2','selected_any_literal_span','packed_any_literal_span','target_literal_packed','generated_tokens','generation_length_capped','top4_overlap']:
 z=delta[[a,'agent_response_f1']].dropna(); out['document_associations'].append({'x':a,'n':len(z),'pearson':float(z.corr().iloc[0,1]) if z[a].nunique()>1 else None,'spearman':float(z.corr(method='spearman').iloc[0,1]) if z[a].nunique()>1 else None,'scope':'document-level paired differences; overlap is raw document mean; no p-values or causal interpretation'})
out['all_pairwise_descriptive']=[]
for a,bm in itertools.combinations(['H','U','B','E'],2):
 out['all_pairwise_descriptive'].append({'contrast':bm+' minus '+a,**{v:float(doc.xs(bm,level='system')[v].mean()-doc.xs(a,level='system')[v].mean()) for v in metrics[:4]}})
out['training_grid']=read('training_grid/grid_training_manifest.json')
out['final_refit_manifests']=[]
for p in sorted(PS4.glob('**/terminal_manifest.json')):
 if 'refit' in str(p): out['final_refit_manifests'].append({'path':str(p),'sha256':psha(p),'value':json.loads(p.read_text())})
out['retrieval_timing_seconds']={a:summ(x['timing_seconds'][a] for x in cells) for a in ['bm25','dense','fusion','total']}
for rel in ['evaluation/reranking_manifest.json','evaluation/generation_manifest.json','resource/allocation_ledger.json']: out[Path(rel).stem]=read(rel)
out['ndcg_verified_outputs']=[]
nm=json.loads((P3NC/'manifest.json').read_text())
for p in sorted((P3NC/'public').glob('*')): out['ndcg_verified_outputs'].append({'path':str(p),'sha256':psha(p)})
assert psha(P3NC/'public/corrected_aggregates.json')=='d462a564cc00741d5a08f24f5fca7e98ed65255a0dd4f5a5d16df21d3ddd8d4f'
out['phase3_ndcg']=json.loads((P3NC/'public/corrected_aggregates.json').read_text())
out['phase3_evidence_flow']=json.loads((P3EF/'public/aggregate_summary.json').read_text()) if (P3EF/'public/aggregate_summary.json').exists() else {'public_files':[str(p) for p in (P3EF/'public').glob('*')]}
out['source_reads']=reads
for rel,want in PSHASH.items(): assert psha(PS4/rel)==want,rel
out['post_analysis_original_rehash']={'checks':len(PSHASH),'changed':0,'missing':0}
f.to_json(PSO/'private/document_chunker_instance_diagnostics_v1.jsonl',orient='records',lines=True); o.to_json(PSO/'private/BE_top4_overlap_v1.jsonl',orient='records',lines=True)
def clean(x):
 if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list): return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x): return None
 return x
out=clean(out)
pp=PSO/'private/diagnostic_aggregate_results_v1.json'; pp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('DIAGNOSTIC_RESULT',pp.read_text()); print('DIAGNOSTIC_SHA',psha(pp))
