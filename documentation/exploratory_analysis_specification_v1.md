# Post-study exploratory specification

Recorded September 8, 2026 (UTC), before any new diagnostic calculation. This is a post-outcome specification, not preregistration or a retrospective freeze. Existing primary and secondary outcomes are already known from the handoff. CPU only; no fitting, relabeling, generation, prompt tuning, selection, or new hypothesis test.

## Gate and reuse

Acquire and hash-check the final Phase 4 manifest, verification, primary result, finalized protocol, secondary results, repair ledger, and required immutable ranking/candidate/generation artifacts. Inventory completed diagnostics first; reuse each matching completed diagnostic and its checks. Do not repeat the bootstrap/sign flips. No diagnostic computation until the input identity and field meanings are established. Missing fields produce unavailable entries, never inferred observations.

## Full population and definitions

Use all 273 selected documents, four frozen chunkers, and all three paired B/E seeds. H/U have one system instance each; never replicate them as independent seed runs. Summarize B/E paired rows within document over chunkers and seeds. Preserve domain and dependency-family structure. Report all four domains and chunkers, including missingness, without selecting favorable subsets. Diagnostic associations are descriptive, with no new p-values or multiplicity claims.

1. Candidate availability: per shared document/chunker pool, count frozen grades 0/1/2 and whether any grade 2 exists; separately report any grade >=1. Grade 2 means normalized annotated-span containment, not human-rated sufficiency; grade 1 is gold-document membership, not direct support. Use the recorded common pool, not newly retrieved candidates.
2. Ranking overlap: paired B/E top-four chunk-ID intersection divided by four, Jaccard intersection/union, exact ordered-top-four match, and full-common-pool Spearman rank correlation (only complete shared permutations). Report distributions and document-weighted means for every fixed seed/chunker. Ties use saved deterministic order.
3. Retrieval endpoints: reuse corrected common-pool NDCG@4; report top-four grade-2 availability and fraction of available grade-2 pool chunks selected, defining zero-available pools as undefined for the latter and reporting their count separately. Never substitute this custom coverage for prespecified coverage or call it answerability.
4. Context packing: use saved packed context and saved selected chunks, with the already-frozen normalization. Count selected annotated spans literally present before packing and retained after packing; separately count literal target response overlap if recorded/computable. Do not reconstruct exact token packing without pinned tokenizer/template and token-identical verification. No semantic support inference.
5. Relationships: at document level, average paired E-minus-B F1 and paired retrieval/packing differences across all seeds/chunkers. Report Pearson and Spearman correlations for available nonconstant measures and full-case denominator; no causal language. If any measure is constant, report correlation undefined. Plot aggregate bins only where individual document data are protected; retain private diagnostic rows privately. Correlations cannot identify causal evidence use.
6. Generation/failures: summarize recorded output tokens, input tokens, finish reasons, empty outputs, cap hits, missing/duplicate records, and operational failures by all methods and fixed B/E seeds. A cap hit is not proof of semantic incompleteness. No new tokenizer-based length claim if the saved field is unavailable.
7. Costs: extract existing stage timers and their documented boundaries, unit, batch size, hardware, dtype, workload counts, and cold-load inclusion. Keep allocated research time separate from measured operation time. Report per-request latency only from matched request timers; do not divide total allocated GPU time by requests and call it deployment latency.

No equivalence test. No pooling Phase 3 and Phase 4. No favorable weighting. If the final protocol specifies a different existing endpoint, retain it under its original prespecified name and keep these additions exploratory. Record any deviation in a new dated file before affected computation.

## Figure contract

Publication figures use standalone static exports with labeled units and captions. Primary intervals, if verified, occupy separate Phase 3 and Phase 4 panels because their estimands differ; show zero and the original +0.010 target. All-method metric figures use fixed H/U/B/E order and distinguish agent-response F1, grounding-span F1, and corrected NDCG. Seed points are descriptive, not independent dataset replications. No post-hoc subgroup significance coloring. Use dark gray/blue and marker shapes, legible at manuscript width. Numeric tables provide exact lookup; source hashes and QA accompany figures.
