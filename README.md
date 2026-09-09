# ChunkRAG v1.0.0 staging — PRIVATE / NOT PUBLISHED

Prepared for author review only. The authors approved Apache-2.0 for code/configurations and CC-BY-4.0 for aggregate results, figures and documentation. See LICENSE.md for scope. The private staging repository is https://github.com/zukhriddingit/chunkrag-reranking-study. Zenodo is planned as a manually managed draft; its account and DOI remain unresolved because the service is currently inaccessible. No public release has occurred. This is a selected source-and-aggregate companion, not the complete experiment archive.

## Included

- 20 unchanged non-human aggregate CSV tables, including both studies, every supplied method/domain/chunker/fixed-seed summary, evidence-flow diagnostics and measured cost summaries.
- Exact Phase 4 primary values extracted from the sealed snapshot in a clearly labeled new JSON representation.
- Six unchanged aggregate figure files (three PDF/PNG pairs), captions and the dated exploratory specification.
- The 20-file local Phase 4 preparation source bundle, selected retained Phase 3 analysis sources, the versioned NDCG correction source/patch and historical post-study diagnostic code.
- Finalized pre-execution protocol, frozen configuration in the source bundle, selected recorded environment/model identities and operational repair metadata.

## Safe offline check and viewing

Use Python 3.9 or later with its standard library. From a fresh extraction:

    python3 tools/inspect_aggregates.py --output /new/nonexistent/output_directory

This verifies the exact payload hashes and formats the included CSV cells as an offline HTML page. It performs no training, generation, reranking, bootstrap, new statistical test or metric calculation. It cannot access the original private records. The output directory must not exist. Open its report.html locally.

Historical files under code/ are reference material, not the supported entry point. Do not execute experiment scripts as part of package verification. Their dependencies and input paths are incomplete in this companion. The copied sources are unchanged and can contain original path assumptions.

## Scientific interpretation

Phase 3 and Phase 4 are separate studies. Neither passed its prespecified success rule. Phase 4 E minus B response F1 is -0.0022110633468929476, with 95% family-cluster interval [-0.006609704843055635, 0.0019202852812187537] and two-sided family sign-flip p=0.3122168778312217. The interval excludes the targeted +0.010 gain; this is not equivalence. There are 273 documents and 235 dependency families, not 8,736 independent observations. The final executed secondary specification is descriptive; post-study diagnostics remain exploratory.

Phase 3 reserve questions were excluded from Phase 3 fitting and selection, but were not globally untouched. Phase 3 answer generation covers one seed; Phase 4 B/E covers all three fixed seeds. Corrected common-pool reserve NDCG does not alter the original F1 results. Literal overlap, annotated evidence and human-rated support are different constructs. Cost rows have their recorded workload/session scopes and are not interchangeable deployment latency measurements.

Human-evaluation material is intentionally excluded from this release candidate pending provenance/ethics and publication-treatment decisions. It remains in the internal manuscript/audit; this package does not delete or replace the planned evaluation.

See LICENSE.md, CITATION.cff, DESTINATION_PLAN.md, RELEASE_ACCESS_MATRIX.md, REPRODUCIBILITY.md, RIGHTS_AND_RELEASE_GATES.md and DATA_AVAILABILITY_DRAFT.md. Full source traceability appears in provenance/source_identities.json; the exact shipped inventory is PAYLOAD_MANIFEST.json. Source hashes identify bytes, not licenses or human provenance.

This is a private staging checkpoint. Do not treat its manifest as the final DOI-bearing distribution seal. A real DOI must be reserved and inserted before final release sealing. No automatic GitHub-to-Zenodo publication route is configured by this workflow.
