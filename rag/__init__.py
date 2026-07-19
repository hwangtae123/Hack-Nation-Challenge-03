"""RealDoor RAG - the rule-retrieval layer for Stages 2 (Understand) and 3 (Prepare).

Scope is fixed to LIHTC (Section 42) / San Diego-Chula Vista-Carlsbad, CA MSA /
FY2026 (effective 2026-05-01). This package retrieves cited rule *prose* and runs
*deterministic* calculations. It never decides eligibility: income and limits are
shown side by side, the final determination stays human. See ``rag/claude.md`` for
the governing rules.
"""
