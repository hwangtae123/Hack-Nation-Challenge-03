"""RealDoor Stage 1 — document extraction pipeline.

Extracts an allowlisted set of fields from synthetic income documents, attaches a
source box (bbox) and confidence to every value, and keeps every value locked
(`confirmed = False`) until a renter confirms it. This package never makes an
eligibility, approval, or scoring decision: it stops at the extraction ->
confirmation hand-off.
"""
