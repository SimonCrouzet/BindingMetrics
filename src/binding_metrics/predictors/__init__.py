"""Readers for the output of structure-prediction models.

Each supported model gets an adapter that turns its files into one neutral record, so the
confidence metrics and the EvoBind check need no per-model code. Importing this package
imports nothing heavy; biotite is imported only when a structure is read.
"""
