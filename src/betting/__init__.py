"""Betting evaluation: odds math, de-vigging, and a CLV/ROI backtest harness.

This package is deliberately independent of the win/outcome models. Its only
contract with a model is a per-game home-win probability plus the game's
final result; everything else (prices, de-vigging, bet selection, staking,
settlement, CLV) lives here so any probability source can be backtested the
same way.
"""
