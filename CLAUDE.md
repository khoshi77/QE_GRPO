# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

verl-based GRPO 学習を **NQSV (PBS) + OpenMPI + Ray** で複数ノード分散実行するためのプロジェクト。`verl/` はベンダリングされた upstream（手を入れない、独自の AGENTS.md がある）。プロジェクト固有のコードは `scripts/` のみ。

クラスタ構成:
- ノードあたり **H100 80GB が 1 枚**（= GPU数 == ノード数）
- ジョブ投入: `qsub` (NQSV/PBS) — `gpu` キューと `debug` キューがある
- マルチノード分散: OpenMPI で各ノードに 1 プロセス起動 → Ray クラスタ構築
