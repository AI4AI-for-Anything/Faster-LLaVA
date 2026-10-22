#!/usr/bin/env python3
"""Render compact crimson README assets from archived measurements; no GPU work."""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-llava-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def hero():
    data = json.loads((ROOT/'history.json').read_text())['rows'][-1]
    values = [data['reference_ms'], data['candidate_ms']]
    assert all(math.isfinite(v) and v > 0 for v in values)
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-LLaVA, optimized by AI agents', color=RED,
             weight='bold', fontsize=19)
    fig.text(.06, .78, 'LLaVA-1.5-7B · 4 H20 GPUs · global batch 128 · BF16 / TF32 / ZeRO-3',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.20, .24, .43, .45])
    ax.barh([1, 0], values, color=['#D7BDBF', RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Baseline', 'Faster-LLaVA'])
    for y, value in zip([1, 0], values):
        ax.text(value+550, y, f'{value:,.1f} ms', va='center',
                color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 28500)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Time per optimizer step (ms) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{values[0]/values[1]:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'step speedup', color=MUTED, fontsize=12, ha='center')
    save(fig, ROOT/'01-h20-benchmark')


def history():
    rows = json.loads((ROOT/'history.json').read_text())['rows']
    early = [row for row in rows if row['step'] < 12]       # rounds 1-52: accepted speedup points only
    later = [row for row in rows if row['step'] >= 12]      # rounds 53-69, one step each
    fig = plt.figure(figsize=(9, 9), facecolor='white')
    fig.text(.09, .96, 'Faster-LLaVA', fontsize=11, weight='bold', color=RED)
    fig.text(.09, .913, 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, .928, f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, .902, 'step speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, .855, 'Cumulative speedup · 4 H20 GPUs · global batch 128 · 28 rounds', fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, .535, .85, .29])
    x = [row['step'] for row in rows]
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    for row in rows:
        quick = row['check'] == 'quick'
        ax.plot([row['step']], [row['speedup']], marker='o', markersize=6.5, zorder=3, color=RED,
                markerfacecolor='white' if quick else RED, markeredgewidth=1.8)
    labelled = [early[0], early[-1]] + later
    for i, row in enumerate(labelled):
        ax.annotate(f"{row['speedup']:.2f}×", (row['step'], row['speedup']),
                    xytext=(0, 11 if i % 2 == 0 else -23), textcoords='offset points', ha='center', fontsize=12,
                    color=RED if row is rows[-1] else INK, weight='bold')
    ticks = [1, 5, 10, 15, 20, 25, 28]
    ax.set_xticks(ticks, [str(step) for step in ticks])
    ax.set_yticks([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.2f}×'))
    ax.set_ylim(.93, 2.57)
    ax.set_xlim(.3, 28.7)
    frame(ax)
    cards = [(f"{early[0]['step']}–{early[-1]['step']}",
              f"Early rounds, {early[0]['speedup']:.2f}× → {early[-1]['speedup']:.2f}×")]
    cards += [(str(row['step']), row['label']) for row in later]
    for i, (tag, text) in enumerate(cards):
        x, y = .09+(i % 2)*.44, .425-(i // 2)*.082
        fig.add_artist(FancyBboxPatch((x, y-.042), .41, .065,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(x+.014, y-.009, tag, fontsize=12, weight='bold', color=RED)
        label = textwrap.fill(text, width=24, break_long_words=False, break_on_hyphens=False)
        fig.text(x+.079, y-.009, label, fontsize=11.5, color=INK,
                 weight='bold', va='center', linespacing=1.35)
    fig.text(.09, .022, 'Filled: full check (all four gates). Open: timing screen + quick check. '
             'Steps 1–11 are the accepted speedup points of rounds 1–52.', fontsize=9, color=MUTED)
    save(fig, ROOT/'02-history')


def modulemap():
    fig = plt.figure(figsize=(9, 6.6), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .92, 'What changed in Faster-LLaVA', fontsize=23, color=INK, weight='bold')
    cards = [
        ('Data & sampling', 'Token grouping · step balancing\nGlobal supervised-token loss',
         'train.py\nllava_trainer.py', 2),
        ('Packed computation', 'Keep valid tokens packed\nAcross attention, MLP and layers',
         'unpad_mlp.py · unpad_attention.py\npacked_attention.py · packed_decoder.py\npacked_rope.py', 5),
        ('Activations & loss', 'Recompute SwiGLU · supervised-row head\nOptional chunking (off in final preset)',
         'recompute_swiglu.py\nfused_head_loss.py · chunked_head.py', 3),
        ('Integration & communication', 'Fused norms · activation retention\nZeRO-3 parameter persistence',
         'language_model/llava_llama.py\nscripts/zero3.json', 2),
    ]
    for i, (title, description, paths, count) in enumerate(cards):
        x, y = .055+(i % 2)*.455, .49-(i // 2)*.39
        ax.add_patch(FancyBboxPatch((x, y), .425, .34,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x+.425], [y+.34, y+.34], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x+.014, y+.30, f'{i+1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x+.41, y+.30, f'{count} files', fontsize=10, color=MUTED, ha='right')
        ax.text(x+.014, y+.254, title, fontsize=13, color=INK, weight='bold')
        ax.text(x+.014, y+.198, description, fontsize=10.3, color=INK, va='top', linespacing=1.6)
        ax.text(x+.014, y+.035, paths, fontsize=9.5, color=MUTED, linespacing=1.5)
    save(fig, ROOT.parent/'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    modulemap()
