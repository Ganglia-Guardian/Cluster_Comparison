---
name: utd-cohort-colors
description: Apply the UTD cohort color scheme (controls = medium-bright orange shades, MitoPark = medium-dark green shades, all lines solid) to a plot in this repo. Use when asked to recolor a plot by cohort, apply "the UTD colors"/"our color scheme"/"the usual green-orange", make control/MitoPark lines consistent, drop dashed-for-controls line styling, or swap which cohort gets which hue.
---

# UTD cohort color scheme

Color encodes **cohort** by hue and **individual mouse** by lightness:

- littermate controls (`*lc`) → shades of medium-bright orange (`UTD_ORANGE`, `#F08C1E`)
- MitoPark (`*mp`) → shades of medium-dark green (`UTD_GREEN`, `#2E6F4E`)

A nod to UT Dallas's colors. Because hue already carries the cohort, **every line is
solid** — the old dashed-for-controls convention is redundant and must go.

## Swapping which cohort gets which hue

This has been flipped once already, so expect it again. The hue→cohort mapping lives
in exactly one place — the loop in `cohort_colors` (repo-root `utils.py`):

```python
for coh, base in (("lc", UTD_ORANGE), ("mp", UTD_GREEN)):
```

Swap the two constants there. `UTD_GREEN`/`UTD_ORANGE` are named for the *colour*,
not the cohort, so they stay correct either way and no call site changes. Then fix
the captions — grep the repo for `green`/`orange` and update every title, docstring
and comment that names a cohort (see step 3 below); those are the only things that
go stale, and nothing catches them automatically. Finally re-run the affected plots.

## How to apply it

`cohort_colors` in the repo-root `utils.py` does the work. Never hand-roll the shades.

```python
from utils import cohort_colors      # subfolder scripts also need the repo root on sys.path

colors = cohort_colors(names)        # {name: rgb} for a list of mouse/dataset names
for name in names:
    ax.plot(x, y, "-", color=colors[name], label=name)   # solid, always
```

`cohort_colors` sorts within each cohort and spreads shades darkest→lightest across
however many mice are present, so the same set of mice always gets the same colors.
`cohort_of(name)` returns `"lc"`/`"mp"` if you need the cohort itself.

## Converting an existing plot

Work through all four steps — a half-conversion looks like a bug:

1. **Replace the per-mouse colormap.** Drop `cmap = plt.get_cmap("tab10")` /
   `color = cmap(i % 10)` and any hardcoded dict like
   `{"1mp": "tab:blue", "1lc": "tab:red"}`. Build `cohort_colors(names)` once
   before the loop and index it by name.
2. **Make every line solid.** Delete cohort line-style switches such as
   `style = "--" if cohort(name) == "lc" else "-"` and pass `"-"`.
3. **Fix the title/legend text.** Captions like `"dashed = control (lc), solid =
   MitoPark (mp)"` become false the moment step 2 lands. Replace with something
   like `"orange = control (lc), green = MitoPark (mp)"`.
4. **Drop shaded CI bands** on any plot that overlays several mice. Same-hue
   translucent bands smear into each other and hide the medians. Remove the
   `ax.fill_between(...)` *and* the now-unused bootstrap that fed it (it is
   usually the expensive part of the plot). Per-point `errorbar`s on isolated
   markers are fine — they don't overlap.

## Don't touch

Dashing that encodes something *other* than cohort — `ARENA_STYLE = {"2D": "--",
"3D": "-"}` in the arena scripts is 2D-vs-3D, not control-vs-MitoPark. Leave it.
Same for per-category palettes (`CAT_COLORS` for early/mid/late/sustained) and
sequential colormaps (viridis by week, magma for presence).

## Plots already converted

`elevation_analysis/slope_transitions.py` (module-level `COLORS`, so all three of
its plots), `within_mouse_drift.py` (`plot_combined`), `cluster_successor_diversity.py`
(`plot_summary`), `cluster_transition_compare.py` (`plot_top_change`, both the
expansion and contraction directions).

Known holdout: `plot_fanout` in `cluster_transition_compare.py` still uses tab10 +
dashed-for-controls. It was outside the original request; convert it with the steps
above if asked.
