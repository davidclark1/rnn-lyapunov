"""Make every figure of the paper from the bundled data (about a minute on a CPU).

    python -m paper.make_figures [--out DIR] [--root DIR]

--root points the data-based figures at another directory with the same file formats as data/ (e.g. results/ after
a recomputation, see reproduce/README.md).  Each figure also has its own script: python -m paper.fig_<name>.
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")         # plotting needs no GPU

from paper import fig_dimension_entropy, fig_max_exponent, fig_spectra, fig_trivial_fixed_point  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output directory (default: figures/)")
    p.add_argument("--root", default=None, help="data directory (default: data/)")
    a = p.parse_args()
    fig_spectra.main(a.out, a.root)
    fig_dimension_entropy.main(a.out, a.root)
    fig_trivial_fixed_point.main(a.out)
    fig_max_exponent.main(a.out, a.root)
