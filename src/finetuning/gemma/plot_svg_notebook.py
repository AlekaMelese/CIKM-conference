#!/usr/bin/env python3
"""
Simple SVG Plotting for Jupyter Notebooks
Use this in Jupyter notebooks to display SVG plots inline

Usage in Jupyter notebook:
    from plot_svg_notebook import SVGPlotter

    plotter = SVGPlotter()
    plotter.show_all()                  # Show all plots
    plotter.show(1)                     # Show plot 1
    plotter.show_by_name('loss')        # Show plots with 'loss' in name
"""

from IPython.display import SVG, display, HTML
from pathlib import Path
import glob


class SVGPlotter:
    """Simple SVG plotter for Jupyter notebooks"""

    def __init__(self, plots_dir="outputs/plots"):
        self.plots_dir = Path(plots_dir)

        if not self.plots_dir.exists():
            print(f"⚠️ Plots directory not found: {plots_dir}")
            print(f"   Current working directory: {Path.cwd()}")
            self.svg_files = []
        else:
            self.svg_files = sorted(self.plots_dir.glob("*.svg"))
            print(f"✓ Found {len(self.svg_files)} SVG plots in {plots_dir}")

    def list_plots(self):
        """List all available SVG plots"""
        if not self.svg_files:
            print("❌ No SVG files found")
            return

        print("\n📊 Available SVG Plots:\n")
        print("="*80)
        for i, svg_file in enumerate(self.svg_files, 1):
            file_size = svg_file.stat().st_size / 1024  # KB
            print(f"{i}. {svg_file.name} ({file_size:.2f} KB)")
        print("="*80 + "\n")

    def show(self, plot_number):
        """Show a specific plot by number"""
        if not self.svg_files:
            print("❌ No SVG files found")
            return

        if plot_number < 1 or plot_number > len(self.svg_files):
            print(f"❌ Invalid plot number. Available: 1-{len(self.svg_files)}")
            return

        svg_file = self.svg_files[plot_number - 1]
        print(f"📊 Displaying: {svg_file.name}")
        display(SVG(filename=str(svg_file)))

    def show_all(self, max_width='100%'):
        """Show all SVG plots"""
        if not self.svg_files:
            print("❌ No SVG files found")
            return

        print(f"\n📊 Displaying all {len(self.svg_files)} SVG plots...\n")

        for i, svg_file in enumerate(self.svg_files, 1):
            print(f"\n{'='*80}")
            print(f"Plot {i}: {svg_file.name}")
            print('='*80)

            # Display with custom width
            display(HTML(f'<div style="max-width: {max_width};">'))
            display(SVG(filename=str(svg_file)))
            display(HTML('</div>'))

    def show_by_name(self, keyword):
        """Show plots that contain keyword in filename"""
        matching_files = [f for f in self.svg_files if keyword.lower() in f.name.lower()]

        if not matching_files:
            print(f"❌ No plots found matching '{keyword}'")
            return

        print(f"\n📊 Found {len(matching_files)} plots matching '{keyword}':\n")

        for svg_file in matching_files:
            print(f"\n{'='*80}")
            print(f"{svg_file.name}")
            print('='*80)
            display(SVG(filename=str(svg_file)))

    def get_plot_path(self, plot_number):
        """Get the full path of a specific plot"""
        if not self.svg_files:
            return None

        if plot_number < 1 or plot_number > len(self.svg_files):
            return None

        return str(self.svg_files[plot_number - 1])


# Convenience functions for quick usage
def show_all_plots(plots_dir="outputs/plots"):
    """Quick function to show all plots"""
    plotter = SVGPlotter(plots_dir)
    plotter.show_all()


def show_plot(plot_number, plots_dir="outputs/plots"):
    """Quick function to show specific plot"""
    plotter = SVGPlotter(plots_dir)
    plotter.show(plot_number)


def list_plots(plots_dir="outputs/plots"):
    """Quick function to list all plots"""
    plotter = SVGPlotter(plots_dir)
    plotter.list_plots()


# Example usage when run directly
if __name__ == "__main__":
    print("""
    This script is designed for Jupyter notebooks.

    In a Jupyter notebook, use:

        from plot_svg_notebook import SVGPlotter

        plotter = SVGPlotter()
        plotter.list_plots()           # List all plots
        plotter.show(1)                # Show plot 1
        plotter.show_all()             # Show all plots
        plotter.show_by_name('loss')   # Show plots with 'loss' in name

    Or use quick functions:

        from plot_svg_notebook import show_plot, show_all_plots, list_plots

        list_plots()                   # List all plots
        show_plot(1)                   # Show plot 1
        show_all_plots()               # Show all plots
    """)
