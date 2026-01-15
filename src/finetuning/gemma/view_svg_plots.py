#!/usr/bin/env python3
"""
SVG Plot Viewer and Utility Script
Displays and manipulates SVG images generated from training

Usage:
    python view_svg_plots.py                    # Display all SVG plots
    python view_svg_plots.py --list             # List all available SVG files
    python view_svg_plots.py --plot 1           # Display specific plot number
    python view_svg_plots.py --convert-to-png   # Convert all SVG to PNG
"""

import os
import glob
import argparse
from pathlib import Path

# Try to import required libraries
try:
    from IPython.display import SVG, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("⚠️ IPython not available. SVG display will use alternative method.")

try:
    from cairosvg import svg2png
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("⚠️ cairosvg not available. Install: pip install cairosvg")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available. Install: pip install Pillow")

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class SVGPlotViewer:
    """Utility class for viewing and converting SVG plots"""

    def __init__(self, plots_dir="outputs/plots"):
        self.plots_dir = Path(plots_dir)
        if not self.plots_dir.exists():
            raise FileNotFoundError(f"Plots directory not found: {plots_dir}")

    def list_svg_files(self):
        """List all SVG files in the plots directory"""
        svg_files = sorted(self.plots_dir.glob("*.svg"))

        if not svg_files:
            print(f"❌ No SVG files found in {self.plots_dir}")
            return []

        print(f"\n📊 Found {len(svg_files)} SVG plots:\n")
        print("="*80)
        for i, svg_file in enumerate(svg_files, 1):
            file_size = svg_file.stat().st_size / 1024  # KB
            print(f"{i:2d}. {svg_file.name:50s} ({file_size:8.2f} KB)")
        print("="*80)

        return svg_files

    def display_svg_inline(self, svg_path):
        """Display SVG inline (works in Jupyter notebooks)"""
        if not IPYTHON_AVAILABLE:
            print("⚠️ IPython not available. Cannot display inline.")
            return False

        print(f"\n📊 Displaying: {svg_path.name}")
        display(SVG(filename=str(svg_path)))
        return True

    def display_svg_matplotlib(self, svg_path):
        """Display SVG using matplotlib (requires conversion)"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available.")
            return False

        if not CAIRO_AVAILABLE or not PIL_AVAILABLE:
            print("⚠️ cairosvg and/or PIL not available. Cannot convert SVG.")
            return False

        # Convert SVG to PNG in memory
        png_data = svg2png(url=str(svg_path))
        img = Image.open(io.BytesIO(png_data))

        # Display using matplotlib
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{svg_path.name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return True

    def display_svg_browser(self, svg_path):
        """Open SVG in default web browser"""
        import webbrowser

        abs_path = svg_path.absolute()
        print(f"\n🌐 Opening in browser: {svg_path.name}")
        webbrowser.open(f'file://{abs_path}')
        return True

    def convert_svg_to_png(self, svg_path, output_path=None, dpi=300):
        """Convert SVG to PNG with specified DPI"""
        if not CAIRO_AVAILABLE:
            print("⚠️ cairosvg not available. Install: pip install cairosvg")
            return False

        if output_path is None:
            output_path = svg_path.with_suffix('.png')

        # Calculate scale factor for DPI
        scale = dpi / 96.0  # 96 DPI is default

        svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            scale=scale
        )

        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✓ Converted: {svg_path.name} → {output_path.name} ({file_size:.2f} KB)")

        return True

    def convert_all_to_png(self, dpi=300):
        """Convert all SVG files to PNG"""
        svg_files = self.list_svg_files()

        if not svg_files:
            return

        print(f"\n🔄 Converting all SVG files to PNG (DPI={dpi})...\n")

        success_count = 0
        for svg_file in svg_files:
            if self.convert_svg_to_png(svg_file, dpi=dpi):
                success_count += 1

        print(f"\n✓ Successfully converted {success_count}/{len(svg_files)} files")

    def display_all_svgs(self, method='browser'):
        """Display all SVG files using specified method"""
        svg_files = self.list_svg_files()

        if not svg_files:
            return

        print(f"\n📊 Displaying all SVG plots using method: {method}\n")

        for svg_file in svg_files:
            if method == 'inline':
                self.display_svg_inline(svg_file)
            elif method == 'matplotlib':
                self.display_svg_matplotlib(svg_file)
            elif method == 'browser':
                self.display_svg_browser(svg_file)
            else:
                print(f"❌ Unknown display method: {method}")

    def display_specific_plot(self, plot_number, method='browser'):
        """Display a specific plot by number"""
        svg_files = sorted(self.plots_dir.glob("*.svg"))

        if not svg_files:
            print(f"❌ No SVG files found in {self.plots_dir}")
            return

        if plot_number < 1 or plot_number > len(svg_files):
            print(f"❌ Invalid plot number. Available: 1-{len(svg_files)}")
            return

        svg_file = svg_files[plot_number - 1]

        if method == 'inline':
            self.display_svg_inline(svg_file)
        elif method == 'matplotlib':
            self.display_svg_matplotlib(svg_file)
        elif method == 'browser':
            self.display_svg_browser(svg_file)
        else:
            print(f"❌ Unknown display method: {method}")

    def get_svg_info(self):
        """Get information about all SVG files"""
        svg_files = sorted(self.plots_dir.glob("*.svg"))

        if not svg_files:
            print(f"❌ No SVG files found in {self.plots_dir}")
            return

        print(f"\n📊 SVG Files Information:\n")
        print("="*100)
        print(f"{'#':<4} {'Filename':<50} {'Size (KB)':<12} {'Full Path':<30}")
        print("="*100)

        total_size = 0
        for i, svg_file in enumerate(svg_files, 1):
            file_size = svg_file.stat().st_size / 1024  # KB
            total_size += file_size
            print(f"{i:<4} {svg_file.name:<50} {file_size:<12.2f} {str(svg_file):<30}")

        print("="*100)
        print(f"Total: {len(svg_files)} files, {total_size:.2f} KB\n")


def main():
    parser = argparse.ArgumentParser(
        description="SVG Plot Viewer and Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_svg_plots.py --list
  python view_svg_plots.py --plot 1 --method browser
  python view_svg_plots.py --convert-to-png --dpi 600
  python view_svg_plots.py --all --method matplotlib
        """
    )

    parser.add_argument('--plots-dir', type=str, default='outputs/plots',
                       help='Path to plots directory (default: outputs/plots)')
    parser.add_argument('--list', action='store_true',
                       help='List all SVG files')
    parser.add_argument('--plot', type=int, metavar='N',
                       help='Display specific plot number (1-7)')
    parser.add_argument('--all', action='store_true',
                       help='Display all SVG plots')
    parser.add_argument('--method', type=str, choices=['inline', 'matplotlib', 'browser'],
                       default='browser',
                       help='Display method (default: browser)')
    parser.add_argument('--convert-to-png', action='store_true',
                       help='Convert all SVG files to PNG')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PNG conversion (default: 300)')
    parser.add_argument('--info', action='store_true',
                       help='Show detailed information about SVG files')

    args = parser.parse_args()

    # Create viewer instance
    try:
        viewer = SVGPlotViewer(plots_dir=args.plots_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Execute requested action
    if args.list:
        viewer.list_svg_files()
    elif args.info:
        viewer.get_svg_info()
    elif args.convert_to_png:
        viewer.convert_all_to_png(dpi=args.dpi)
    elif args.plot:
        viewer.display_specific_plot(args.plot, method=args.method)
    elif args.all:
        viewer.display_all_svgs(method=args.method)
    else:
        # Default: list files
        print("\n" + "="*80)
        print("SVG PLOT VIEWER - Gemma 3 12B IT Fine-tuning Visualizations")
        print("="*80)
        viewer.list_svg_files()
        print("\n💡 Use --help for more options")
        print("💡 Example: python view_svg_plots.py --plot 1 --method browser")


if __name__ == "__main__":
    main()
