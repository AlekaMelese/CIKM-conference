#!/usr/bin/env python3
"""
Create an HTML gallery to view all SVG plots in a web browser

Usage:
    python create_svg_gallery.py
    python create_svg_gallery.py --plots-dir outputs/plots --output gallery.html
"""

import argparse
from pathlib import Path
from datetime import datetime


def create_svg_gallery(plots_dir="outputs/plots", output_file="svg_gallery.html"):
    """Create an HTML gallery of all SVG plots"""

    plots_dir = Path(plots_dir)

    if not plots_dir.exists():
        print(f"❌ Plots directory not found: {plots_dir}")
        return

    svg_files = sorted(plots_dir.glob("*.svg"))

    if not svg_files:
        print(f"❌ No SVG files found in {plots_dir}")
        return

    print(f"✓ Found {len(svg_files)} SVG plots")

    # Read SVG content
    svg_contents = []
    for svg_file in svg_files:
        with open(svg_file, 'r') as f:
            svg_content = f.read()
        svg_contents.append({
            'name': svg_file.name,
            'content': svg_content,
            'size': svg_file.stat().st_size / 1024  # KB
        })

    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemma 3 12B IT - Training Visualizations</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .stats {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            display: inline-block;
        }}

        .stats span {{
            margin: 0 15px;
            font-weight: bold;
        }}

        nav {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 3px solid #667eea;
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        nav ul {{
            list-style: none;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
        }}

        nav a {{
            display: block;
            padding: 12px 24px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 2px solid #667eea;
        }}

        nav a:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}

        .plot-section {{
            padding: 40px;
            border-bottom: 2px solid #e9ecef;
        }}

        .plot-section:last-child {{
            border-bottom: none;
        }}

        .plot-header {{
            margin-bottom: 20px;
        }}

        .plot-header h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 10px;
        }}

        .plot-info {{
            color: #6c757d;
            font-size: 0.95em;
        }}

        .svg-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .svg-container:hover {{
            transform: scale(1.01);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }}

        .svg-container svg {{
            max-width: 100%;
            height: auto;
        }}

        .download-btn {{
            display: inline-block;
            margin-top: 15px;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}

        .download-btn:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}

        footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 30px;
            font-size: 0.9em;
        }}

        footer a {{
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        .back-to-top {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            text-decoration: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }}

        .back-to-top:hover {{
            background: #764ba2;
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        }}

        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8em;
            }}

            nav ul {{
                flex-direction: column;
                align-items: center;
            }}

            .plot-section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Gemma 3 12B IT Fine-tuning</h1>
            <p>Medical Discharge Summary Generation - Training Visualizations</p>
            <div class="stats">
                <span>📊 {len(svg_files)} Plots</span>
                <span>📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </header>

        <nav>
            <ul>
"""

    # Add navigation links
    for i, svg_data in enumerate(svg_contents, 1):
        plot_name = svg_data['name'].replace('.svg', '').replace('_', ' ').title()
        html_content += f'                <li><a href="#plot{i}">{i}. {plot_name}</a></li>\n'

    html_content += """            </ul>
        </nav>

        <main>
"""

    # Add plot sections
    for i, svg_data in enumerate(svg_contents, 1):
        plot_name = svg_data['name'].replace('.svg', '').replace('_', ' ').title()

        html_content += f"""
        <section class="plot-section" id="plot{i}">
            <div class="plot-header">
                <h2>Plot {i}: {plot_name}</h2>
                <p class="plot-info">File: {svg_data['name']} | Size: {svg_data['size']:.2f} KB</p>
            </div>
            <div class="svg-container">
                {svg_data['content']}
            </div>
            <div style="text-align: center;">
                <a href="#{i+1 if i < len(svg_contents) else 'top'}" class="download-btn">
                    {'Next Plot →' if i < len(svg_contents) else '↑ Back to Top'}
                </a>
            </div>
        </section>
"""

    html_content += """
        </main>

        <footer>
            <p>Generated by <a href="https://github.com/anthropics/claude-code" target="_blank">Claude Code</a></p>
            <p>Fine-tuned using <a href="https://github.com/unslothai/unsloth" target="_blank">Unsloth</a> on Gemma 3 12B IT</p>
            <p style="margin-top: 10px; opacity: 0.8;">Medical AI Research - 5000 Samples Dataset</p>
        </footer>
    </div>

    <a href="#top" class="back-to-top">↑</a>

    <script>
        // Smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Show/hide back to top button
        const backToTop = document.querySelector('.back-to-top');
        window.addEventListener('scroll', () => {
            if (window.scrollY > 300) {
                backToTop.style.opacity = '1';
            } else {
                backToTop.style.opacity = '0';
            }
        });
    </script>
</body>
</html>
"""

    # Write HTML file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✓ HTML gallery created: {output_path.absolute()}")
    print(f"📊 Total plots: {len(svg_files)}")
    print(f"\n🌐 Open in browser: file://{output_path.absolute()}")

    return str(output_path.absolute())


def main():
    parser = argparse.ArgumentParser(description="Create HTML gallery of SVG plots")
    parser.add_argument('--plots-dir', type=str, default='outputs/plots',
                       help='Path to plots directory (default: outputs/plots)')
    parser.add_argument('--output', type=str, default='svg_gallery.html',
                       help='Output HTML file (default: svg_gallery.html)')

    args = parser.parse_args()

    create_svg_gallery(plots_dir=args.plots_dir, output_file=args.output)


if __name__ == "__main__":
    main()
