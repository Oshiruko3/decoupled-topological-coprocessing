#!/usr/bin/env python3
"""
compile_pdf.py - Automated PDF compilation script for DTC papers (EN & JA)
Supports compiling v3 (default) or v2.
"""
import os
import sys
import subprocess
import shutil

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))

def check_command(cmd):
    return shutil.which(cmd) is not None

def run_command(args, cwd):
    print(f"Running: {' '.join(args)} in {cwd}")
    res = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    if res.returncode != 0:
        print(f"Error executing command: {' '.join(args)}")
        print(res.stdout[-1500:])
        return False
    return True

def clean_temp_files(target_dir):
    exts = ['.aux', '.log', '.out', '.toc', '.synctex.gz', '.fls', '.fdb_latexmk']
    for f in os.listdir(target_dir):
        for ext in exts:
            if f.endswith(ext):
                try:
                    os.remove(os.path.join(target_dir, f))
                except OSError:
                    pass

def compile_version(version="v3"):
    target_dir = os.path.join(PAPER_DIR, version)
    if not os.path.exists(target_dir):
        print(f"Error: Version directory {target_dir} not found.")
        return False, False

    tex_en = f"dtc_{version}_paper_en.tex"
    pdf_en = f"dtc_{version}_paper_en.pdf"
    tex_ja = f"dtc_{version}_paper_ja.tex"
    pdf_ja = f"dtc_{version}_paper_ja.pdf"

    print(f"\n==========================================")
    print(f" Compiling DTC {version.upper()} Papers (EN & JA)")
    print(f"==========================================")

    # Compile English (pdflatex)
    en_ok = False
    full_tex_en = os.path.join(target_dir, tex_en)
    full_pdf_en = os.path.join(target_dir, pdf_en)
    if os.path.exists(full_tex_en):
        if not check_command("pdflatex"):
            print("Error: pdflatex not found in PATH.")
        else:
            cmd_en = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_en]
            if run_command(cmd_en, cwd=target_dir) and run_command(cmd_en, cwd=target_dir):
                size = os.path.getsize(full_pdf_en)
                print(f"[+] SUCCESS: Generated {pdf_en} ({size:,} bytes)")
                en_ok = True

    # Compile Japanese (lualatex)
    ja_ok = False
    full_tex_ja = os.path.join(target_dir, tex_ja)
    full_pdf_ja = os.path.join(target_dir, pdf_ja)
    if os.path.exists(full_tex_ja):
        if not check_command("lualatex"):
            print("Error: lualatex not found in PATH.")
        else:
            cmd_ja = ["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex_ja]
            if run_command(cmd_ja, cwd=target_dir) and run_command(cmd_ja, cwd=target_dir):
                size = os.path.getsize(full_pdf_ja)
                print(f"[+] SUCCESS: Generated {pdf_ja} ({size:,} bytes)")
                ja_ok = True

    clean_temp_files(target_dir)
    return en_ok, ja_ok

def main():
    ver = sys.argv[1].lower() if len(sys.argv) > 1 else "v3"
    en_ok, ja_ok = compile_version(ver)

    print("\n==========================================")
    print(f" English ({ver.upper()}) Build:  {'SUCCESS' if en_ok else 'FAILED'}")
    print(f" Japanese ({ver.upper()}) Build: {'SUCCESS' if ja_ok else 'FAILED'}")
    print("==========================================")

    if not (en_ok and ja_ok):
        sys.exit(1)

if __name__ == "__main__":
    main()
