#!/usr/bin/env python3
"""
compile_pdf.py - Automated PDF compilation script for DTC v2.0 paper (EN & JA)
"""
import os
import sys
import subprocess
import shutil

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))

def check_command(cmd):
    return shutil.which(cmd) is not None

def run_command(args, cwd=PAPER_DIR):
    print(f"Running: {' '.join(args)}")
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

def compile_english():
    tex_file = "dtc_v2_paper_en.tex"
    pdf_file = "dtc_v2_paper_en.pdf"
    full_tex = os.path.join(PAPER_DIR, tex_file)
    full_pdf = os.path.join(PAPER_DIR, pdf_file)
    
    if not os.path.exists(full_tex):
        print(f"Error: {tex_file} not found.")
        return False
    
    print("\n==========================================")
    print(" Compiling English Paper (pdflatex, 2 passes)...")
    print("==========================================")
    
    compiler = "pdflatex"
    if not check_command(compiler):
        print(f"Error: {compiler} not found in PATH.")
        return False
        
    cmd = [compiler, "-interaction=nonstopmode", "-halt-on-error", tex_file]
    
    # 2 passes for references and labels
    if run_command(cmd) and run_command(cmd):
        size = os.path.getsize(full_pdf)
        print(f"SUCCESS: Generated {pdf_file} ({size} bytes)")
        
        return True
    return False

def compile_japanese():
    tex_file = "dtc_v2_paper_ja.tex"
    pdf_file = "dtc_v2_paper_ja.pdf"
    full_tex = os.path.join(PAPER_DIR, tex_file)
    full_pdf = os.path.join(PAPER_DIR, pdf_file)
    
    if not os.path.exists(full_tex):
        print(f"Error: {tex_file} not found.")
        return False
    
    print("\n==========================================")
    print(" Compiling Japanese Paper (lualatex, 2 passes)...")
    print("==========================================")
    
    compiler = "lualatex"
    if not check_command(compiler):
        print(f"Error: {compiler} not found in PATH.")
        return False
        
    cmd = [compiler, "-interaction=nonstopmode", "-halt-on-error", tex_file]
    
    # 2 passes for references and labels
    if run_command(cmd) and run_command(cmd):
        size = os.path.getsize(full_pdf)
        print(f"SUCCESS: Generated {pdf_file} ({size} bytes)")
        
        return True
    return False

def main():
    en_ok = compile_english()
    ja_ok = compile_japanese()
    
    print("\n==========================================")
    print(f" English Build:  {'SUCCESS' if en_ok else 'FAILED'}")
    print(f" Japanese Build: {'SUCCESS' if ja_ok else 'FAILED'}")
    print("==========================================")
    
    if not (en_ok and ja_ok):
        sys.exit(1)

if __name__ == "__main__":
    main()
