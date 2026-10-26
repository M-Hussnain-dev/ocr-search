#!/usr/bin/env python3
"""
One-command setup for OCR Search.

    python setup.py            install everything and prepare the project
    python setup.py --run      same, then start the app
    python setup.py --test     same, then run the automated tests
    python setup.py --key      save your Gemini API key on this computer (never upload that file)
"""
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WANT_LANGS = ("eng", "urd")
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{}.traineddata"
WIN_TESS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def step(msg):
    print(f"\n==> {msg}")


def ask(q):
    try:
        return input(f"{q} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def find_tesseract():
    return shutil.which("tesseract") or (str(WIN_TESS) if WIN_TESS.exists() else None)


def install_python_packages():
    step("Installing Python libraries")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])


def install_tesseract():
    system = platform.system()
    if system == "Windows":
        print("Tesseract is not installed. Install it manually:\n"
              "  1. Download the installer: https://github.com/UB-Mannheim/tesseract/wiki\n"
              "  2. In 'Additional language data', tick Urdu.\n"
              "  3. Keep the default folder (C:\\Program Files\\Tesseract-OCR).\n"
              "  4. Run this setup again.")
        return False
    cmd = None
    if system == "Linux" and shutil.which("apt-get"):
        cmd = ["sudo", "apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng", "tesseract-ocr-urd"]
    elif system == "Darwin" and shutil.which("brew"):
        cmd = ["brew", "install", "tesseract", "tesseract-lang"]
    if cmd and ask(f"Tesseract is missing. Run `{' '.join(cmd)}` now?"):
        subprocess.call(cmd)
        return find_tesseract() is not None
    print("Please install Tesseract OCR (with English and Urdu data), then run setup again.")
    return False


def tess_langs(tess):
    out = subprocess.run([tess, "--list-langs"], capture_output=True, text=True)
    lines = (out.stdout + out.stderr).splitlines()
    m = re.search(r'in "(.+?)"', lines[0]) if lines else None
    return (Path(m.group(1)) if m else None), {l.strip() for l in lines[1:] if l.strip()}


def download(lang, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{lang}.traineddata"
    print(f"  downloading {lang}.traineddata ...")
    urllib.request.urlretrieve(TESSDATA_URL.format(lang), target)


def ensure_languages(tess):
    step("Checking OCR language data (English + Urdu)")
    sysdir, have = tess_langs(tess)
    missing = [l for l in WANT_LANGS if l not in have]
    if not missing:
        print("  all language data present")
        return
    try:
        for l in missing:
            download(l, sysdir)
    except Exception as e:
        print(f"  could not write to the system folder ({type(e).__name__}); using ./tessdata instead")
        for l in WANT_LANGS:
            download(l, ROOT / "tessdata")


def save_key():
    import getpass
    key = getpass.getpass("Paste your Gemini API key (hidden while typing): ").strip()
    if not key:
        print("No key entered.")
        return
    folder = ROOT / ".streamlit"
    folder.mkdir(exist_ok=True)
    f = folder / "secrets.toml"
    lines = [l for l in f.read_text().splitlines() if not l.startswith("GEMINI_API_KEY")] if f.exists() else []
    f.write_text("\n".join(lines + [f'GEMINI_API_KEY = "{key}"']) + "\n")
    print(f"Saved to {f}\nKeep this file on your computer only: never upload the .streamlit folder to GitHub.")


def main():
    if "--key" in sys.argv:
        save_key()
        return
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    install_python_packages()

    step("Checking Tesseract OCR")
    tess = find_tesseract() or (install_tesseract() and find_tesseract())
    if tess:
        print(f"  found: {tess}")
        ensure_languages(tess)
    else:
        print("  Tesseract still missing: the app will open but cannot read images until it is installed.")

    step("Preparing folders and database")
    sys.path.insert(0, str(ROOT))
    import database as db
    db.init()
    print("  ready")

    print("\nSetup finished. Start the app with:\n    streamlit run app.py\n")
    if "--test" in sys.argv:
        subprocess.call([sys.executable, "test_system.py"], cwd=ROOT)
    if "--run" in sys.argv:
        subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
