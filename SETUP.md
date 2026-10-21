# Running the website (Windows / Mac)

You only need Python. Everything else is already in this folder.

## 1. Install Python (once)

Download Python 3.12 from https://www.python.org/downloads/ and run the installer.

- **Windows:** tick **"Add python.exe to PATH"** on the first screen, then Install.
- **Mac:** open the downloaded `.pkg` and click through.

## 2. Open a terminal in this folder

- **Windows:** open the folder in File Explorer, click the address bar, type `cmd`, press Enter.
- **Mac:** right-click the folder in Finder, choose **New Terminal at Folder**.

## 3. Start the website

**Windows:**
```
python -m http.server 8765 --directory site
```

**Mac:**
```
python3 -m http.server 8765 --directory site
```

Open http://localhost:8765 in your browser. Press `Ctrl+C` in the terminal to stop.

This shows the full site and the 3-D demo. Two extras are off in this mode:
the "Live model" toggle (upload your own photo) and hover previews of the dataset photos.

## Optional: turn on the live model and photo previews

This needs a one-time download of about 1 GB and an internet connection.

**Windows:**
```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision numpy pillow
python scripts\serve_site.py
```

**Mac:**
```
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy pillow
python scripts/serve_site.py
```

Open http://127.0.0.1:8765. The first start also downloads a small ImageNet model, so it can take a minute.

## If something goes wrong

- `python is not recognized` (Windows): reinstall Python and tick "Add to PATH", then open a new terminal.
- `Address already in use`: change `8765` to `9000` in the command and in the browser address.
- Blank page: make sure the terminal is open **in this folder** (it should contain a `site` folder when you type `dir` on Windows or `ls` on Mac).
