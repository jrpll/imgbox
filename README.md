![imgbox logo](frontend/src/assets/imgbox_2.png)

## Test app

Open UI

```bash
~/imgbox/src-tauri/target/debug/app
```

Launch side car

```bash
cd ~/imgbox/server && HUGGING_FACE_TOKEN=hf_yourtoken uv run uvicorn app:app --reload --port 8080
```

## Todo

- [ ] Settings UI — HuggingFace token input, local/remote mode switch
- [ ] Model loading screen — poll `/health` on startup, show progress
- [ ] PyInstaller — bundle FastAPI server into standalone binary (needed for local mode)
- [ ] GitHub Actions — build installers (.deb, .AppImage, .exe, .dmg) on release
- [ ] Code signing — Windows + macOS (requires paid certificates)