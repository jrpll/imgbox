![imgbox logo](frontend/src/assets/imgbox_2.png)

## Testing

Start backend

```bash
cd server && uv run uvicorn app:app --reload --port 8080
```

Start the UI

```bash
cd frontend && npm run dev
```

## Build frontend for single command launch

```bash
npm run build
```

Then we can just do
```bash
uv run python app.py
```

## Todo

- [ ] identity matching avec une option "already cropped"
- [ ] add advanced settings
- [ ] add stop button 
- [ ] drop folder working pour identity
- [ ] dissocier chaque field de chaque mode
- [ ] option pour matcher qu'avec les visages sains, après comparar tu peux mettre un filtre
- [ ] cropping automatique centré sur les visages

## Valider les changements

```bash
git add .
git commit -m ""
git push origin main
```