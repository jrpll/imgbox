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

- [ ] Ajouter clear cache option dans le panel de droite
- [ ] Renommer les items dans le drop down "edit" -> "sd3 + gommette edit", ainsi que les routes
- [ ] Mettre des pastilles i à côté des champs pour donner des indications
- [ ] dark theme ?

## Valider les changements

```bash
git add .
git commit -m ""
git push origin main
```