![imgbox logo](frontend/src/assets/imgbox3.png)

![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white)
![MPS](https://img.shields.io/badge/Apple%20Silicon-MPS-black?logo=apple&logoColor=white)
[![GitHub stars](https://img.shields.io/github/stars/jrpll/imgbox?style=social)](https://github.com/jrpll/imgbox/stargazers)

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

## Clean-install test (Docker + GPU)

Emulate a brand-new user installing and running the web app from scratch — clean
image, real model download, real GPU inference. Full guide:
[docs/clean-install-test.md](docs/clean-install-test.md).

```bash
docker compose build                          # build the image (once)
docker compose run --rm --service-ports app   # run it, then open http://localhost:8080
```

## Todo

- [ ] identity matching avec une option "already cropped"
- [ ] add stop button 
- [ ] dissocier chaque field de chaque mode
- [ ] option pour matcher qu'avec les visages sains, après comparar tu peux mettre un filtre
- [ ] cropping automatique centré sur les visages

## Valider les changements

```bash
git add .
git commit -m ""
git push origin main
```

---

If imgbox is useful to you, a ⭐ goes a long way.