# TODO

- [x] Inspect runtime entrypoints for image-to-image: `backend/routes/image_routes.py` and `backend/services/image2image_service.py`.
- [x] Add required runtime logging to `backend/routes/image_routes.py` (route called, service module, local img2img message).
- [x] Add runtime log to `backend/services/image2image_service.py` to confirm local module import.
- [x] Search entire project for any remaining HuggingFace API / remote inference strings (`api-inference.huggingface.co`, `HF_IMG2IMG_URL`, `requests.post(`, `instruct-pix2pix` remote logic) => none found in repo text files.

- [x] Kill old python/Flask processes and restart server (best-effort; python.exe may not be present in task list).

- [ ] Click Transform in browser and verify terminal logs match expected output.
- [ ] If browser still shows `api-inference.huggingface.co`, trace the exact remaining runtime code path and remove it completely.
- [ ] Re-run the full verification loop until NO remote HuggingFace API usage remains for image-to-image.

