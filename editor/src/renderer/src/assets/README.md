# Asset thumbnail helpers

Material thumbnails reuse the production material preview sphere through a hidden streamed viewport. The preview is rendered at 2x the requested thumbnail resolution, with sky/fog/editor overlays disabled. The uniform clear background is flood-masked to transparent before the image is downsampled and encoded as PNG.

Thumbnail requests are serialized so browsing a folder with several materials does not create competing preview renders. The in-memory cache is keyed by material GUID, asset generation, and requested size; a new asset generation therefore invalidates the previous thumbnail automatically.
