#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/scripts')
from morph_v2 import detect_faces

image = sys.argv[1] if len(sys.argv) > 1 else '/app/input/self_style_samples/IMG_4160.jpeg'
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

faces = detect_faces(image, confidence_threshold=threshold)
print(f"Found {len(faces)} faces (threshold={threshold}):")
for f in faces:
    x, y, w, h = f['bbox']
    fid = f['id']
    conf = f['confidence']
    cov = f['coverage']
    print(f"  Face #{fid}: {w}x{h} at ({x},{y}) - {cov:.2f}% coverage, {conf:.2f} confidence")
