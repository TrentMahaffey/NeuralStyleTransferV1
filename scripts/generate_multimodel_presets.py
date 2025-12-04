#!/usr/bin/env python3
"""Generate expanded multi-model presets for 10 categories with 20-30 presets each."""

import sqlite3
import itertools
import random

# Available resources
MAGENTA_STYLES = [
    '/app/models/magenta_styles/canyon.jpg',
    '/app/models/magenta_styles/starry_night.jpg',
    '/app/models/magenta_styles/rainbow.jpg',
    '/app/models/magenta_styles/atoms.jpg',
    '/app/models/magenta_styles/style_rainforest.jpg',
    '/app/models/magenta_styles/dunes2.jpg',
    '/app/models/magenta_styles/frame.jpg',
    '/app/models/magenta_styles/style_gpt.jpg',
    '/app/models/magenta_styles/gpt_style2.jpg',
    '/app/models/magenta_styles/gpt_style3.jpg',
    '/app/models/magenta_styles/gptstyle4.jpg',
    '/app/models/magenta_styles/mountain_geo.jpg',
]

PYTORCH_MODELS = [
    '/app/models/pytorch/candy.pth',
    '/app/models/pytorch/mosaic.pth',
    '/app/models/pytorch/rain_princess.pth',
    '/app/models/pytorch/udnie.pth',
]

TORCH7_MODELS = [
    '/app/models/torch/starry_night_eccv16.t7',
    '/app/models/torch/the_scream.t7',
    '/app/models/torch/composition_vii_eccv16.t7',
    '/app/models/torch/la_muse_eccv16.t7',
]

TILE_SIZES = [256, 384, 512, 768, 1024]
REGION_COUNTS = [2, 3, 4, 5, 6]
FEATHER_VALUES = [10, 20, 30, 50]

def get_style_name(path):
    """Extract short name from style path."""
    name = path.split('/')[-1].replace('.jpg', '').replace('.png', '').replace('.pth', '').replace('.t7', '')
    return name.replace('_', ' ').title()

def short_name(path):
    """Get very short name for combining."""
    name = path.split('/')[-1].split('.')[0]
    mappings = {
        'canyon': 'Canyon',
        'starry_night': 'Starry',
        'rainbow': 'Rainbow',
        'atoms': 'Atoms',
        'style_rainforest': 'Forest',
        'dunes2': 'Dunes',
        'frame': 'Frame',
        'style_gpt': 'GPT1',
        'gpt_style2': 'GPT2',
        'gpt_style3': 'GPT3',
        'gptstyle4': 'GPT4',
        'mountain_geo': 'Mountain',
        'candy': 'Candy',
        'mosaic': 'Mosaic',
        'rain_princess': 'Rain',
        'udnie': 'Udnie',
        'starry_night_eccv16': 'VanGogh',
        'the_scream': 'Scream',
        'composition_vii_eccv16': 'Kandinsky',
        'la_muse_eccv16': 'Muse',
    }
    return mappings.get(name, name.title())

def create_categories(conn):
    """Create new multi-model categories."""
    categories = [
        (10, 'Blob Morph', 'Organic blob morphing region effects', 10),
        (11, 'Tentacle Morph', 'Dynamic tentacle morphing patterns', 11),
        (12, 'Wave Morph', 'Flowing wave morphing effects', 12),
        (13, 'Pulse Morph', 'Pulsing radial morph animations', 13),
        (14, 'Voronoi Static', 'Static voronoi multi-style patterns', 14),
        (15, 'Radial Patterns', 'Radial burst multi-style effects', 15),
        (16, 'Spiral Patterns', 'Spiral and concentric patterns', 16),
        (17, 'Grid Patterns', 'Grid and diagonal style divisions', 17),
        (18, 'Resolution Mix', 'Mixed resolution magenta styles', 18),
        (19, 'Model Mix', 'Combining different model types', 19),
    ]

    cursor = conn.cursor()
    for cat_id, name, desc, sort_order in categories:
        cursor.execute("""
            INSERT OR REPLACE INTO categories (id, name, description, sort_order)
            VALUES (?, ?, ?, ?)
        """, (cat_id, name, desc, sort_order))
    conn.commit()
    return {name: cat_id for cat_id, name, _, _ in categories}

def generate_blob_morph_presets(conn, cat_id):
    """Generate 25 blob morph presets."""
    presets = []
    cursor = conn.cursor()

    # Different style combinations with blob morph
    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 3))
    random.shuffle(combos)

    for i, (s1, s2, s3) in enumerate(combos[:25]):
        name = f"Blob {short_name(s1)}-{short_name(s2)}-{short_name(s3)}"
        desc = f"Blob morph with {short_name(s1)}, {short_name(s2)}, {short_name(s3)}"

        # Vary the morph speed and amplitude
        morph_speed = random.choice([0.5, 1.0, 1.5, 2.0])
        morph_amp = random.choice([0.1, 0.15, 0.2, 0.25])
        morph_freq = random.choice([2.0, 3.0, 4.0])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 magenta_style_c, magenta_tile_c, magenta_overlap_c,
                                 region_mode, region_count, region_feather, region_morph)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, ?, 512, 64, 'voronoi', 3, 30, ?)
        """, (name, desc, cat_id, s1, s2, s3, f"{morph_speed},{morph_amp},{morph_freq},blob"))
        presets.append(name)

    conn.commit()
    return presets

def generate_tentacle_morph_presets(conn, cat_id):
    """Generate 25 tentacle morph presets."""
    presets = []
    cursor = conn.cursor()

    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 4))
    random.shuffle(combos)

    for i, (s1, s2, s3, s4) in enumerate(combos[:25]):
        name = f"Tentacle {short_name(s1)}-{short_name(s2)}-{short_name(s3)}-{short_name(s4)}"[:60]
        desc = f"Tentacle morph quad pattern"

        morph_speed = random.choice([0.8, 1.0, 1.2, 1.5])
        morph_amp = random.choice([0.12, 0.15, 0.18, 0.22])
        morph_freq = random.choice([2.5, 3.0, 3.5, 4.0])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 magenta_style_c, magenta_tile_c, magenta_overlap_c,
                                 magenta_style_d, magenta_tile_d, magenta_overlap_d,
                                 region_mode, region_count, region_feather, region_morph)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, ?, 512, 64, ?, 512, 64, 'voronoi', 4, 25, ?)
        """, (name, desc, cat_id, s1, s2, s3, s4, f"{morph_speed},{morph_amp},{morph_freq},tentacle"))
        presets.append(name)

    conn.commit()
    return presets

def generate_wave_morph_presets(conn, cat_id):
    """Generate 25 wave morph presets."""
    presets = []
    cursor = conn.cursor()

    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 2))
    random.shuffle(combos)

    for i, (s1, s2) in enumerate(combos[:25]):
        name = f"Wave {short_name(s1)}-{short_name(s2)}"
        desc = f"Wave morph duo with {short_name(s1)} and {short_name(s2)}"

        morph_speed = random.choice([0.6, 0.8, 1.0, 1.2])
        morph_amp = random.choice([0.15, 0.2, 0.25, 0.3])
        morph_freq = random.choice([1.5, 2.0, 2.5, 3.0])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 region_mode, region_count, region_feather, region_morph)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, 'waves', 2, 40, ?)
        """, (name, desc, cat_id, s1, s2, f"{morph_speed},{morph_amp},{morph_freq},wave"))
        presets.append(name)

    conn.commit()
    return presets

def generate_pulse_morph_presets(conn, cat_id):
    """Generate 25 pulse morph presets."""
    presets = []
    cursor = conn.cursor()

    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 3))
    random.shuffle(combos)

    for i, (s1, s2, s3) in enumerate(combos[:25]):
        name = f"Pulse {short_name(s1)}-{short_name(s2)}-{short_name(s3)}"
        desc = f"Pulse morph radial pattern"

        morph_speed = random.choice([0.5, 0.7, 1.0, 1.3])
        morph_amp = random.choice([0.1, 0.15, 0.2])
        morph_freq = random.choice([2.0, 2.5, 3.0])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 magenta_style_c, magenta_tile_c, magenta_overlap_c,
                                 region_mode, region_count, region_feather, region_morph)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, ?, 512, 64, 'radial', 3, 35, ?)
        """, (name, desc, cat_id, s1, s2, s3, f"{morph_speed},{morph_amp},{morph_freq},pulse"))
        presets.append(name)

    conn.commit()
    return presets

def generate_voronoi_static_presets(conn, cat_id):
    """Generate 30 static voronoi presets."""
    presets = []
    cursor = conn.cursor()
    preset_idx = 0

    # 2-style, 3-style, 4-style, 5-style, 6-style combinations
    for count in [2, 3, 4, 5, 6]:
        combos = list(itertools.combinations(MAGENTA_STYLES[:10], count))
        random.shuffle(combos)

        for styles in combos[:6]:  # 6 presets per count = 30 total
            preset_idx += 1
            names = [short_name(s) for s in styles]
            name = f"VS{preset_idx} {count}x " + "-".join(names[:2])
            if len(names) > 2:
                name += f"+{len(names)-2}"
            name = name[:60]
            desc = f"Static {count}-style voronoi pattern"

            feather = random.choice([15, 25, 35, 45])

            # Build insert with dynamic number of styles
            style_cols = ['magenta_style', 'magenta_style_b', 'magenta_style_c', 'magenta_style_d'][:count]
            tile_cols = ['magenta_tile', 'magenta_tile_b', 'magenta_tile_c', 'magenta_tile_d'][:count]
            overlap_cols = ['magenta_overlap', 'magenta_overlap_b', 'magenta_overlap_c', 'magenta_overlap_d'][:count]

            cols = ['name', 'description', 'category_id', 'model_type', 'region_mode', 'region_count', 'region_feather']
            vals = [name, desc, cat_id, 'magenta', 'voronoi', count, feather]

            for i, style in enumerate(styles[:4]):
                cols.append(style_cols[i] if i < len(style_cols) else f'magenta_style_{chr(98+i)}')
                vals.append(style)
                if i < len(tile_cols):
                    cols.append(tile_cols[i])
                    vals.append(512)
                    cols.append(overlap_cols[i])
                    vals.append(64)

            placeholders = ', '.join(['?'] * len(vals))
            col_str = ', '.join(cols)

            cursor.execute(f"INSERT INTO presets ({col_str}) VALUES ({placeholders})", vals)
            presets.append(name)

    conn.commit()
    return presets

def generate_radial_patterns_presets(conn, cat_id):
    """Generate 25 radial pattern presets."""
    presets = []
    cursor = conn.cursor()

    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 3))
    random.shuffle(combos)

    for i, (s1, s2, s3) in enumerate(combos[:25]):
        name = f"Radial {short_name(s1)}-{short_name(s2)}-{short_name(s3)}"
        desc = f"Radial burst pattern"

        feather = random.choice([20, 30, 40, 50])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 magenta_style_c, magenta_tile_c, magenta_overlap_c,
                                 region_mode, region_count, region_feather)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, ?, 512, 64, 'radial', 3, ?)
        """, (name, desc, cat_id, s1, s2, s3, feather))
        presets.append(name)

    conn.commit()
    return presets

def generate_spiral_patterns_presets(conn, cat_id):
    """Generate 25 spiral pattern presets."""
    presets = []
    cursor = conn.cursor()

    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 2))
    random.shuffle(combos)

    modes = ['spiral', 'concentric']

    for i, (s1, s2) in enumerate(combos[:25]):
        mode = modes[i % 2]
        name = f"{mode.title()} {short_name(s1)}-{short_name(s2)}"
        desc = f"{mode.title()} pattern with two styles"

        feather = random.choice([25, 35, 45])
        count = random.choice([2, 3, 4])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 region_mode, region_count, region_feather)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, ?, ?, ?)
        """, (name, desc, cat_id, s1, s2, mode, count, feather))
        presets.append(name)

    conn.commit()
    return presets

def generate_grid_patterns_presets(conn, cat_id):
    """Generate 25 grid pattern presets."""
    presets = []
    cursor = conn.cursor()

    combos = list(itertools.combinations(MAGENTA_STYLES[:8], 4))
    random.shuffle(combos)

    modes = ['grid', 'diagonal']

    for i, (s1, s2, s3, s4) in enumerate(combos[:25]):
        mode = modes[i % 2]
        name = f"{mode.title()} {short_name(s1)}-{short_name(s2)}-{short_name(s3)}-{short_name(s4)}"[:60]
        desc = f"{mode.title()} pattern with four styles"

        feather = random.choice([15, 25, 35])

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                 magenta_style_c, magenta_tile_c, magenta_overlap_c,
                                 magenta_style_d, magenta_tile_d, magenta_overlap_d,
                                 region_mode, region_count, region_feather)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, ?, 512, 64, ?, 512, 64, ?, 512, 64, ?, 4, ?)
        """, (name, desc, cat_id, s1, s2, s3, s4, mode, feather))
        presets.append(name)

    conn.commit()
    return presets

def generate_resolution_mix_presets(conn, cat_id):
    """Generate 25 resolution mix presets - same style at different resolutions."""
    presets = []
    cursor = conn.cursor()

    for style in MAGENTA_STYLES[:8]:
        # Generate 3 presets per style with different resolution combos
        for res_combo in [(256, 512, 1024), (384, 768, 1024), (256, 512, 768)]:
            style_name = short_name(style)
            res_str = "-".join([f"{r}px" for r in res_combo])
            name = f"ResMix {style_name} {res_str}"[:60]
            desc = f"{style_name} at resolutions {res_combo}"

            cursor.execute("""
                INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                     magenta_style_b, magenta_tile_b, magenta_overlap_b,
                                     magenta_style_c, magenta_tile_c, magenta_overlap_c,
                                     region_mode, region_count, region_feather)
                VALUES (?, ?, ?, 'magenta', ?, ?, 32, ?, ?, 64, ?, ?, 128, 'voronoi', 3, 30)
            """, (name, desc, cat_id, style, res_combo[0], style, res_combo[1], style, res_combo[2]))
            presets.append(name)

            if len(presets) >= 25:
                break
        if len(presets) >= 25:
            break

    conn.commit()
    return presets

def generate_model_mix_presets(conn, cat_id):
    """Generate 25 mixed model type presets (magenta + pytorch/torch7)."""
    presets = []
    cursor = conn.cursor()

    # Mix magenta with pytorch models
    magenta_pytorch_combos = list(itertools.product(MAGENTA_STYLES[:6], PYTORCH_MODELS))
    random.shuffle(magenta_pytorch_combos)

    for i, (mag_style, pytorch_model) in enumerate(magenta_pytorch_combos[:15]):
        name = f"Mix {short_name(mag_style)}-{short_name(pytorch_model)}"
        desc = f"Magenta + PyTorch model blend"

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 model_b_type, model_b_path, io_preset_b,
                                 region_mode, region_count, region_feather, blend_models_weights)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, 'transformer', ?, 'auto', 'voronoi', 2, 25, '0.5,0.5')
        """, (name, desc, cat_id, mag_style, pytorch_model))
        presets.append(name)

    # Mix magenta with torch7 models
    magenta_torch7_combos = list(itertools.product(MAGENTA_STYLES[:6], TORCH7_MODELS))
    random.shuffle(magenta_torch7_combos)

    for i, (mag_style, torch7_model) in enumerate(magenta_torch7_combos[:10]):
        name = f"Mix {short_name(mag_style)}-{short_name(torch7_model)}"
        desc = f"Magenta + Torch7 model blend"

        cursor.execute("""
            INSERT INTO presets (name, description, category_id, model_type, magenta_style, magenta_tile, magenta_overlap,
                                 model_b_type, model_b_path, io_preset_b,
                                 region_mode, region_count, region_feather, blend_models_weights)
            VALUES (?, ?, ?, 'magenta', ?, 512, 64, 'torch7', ?, 'auto', 'voronoi', 2, 25, '0.5,0.5')
        """, (name, desc, cat_id, mag_style, torch7_model))
        presets.append(name)

    conn.commit()
    return presets

def main():
    random.seed(42)  # Reproducible results

    conn = sqlite3.connect('presets.db')

    # Create new categories
    print("Creating new categories...")
    cat_ids = create_categories(conn)

    # Generate presets for each category
    generators = [
        ('Blob Morph', generate_blob_morph_presets),
        ('Tentacle Morph', generate_tentacle_morph_presets),
        ('Wave Morph', generate_wave_morph_presets),
        ('Pulse Morph', generate_pulse_morph_presets),
        ('Voronoi Static', generate_voronoi_static_presets),
        ('Radial Patterns', generate_radial_patterns_presets),
        ('Spiral Patterns', generate_spiral_patterns_presets),
        ('Grid Patterns', generate_grid_patterns_presets),
        ('Resolution Mix', generate_resolution_mix_presets),
        ('Model Mix', generate_model_mix_presets),
    ]

    total_presets = 0
    for cat_name, generator in generators:
        print(f"\nGenerating {cat_name} presets...")
        presets = generator(conn, cat_ids[cat_name])
        print(f"  Created {len(presets)} presets")
        total_presets += len(presets)

    conn.close()

    print(f"\n{'='*50}")
    print(f"Total presets created: {total_presets}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
