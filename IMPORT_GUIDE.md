# Import Guide for O3D Project

## Package Structure

The project is now organized into proper Python packages:

```
o3d/
├── pcd_processing/          # Point cloud processing
│   ├── __init__.py
│   ├── pcd_dataset.py
│   └── point_cloud_processor.py
├── planes_detection/        # Floor, ceiling, wall detection
│   ├── __init__.py
│   ├── detect_floor.py
│   ├── detect_ceiling.py
│   └── detect_walls.py
├── dimensions/              # Dimension calculations
│   ├── __init__.py
│   ├── finetune.py
│   ├── get_boundary_concave.py
│   ├── get_dimentions_concave.py
│   ├── get_dimensions_lines.py
│   ├── get_simplification.py
│   ├── manhattan_world_assumption.py
│   └── merge_walls.py
└── notebooks/               # Jupyter notebooks
```

## How to Import in Notebooks

### Step 1: Add parent directory to path (First cell)
```python
import sys
from pathlib import Path

# Add parent directory to path to enable package imports
parent_dir = Path('.').resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
```

### Step 2: Import packages (Second cell)
```python
# Point cloud processing
from pcd_processing import PointCloudDataset

# Plane detection
from planes_detection import detect_ceiling_correct, detect_floor, detect_walls
# Or import everything:
# from planes_detection import *

# Dimensions
from dimensions import *
# Or import specific functions:
# from dimensions import finetune, merge_walls, get_boundary_concave

# Standard libraries
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
```

## How to Import in Python Scripts

For Python scripts in the root directory or subdirectories:

```python
from pcd_processing import PointCloudDataset
from planes_detection import detect_ceiling_correct, detect_floor, detect_walls
from dimensions import finetune, merge_walls
```

## Cross-Package Imports (within package files)

When one package needs to import from another (e.g., `dimensions/finetune.py` importing from `planes_detection`), use absolute imports:

```python
# In dimensions/finetune.py
from planes_detection.detect_floor import *
from planes_detection.detect_walls import *
from dimensions.get_boundary_concave import *
```

**Note:** Use absolute imports (not relative imports like `..planes_detection`) to ensure compatibility with both direct execution and notebook imports.

## Files Updated

The following files have been updated to use the new package structure:
- ✅ All notebooks in `notebooks/` folder
- ✅ `dimensions/finetune.py`
- ✅ `s3dis/s3dis_run_experiment.py`
- ✅ Package `__init__.py` files created

## Template

See `notebooks/IMPORT_TEMPLATE.md` for a quick copy-paste template.
